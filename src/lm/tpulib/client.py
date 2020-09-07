# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# Lint as: python3
"""Cloud TPU Client."""

from __future__ import absolute_import, division, print_function

import datetime
import json
import logging
import os
import time
from concurrent import futures

from absl import flags

from six.moves.urllib import request
from six.moves.urllib.error import HTTPError

_GOOGLE_API_CLIENT_INSTALLED = True
try:
    from googleapiclient import discovery  # pylint: disable=g-import-not-at-top
    from oauth2client import client  # pylint: disable=g-import-not-at-top
except ImportError:
    _GOOGLE_API_CLIENT_INSTALLED = False

_GKE_ENV_VARIABLE = "KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS"
_ENDPOINTS_SEPARATOR = ","
_DEFAULT_ENV_VARIABLE = "TPU_NAME"
_DISCOVERY_SERVICE_URL_ENV_VARIABLE = "TPU_API_DISCOVERY_URL"
_GCE_METADATA_ENDPOINT = "http://metadata.google.internal"
_DEFAULT_ENDPOINT_PORT = "8470"
_OOM_EVENT_COOL_TIME_SEC = 90
_VERSION_SWITCHER_ENDPOINT = "http://{}:8475/requestversion"


def _utcnow():
    """A wrapper function around datetime.datetime.utcnow.

  This function is created for unit testing purpose. It's not easy to do
  StubOutWithMock with datetime.datetime package.

  Returns:
    datetime.datetime
  """
    return datetime.datetime.utcnow()


def _environment_discovery_url():
    return os.environ.get(_DISCOVERY_SERVICE_URL_ENV_VARIABLE)


def _request_compute_metadata(path):
    req = request.Request(
        "%s/computeMetadata/v1/%s" % (_GCE_METADATA_ENDPOINT, path),
        headers={"Metadata-Flavor": "Google"},
    )
    resp = request.urlopen(req)
    return _as_text(resp.read())


def _environment_var_to_network_endpoints(endpoints):
    """Yields a dict with ip address and port."""
    for endpoint in endpoints.split(","):
        grpc_prefix = "grpc://"
        if endpoint.startswith(grpc_prefix):
            endpoint = endpoint.split(grpc_prefix)[1]
        parts = endpoint.split(":")
        ip_address = parts[0]
        port = _DEFAULT_ENDPOINT_PORT
        if len(parts) > 1:
            port = parts[1]
        yield {"ipAddress": ip_address, "port": port}


def _get_tpu_name(tpu):
    if tpu:
        return tpu

    for e in [_GKE_ENV_VARIABLE, _DEFAULT_ENV_VARIABLE]:
        if e in os.environ:
            return os.environ[e]
    return None


def _as_text(s):
    if isinstance(s, bytes):
        return s.decode("utf-8")
    return s


class Client(object):
    """Client for working with the Cloud TPU API.

  This client is intended to be used for resolving tpu name to ip addresses.

  It's recommended to use this library as a contextlib to utilize all
  functionality.
  """

    def __init__(
        self,
        tpu=None,
        zone=None,
        project=None,
        credentials="default",
        service=None,
        discovery_url=None,
    ):
        if isinstance(tpu, list):
            if not tpu:
                raise ValueError("At least one TPU must be specified.")
            if len(tpu) != 1:
                raise NotImplementedError(
                    "Using multiple TPUs in a single session is not yet implemented"
                )
            tpu = tpu[0]

        tpu = _get_tpu_name(tpu)

        if tpu is None:
            raise ValueError("Please provide a TPU Name to connect to.")

        self._tpu = _as_text(tpu)

        self._use_api = not self._tpu.startswith("grpc://")
        self._service = service

        self._credentials = None
        self._project = None
        self._zone = None
        self._discovery_url = None
        if self._use_api:
            if credentials != "default":
                self._credentials = credentials
            # Automatically detect project and zone if unspecified.
            if project:
                self._project = _as_text(project)
            else:
                self._project = _request_compute_metadata("project/project-id")
            if zone:
                self._zone = _as_text(zone)
            else:
                zone_path = _request_compute_metadata("instance/zone")
                self._zone = zone_path.split("/")[-1]
            self._discovery_url = _environment_discovery_url() or discovery_url

    def _symptom_msg(self, msg):
        """Return the structured Symptom message."""
        return "Symptom: " + msg

    def _oom_event(self):
        """Check if a runtime OOM event is reported."""
        symptoms = self.symptoms()
        if not symptoms:
            return False
        for symptom in reversed(symptoms):
            if symptom["symptomType"] != "OUT_OF_MEMORY":
                continue
            oom_datetime_str = symptom["createTime"].split(".")[0]
            oom_datetime = datetime.datetime.strptime(
                oom_datetime_str, "%Y-%m-%dT%H:%M:%S"
            )
            time_diff = _utcnow() - oom_datetime
            if time_diff < datetime.timedelta(seconds=_OOM_EVENT_COOL_TIME_SEC):
                logging.warning(
                    self._symptom_msg(
                        "a recent runtime OOM has occurred ~{} seconds ago. The model "
                        "script will terminate automatically. To prevent future OOM "
                        "events, please consider reducing the model size. To disable this "
                        "behavior, set flag --runtime_oom_exit=false when starting the "
                        "script.".format(time_diff.seconds)
                    )
                )
                return True
        return False

    def _tpu_service(self):
        """Creates a new Cloud TPU API object.

    This works around an issue where the underlying HTTP connection sometimes
    times out when the script has been running for too long. Other methods in
    this object call this method to get a new API object whenever they need
    to communicate with the Cloud API.

    Raises:
      RuntimeError: If the dependent Python packages are missing.

    Returns:
      A Google Cloud TPU API object.
    """
        if self._service:
            return self._service

        if not _GOOGLE_API_CLIENT_INSTALLED:
            raise RuntimeError(
                "Missing runtime dependency on the Google API client. "
                "Run `pip install cloud-tpu-client` to fix."
            )

        credentials = self._credentials
        if credentials is None or credentials == "default":
            credentials = client.GoogleCredentials.get_application_default()

        if self._discovery_url:
            return discovery.build(
                "tpu",
                "v1",
                credentials=credentials,
                discoveryServiceUrl=self._discovery_url,
                cache_discovery=False,
            )
        else:
            return discovery.build(
                "tpu", "v1", credentials=credentials, cache_discovery=False
            )

    def _full_name(self):
        """Returns the full Cloud name for this TPU."""
        return "projects/%s/locations/%s/nodes/%s" % (
            self._project,
            self._zone,
            self._tpu,
        )

    def _fetch_cloud_tpu_metadata(self):
        """Returns the TPU metadata object from the TPU Get API call."""
        service = self._tpu_service()
        try:
            r = service.projects().locations().nodes().get(name=self._full_name())
            return r.execute()
        except Exception as e:
            raise ValueError(
                "Could not lookup TPU metadata from name '%s'. Please "
                "doublecheck the tpu argument in the TPUClusterResolver "
                "constructor. Exception: %s" % (self._tpu, e)
            )

    def _get_tpu_property(self, key):
        if self._use_api:
            metadata = self._fetch_cloud_tpu_metadata()
            return metadata.get(key)

        return None

    def __enter__(self):
        self._open = True

    def __exit__(self, type, value, traceback):  # pylint: disable=redefined-builtin
        del type, value, traceback

    def recoverable(self):
        """Returns true if the TPU is in a state where training should eventually resume.

    If false the TPU is in a unrecoverable state and should be recreated.
    """
        state = self.state()
        if state and state in ["TERMINATED", "PREEMPTED"]:
            return False
        elif FLAGS.runtime_oom_exit and self._oom_event():
            return False
        return True

    def symptoms(self):
        """Return Cloud TPU Symptoms of the TPU."""
        return self._get_tpu_property("symptoms")

    def state(self):
        """Return state of the TPU."""
        return self._get_tpu_property("state")

    def health(self):
        """Return health of the TPU."""
        return self._get_tpu_property("health")

    def runtime_version(self):
        """Return runtime version of the TPU."""

        if not self._use_api:
            # Fallback on getting version directly from TPU.
            url = _VERSION_SWITCHER_ENDPOINT.format(
                self.network_endpoints()[0]["ipAddress"]
            )
            try:
                req = request.Request(url)
                resp = request.urlopen(req)
                version_details = json.loads(resp.read())
                return version_details.get("currentVersion")
            except HTTPError as e:
                status_code = e.code
                if status_code == 404:
                    return None
                else:
                    raise e
        return self._get_tpu_property("tensorflowVersion")

    def accelerator_type(self):
        """Return accelerator type of the TPU."""
        return self._get_tpu_property("acceleratorType")

    def api_available(self):
        """Return if the Cloud TPU API is available, if not certain features will not work."""
        return self._use_api

    def name(self):
        """Return the name of the tpu, or the ip address if name is not provided."""
        return self._tpu

    def get_local_ip(self):
        """Return the local ip address of the Google Cloud VM the workload is running on."""
        return _request_compute_metadata("instance/network-interfaces/0/ip")

    def network_endpoints(self):
        """Return a list of tpu endpoints."""
        if not self._use_api:
            return list(_environment_var_to_network_endpoints(self._tpu))
        response = self._fetch_cloud_tpu_metadata()

        if response.get("state") != "READY":
            raise RuntimeError(
                'TPU "%s" is not yet ready; state: "%s"'
                % (self._tpu, response.get("state"))
            )
        if "networkEndpoints" in response:
            return response["networkEndpoints"]
        else:
            return [{"ipAddress": response["ipAddress"], "port": response["port"]}]

    def wait_for_healthy(self, timeout_s=1200, interval=30):
        """Wait for TPU to become healthy or raise error if timeout reached.

    Args:
      timeout_s (int): The timeout in seconds for waiting TPU to become healthy.
      interval (int): The interval in seconds to poll the TPU for health.

    Raises:
      RuntimeError: If the TPU doesn't become healthy by the timeout.
    """
        timeout = time.time() + timeout_s
        while self.health() != "HEALTHY":
            logging.warning(
                (
                    'Waiting for TPU "%s" with state "%s" '
                    'and health "%s" to become healthy'
                ),
                self.name(),
                self.state(),
                self.health(),
            )
            if time.time() + interval > timeout:
                raise RuntimeError(
                    'Timed out waiting for TPU "%s" to become healthy' % self.name()
                )
            time.sleep(interval)

        logging.warning('TPU "%s" is healthy.', self.name())

    def configure_tpu_version(self, version, restart_type="always"):
        """Configure TPU software version.

    Args:
      version (string): Version of software to configure the TPU with.
      restart_type (string): Restart behavior when switching versions,
        defaults to always restart. Options are 'always', 'ifNeeded'.

    """

        def configure_worker(worker):
            """Configure individual TPU worker.

      Args:
        worker: A dict with the field ipAddress where the configure request will
          be sent.
      """
            ip_address = worker["ipAddress"]
            url = (_VERSION_SWITCHER_ENDPOINT + "/{}?restartType={}").format(
                ip_address, version, restart_type
            )
            req = request.Request(url, data=b"")
            try:
                request.urlopen(req)
            except HTTPError as e:
                status_code = e.code
                if status_code == 404:
                    raise Exception(
                        "Tensorflow version {} is not available on Cloud TPU, "
                        "try a previous nightly version or refer to "
                        "https://cloud.google.com/tpu/docs/release-notes for "
                        "the latest official version.".format(version)
                    )
                else:
                    raise Exception("Failed to configure worker {}".format(ip_address))

        workers = self.network_endpoints()

        with futures.ThreadPoolExecutor(max_workers=len(workers)) as executor:
            results = executor.map(configure_worker, workers)
            for result in results:
                if result:
                    result.result()


# originally from https://github.com/shawwn/tpunicorn/blob/master/tpunicorn/tpu.py
# Copyright (C) 2020 Shawn Presser. All rights reserved. MIT license.


def parse_tpu_project(tpu):
    fqn = tpu if isinstance(tpu, str) else tpu["name"]
    return fqn.split("/")[-5]


def parse_tpu_zone(tpu):
    fqn = tpu if isinstance(tpu, str) else tpu["name"]
    return fqn.split("/")[-3]


def parse_tpu_id(tpu):
    fqn = tpu if isinstance(tpu, str) else tpu["name"]
    return fqn.split("/")[-1]


def parse_tpu_index(tpu):
    fqn = tpu if isinstance(tpu, str) else tpu["name"]
    idx = re.findall(r"([0-9]+)$", fqn)
    if len(idx) <= 0:
        idx = -1
    else:
        idx = int(idx[0])
    return idx


def parse_tpu_network(tpu):
    net = tpu if isinstance(tpu, str) else tpu["network"]
    return net.split("/")[-1]


import google.auth


def _determine_default_project(project=None):
    """Determine default project ID explicitly or implicitly as fall-back.
    See :func:`google.auth.default` for details on how the default project
    is determined.
    :type project: str
    :param project: Optional. The project name to use as default.
    :rtype: str or ``NoneType``
    :returns: Default project if it can be determined.
    """
    if project is None:
        _, project = google.auth.default()
    return project


@ring.lru(expire=3600)  # cache default project for an hour
def get_default_project(project=None):
    return _determine_default_project(project=project)


@ring.lru(expire=3600)  # cache tpu zones for an hour
def get_tpu_zones(project=None):
    zones = (
        api.projects()
        .locations()
        .list(name="projects/" + get_default_project(project=project))
        .execute()
        .get("locations", [])
    )
    return [zone["locationId"] for zone in zones]


@ring.lru(expire=15)  # cache tpu info for 15 seconds
def fetch_tpus(zone=None, project=None):
    if zone is None:
        zones = get_tpu_zones(project=project)
    if isinstance(zone, str):
        zones = zone.split(",")
    tpus = []
    for zone in zones:
        results = list_tpus(zone)
        tpus.extend(results)
    return tpus


def list_tpus(zone, project=None):
    if "/" not in zone:
        zone = "projects/" + get_default_project(project=project) + "/locations/" + zone
    tpus = (
        api.projects().locations().nodes().list(parent=zone).execute().get("nodes", [])
    )
    return list(sorted(tpus, key=parse_tpu_index))


def get_tpus(zone=None, project=None):
    tpus = fetch_tpus(zone=zone, project=project)
    if zone is None:
        return tpus
    else:
        return [tpu for tpu in tpus if "/{}/".format(zone) in tpu["name"]]


def get_tpu(tpu, zone=None, project=None, silent=False):
    if isinstance(tpu, dict):
        tpu = parse_tpu_id(tpu)
    if isinstance(tpu, str) and re.match("^[0-9]+$", tpu):
        tpu = int(tpu)
    if isinstance(tpu, int):
        which = "index"
        tpus = [
            x for x in get_tpus(zone=zone, project=project) if parse_tpu_index(x) == tpu
        ]
    else:
        which = "id"
        tpus = [
            x for x in get_tpus(zone=zone, project=project) if parse_tpu_id(x) == tpu
        ]
    if len(tpus) > 1:
        raise ValueError(
            "Multiple TPUs matched {} {!r}. Try specifying --zone".format(which, tpu)
        )
    if len(tpus) <= 0:
        if silent:
            return None
        raise ValueError("No TPUs matched {} {!r}".format(which, tpu))
    return tpus[0]


from string import Formatter


class NamespaceFormatter(Formatter):
    def __init__(self, namespace={}):
        Formatter.__init__(self)
        self.namespace = namespace

    def get_value(self, key, args, kwds):
        if isinstance(key, str):
            try:
                # Check explicitly passed arguments first
                return kwds[key]
            except KeyError:
                return self.namespace[key]
        else:
            return Formatter.get_value(key, args, kwds)


from collections import defaultdict


@ring.lru(expire=1)  # seconds
def format_widths():
    headers = format_headers()
    tpus = get_tpus()
    r = defaultdict(int)
    for tpu in tpus:
        args = _format_args(tpu)
        for k, v in args.items():
            s = "{}".format(v)
            r[k + "_w"] = max(r[k + "_w"], len(s) + 1, len(headers[k]) + 1)
    return r


def _normalize_tpu_isodate(iso):
    r = re.findall("(.*[.][0-9]{6})[0-9]*Z", iso)
    if len(r) > 0:
        return r[0] + "Z"
    raise ValueError("Could not parse TPU date {!r}".format(iso))


import datetime
import time

import moment


def get_timestamp(timestamp=None, utc=True):
    if timestamp is None:
        timestamp = time.time()
    # https://stackoverflow.com/a/52606421/9919772
    # dt = datetime.datetime.fromtimestamp(timestamp).astimezone()
    dt = moment.unix(timestamp, utc=utc)
    dt = dt.timezone(current_tzname())
    return dt.strftime("%m-%d-%Y %I:%M:%S%p %Z")


def current_timezone():
    if time.daylight:
        return datetime.timezone(
            datetime.timedelta(seconds=-time.altzone), time.tzname[1]
        )
    else:
        return datetime.timezone(
            datetime.timedelta(seconds=-time.timezone), time.tzname[0]
        )


def current_tzname():
    return current_timezone().tzname(None)


def since(iso):
    dt = moment.utcnow() - moment.utc(
        _normalize_tpu_isodate(iso), "%Y-%m-%dT%H:%M:%S.%fZ"
    )
    return dt.total_seconds()


def minutes_since(iso):
    return since(iso) / 60


def hours_since(iso):
    return since(iso) / 3600


def days_since(iso):
    return since(iso) / 86400


def nice_since(iso):
    t = int(since(iso))
    m = (t // 60) % 60
    h = (t // 3600) % 24
    d = t // 86400
    r = []
    out = False
    if d > 0 or out:
        out = True
        r += ["{:02d}d".format(d)]
    else:
        r += ["   "]
    if h > 0 or out:
        out = True
        r += ["{:02d}h".format(h)]
    else:
        r += ["   "]
    if m > 0 or out:
        out = True
        r += ["{:02d}m".format(m)]
    else:
        r += ["   "]
    return "".join(r)


def format_headers():
    return {
        "kind": "header",
        "project": "PROJECT",
        "zone": "ZONE",
        "id": "ID",
        "fqn": "FQN",
        "ip": "IP",
        "port": "PORT",
        "master": "MASTER",
        "range": "RANGE",
        "type": "TYPE",
        "created": "CREATED",
        "age": "AGE",
        "preemptible": "PREEMPTIBLE?",
        "status": "STATUS",
        "health": "HEALTH",
        "index": "INDEX",
        "version": "VERSION",
        "network": "NETWORK",
    }


def _format_args(tpu):
    return {
        "kind": "tpu",
        "project": parse_tpu_project(tpu),
        "zone": parse_tpu_zone(tpu),
        "id": parse_tpu_id(tpu),
        "fqn": tpu["name"],
        "ip": parse_tpu_ip(tpu),
        "port": tpu["port"],
        "master": parse_tpu_master(tpu),
        "range": parse_tpu_range(tpu),
        "type": parse_tpu_type(tpu),
        "created": tpu["createTime"],
        "age": nice_since(tpu["createTime"]),
        "preemptible": "yes" if parse_tpu_preemptible(tpu) else "no",
        "status": tpu["state"],
        "health": tpu.get("health", "UNKNOWN"),
        "index": parse_tpu_index(tpu),
        "version": parse_tpu_version(tpu),
        "network": parse_tpu_network(tpu),
    }


def parse_tpu_preemptible(tpu):
    return tpu.get("schedulingConfig", {"preemptible": False}).get("preemptible", False)


def parse_tpu_ip(tpu):
    return tpu.get("ipAddress", "")


def parse_tpu_master(tpu):
    return "{}:{}".format(tpu.get("ipAddress", ""), tpu.get("port", 8470))


def parse_tpu_range(tpu):
    return tpu.get("cidrBlock", None)


def parse_tpu_version(tpu):
    return tpu["tensorflowVersion"]


def parse_tpu_type(tpu):
    return tpu["acceleratorType"]


def parse_tpu_description(tpu):
    return tpu.get("description", None)


def format_args(tpu):
    r = _format_args(tpu)
    r.update(format_widths())
    return r


def get_default_format_specs(thin=False):
    specs = [
        "{zone:{zone_w}}",
        "{index:<{index_w}}",
        "{type:{type_w}}",
        "{age:{age_w}}",
        "{id:{id_w}}",
        "{status:{status_w}}",
        "{health:{health_w}}",
        "{version:{version_w}}",
        "{network:{network_w}}",
        "{master:{master_w}}",
        "{range:{range_w}}",
        "{preemptible!s:{preemptible_w}}",
    ]
    if thin:
        return ["{" + re.findall("{([^:]+)[:]", x)[0] + "}" for x in specs]
    else:
        return specs


def get_default_format_spec(thin=False):
    return " ".join(get_default_format_specs(thin=thin))


def format(tpu, spec=None, formatter=NamespaceFormatter):
    if tpu.get("kind", "tpu") == "tpu":
        args = format_args(tpu)
    else:
        args = {}
        args.update(tpu)
        args.update(format_widths())
    args = {k: v if v is not None else "" for k, v in args.items()}
    fmt = formatter(args)
    if spec is None:
        spec = get_default_format_spec(thin=len(format_widths()) == 0)
    return fmt.format(spec)


def create_tpu_command(
    tpu, zone=None, project=None, version=None, description=None, preemptible=None
):
    if zone is None:
        zone = parse_tpu_zone(tpu)
    if project is None:
        project = parse_tpu_project(tpu)
    if version is None:
        version = parse_tpu_version(tpu)
    if description is None:
        description = parse_tpu_description(tpu)
    if preemptible is None:
        preemptible = True if parse_tpu_preemptible(tpu) else None
    return build_commandline(
        "gcloud compute tpus create",
        parse_tpu_id(tpu),
        zone=zone,
        project=project,
        network=parse_tpu_network(tpu),
        range=parse_tpu_range(tpu),
        version=version,
        accelerator_type=parse_tpu_type(tpu),
        preemptible=preemptible,
        description=description,
    )


def delete_tpu_command(tpu, zone=None, project=None):
    if zone is None:
        zone = parse_tpu_zone(tpu)
    if project is None:
        project = parse_tpu_project(tpu)
    return build_commandline(
        "gcloud compute tpus delete",
        parse_tpu_id(tpu),
        zone=zone,
        project=project,
        quiet=True,
    )


def start_tpu_command(tpu, zone=None, project=None):
    if zone is None:
        zone = parse_tpu_zone(tpu)
    if project is None:
        project = parse_tpu_project(tpu)
    return build_commandline(
        "gcloud compute tpus start",
        parse_tpu_id(tpu),
        zone=zone,
        project=project,
        quiet=True,
    )


def stop_tpu_command(tpu, zone=None, project=None):
    if zone is None:
        zone = parse_tpu_zone(tpu)
    if project is None:
        project = parse_tpu_project(tpu)
    return build_commandline(
        "gcloud compute tpus stop",
        parse_tpu_id(tpu),
        zone=zone,
        project=project,
        quiet=True,
    )


def reimage_tpu_command(tpu, zone=None, project=None, version=None):
    if zone is None:
        zone = parse_tpu_zone(tpu)
    if project is None:
        project = parse_tpu_project(tpu)
    if version is None:
        version = parse_tpu_version(tpu)
    return build_commandline(
        "gcloud compute tpus reimage",
        parse_tpu_id(tpu),
        zone=zone,
        project=project,
        version=version,
        quiet=True,
    )
