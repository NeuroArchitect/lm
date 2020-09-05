import collections
import functools
import inspect
from typing import Dict

from absl import logging

REGISTRY = collections.defaultdict(dict)
CLASS_NAMES = set()


def register(cls, kind, name, config):
    assert not (name in REGISTRY), "%s with that name already present" % kind
    REGISTRY[kind][name] = cls

    @functools.wraps(cls)
    def wrapper(*args, **kwds):
        return cls(*args, **kwds)

    return wrapper


def register_multi(cls, kind, names):
    for name in names:
        assert not (name in REGISTRY), "%s with that name already present" % kind
        if inspect.isfunction(cls):
            factory = lambda *args, **kwds: cls
        REGISTRY[kind][name] = factory

    @functools.wraps(cls)
    def wrapper(*args, **kwds):
        return cls(*args, **kwds)

    return wrapper


def register_task(name, config=None):
    logging.debug("task %s registered", name)
    return functools.partial(register, kind="tasks", name=name, config=config)


def register_infeed(name, config=None):
    logging.debug("infeed %s registered", name)
    return functools.partial(register, kind="infeeds", name=name, config=config)


def register_dataset(name, config=None):
    logging.debug("dataset %s registered", name)
    return functools.partial(register, kind="datasets", name=name, config=config)


def register_encoder(name, config=None):
    logging.debug("encoders %s registered", name)
    return functools.partial(register, kind="encoders", name=name, config=config)


def register_model(name, config=None):
    logging.debug("model %s registered", name)
    return functools.partial(register, kind="models", name=name, config=config)


def register_parser(name, *fmts):
    names = ["%s:%s" % (name, ext) for ext in fmts]
    return functools.partial(register_multi, kind="parsers", names=names)


def model_from_config(config: Dict):
    model = config["kind"]
    return REGISTRY["models"][model](**config)


def infeed_from_config(config: Dict):
    model = config["kind"]
    return REGISTRY["infeeds"][model](**config)


def dataset_from_config(config: Dict):
    model = config["kind"]
    return REGISTRY["datasets"][model](**config)


def parser_from_config(config: Dict):
    model = config["kind"]
    return REGISTRY["parsers"][model](**config)


def get_object(group, config):
    if isinstance(config, dict):
        kind = config.get("kind", None)
        if kind is None:
            raise ValueError(
                'invalid task configuration. "kind" key was not found in dictionary'
            )
    elif isinstance(config, str):
        kind = config
        config = {}
    else:
        kind = config.kind
        config = dict(config)
    return REGISTRY[group][kind](**config)


get_dataset = functools.partial(get_object, "datasets")
get_model = functools.partial(get_object, "models")
get_task = functools.partial(get_object, "tasks")
get_infeed = functools.partial(get_object, "infeeds")
get_parser = functools.partial(get_object, "parsers")
