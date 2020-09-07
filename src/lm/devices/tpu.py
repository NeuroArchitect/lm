"TPU Configuration Module"
import ipaddress
from typing import Any, Callable, Dict, Optional, Union

import tensorflow as tf
from absl import logging
from pydantic.dataclasses import dataclass
from tensorflow.python.tpu import tpu_config, tpu_estimator

from tensorflow_estimator.python.estimator import estimator as estimator_lib

import mesh_tensorflow as mtf

from enum import Enum

class Device:
    pass

@dataclass
class TPUConfig:
    address: Optional[Union[str, ipaddress.IPv4Address]] = None
    num_cores: int = 8

@dataclass
class TPUInfeedSpec:
    batch_size: int
    function: Callable[[Dict[str, Any]], Any]
    params: Dict

class TPUPrecision(str, Enum):
    float32 = 'float32'
    bfloat16 = 'bfloat16'

@dataclass
class TPUJobSpec:
    # steps_per_iteration: int
    # steps_per_checkpoint: int
    max_steps: int
    model_path: str
    function: Callable[[Dict[str, Any]], Any]
    params: Dict
    infeed: TPUInfeedSpec
    train: bool = False
    test: bool = False
    predict: bool = False
    use_tpu: bool = False
    export: Optional[str] = None
    signature: Optional[Callable] = None
    precision: TPUPrecision = TPUPrecision


class TPU:
    def __init__(self, config: TPUConfig):
        self.config = config
        self._cluster = None

    def resolve(self):
        if not self.config.address:
            return

        if not self._cluster:
            self._cluster = tf.distribute.cluster_resolver.TPUClusterResolver(
                tpu=self.config.address
            )
        return self._cluster

    def check_connection(self):
        pass

    def execute(self, job: TPUJobSpec):
        "execute the give job spec"
        cluster = self.resolve()
        mesh_shape = mtf.convert_to_shape(job.function.mesh_shape)
        # layout_rules = mtf.convert_to_layout_rules(self.mesh_layout)
        
        my_tpu_config = tpu_config.TPUConfig(
            tpu_job_name=None,
            num_shards=mesh_shape.size,
            iterations_per_loop=job.params["steps_per_iteration"],
            num_cores_per_replica=1, # SIMD Mesh abstractiong
            per_host_input_for_training=tpu_config.InputPipelineConfig.BROADCAST,
        )

        run_config = tpu_config.RunConfig(
            cluster=cluster,
            model_dir=job.model_path,
            save_checkpoints_steps=None,  # Disable the default saver
            save_checkpoints_secs=None,  # Disable the default saver
            log_step_count_steps=job.params["steps_per_iteration"],
            save_summary_steps=job.params["steps_per_checkpoint"],
            tpu_config=my_tpu_config,
        )

        transformer_model = job.function
        # tpu_estimator_model_fn

        estimator = tpu_estimator.TPUEstimator(
            model_fn=transformer_model,
            model_dir=job.model_path, # same as run_config
            config=run_config,
            params=job.params,
            use_tpu=job.use_tpu,
            train_batch_size=job.infeed.batch_size,  # these change with the configuration
            eval_batch_size=job.infeed.batch_size,
            predict_batch_size=job.infeed.batch_size,
            batch_axis=None,
            eval_on_tpu=True,
            export_to_tpu=True,
            export_to_cpu=True,
            warm_start_from=None,
            embedding_config_spec=None,
        )

        assert job.train or job.eval

        if job.train:
            if tf.io.gfile.exists(job.model_path):
                logging.info("restoring checkpoint steps from %s", job.model_path)
                current_step = int(
                    estimator_lib._load_global_step_from_checkpoint_dir(job.model_path)
                )
                logging.info("current step is now at %d", current_step)
            else:
                current_step = 0

            from tqdm import auto as tqdm
            pbar = tqdm.tqdm(total=job.max_steps)
            while current_step < job.max_steps:
                steps_between_checkpoints = job.params['steps_per_checkpoint']
                next_step = min(current_step + steps_between_checkpoints, job.max_steps)
                logging.info("running train for %d steps", next_step)
                estimator.train(input_fn=job.infeed.function, max_steps=next_step)
                current_step = int(
                    estimator_lib._load_global_step_from_checkpoint_dir(job.model_path)
                )
                pbar.update(current_step)
                logging.info("step %s", current_step)

            logging.info("completed device execution after %s steps", current_step)

            if job.export:
                estimator.export_saved_model(
                         export_dir_base=job.export,
                         serving_input_receiver_fn=job.signature,
                         assets_extra=None,
                         as_text=False,
                         checkpoint_path=None,
                         experimental_mode=tf.estimator.ModeKeys.PREDICT)

            return {"current_step": current_step}

        if job.eval:
            # If eval is on - stop and eval every ckpt
            logging.info("starting to evaluate.")
            eval_results = estimator.evaluate(
                input_fn=job.infeed.function, steps=job.max_steps
            )
            logging.info("completed eval. results: %s", eval_results)
            return eval_results
