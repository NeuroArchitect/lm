import datetime
from contextlib import ContextDecorator, ExitStack
from typing import Dict

import tensorflow as tf
import tensorflow.compat.v1 as v1
from absl import app, logging
from absl.flags import argparse_flags
from pydantic import BaseModel

import lm
from lm.devices.tpu import TPUConfig, TPUInfeedSpec, TPUJobSpec, TPU


def serving_input_receiver_fn():
    feature = v1.placeholder(tf.int32, shape=[None, 8], name="tokens")
    return tf.estimator.export.TensorServingInputReceiver(feature, feature)


class TrainerConfig(BaseModel):
    device: Dict
    optimizer: Dict
    learning_rate: Dict
    weight_decay: float = 0.1
    gradient_clipping: float = 0.5
    def use_tpu(self, address):
        self.device = TPUConfig(address=address)


class Trainer:
    def __init__(self, infeed, model, schedule, **kwds):
        self.__dict__.update(dict(TrainerConfig(**kwds)))
        self.infeed = infeed
        self.model = model
        self._optimizer = None
        self.schedule = schedule

    def save_checkpoint(self):
        state = self.model.state_dict()
        logging.info("saving model checkpoint to %s", self.config.ckpt_path)
        self.save(state, self.config.ckpt_path)
        logging.info("saved model checkpoint to %s", self.config.ckpt_path)

    def create_train_jobspec(self, checkpoint_path):
        return TPUJobSpec(
            function=self.model,
            params={
                "optimizer": self.optimizer,
                "n_tokens": self.infeed.config.n_tokens,
                "len_sequence": self.infeed.config.len_sequence,
                "n_channels": self.model.config.n_channels,
                "total_steps": self.schedule.steps,
                "steps_per_checkpoint": self.schedule.steps_per_checkpoint,
                "steps_per_iteration": self.schedule.steps_per_iteration,
                "checkpoint_dir": checkpoint_path,
                "precision": 'float32',
                "learning_rate": self.learning_rate,
                "predict_batch_size": 8,
                # **self.config.runspec.optimizer,
                # **self.config.runspec.learning_rate,
            },
            max_steps=self.schedule.steps,
            use_tpu=self.device.get("kind", "cpu") == "tpu",
            model_path=checkpoint_path,
            # steps_per_iteration=self.config.schedule.steps_per_iteration,
            # steps_per_checkpoint=self.config.schedule.steps_per_checkpoint,
            infeed=TPUInfeedSpec(
                batch_size=self.infeed.config.batch_size,
                function=self.infeed,
                params={},
            ),
            export=None, #FIXME: disabled for now
            signature=serving_input_receiver_fn,
        )

    def create_export_jobspec(self):
        model = self.load_model()
        infeed = self.load_infeed()
        EOS = 1

        return TPUJobSpec(
            function=model,
            params={
                # patch legacy config
                "opt_name": self.config.runspec.optimizer["name"],
                "train_steps": self.config.schedule.steps,
                "steps_per_checkpoint": self.config.schedule.steps_per_checkpoint,
                "steps_per_iteration": self.config.schedule.steps_per_iteration,
                "model_path": self.config.model_path,
                "stop_at_token": EOS,
                "learning_rate": self.config.learning_rate,
                **self.config.runspec.optimizer,
                **self.config.runspec.learning_rate,
            },
            max_steps=self.config.schedule.steps,
            use_tpu=self.config.device.get("kind", "cpu") == "tpu",
            model_path=self.config.model_path,
            # steps_per_iteration=self.config.schedule.steps_per_iteration,
            # steps_per_checkpoint=self.config.schedule.steps_per_checkpoint,
            infeed=TPUInfeedSpec(
                batch_size=infeed.config.batch_size,
                function=infeed.train,
                params={},
            ),
            export="/tmp/export",
            signature=serving_input_receiver_fn,
        )

    def execute(self, jobspec):
        # if self.device is None:
        #     self.device = lm.devices.from_config(self.config.device)
        return TPU(TPUConfig()).execute(jobspec)
        #return self.device.execute(jobspec)
