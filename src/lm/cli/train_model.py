import json
import os
from datetime import datetime
from typing import Dict

import tensorflow as tf
import tensorflow.compat.v1 as v1
from absl import app, logging
from absl.flags import argparse_flags
from pydantic.dataclasses import dataclass
from pydantic import BaseModel

import lm.config
import lm.datasets
import lm.devices
import lm.infeeds
import lm.models
import lm.training
import lm.optimizers


def parse_args(args, parser=None):
    # Parse command line arguments
    parser.add_argument(
        "trainspec",
        type=str,
        help="the json file specifiing the configuration for this run",
    )
    parser.add_argument(
        "--save-settings",
        type=str,
        help="freeze and save the final configuration settings.",
    )
    parser.add_argument(
        "--check-infeed",
        action="store_true",
        default=False,
        help="creates the and fetches once from the infeed",
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Name of the device to train on, (TPUv3-8, v3-32, etc if any)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="model path to write the checkpoints",
    )
    parser.add_argument("--steps", type=int, help="max steps to run the train")
    parser.add_argument(
        "--dataset", type=str, help="location to a dataset jsonnet configuration file."
    )
    parser.add_argument(
        "--task", type=str, help="location to a task jsonnet configuration file."
    )
    parser.add_argument("--id", type=str)

    # parser.add_argument(
    #     "--project",
    #     default=None,
    #     help="Project name for the Cloud TPU-enabled project. If not specified, we "
    #     "will attempt to automatically detect the GCE project from metadata.")

    # parser.add_argument(
    #     "--zone",
    #     default=None,
    #     help="GCE zone where the Cloud TPU is located in. If not specified, we "
    #     "will attempt to automatically detect the GCE project from metadata.")


def local_parse_args(args):
    parser = argparse_flags.ArgumentParser()
    parse_args(args, parser)
    return parser.parse_args(args[1:])


class ScheduleConfig(BaseModel):
    steps: int = 10
    steps_per_checkpoint: int = 100
    steps_per_iteration: int = 100

@dataclass
class ExperimentConfig:
    infeed: Dict
    model: Dict
    schedule: ScheduleConfig
    trainer: lm.training.TrainerConfig
    model_path: str
    task: Dict


class Experiment:
    def __init__(self, config: ExperimentConfig):
        self.config = config

    @property
    def model(self):
        return self.config.model

    @property
    def infeed(self):
        return self.config.infeed

    @property
    def schedule(self):
        return self.config.schedule

    def load_model(self):
        return lm.get_model(self.config.model)

    def load_infeed(self):
        return lm.get_infeed(self.config.infeed)

    def load_trainer(self):
        model = self.load_model()
        infeed = self.load_infeed()
        schedule = self.schedule
        return lm.training.Trainer(infeed, model, schedule, **dict(self.config.trainer))

    def train(self, checkpoint_path):
        trainer = self.load_trainer()
        j = trainer.create_train_jobspec(checkpoint_path)
        j.train = True
        trainer.execute(j)

    def check_infeed(self):
        trainer = self.load_trainer()

        steps = self.config.schedule.steps

        logging.info("running for %d steps", steps)
        with v1.Session(graph=tf.Graph()) as sess:
            ds = trainer.infeed(self.config.infeed)
            #     "batch_size": trainer.infeed.config.batch_size,
            #     "lem_sequence": tainer.infeed.config.,
            # })

            it = ds.make_one_shot_iterator()
            example = it.get_next()
            for i in range(steps):
                try:
                    result = sess.run(example)
                    if i % 1000 == 0:
                        logging.info("%d/%d: %r", i, steps, result)
                except tf.errors.OutOfRangeError:
                    logging.error(
                        "dataset ended prematurely after only %d of the %d expected steps",
                        i,
                        steps,
                    )


def main(args):
    logging.info("started train process")

    dataset = args.dataset
    # TODO: fix this logic with proper dataset loading
    if dataset:
        if tf.io.gfile.exists(dataset) and not tf.io.gfile.isdir(dataset):
            dataset = os.path.split(args.dataset)[0]
        dataset = os.path.join(dataset, "*.tfrecord*")
    logging.info('use %s as dataset path', dataset)
    settings = lm.config.load(args.trainspec, dataset=dataset)

    # patch config
    if args.task:
        settings["task"] = lm.config.load(args.task)

        # settings["infeed"]["dataset"] = dscfg
        # settings["infeed"]["file_pattern"] = ds_location
        # settings["infeed"]["max_sequence_length"] = dscfg["max_sequence_length"]

    trainer_config = lm.training.TrainerConfig(**settings["trainer"])
    if args.device:
        trainer_config.use_tpu(address=args.device)

    if args.steps:
        settings["schedule"]["steps"] = args.steps

    if args.model_path:
        model_path = args.model_path
    else:
        model_path = settings['model_path']

    if args.id is None:
        dt = datetime.utcnow()
        timestamp = int(dt.timestamp())

        tstamp = dt.strftime("%Y%b%d-" + str(timestamp))
        model_path = os.path.join(model_path, tstamp)
    else:
        model_path = os.path.join(model_path, args.id)

    logging.info("final config %r", settings)

    if args.save_settings:
        runspec = args.save_settings
    else:
        dt = datetime.now().strftime("%Y%M%d_%H%M%S")
        runspec = "run-%s.json" % dt

    logging.info("generating run settings %s", runspec)
    with tf.io.gfile.GFile(runspec, "w") as fd:
        json.dump(settings, fd, indent=2)

    # reload the settings from the saved configuration
    settings = lm.config.load(runspec)

    exconfig = ExperimentConfig(**settings)
    experiment = Experiment(exconfig)

    if args.check_infeed:
        experiment.check_infeed()

    experiment.train(model_path)

    # train
    logging.info("completed train process")


if __name__ == "__main__":
    tf.disable_v2_behavior()
    app.run(main, flags_parser=local_parse_args)
