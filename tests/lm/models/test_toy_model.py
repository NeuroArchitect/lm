import multiprocessing
from typing import Callable

import mesh_tensorflow as mtf
import numpy as np
import tensorflow as tf
from absl import logging
from absl.testing import absltest, parameterized
from pydantic.dataclasses import dataclass
from tensorflow.python.framework import test_util
from tensorflow.python.tpu import (  # pylint: disable=g-direct-tensorflow-import
    tpu_config,
    tpu_estimator,
)

import lm.examples
import lm.models.toy as toy
import lm.tf
from tensorflow_estimator.python.estimator import estimator as estimator_lib


class MockContext:
    num_hosts: int = 2


class TestToyTransformerModel(parameterized.TestCase, tf.test.TestCase):

    #     @parameterized.parameters(
    # #        ('bfloat16', ),
    #         ('float32', ),
    #     )
    #     @test_util.run_v1_only(reason='is better')
    #     def test_as_dataset(self, precision):
    #         batch = 2
    #         seq = 4
    #         channels = 4
    #         n_tokens = 10

    #         # seq 2 seq
    #         inputs = tf.ones([batch, seq ], dtype=tf.int32)
    #         outputs = tf.ones([batch, seq ], dtype=tf.float32)

    #         cfg = toy.ToyTransformerConfig(
    #             mesh_shape="all:1",
    #             mesh_layout="",
    #             use_bias=True,
    #             optimizer="SGD",
    #             learning_rate=2e-5,
    #             num_hidden_layers=2,
    #         )

    #         model_builder = toy.ToyTransformer(config=cfg)
    #         mode = tf.estimator.ModeKeys.PREDICT
    #         context = MockContext()
    #         params = {
    #             "batch_size": batch,
    #             "tokens_size": 1,
    #             "n_tokens": n_tokens,
    #             "n_sequences": seq,
    #             "n_channels": channels,
    #             "use_tpu": False,
    #             "context": context,
    #             "precision": precision,
    #             "checkpoint_dir": '',
    #         }

    #         estimator, lowering = model_builder(inputs, outputs, mode, params=params)

    #         # print(tf_model_graph)
    #         # actual_outputs = lowering.export_to_tf_tensor(estimator)

    #         # expected_outputs = tf.cast(tf.not_equal(inputs, 0), tf.float32)
    #         # expected = tf.identity(tf_model_graph, name=mode)
    #         init = tf.global_variables_initializer()
    #         self.evaluate(init)
    #         tf_group = lowering.copy_masters_to_slices()
    #         self.evaluate(tf_group)
    #         results = self.evaluate([estimator.predictions['predictions']])
    #         expected = (batch, seq, n_tokens)
    #         self.assertEqual(results[0].shape, expected)

    @parameterized.parameters(("float32",),)
    @test_util.run_v1_only(reason="is better")
    def test_can_train(self, precision):
        batch = 2
        seq = 4
        channels = 4
        n_tokens = 10

        def build_model_fn(params):

            cfg = toy.ToyTransformerConfig(
                mesh_shape=params["mesh_shape"],
                mesh_layout=params["mesh_layout"],
                use_bias=True,
                optimizer="SGD",
                learning_rate=2e-5,
                num_hidden_layers=2,
            )

            model_builder = toy.ToyTransformer(config=cfg)
            return model_builder

            # mode = tf.estimator.ModeKeys.PREDICT
            # context = MockContext()
            # params = {
            #     "batch_size": batch,
            #     "tokens_size": 1,
            #     "n_tokens": n_tokens,
            #     "n_sequences": seq,
            #     "n_channels": channels,
            #     "use_tpu": False,
            #     # "context": context,
            #     "precision": precision,
            #     "checkpoint_dir": '/tmp',
            # }

            # params.update(**params)

            # estimator_spec, lowering = model_builder(features, labels, mode, params=params)
            # return estimator_spec

            # return wrapper

        def input_fn(batch_size):
            def simplegen():
                for i in range(batch_size):
                    yield lm.examples.Seq2SeqSimpleExample(
                        np.ones(8, dtype=np.int64) * i, np.zeros(8, dtype=np.int64)
                    ).serialize()

            ds = lm.tf.from_generator(lambda: simplegen)
            ds = ds.batch(batch_size)
            ds_batched = ds.cache().shuffle(buffer_size=50_000).batch(batch_size)
            epochs_between_evals = 1
            # Iterate through the dataset a set number (`epochs_between_evals`) of times
            # during each training session.
            ds = ds_batched.repeat(epochs_between_evals)
            return ds

        # expected_outputs = tf.cast(tf.not_equal(inputs, 0), tf.float32)
        # expected = tf.identity(tf_model_graph, name=mode)
        # init = tf.global_variables_initializer()
        # self.evaluate(init)
        # tf_group = lowering.copy_masters_to_slices()
        # self.evaluate(tf_group)
        # results = self.evaluate([estimator.predictions['predictions']])
        # expected = (batch, seq, n_tokens)
        # self.assertEqual(results[0].shape, expected)

        # """Run a toy model on TPU."""
        @dataclass
        class RunConfig:
            mesh_shape: str
            mesh_layout: str
            iterations: int
            steps_per_checkpoint: int
            model_dir: str
            batch_size: int
            train_steps: int

        runconfig = RunConfig(
            mesh_shape="all:%d" % multiprocessing.cpu_count(),
            mesh_layout="batch:all",
            iterations=200,
            train_steps=1000,
            # model_fn=model_fn,
            steps_per_checkpoint=100,
            model_dir="/tmp/model_dir_test",
            batch_size=8,
        )

        iterations_per_loop = runconfig.iterations
        mesh_shape = mtf.convert_to_shape(runconfig.mesh_shape)
        config = tpu_config.RunConfig(
            cluster=None,  # cpu
            model_dir=runconfig.model_dir,
            save_checkpoints_steps=None,  # Disable the default saver
            save_checkpoints_secs=None,  # Disable the default saver
            log_step_count_steps=iterations_per_loop,
            save_summary_steps=iterations_per_loop,
            tpu_config=tpu_config.TPUConfig(
                num_shards=mesh_shape.size,
                iterations_per_loop=iterations_per_loop,
                num_cores_per_replica=1,
                per_host_input_for_training=tpu_config.InputPipelineConfig.BROADCAST,
            ),
        )

        model_fn = build_model_fn(
            dict(mesh_shape=mesh_shape, mesh_layout=runconfig.mesh_layout)
        )

        # Estimator
        classifier = tpu_estimator.TPUEstimator(
            use_tpu=False,
            model_fn=model_fn,
            config=config,
            train_batch_size=runconfig.batch_size,
            eval_batch_size=runconfig.batch_size,
            predict_batch_size=runconfig.batch_size,
        )

        current_step = estimator_lib._load_global_step_from_checkpoint_dir(
            runconfig.model_dir
        )  # pylint: disable=protected-access,line-too-long
        logging.info("Current step %d", current_step)
        if runconfig.steps_per_checkpoint == 0:
            classifier.train(
                input_fn=input_fn(runconfig.batch_size), max_steps=runconfig.train_steps
            )
            return

        while current_step < runconfig.train_steps:
            next_checkpoint = min(
                current_step + runconfig.steps_per_checkpoint, runconfig.train_steps
            )
            classifier.train(
                input_fn=input_fn(runconfig.batch_size), max_steps=next_checkpoint
            )
            current_step = next_checkpoint
            logging.info("Starting to evaluate.")
            eval_results = classifier.evaluate(
                input_fn=input_fn(runconfig.batch_size), steps=156
            )  # since we have 10000 examples and batch_size = 64 per host
            logging.info("Eval results: %s", eval_results)

    # def run_toy_model_tpu(self):
    #     """Run a toy model on TPU."""
    #     tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
    #         FLAGS.tpu, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    #     iterations_per_loop = FLAGS.iterations
    #     mesh_shape = mtf.convert_to_shape(FLAGS.mesh_shape)
    #     config = tpu_config.RunConfig(
    #         cluster=tpu_cluster_resolver,
    #         model_dir=FLAGS.model_dir,
    #         save_checkpoints_steps=None,  # Disable the default saver
    #         save_checkpoints_secs=None,  # Disable the default saver
    #         log_step_count_steps=iterations_per_loop,
    #         save_summary_steps=iterations_per_loop,
    #         tpu_config=tpu_config.TPUConfig(
    #             num_shards=mesh_shape.size,
    #             iterations_per_loop=iterations_per_loop,
    #             num_cores_per_replica=1,
    #             per_host_input_for_training=tpu_config.InputPipelineConfig.BROADCAST))
    #     # Estimator
    #     classifier = tpu_estimator.TPUEstimator(
    #         use_tpu=True,
    #         model_fn=model_fn,
    #         config=config,
    #         train_batch_size=FLAGS.batch_size,
    #         eval_batch_size=FLAGS.batch_size)
    #     current_step = estimator_lib._load_global_step_from_checkpoint_dir(FLAGS.model_dir)  # pylint: disable=protected-access,line-too-long
    #     logging.info('Current step %d', current_step)
    #     if FLAGS.steps_per_checkpoint == 0:
    #         classifier.train(input_fn=ToyModelInput(), max_steps=FLAGS.train_steps)
    #         return
    #     while current_step < FLAGS.train_steps:
    #         next_checkpoint = min(current_step + FLAGS.steps_per_checkpoint,
    #                             FLAGS.train_steps)
    #         classifier.train(input_fn=ToyModelInput(), max_steps=next_checkpoint)
    #         current_step = next_checkpoint
    #         logging.info('Starting to evaluate.')
    #         eval_results = classifier.evaluate(
    #             input_fn=ToyModelInput(),
    #             steps=156)  # since we have 10000 examples and batch_size = 64 per host
    #         logging.info('Eval results: %s', eval_results)


if __name__ == "__main__":
    absltest.main()
