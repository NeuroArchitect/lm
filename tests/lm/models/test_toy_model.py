import numpy as np
import tensorflow as tf
from absl.testing import absltest, parameterized
from tensorflow.python.framework import test_util

import lm.examples
import lm.models.toy as toy
import lm.tf


class MockContext:
    num_hosts: int = 2


class TestToyTransformerModel(parameterized.TestCase, tf.test.TestCase):
    
    @parameterized.parameters(
#        ('bfloat16', ),
        ('float32', ),
    )
    @test_util.run_v1_only(reason='is better')
    def test_as_dataset(self, precision):
        batch = 2
        seq = 4
        channels = 4
        n_tokens = 10

        # seq 2 seq
        inputs = tf.ones([batch, seq ], dtype=tf.int32)
        outputs = tf.ones([batch, seq ], dtype=tf.float32)

        cfg = toy.ToyTransformerConfig(
            mesh_shape="all:1",
            mesh_layout="",
            use_bias=True,
            optimizer="SGD",
            learning_rate=2e-5,
            num_hidden_layers=2,
        )

        model_builder = toy.ToyTransformer(config=cfg)
        mode = tf.estimator.ModeKeys.PREDICT
        context = MockContext()
        params = {
            "batch_size": batch,
            "tokens_size": 1,
            "n_tokens": n_tokens,
            "n_sequences": seq,
            "n_channels": channels,
            "use_tpu": False,
            "context": context,
            "precision": precision,
            "checkpoint_dir": '',
        }
        
        estimator, lowering = model_builder(inputs, outputs, mode, params=params)

        # print(tf_model_graph)
        # actual_outputs = lowering.export_to_tf_tensor(estimator)

        # expected_outputs = tf.cast(tf.not_equal(inputs, 0), tf.float32)
        # expected = tf.identity(tf_model_graph, name=mode)
        init = tf.global_variables_initializer()
        self.evaluate(init)
        tf_group = lowering.copy_masters_to_slices()
        self.evaluate(tf_group)
        results = self.evaluate([estimator.predictions['predictions']])
        expected = (batch, seq, n_tokens)
        self.assertEqual(results[0].shape, expected)


if __name__ == "__main__":
    absltest.main()
