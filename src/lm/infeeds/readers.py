"an infeed generator "
from enum import Enum
from typing import Dict

import tensorflow as tf
from absl import logging

import lm

from .base import Infeed, InfeedConfig


class CompressionType(str, Enum):
    gzip = "gzip"
    zlib = "zlib"
    none = "none"


class TFRecordDatasetReaderConfig(InfeedConfig):
    batch_size: int
    sources: str
    compression_type: CompressionType = CompressionType.none

    n_tokens: int = 10
    len_sequence: int = 8

@lm.register_infeed("lm.infeeds.TFRecordDatasetReader", TFRecordDatasetReaderConfig)
class TFRecordDatasetReader(Infeed):
    def __init__(self, config: TFRecordDatasetReaderConfig):
        super().__init__()
        self.config = config

    @property
    def sources(self):
        return self.config.sources

    def __call__(self, params: Dict) -> tf.data.Dataset:
        """Input function which provides a single batch for train or eval."""

        batch_size = params["batch_size"]
        len_sequence = params["len_sequence"]

        logging.info(
            "call Seq2SeqTFRecordDataset() with batch size %d and sequence length %d",
            batch_size,
            len_sequence,
        )

        if self.sources.endswith("/") and tf.io.gfile.exists(self.sources + "/"):
            raise ValueError("invalid dataset directory. contains nested empty folders")

        filenames = tf.io.gfile.glob(self.config.sources)
        logging.info("Found %s files matching %s" % (len(filenames), self.sources))
        if not filenames:
            raise ValueError(
                "No matching files found for pattern %s" % self.config.sources
            )
        ds = tf.data.TFRecordDataset(filenames, buffer_size=64 * 1024 * 1024)
        keys = ["content", "target"]

        def format_tokens(tokens):
            return tf.reshape(tf.cast(tokens, tf.int32), [len_sequence, ])

        # Examples are already pre-processed
        def decode_example(serialized_example):
            """Return a dict of Tensors from a serialized tensorflow.Example."""
            decoded = tf.io.parse_example(
                serialized=[serialized_example],
                features={k: tf.VarLenFeature(tf.int64) for k in keys},
            )
            # cast the features to int32, shape it to [batch_size, len_sequence]
            decoded = {k: format_tokens(v.values) for k, v in decoded.items()}
            return decoded["content"], decoded["target"]

        ds = ds.map(decode_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.batch(batch_size=batch_size, drop_remainder=True) 
        return ds
