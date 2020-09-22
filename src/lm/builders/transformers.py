"Block utils library"
from typing import Dict

import mesh_tensorflow as mtf
import tensorflow as tf
from pydantic.dataclasses import dataclass

from .attention import MultiHeadAttentionConfig, MultiHeadAttentionBuilder
from .nn import layer_norm
from .feedforward import FeedForward, FeedForwardConfig

@dataclass
class TransformerConfig:
    dst_seq_len: int
    attention: Dict
    layer_norm: Dict
    feedforward: Dict


class TransformerBuilder:
    "Classic Transformer Decoder"
    def __init__(self, config: TransformerConfig):
        self._config = config
        self.attention_builder = MultiHeadAttentionBuilder(config.attention)
        self.feedforward_builder = FeedForward(FeedForwardConfig(config.feedforward))
    
    def add_layer_norm(self, inputs):
        return layer_norm(
            inputs,
            self._config.io_dim,
        )

    def add_feed_forward(self, x):
        return self.feedforward_builder(x)

    def add_attention(self, x):
        return self.attention_builder(x)

    def __call__(self, name, x):
        with tf.variable_scope(name):
            x = x + self.add_layer_norm(self.add_attention(x))
            x = x + self.add_layer_norm(self.add_feed_forward(x))
        return x