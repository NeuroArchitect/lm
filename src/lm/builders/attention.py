"Collection of Attention Layers"
import mesh_tensorflow as mtf
import tensorflow as tf
from pydantic.dataclasses import dataclass


@dataclass
class MultiHeadAttentionConfig:
    src_seq_len: int
    dst_seq_len: int
    n_head: int
    m_dim: int
    q_dim: int
    k_dim: int
    v_dim: int
    o_dim: int
    dtype: str


def create_initializer(shape):
    return mtf.random_uniform_initializer(shape=shape)


class MultiHeadAttentionBuilder:

    def __init__(self, config: MultiHeadAttentionConfig):
        self.config = config

    def add_random_uniform(self, shape):
        return mtf.random_uniform_initializer(shape=shape)

    def add_var(self, name, shape):
        return mtf.get_variable(name, shape=shape, initializer=self.add_random_uniform)

    def add_triangular_mask(self):
        """
        language model next token prediction mask
        returns: [batch, heads, dst_seq, src_seq]
        """
        nd = self.config.dst_seq_len
        ns = self.config.src_seq_len
        # add one dimension
        i = tf.range(nd)[:, None]
        j = tf.range(ns)
        mask = i >= (j - ns + nd)

        mask = tf.reshape(mask, (1, 1, nd, ns))

        return tf.cast(mask, self.config.dtype)

    def __call__(self, x):
        """
        x: [batch_dim, sequence, embedding_dim]
        """
        d = self.config.src_seq_len
        n_head = self.config.n_head
        m_dim = self.config.m_dim
        q_dim = self.config.q_dim
        k_dim = self.config.k_dim
        v_dim = self.config.v_dim
        o_dim = self.config.o_dim

        mask = self.add_triangular_mask()  # [batch, heads, dest, src]
        M = self.add_var("M", [1, m_dim, d])
        Q = self.add_var("Q", [n_head, d, q_dim])
        K = self.add_var("K", [n_head, d, k_dim])
        V = self.add_var("V", [n_head, d, v_dim])
        O = self.add_var("O", [n_head, d, o_dim])

        batch_dim = mtf.Dimension("batch")
        sequence_dim = mtf.Dimension("sequence")
        k_dim = mtf.Dimension("k")
        q_dim = mtf.Dimension("q")
        v_dim = mtf.Dimension("v")
        
        # Project Key and Query
        # x : [ batch, sequence, embedding_dim]
        # K : [ batch, embedding_dim, k_dim]
        xK = mtf.einsum([x, K], output_shape=[batch_dim, sequence_dim, k_dim])
        xQ = mtf.einsum([x, Q], output_shape=[batch_dim, sequence_dim, q_dim])

        # Attention Weights [ batch, sequence, 1] 
        W = mtf.einsum([xK, xQ], output_shape=[batch_dim, sequence_dim, 1])
        # Weight Scaling
        W = W * mtf.rsqrt(k_dim)

        # Mask
        mask = self.add_triangular_mask() # [ batch, sequence, 1]
        W_masked =mtf.einsum([W, mask])
        attention = mtf.softmax(W_masked, sequence_dim)

        xV = mtf.einsum([x, V], output_shape=[batch_dim, sequence_dim, v_dim])
        values = mtf.einsum([attention, xV], output_shape=[batch_dim, sequence_dim, v_dim])
        io = mtf.einsum([values, O], output_shape=[batch_dim, sequence_dim, o_dim])
        return io
