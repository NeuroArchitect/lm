"The only valid example formats accepted by the framework"
import collections
import os

import farmhash
import numpy as np
import tensorflow as tf

import lm.parsers

CONTENT_KEY = "content"
TARGET_KEY = "target"

PreProcessedTextLine = collections.namedtuple(
    "PreProcessedTextLine", ["id", "content", "target", "offset_start", "offset_end"]
)

Seq2SeqSimpleExample = collections.namedtuple(
    "Seq2SeqSimpleExample", [CONTENT_KEY, TARGET_KEY]
)


def _uint64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(
        int64_list=tf.train.Int64List(value=np.int64(np.array(value, dtype=np.uint64)))
    )


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def read_example(example_proto) -> dict:
    features = {
        "id": tf.io.VarLenFeature(tf.int64),
        "content": tf.io.FixedLenFeature([], tf.string),
        "target": tf.io.VarLenFeature(tf.int64),
        "offset_start": tf.io.VarLenFeature(tf.int64),
        "offset_end": tf.io.VarLenFeature(tf.int64),
    }
    parsed_features = tf.io.parse_single_example(example_proto, features)
    return {
        "id": tf.cast(parsed_features["id"], tf.uint64),
        "content": parsed_features["content"],
        # WARNING: remapping from target to targets
        "targets": tf.sparse.to_dense(tf.cast(parsed_features["target"], tf.int64)),
        "offset_start": tf.sparse.to_dense(
            tf.cast(parsed_features["offset_start"], tf.uint64)
        ),
        "offset_end": tf.sparse.to_dense(
            tf.cast(parsed_features["offset_end"], tf.uint64)
        ),
    }


# def read_example(example_proto, max_seq_len=1024) -> dict:
#     features = {
#         "id": tf.VarLenFeature(tf.uint64, default=-1),
#         "content": tf.VarLenFeature(tf.bytes, default=0),
#         "target": tf.VarLenFeature(tf.uint64, default=0),
#         "offset_start": tf.VarLenFeature(tf.uint64, default=0),
#         "offset_end": tf.VarLenFeature(tf.uint64, default=0),
#     }
#     return tf.parse_single_example(example_proto, features)


def create_example(features: PreProcessedTextLine) -> tf.train.Example:
    feature = {
        "id": _uint64_feature([features.id]),
        "content": _bytes_feature(features.content.encode("utf-8")),
        "target": _uint64_feature(features.target),
        "offset_start": _uint64_feature(features.offset_start),
        "offset_end": _uint64_feature(features.offset_end),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def _trec_options(compression):
    # TODO: do we get some speedups here with better buffer inputs?
    return tf.io.TFRecordOptions(
        compression_type=compression,
        flush_mode=None,
        input_buffer_size=None,
        output_buffer_size=None,
        window_bits=None,
        compression_level=None,
        compression_method=None,
        mem_level=None,
        compression_strategy=None,
    )


def transform_many_and_write_one_tfrecord(job):
    tokenizer, sources, dst, parse_strategy = job
    token_count = 0
    example_count = 0
    *_, ext = os.path.splitext(dst)
    if ext in (".gz", ".gzip"):
        options = _trec_options(tf.io.TFRecordCompressionType.GZIP)
    elif ext in (".lz",):
        options = _trec_options(tf.io.TFRecordCompressionType.ZLIB)
    else:
        options = _trec_options(tf.io.TFRecordCompressionType.NONE)

    with tf.io.TFRecordWriter(dst, options) as w:
        for source in sources:
            for uids, sources, tokens, start_offsets, end_offsets in batch_tokenizer(
                tokenizer, source, strategy=parse_strategy
            ):
                result = PreProcessedTextLine(
                    uids, sources, tokens, start_offsets, end_offsets
                )
                example = create_example(result)
                w.write(example.SerializeToString())
                token_count += len(tokens)
                example_count += 1
    return token_count, example_count


def batch_tokenizer(tokenizer, txtfile_location, strategy="file"):
    # just convert to the token ids, we will do adaptative padding on training time.
    sources = lm.parsers.parse_url(txtfile_location, strategy=strategy)
    if len(sources) == 0:
        return
    uids = [farmhash.fingerprint64(source) for source in sources]
    batches = tokenizer.batch_encode_plus(
        sources,
        return_token_type_ids=True,
        pad_to_max_length=False,
        truncation=False,
        add_special_tokens=True,
        return_offsets_mapping=True,
        verbose=False,
    )

    yield from zip(
        uids,
        sources,
        batches["input_ids"],
        [[start for start, end in offsets] for offsets in batches["offset_mapping"]],
        [[end for start, end in offsets] for offsets in batches["offset_mapping"]],
    )


def _int64_list_feature(int_list):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=int_list))


def gen_serialization(ndigit):
    def serialize(tokens, idx):
        """
        Creates a tf.Example message ready to be written to a file.
        """

        # Create a dictionary mapping the feature name to the tf.Example-compatible
        # data type.
        feature = {
            "tokens": _int64_list_feature(tokens),
            "idx": _int64_list_feature(idx),
        }

        example = tf.train.Example(features=tf.train.Features(feature=feature))

        return example.SerializeToString()

    feature_spec = {
        "tokens": tf.io.FixedLenFeature([ndigit * 3], dtype=tf.dtypes.int64),
        "idx": tf.io.FixedLenFeature([ndigit * 3], dtype=tf.dtypes.int64),
    }

    def deserialize(example):
        return tf.io.parse_single_example(example, features=feature_spec)

    return serialize, deserialize


def _serialize_seq2seq(self):
    feature = {
        CONTENT_KEY: _int64_list_feature(self.content),
        TARGET_KEY: _int64_list_feature(self.target),
    }

    example = tf.train.Example(features=tf.train.Features(feature=feature))

    return example.SerializeToString()
    # raise ValueError('type %r not yet supported' % type(ex))
    # feature_spec = {
    #     "tokens": tf.io.FixedLenFeature([ndigit * 3], dtype=tf.dtypes.int64),
    #     "idx": tf.io.FixedLenFeature([ndigit * 3], dtype=tf.dtypes.int64),
    # }


Seq2SeqSimpleExample.serialize = _serialize_seq2seq


def from_file_list(file_list):
    has_any_gz = any(f.endswith("gz") for f in file_list)
    has_all_gz = all(f.endswith("gz") for f in file_list)
    if has_any_gz and not has_all_gz:
        raise ValueError("invalid mix of gz and non gz records")
    if has_all_gz:
        return lambda *args, **kwds: tf.data.TFRecordDataset(
            *args, **kwds, compression_type="GZIP", buffer_size=None
        )

    return tf.data.TFRecordDataset
