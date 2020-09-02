import argparse
import collections
import json
import numpy as np
import os
import random
import time
from glob import glob
from multiprocessing import Pool, cpu_count

import farmhash
import numpy as np
import tensorflow as tf
from absl import app, logging
from absl.flags import argparse_flags
from lm_dataformat import Reader
from tokenizers import Tokenizer
from tqdm import auto as tqdm
from transformers import GPT2Config, GPT2Tokenizer, GPT2TokenizerFast

from tensorflow.compat import v1


def parse_args(argv):
    parser = argparse_flags.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to where your files are located. Files ending in .zst are treated as \
                        archives, all others as raw text.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=50 * 2 ** 20,
        help="the size in MiB of uncompressed text to add to a record, default 50MiB",
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Name of output files will be {name}_%%05d.tfrecord where i is the number of the file",
    )
    parser.add_argument(
        "--output", type=str, default="output", help="Where to write tfrecords"
    )
    parser.add_argument(
        "--tokenizer", type=str, required=True, help="Name or path of a tokenizer spec"
    )
    parser.add_argument("--random_seed", type=int, default=1337, help="seed")
    args = parser.parse_args(argv[1:])
    return args


def readlines_txt(src):
    with open(src) as fd:
        return fd.readlines()


LINE_READER = {
    ".txt": readlines_txt,
    ".tsv": readlines_txt,
    # '.ztx':
}


def readlines(src):
    _, ext = os.path.splitext(src)
    f = LINE_READER.get(ext, None)
    if f is None:
        logging.warning("no readlines for file %s", src)
        return
    return f(src)


# Helper functions and classes
def sizechunks(l, n):
    out = []
    chunk = []
    sz = 0
    for fpath in l:
        chunk.append(fpath)
        sz += tf.io.gfile.stat(fpath).length
        if sz >= n:
            out.append(chunk)
            sz = 0
            chunk = []
    if chunk:
        out.append(chunk)
    return out


# END_OF_TEXT_TOKEN_ID = 0
# def pad_sequence(buff, missing_elements):
#     buff.extend([END_OF_TEXT_TOKEN_ID] * missing_elements)
#     return buff

# def token_generator(tokenizer, sources, max_seq_len):
#     buff = []
#     for src in sources:
#         for line in readlines(src):
#             enctext = tokenizer.encode(line)
#             if not buff:
#                 # skip the line if is too long
#                 # most likely it does not make sense
#                 if len(enctext) > max_seq_len:
#                     continue
#                 else:
#                     buff = enctext.ids
#                     continue

#             if len(buff) + len(enctext.ids) > max_seq_len:
#                 padded = pad_sequence(buff, max_seq_len - len(buff))
#                 yield padded
#                 buff = enctext.ids
#             else:
#                 buff.extend(enctext.ids)
#         # flush buffer when finish one file
#         if buff:
#             padded = pad_sequence(buff, max_seq_len - len(buff))
#             yield padded
#             buff = []
#     if buff:
#         padded = pad_sequence(buff, max_seq_len - len(buff))
#         yield padded
#         buff = []


def batch_tokenizer(tokenizer, txtfile_location):
    # just convert to the token ids, we will do adaptative padding on training time.
    lines = [
        l.decode("utf-8") for l in tf.io.gfile.GFile(txtfile_location, "rb").readlines()
    ]
    uids = [farmhash.fingerprint64(line) for line in lines]
    batches = tokenizer.batch_encode_plus(
        lines,
        return_token_type_ids=True,
        pad_to_max_length=False,
        truncation=False,
        add_special_tokens=True,
        return_offsets_mapping=True,
        verbose=False,
    )

    return zip(
        uids,
        lines,
        batches["input_ids"],
        [[start for start, end in offsets] for offsets in batches["offset_mapping"]],
        [[end for start, end in offsets] for offsets in batches["offset_mapping"]],
    )


PreProcessedTextLine = collections.namedtuple(
    "PreProcessedTextLine", ["id", "content", "target", "offset_start", "offset_end"]
)


def _uint64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(
        int64_list=tf.train.Int64List(value=np.int64(np.array(value, dtype=np.uint64)))
    )


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def read_example(example_proto, max_seq_len=1024) -> dict:
    features = {
        "id": tf.VarLenFeature(tf.uint64, default=-1),
        "content": tf.VarLenFeature(tf.bytes, default=0),
        "target": tf.VarLenFeature(tf.uint64, default=0),
        "offset_start": tf.VarLenFeature(tf.uint64, default=0),
        "offset_end": tf.VarLenFeature(tf.uint64, default=0),
    }
    return tf.parse_single_example(example_proto, features)


def create_example(features: PreProcessedTextLine) -> tf.train.Example:
    feature = {
        "id": _uint64_feature([features.id]),
        "content": _bytes_feature(features.content.encode("utf-8")),
        "target": _uint64_feature(features.target),
        "offset_start": _uint64_feature(features.offset_start),
        "offset_end": _uint64_feature(features.offset_end),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def transform_many_and_write_one_tfrecord(job):
    tokenizer, sources, dst = job
    with tf.io.TFRecordWriter(dst) as w:
        for source in sources:
            for features in batch_tokenizer(tokenizer, source):
                example = create_example(PreProcessedTextLine(*features))
                w.write(example.SerializeToString())
    return len(sources)


def parallel(src_dst_list, total):
    count = cpu_count() - 1 or 1
    pool = Pool(processes=count)
    ret = 0
    for i in tqdm.tqdm(
        pool.imap(transform_many_and_write_one_tfrecord, src_dst_list), total=total
    ):
        ret += i
    return ret


def load_tokenizer(location):
    if tf.io.gfile.exists(os.path.join(location, "merges.txt")):
        # use tf gfile in case the dictionary is remote
        fastok = GPT2TokenizerFast.from_pretrained(location)
        fastok.add_special_tokens(
            {"eos_token": "[EOS]", "pad_token": "[PAD]", "pad_token": "[UNK]",}
        )
    else:
        if location.startswith("/"):
            raise ValueError("invalid location %s", location)
        else:
            fastok = GPT2TokenizerFast.from_pretrained(location)
    return fastok


def listfiles(location):
    txt_files = list(p for p in glob(location) if not os.path.isdir(p))

    # try with general glob
    if not txt_files:
        txt_files = list(glob(os.path.join(location, "*.*")))

    txt_files = list(p for p in txt_files if not os.path.isdir(p))
    return txt_files


def main(args):
    random.seed(args.random_seed)
    v1.set_random_seed(args.random_seed)

    txt_files = listfiles(args.input)
    if not txt_files:
        logging.error("no data files found")
        return

    os.makedirs(args.output, exist_ok=True)

    tokenizer = load_tokenizer(args.tokenizer)

    file_chunks = sizechunks(
        txt_files, args.size
    )  # Assign files_per file to a tfrecord file each

    logging.info(
        "Got %d files, divided into %d chunks.", len(txt_files), len(file_chunks)
    )

    def getdst(name, idx, total):
        return os.path.join(args.output, "%s_%05d_%05d.tfrecord" % (name, idx, total))

    jobs = (
        (tokenizer, chunks, getdst(args.name, idx, len(file_chunks)))
        for idx, chunks in enumerate(file_chunks)
    )

    start = time.time()
    ret = parallel(jobs, total=len(txt_files))
    end = time.time()

    logging.info(
        "job completed in %.2fs, %d / %d good files.", end - start, ret, len(txt_files)
    )


if __name__ == "__main__":
    app.run(main, flags_parser=parse_args)
