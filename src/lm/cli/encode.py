import os
import time
from multiprocessing import Pool, cpu_count

import numpy as np
import tensorflow as tf
from absl import app, flags, logging
from tqdm import auto as tqdm

import lm.config
import lm.encoders
import lm.examples
import lm.human


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


def executor(nproc):
    if nproc == 1:
        return map
    if nproc == 0:
        nproc = cpu_count() - 1
    pool = Pool(processes=nproc)
    return pool.imap_unordered


def run(nproc, src_dst_list, total):
    token_total = 0
    example_total = 0
    execute = executor(nproc)
    for token_count, example_count in tqdm.tqdm(
        execute(lm.examples.transform_many_and_write_one_tfrecord, src_dst_list),
        total=total,
    ):
        token_total += token_count
        example_total += example_count
    return token_total, example_total


def parse_args(args, parser):
    parser.add_argument(
        "input",
        type=str,
        help="path to: 1) a directory, 2) an index file with one file path per line.",
    )
    parser.add_argument(
        "output", type=str, default="output", help="Where to write tfrecords"
    )

    parser.add_argument(
        "--size",
        type=float,
        default=300.0,
        help="the size in MB of uncompressed text to add to each tfrecord file, default 300MB",
    )
    parser.add_argument(
        "--name", type=str, default="dataset", help="prefix name for the output files."
    )
    parser.add_argument(
        "--encoder", type=str, default="gpt2", help="Name or path of an encoder spec"
    )

    parser.add_argument(
        "--by",
        default="file",
        type=str.lower,
        choices=["file", "passage", "line"],
        help="how to encode the inputs: file: each file is a record. passage: each file is splitted in passages, each passage is a record. line: is an example.",
    )

    parser.add_argument(
        "--compress",
        default="lz",
        type=str.lower,
        choices=["none", "gz", "gzip", "lz", "zlib"],
        help="compression to use for the records. the suffix will be added to the final files",
    )

    parser.add_argument(
        "--nproc",
        type=int,
        default=0,
        help="the number of processes to use for multiprocess encoding (0=all CPUs, 1=disable multiprocessing)",
    )


def is_integer(x):
    return np.can_cast(x, np.int32)


def is_float(x):
    return np.can_cast(x, np.float32)


def is_exact(x):
    return is_integer(x) or is_float(x) and x == int(x)


def num(x, digits_after_decimal=2):
    if is_integer(x):
        spec = "{:,d}"
    else:
        spec = "{:,.%df}" % digits_after_decimal
    return spec.format(x)


def local_parse_args(args):
    parser = flags.argparse_flags.ArgumentParser()
    parse_args(args, parser)
    return parser.parse_args(args[1:])


def main(args):

    txt_files = lm.human.filepaths_from_user_input(args.input)
    if not txt_files:
        logging.error("no input data files found with input: %s", args.input)
        return

    os.makedirs(args.output, exist_ok=True)

    if tf.io.gfile.exists(args.encoder):
        enccfg = lm.config.load(args.encoder)
        encoder = lm.encoders.from_config(enccfg)
    else:
        encoder = lm.encoders.from_config(dict(kind="hf", location=args.encoder))

    megabytes_per_tfrecord = int(args.size * 1e6)
    file_chunks = sizechunks(
        txt_files, megabytes_per_tfrecord
    )  # Assign files_per file to a tfrecord file each

    logging.info(
        "got %d files, divided into %d chunks.", len(txt_files), len(file_chunks)
    )

    def getdst(name, idx, compress, total):
        if compress in ("gzip", "gz"):
            ext = "tfrecord.gz"
        elif compress in ("zlib", "lz"):
            ext = "tfrecord.lz"
        else:
            ext = "tfrecord"

        return os.path.join(args.output, "%s_%05d_%05d.%s" % (name, idx, total, ext))

    if len(file_chunks) == 0:
        logging.error(
            "cannot split %d files into %d chunks", len(txt_files), len(file_chunks)
        )
        exit(-1)

    # check output directory
    if tf.io.gfile.exists(args.output) and tf.io.gfile.listdir(args.output):
        logging.error("output directory is not empty. refusing to continue")
        exit(-1)
    tf.io.gfile.makedirs(args.output)

    # start process
    jobs = (
        (
            encoder,
            chunks,
            getdst(args.name, idx, args.compress, len(file_chunks)),
            args.by,
        )
        for idx, chunks in enumerate(file_chunks)
    )

    start = time.time()
    token_total, example_total = run(args.nproc, jobs, total=len(file_chunks))

    end = time.time()
    elapsed = end - start
    tokens_per_second = token_total / elapsed
    tokens_per_record = token_total / len(file_chunks)

    logging.info(
        "finished in %ss: tokenized %d of %d files (%s tokens @ %.2f tokens/sec) in %d tfrecords (~%s tokens per record)",
        num(elapsed),
        example_total,
        len(txt_files),
        num(token_total),
        tokens_per_second,
        len(file_chunks),
        num(tokens_per_record),
    )


if __name__ == "__main__":
    app.run(main, flags_parser=parse_args)
