import base64
import os
import cmd
import subprocess
import tempfile
import time
from multiprocessing import Pool, cpu_count
from typing import Dict

import farmhash
import tensorflow as tf
from absl import app, logging
from absl.flags import argparse_flags
from tqdm import auto as tqdm

import lm.human

# FIXME
from lm.cli.encode import local_parse_args, num

def executor(nproc):
    if nproc == 1:
        return map
    pool = Pool(processes=nproc)
    return pool.imap_unordered


def hashfile(job):
    sources, dst = job
    count = 0
    with tf.io.gfile.GFile(dst, "w") as wf:
        for src in sources:
            count += 1
            with tf.io.gfile.GFile(src, "rb") as fd:
                message_bytes = fd.read()
                base64_bytes = base64.b64encode(message_bytes)
                # there is no relationship here between these two methods
                # the base64 is a trick to allow to dedup gz/compresed files
                hashvalue = farmhash.fingerprint64(base64_bytes.decode("ascii"))
                wf.write("%d\t%s\n" % (hashvalue, src))
    return dst, count


def run(nproc, src_dst_list, total):
    hash_files =[]
    total = 0
    execute = executor(nproc)
    for hash_file, count in tqdm.tqdm(
        execute(hashfile, src_dst_list), total=total,
    ):
        hash_files.append(hash_file)
        total += count
    return hash_files, count


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
        "--nproc",
        type=int,
        default=0,
        help="the number of processes to use for multiprocess encoding (0=all CPUs, 1=disable multiprocessing)",
    )


def chunks(l, n):
    out = []
    chunk = []
    sz = 0
    for fpath in l:
        chunk.append(fpath)
        sz += 1
        if sz >= n:
            out.append(chunk)
            sz = 0
            chunk = []
    if chunk:
        out.append(chunk)
    return out

def sh(cmdline):
    try:
        result = subprocess.check_output(cmdline, shell=True)
        logging.debug(result)
    except subprocess.CalledProcessError as grepexc:
        logging.error("subprocess %s failed with error code: %r\n\n%r", grepexc.returncode, grepexc.output)
        exit()

def main(args):
    logging.info("started evaluation process")
    txt_files = lm.human.filepaths_from_user_input(args.input)
    if not txt_files:
        logging.error("no input data files found with input: %s", args.input)
        return
    for f in txt_files:
        if not os.path.isabs(f):
            logging.error('file paths must be absolute. %s is not', f)
            exit()

    if tf.io.gfile.isdir(args.output):
        output = os.path.join(args.output, 'index.txt')
    else:
        output = args.output
    nproc = args.nproc

    if nproc == 0:
        nproc = cpu_count() - 1

    file_chunks = chunks(
        txt_files, (len(txt_files) // (args.nproc + 1))
    )  # Assign files_per file to a tfrecord file each

    # start process
    jobs = (
        (chunks, tempfile.NamedTemporaryFile(delete=False).name)
        for idx, chunks in enumerate(file_chunks)
    )

    start = time.time()
    hash_files, count = run(args.nproc, jobs, total=len(file_chunks))
    with tempfile.NamedTemporaryFile('wt', delete=True) as index:
        for s in hash_files:
            index.write('%s\n' % s)
        index.flush()
        with tempfile.NamedTemporaryFile('wt', delete=True) as reduce:
            # concatenate all in single txtfile
            sh('{ xargs cat < %s ; } > %s' % (index.name, reduce.name))
            # unique sort using first column
            sh('sort -u --parallel %d -k1,1 %s > %s.farmhash' % (nproc, reduce.name, output))
            sh('cut -f2 <%s.farmhash >%s' % (output, output))

    end = time.time()
    elapsed = end - start

    logging.info(
        "finished in %ss: farmhashed %d of %d files ", # (%s tokens @ %.2f tokens/sec) in %d tfrecords (~%s tokens per record)",
        num(elapsed),
        count,
        len(txt_files),
        # num(token_total),
        # tokens_per_second,
        # len(file_chunks),
        # num(tokens_per_record),
    )

    logging.info(f'created unique index file {output}, hashvalues {output}.index')


if __name__ == "__main__":
    tf.disable_v2_behavior()
    app.run(main, flags_parser=local_parse_args)
