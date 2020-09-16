"""
Given a directory create an index with full path to txt files
"""
import os
import re
import string
import shutil
import time
import collections
from glob import glob
from multiprocessing import Pool, cpu_count

from absl import app, logging
from absl.flags import argparse_flags

from tqdm import auto as tqdm


def chunks(l, n):
    out = []
    chunk = []
    sz = 0
    for path in l:
        chunk.append(path)
        sz += 1
        if sz >= n:
            out.append(chunk)
            sz = 0
            chunk = []
    if chunk:
        out.append(chunk)
    return out


def process_multi_file(files):
    result = []
    excluded = []
    for src in files:
        try:
            if is_text(src):
                result.append(src)
            else:
                excluded.append(src)
        except Exception as exc:
            logging.error("could not process %s: %r", src, exc)
            import traceback; traceback.print_exc() 
    return result, excluded


def parallel(src_dst_list, total):
    count = cpu_count() - 1 or 1
    pool = Pool(processes=count)
    accepted = []
    excluded = []
    for acc, exc in tqdm.tqdm(pool.imap(process_multi_file, src_dst_list), total=total):
        accepted.extend(acc)
        excluded.extend(exc)
    return accepted, excluded


def parse_args(_, parser):
    parser.add_argument(
        "input",
        type=str,
        help="Location of the dataset archives files. Files ending in .zst are treated as \
                        archives, all others as raw text. Can be a glob (/dataset/*.xz, /dataset/*.txt)",
    )

    parser.add_argument(
        "output", type=str, help="Location of the generate index file.",
    )

    parser.add_argument(
        "--force", action="store_true", help="Overwrite output index file",
    )


def local_parse_args(args):
    parser = argparse_flags.ArgumentParser()
    parse_args(args, parser)
    return parser.parse_args(args[1:])


def is_text(filename):
    s = open(filename, 'rb').read(512)
    text_characters = list(range(32, 127)) + [ ord(c) for c in "\n\r\t\b"]
    # null_trans = bytes.maketrans(bytearray(text_characters), bytearray(len(text_characters)))
    null_trans = bytes.maketrans(bytearray(0), bytearray(0))
    if not s:
        # Empty files are considered text
        return True
    if 0 in s:
        # Files with null bytes are likely binary
        return False
    # Get the non-text characters (maps a character to itself then
    # use the 'remove' option to get rid of the text characters.)
    t = s.translate(null_trans, bytearray(text_characters))
    # If more than 30% non-text characters, then
    # this is considered a binary file
    if float(len(t)) / float(len(s)) > 0.30:
        return False
    return True


def main(args):
    if os.path.exists(args.output) and not args.force:
        logging.error('output file exists. aborting.') 
        exit(-1)
        return

    lines = []
    for root, dirs, files in os.walk(args.input):
        for file in files:
            f = os.path.realpath(os.path.join(root, file))
            lines.append(f)

    n = cpu_count() - 1 or 1
    start = time.time()
    accepted, excluded = parallel(chunks(lines, n), total=len(lines))
    end = time.time()

    if not accepted:
        logging.error('no text files found in directory %s', args.input)
        exit(-1)

    with open(args.output, 'w') as fd:
        fd.writelines(f'{l}\n' for l in accepted)

    with open(args.output + '.excluded', 'w') as fd:
        fd.writelines(f'{l}\n' for l in excluded)

    logging.info(
        "processed {} files in {:.2f}s, {} / {} good files. accepted index is at %s. excluded index %s".format(
            len(lines), end - start, len(accepted), len(lines), args.output, args.output + '.excluded'
        )
    )


if __name__ == "__main__":
    app.run(main, flags_parser=local_parse_args)
