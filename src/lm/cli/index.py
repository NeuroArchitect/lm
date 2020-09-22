"""
Given a directory create an index with full path to txt files
"""
import os
import io
import re
import string
import shutil
import time
import functools
import collections
from glob import glob
from multiprocessing import Pool, cpu_count

from absl import app, logging
from absl.flags import argparse_flags

from tqdm import auto as tqdm

from chardet.universaldetector import UniversalDetector

def chunks(l, n, max_size=512):
    out = []
    chunk = []
    sz = 0
    for path in l:
        chunk.append(path)
        sz += 1
        if sz >= n or sz >= max_size:
            out.append(chunk)
            sz = 0
            chunk = []
    if chunk:
        out.append(chunk)
    return out


def binary_check():
    pass

def process_multi_file(files, binary_check:bool=True):
    result = []
    excluded = []
    detector = UniversalDetector()
    for src in files:
        if binary_check is False:
            result.append( (src, 1.0) )
            continue
        try:
            ok = False
            detector.reset()
            with open(src, 'rb') as fd:
                while True:
                    s = fd.read(128)
                    if s:
                        detector.feed(s)
                        if detector.done:
                            detector.close()
                            break
                    if not s or len(s) < 128:
                        break
            p = detector.result['confidence'] 
            if p > 0.99:
                ok = True
            else:
                p = p_of_text(s)
                if p > 0.60: # if less than 40% we consider it text
                    ok = True
            if ok:
                result.append((src, p))
            else:
                excluded.append((src, p))
        except Exception as exc:
            logging.error("could not process %s: %r", src, exc)
            import traceback; traceback.print_exc() 
    return result, excluded


def parallel(src_dst_list, total, nproc=cpu_count() -1 or 1, binary_check=True):
    pool = Pool(processes=nproc)
    accepted = []
    excluded = []

    pbar = tqdm.tqdm(total=total)

    for acc, exc in pool.imap(functools.partial(process_multi_file, binary_check=binary_check), src_dst_list):
        accepted.extend(acc)
        excluded.extend(exc)
        pbar.update(len(acc) + len(exc))
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
    
    parser.add_argument(
        "--no_binary_check", default=False, action="store_true", help="Check if the file is a binary or text file. excludes binary files",
    )
    
    parser.add_argument(
        "--debug", action="store_true", help="Output the probability of being binary for each file",
    )
    
    parser.add_argument(
        "--nproc", type=int, help="number of process to use",
    )


def local_parse_args(args):
    parser = argparse_flags.ArgumentParser()
    parse_args(args, parser)
    return parser.parse_args(args[1:])


def p_of_text(s):
    text_characters = list(range(32, 127)) + [ ord(c) for c in "\n\r\t\b"]
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
    return 1.0 - (float(len(t)) / float(len(s)))


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
    if args.nproc:
        n = args.nproc

    elements_per_chunk = len(lines)//n

    start = time.time()
    accepted, excluded = parallel(chunks(lines, elements_per_chunk), total=len(lines), nproc=args.nproc, binary_check=not args.no_binary_check)
    end = time.time()

    if not accepted:
        logging.error('no text files found in directory %s', args.input)
        exit(-1)

    with open(args.output, 'w') as fd:
        if args.debug:
            fd.writelines(f'{l}\t{p}\n' for l,p in accepted)
        else:
            fd.writelines(f'{l}\n' for l,_ in accepted)

    with open(args.output + '.excluded', 'w') as fd:
        if args.debug:
            fd.writelines(f'{l}\t{p}\n' for l,p in excluded)
        else:
            fd.writelines(f'{l}\n' for l,_ in excluded)

    logging.info(
        "processed {} files in {:.2f}s, {} / {} good files. accepted index is at {}. excluded index {}".format(
            len(lines), end - start, len(accepted), len(lines), args.output, args.output + '.excluded'
        )
    )


if __name__ == "__main__":
    app.run(main, flags_parser=local_parse_args)
