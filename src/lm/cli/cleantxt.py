"""
Extract and cleans text and webarchive files
"""
import os
import sys
import re
import lm
import shutil
import time
import collections
from glob import glob
from multiprocessing import Pool, cpu_count

from absl import app, logging
from absl.flags import argparse_flags
from tqdm import auto as tqdm

import ftfy

from chardet.universaldetector import UniversalDetector


NO_ASCII = re.compile(r"[^\x00-\x7F]+")

CleanTextJob = collections.namedtuple(
    "CleanTextJob", ["files", "ascii_only", "detect_encoding", "encoding"]
)


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


def clean_text(text):
    return ftfy.fix_text(text, normalization="NFKC")


def process_single_file(src_dst):
    src, dst = src_dst
    name, file_ext = os.path.splitext(os.path.basename(src))

    if file_ext in (".txt",):
        with open(src, "r", encoding="UTF-8") as rf, open(
            dst, "w", encoding="UTF-8"
        ) as wf:
            for l in rf.readlines():
                wf.write(clean_text(l))
    else:
        logging.error("unsupported file %s with ext %s" % (src, file_ext))
        return 0
    return 1


def process_multi_file(job):
    count = 0
    detector = UniversalDetector()

    for src_dst in job.files:
        try:
            src, dst = src_dst

                

            with open(src, "rb") as rf:
                data = rf.read()

            if job.detect_encoding:
                detector.reset()
                detector.feed(data)
                if detector.done:
                    encoding = detector.result['encoding']
                else:
                    # could not detect. use default
                    encoding = job.encoding

            else:
                encoding = job.encoding

            txt = data.decode(encoding)

            clean = clean_text(txt)
            
            if job.ascii_only:
                clean = re.sub(NO_ASCII, '', clean)

            # convert all to utf-8
            with open(dst, "w", encoding=job.encoding) as wf:
                wf.write(clean)
            count += 1
        except Exception as exc:
            logging.error("could not process %s: %r", src, exc)
    return count


def parallel(src_dst_list, total):
    count = cpu_count() - 1 or 1
    pool = Pool(processes=count)
    ret = 0
    for i in tqdm.tqdm(pool.imap(process_multi_file, src_dst_list), total=total):
        ret += i
    return ret


def parse_args(_, parser):
    parser.add_argument(
        "input",
        type=str,
        help="Location of the dataset archives files. Files ending in .zst are treated as \
                        archives, all others as raw text. Can be a glob (/dataset/*.xz, /dataset/*.txt)",
    )
    parser.add_argument(
        "output", type=str, help="Location to write the extracted files"
    )
    
    parser.add_argument(
        "--detect_encoding", action="store_true", help="detect encoding of file before opening."
    )
    
    parser.add_argument(
        "--ascii_only", action="store_true", help="filter out non ascii characters"
    )
   
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="removes the output directory if exists",
    )
    parser.add_argument(
        "--nproc",
        type=int,
        default=cpu_count() - 1,
        help="The number of parallel processes to use",
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default="UTF-8",
        help="The encoding to use for the output dataset",
    )


def local_parse_args(args):
    parser = argparse_flags.ArgumentParser()
    parse_args(args, parser)
    return parser.parse_args(args[1:])


def main(args):
    # default
    archives = lm.human.filepaths_from_user_input(args.input)
    if not archives:
        logging.error("no input data files found with input: %s. aborting", args.input)
        sys.exit(-1)
        return

    if os.path.exists(args.output):
        if not args.force:
            logging.error(
                "output directory %s exists. use --force to remove everything inside it",
                args.output,
            )
            return
        logging.error("output directory %s exists. deleting.", args.output)
        shutil.rmtree(args.output)

    os.makedirs(args.output)

    def job_gen(cpu_count, src_list, dst):
        all_dst = []
        for src in src_list:
            dst = os.path.join(
                args.output, os.path.splitext(os.path.basename(src))[0] + ".txt"
            )
            all_dst.append((src, dst))

        for chunk in chunks(all_dst, cpu_count):
            yield CleanTextJob(chunk, ascii_only=False, detect_encoding=args.detect_encoding, encoding=args.encoding)

    start = time.time()
    count = parallel(job_gen(args.nproc, archives, args.output), total=len(archives))
    end = time.time()

    logging.info(
        "processed {} files in {:.2f}s, {} / {} good files.".format(
            len(archives), end - start, count, len(archives)
        )
    )


if __name__ == "__main__":
    app.run(main, flags_parser=local_parse_args)
