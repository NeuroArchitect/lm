import os
import sys
import json

from pathlib import Path
import tensorflow as tf
import _jsonnet
from absl import app
from absl.flags import argparse_flags
from absl import logging


def parse_args(args, parser):
    # Parse command line arguments
    parser = parser if parser else argparse_flags.ArgumentParser()
    parser.add_argument("input", type=str)  # Name of TPU to train on, if any


def local_parse_args(args):
    parser = argparse_flags.ArgumentParser()
    parse_args(args, parser)
    return parser.parse_args(args[1:])


#  Returns content if worked, None if file not found, or throws an exception
def try_path(dir, rel):
    if not rel:
        raise RuntimeError("Got invalid filename (empty string).")
    if rel[0] == "/":
        full_path = rel
    else:
        full_path = dir + rel
    if full_path[-1] == "/":
        raise RuntimeError("Attempted to import a directory")

    if not os.path.isfile(full_path):
        return full_path, None
    with open(full_path) as f:
        return full_path, f.read()


def import_callback(dir, rel):
    full_path, content = try_path(dir, rel)
    if content:
        return full_path, content
    raise RuntimeError("File not found")


def main(args):
    try:
        json_str = _jsonnet.evaluate_file(
            args.input, ext_vars={"MODEL_PATH": "Bob"}, import_callback=import_callback,
        )
    except RuntimeError as e:
        logging.error(e)
        sys.exit(-1)


if __name__ == "__main__":
    tf.disable_v2_behavior()
    app.run(main, flags_parser=parse_args)
