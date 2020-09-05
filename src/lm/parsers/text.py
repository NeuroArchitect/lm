import os

import tensorflow as tf
from absl import logging

import lm

_UTF8 = "UTF-8"
_FILE_EXT = (".csv", ".tsv", ".txt")
_GZIP_EXT = (".gz", ".gzip")


@lm.register_parser("file", *_FILE_EXT)
def readlines_txt(src):
    with tf.io.gfile.GFile(src, "rb") as fd:
        return [fd.read().decode(_UTF8)]


@lm.register_parser("passage", *_FILE_EXT)
def readpassage_txt(src):
    with tf.io.gfile.GFile(src, "rb") as fd:
        return fd.read().decode(_UTF8).split(".")


@lm.register_parser("line", *_FILE_EXT)
def readfile_txt(src):
    with tf.io.gfile.GFile(src, "rb") as fd:
        return fd.read().decode(_UTF8).splitlines(keepends=True)


@lm.register_parser("file", *_GZIP_EXT)
def readfile_gzip(src):
    with tf.io.gfile.GFile(src, "rb") as fd:
        return [gzip.decompress(fd.read()).decode(_UTF8)]


@lm.register_parser("line", *_GZIP_EXT)
def readlines_gzip(src):
    with tf.io.gfile.GFile(src, "rb") as fd:
        return gzip.decompress(fd.read()).decode(_UTF8).splitlines(keepends=True)
