import os

import tensorflow as tf


def filepaths_from_user_input(location):
    "utility method with few input heuristic to transform a 'location' into file URLs"
    location = location.strip()
    # check if is an index file
    txt_files = []
    if tf.io.gfile.exists(location):
        if not tf.io.gfile.isdir(location):
            with tf.io.gfile.GFile(location) as fd:
                for l in fd.readlines():
                    if tf.io.gfile.exists(l):
                        txt_files.append(l)
    if txt_files:
        return txt_files

    txt_files = list(p for p in tf.io.gfile.glob(location) if not tf.io.gfile.isdir(p))

    # try with general glob
    if not txt_files:
        txt_files = list(tf.io.gfile.glob(os.path.join(location, "*")))

    if not txt_files:
        # is the input a list of files?
        txt_files = location.split(" ")
        txt_files = list(p for p in txt_files if tf.io.gfile.exists(p))

    # filter directories
    txt_files = list(p for p in txt_files if not tf.io.gfile.isdir(p))
    return txt_files
