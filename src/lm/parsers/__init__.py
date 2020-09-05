import os

import lm

from .text import (
    readfile_gzip,
    readfile_txt,
    readlines_gzip,
    readlines_txt,
    readpassage_txt,
)

_ = (readfile_txt, readlines_txt, readpassage_txt)


def parse_url(location, strategy):
    _, ext = os.path.splitext(location)
    parse_fn = lm.get_parser("%s:%s" % (strategy, ext))
    if parse_fn is None:
        raise ValueError("unsupported %s:%s" % (strategy, ext))
    return parse_fn(location)
