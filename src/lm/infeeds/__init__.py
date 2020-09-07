" register all the declared infeeds "
from .readers import TFRecordDatasetReader  # noqa: F403 F401

__INFEEDS__ = [TFRecordDatasetReader]
