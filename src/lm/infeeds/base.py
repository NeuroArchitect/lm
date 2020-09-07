"Base infeed class"
import abc
from typing import Dict

import tensorflow as tf
from pydantic import BaseModel


class InfeedConfig(BaseModel):
    batch_size: int


class Infeed(abc.ABC):
    """
    An infeed abstracts the operation of creating a stream of examples.
    """

    @abc.abstractmethod
    def __call__(self, params: Dict) -> tf.data.Dataset:
        """
        Configures and Allocates a tensorflow dataset
        """
