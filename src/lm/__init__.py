"""Top-level package for lm."""

__author__ = """Fabrizio Milo"""
__email__ = "remove-this-fmilo@entropysource.com"
__version__ = "0.2.0-alpha"


from .registry import (
    get_dataset,
    get_infeed,
    get_model,
    get_parser,
    get_task,
    register_dataset,
    register_encoder,
    register_infeed,
    register_model,
    register_parser,
    register_task,
)

__all__ = [
    "register_model",
    "register_dataset",
    "get_infeed",
    "get_model",
    "get_task",
    "get_dataset",
    "get_parser",
    "register_encoder",
    "register_infeed",
    "register_task",
    "register_parser",
    "__version__",
]
