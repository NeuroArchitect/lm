"""Top-level package for lm."""

__author__ = """Fabrizio Milo"""
__email__ = "remove-this-fmilo@entropysource.com"
__version__ = "0.2.2-alpha"


from .registry import (
    get_dataset,
    get_infeed,
    get_model,
    get_parser,
    get_task,
    get_optimizer,
    get_component,
    register_dataset,
    register_encoder,
    register_infeed,
    register_model,
    register_parser,
    register_task,
    register_optimizer,
    register_component,
)

__all__ = [
    "register_model",
    "register_dataset",
    "register_component",
    "get_infeed",
    "get_model",
    "get_task",
    "get_dataset",
    "get_parser",
    "get_optimizer",
    "get_component",
    "register_encoder",
    "register_infeed",
    "register_task",
    "register_parser",
    "__version__",
]
