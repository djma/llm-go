"""Programmatic QA curriculum (stages 1-4). Every answer comes from ``goqueen.board``."""

from goqueen.data.curriculum.core import (
    FORMATS,
    Example,
    Format,
    decode_position,
    encode_position,
)
from goqueen.data.curriculum.generate import STAGE_TASKS, generate, make_example

__all__ = [
    "FORMATS",
    "STAGE_TASKS",
    "Example",
    "Format",
    "decode_position",
    "encode_position",
    "generate",
    "make_example",
]
