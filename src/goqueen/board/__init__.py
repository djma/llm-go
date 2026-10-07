"""Go rules engine: board state, legal moves, captures, ko, liberties, GTP coordinates.

Phase 0. Every curriculum answer is made and checked here, so it must be exact.
"""

from goqueen.board.board import (
    BLACK,
    EMPTY,
    KOMI,
    WHITE,
    Board,
    Color,
    Group,
    IllegalMoveError,
)
from goqueen.board.coords import (
    FILES,
    NUM_POINTS,
    PASS,
    SIZE,
    col_row,
    format_gtp,
    parse_gtp,
    point,
)

__all__ = [
    "BLACK",
    "EMPTY",
    "FILES",
    "KOMI",
    "NUM_POINTS",
    "PASS",
    "SIZE",
    "WHITE",
    "Board",
    "Color",
    "Group",
    "IllegalMoveError",
    "col_row",
    "format_gtp",
    "parse_gtp",
    "point",
]
