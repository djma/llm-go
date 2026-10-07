"""Points and GTP coordinates on a 19x19 board.

A point is an int in 0..360: ``row * 19 + col``. Row 0 is rank 1 (the bottom
edge) and col 0 is file A, so A1 is point 0 and T19 is point 360. This order
matches a ``(19, 19)`` numpy array indexed ``[row, col]``.

``PASS`` (-1) is the pass move.
"""

from typing import Final

SIZE: Final = 19
NUM_POINTS: Final = SIZE * SIZE
PASS: Final = -1

# GTP files skip the letter I.
FILES: Final = "ABCDEFGHJKLMNOPQRST"
_FILE_INDEX: Final = {c: i for i, c in enumerate(FILES)}


def point(col: int, row: int) -> int:
    """Return the point at ``col`` (0 = A) and ``row`` (0 = rank 1)."""
    if not (0 <= col < SIZE and 0 <= row < SIZE):
        raise ValueError(f"off board: col={col}, row={row}")
    return row * SIZE + col


def col_row(p: int) -> tuple[int, int]:
    """Return ``(col, row)`` for point ``p``."""
    if not 0 <= p < NUM_POINTS:
        raise ValueError(f"not a board point: {p}")
    return p % SIZE, p // SIZE


def parse_gtp(text: str) -> int:
    """Parse a GTP vertex such as ``"Q16"`` or ``"pass"``. Case is ignored."""
    s = text.strip().upper()
    if s == "PASS":
        return PASS
    if len(s) < 2 or s[0] not in _FILE_INDEX or not s[1:].isdigit():
        raise ValueError(f"bad GTP vertex: {text!r}")
    rank = int(s[1:])
    if not 1 <= rank <= SIZE:
        raise ValueError(f"bad GTP vertex: {text!r}")
    return (rank - 1) * SIZE + _FILE_INDEX[s[0]]


def format_gtp(p: int) -> str:
    """Format point ``p`` as a GTP vertex, or ``"pass"`` for ``PASS``."""
    if p == PASS:
        return "pass"
    col, row = col_row(p)
    return f"{FILES[col]}{row + 1}"
