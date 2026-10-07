"""Shared parts of the QA curriculum: records, formats, answer formatting.

Every task is a function ``(board, rng, fmt) -> (question, answer) | None``.
It reads the board through ``goqueen.board`` only, so every answer is exact.
It returns None when it cannot make a good question from this position.
Given the same board and the same ``random.Random`` state, a task returns the
same question and answer.
"""

import random
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

from goqueen.board import (
    BLACK,
    EMPTY,
    NUM_POINTS,
    WHITE,
    Board,
    Color,
    format_gtp,
    parse_gtp,
)


class Format(StrEnum):
    OPEN = "open"  # free answer: "What is at Q16?" -> "White"
    YESNO = "yesno"  # check a claim: "Is Q16 Black?" -> "No"
    CHOICE = "choice"  # four options: "... A) 2 B) 3 C) 4 D) 5" -> "C) 4"


FORMATS: tuple[Format, ...] = tuple(Format)

QA = tuple[str, str]
Task = Callable[[Board, random.Random, Format], QA | None]

COLOR_NAME = {BLACK: "Black", WHITE: "White", EMPTY: "empty"}
_SYMBOLS = {EMPTY: ".", BLACK: "X", WHITE: "O"}
_FROM_SYMBOL = {".": EMPTY, "X": BLACK, "O": WHITE}

# Longest point list we put in an answer. Longer lists make poor targets.
MAX_LIST = 30


@dataclass(frozen=True)
class Example:
    """One QA pair with the position it is about."""

    stage: int
    task: str
    format: str
    question: str
    answer: str
    position: dict[str, Any]
    line: list[str]  # moves played before asking (stages 3 and 4), GTP

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


# ----------------------------------------------------------------------
# Positions


def encode_position(board: Board) -> dict[str, Any]:
    """A JSON-safe record of everything the answers depend on."""
    stones = "".join(_SYMBOLS[board[p]] for p in range(NUM_POINTS))
    ko = board.ko
    return {
        "stones": stones,  # point order: A1, B1, ..., T1, A2, ...
        "to_move": "B" if board.to_move == BLACK else "W",
        "ko": None if ko is None else format_gtp(ko),
        "captures": {"B": board.captures[BLACK], "W": board.captures[WHITE]},
    }


def decode_position(record: dict[str, Any]) -> Board:
    """Rebuild a board from ``encode_position`` output."""
    board = Board()
    for p, ch in enumerate(record["stones"]):
        if ch != ".":
            board.set_stone(p, _FROM_SYMBOL[ch])
    board.to_move = BLACK if record["to_move"] == "B" else WHITE
    board.captures[BLACK] = record["captures"]["B"]
    board.captures[WHITE] = record["captures"]["W"]
    board.set_ko(None if record["ko"] is None else parse_gtp(record["ko"]))
    return board


# ----------------------------------------------------------------------
# Answer text


def points_text(points: Sequence[int] | set[int] | frozenset[int]) -> str:
    """Points in point order (A1, B1, ..., T19), comma separated; "none" if empty."""
    if not points:
        return "none"
    return ", ".join(format_gtp(p) for p in sorted(points))


def plural(n: int, one: str, many: str) -> str:
    """``"1 stone"``, ``"2 stones"``."""
    return f"{n} {one if n == 1 else many}"


def yes_no(value: bool) -> str:
    return "Yes" if value else "No"


def choice(
    rng: random.Random, question: str, correct: str, wrong: Sequence[str]
) -> QA | None:
    """Four options in random order. ``wrong`` must hold at least three distinct options."""
    distractors = sorted(set(wrong) - {correct})
    if len(distractors) < 3:
        return None
    options = [correct, *rng.sample(distractors, 3)]
    rng.shuffle(options)
    labels = "ABCD"
    listed = " ".join(f"{labels[i]}) {o}" for i, o in enumerate(options))
    i = options.index(correct)
    return f"{question} {listed}", f"{labels[i]}) {correct}"


def number_distractors(n: int, lo: int = 0) -> list[str]:
    """Nearby wrong numbers, none below ``lo``."""
    return [str(k) for k in range(max(lo, n - 4), n + 5) if k != n]


def point_distractors(rng: random.Random, exclude: set[int], k: int = 6) -> list[str]:
    pool = [
        p for p in rng.sample(range(NUM_POINTS), k + len(exclude)) if p not in exclude
    ]
    return [format_gtp(p) for p in pool[:k]]


def color_name(c: Color) -> str:
    return COLOR_NAME[c]


def random_stone(
    board: Board, rng: random.Random, color: Color | None = None
) -> int | None:
    stones = (
        board.stones(color)
        if color is not None
        else board.stones(BLACK) + board.stones(WHITE)
    )
    return rng.choice(stones) if stones else None
