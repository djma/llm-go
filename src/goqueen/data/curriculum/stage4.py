"""Stage 4 tasks that need search: ladders and capturing races.

Other stage 3 and 4 tasks are stage 1 and 2 tasks asked after a line of
moves (see ``generate``). These two ask about the future directly.
"""

import random

from goqueen.board import BLACK, WHITE, Board, format_gtp
from goqueen.board.tactics import RaceResult, ladder_starts, race
from goqueen.data.curriculum.core import (
    QA,
    Format,
    Task,
    choice,
    color_name,
    points_text,
    yes_no,
)
from goqueen.data.curriculum.positions import race_position

MAX_CANDIDATES = 12


def ladder_task(board: Board, rng: random.Random, fmt: Format) -> QA | None:
    me = board.to_move
    them = me.opponent
    candidates = [g for g in board.groups(them) if len(g.liberties) == 2]
    rng.shuffle(candidates)
    want_capture = rng.random() < 0.5  # balance "works" and "fails"
    found = None
    for g in candidates[:MAX_CANDIDATES]:
        prey = min(g.stones)
        starts = ladder_starts(board, prey)
        if starts is None:
            continue
        if found is None or bool(starts) == want_capture:
            found = (prey, starts)
        if bool(starts) == want_capture:
            break
    if found is None:
        return None
    prey, starts = found
    who, where = color_name(me), format_gtp(prey)
    if fmt is Format.OPEN:
        q = f"{who} to play. Can {who} capture the {color_name(them)} stones at {where} in a ladder? If so, where does it start?"
        return (
            q,
            f"Yes, at {points_text(starts).replace(', ', ' or ')}" if starts else "No",
        )
    libs = sorted(board.liberties(prey))
    if fmt is Format.YESNO:
        start = rng.choice(libs)
        q = f"{who} to play. Does a ladder starting at {format_gtp(start)} capture the {color_name(them)} stones at {where}?"
        return q, yes_no(start in starts)
    if len(starts) != 1:
        return None
    legal = board.legal_moves()
    wrong = [format_gtp(p) for p in libs if p not in starts]
    wrong += [
        format_gtp(p) for p in rng.sample(legal, min(4, len(legal))) if p not in libs
    ]
    return choice(
        rng,
        f"{who} to play. Which move starts a ladder that captures {where}?",
        format_gtp(starts[0]),
        wrong,
    )


_RACE_ANSWER = {
    RaceResult.BLACK: "Black",
    RaceResult.WHITE: "White",
    RaceResult.NEITHER: "neither (seki)",
}


def race_question(
    board: Board, black: int, white: int, rng: random.Random, fmt: Format
) -> QA | None:
    result = race(board, black, white)
    if result is None:
        return None
    who = color_name(board.to_move)
    q = (
        f"{who} to play. In the capturing race between the Black stones at {format_gtp(black)} "
        f"and the White stones at {format_gtp(white)}, who captures the other?"
    )
    if fmt is Format.OPEN:
        return q, _RACE_ANSWER[result]
    if fmt is Format.YESNO:
        claim = rng.choice((RaceResult.BLACK, RaceResult.WHITE))
        verb = f"Does {color_name(BLACK if claim == RaceResult.BLACK else WHITE)} capture the other side"
        q2 = (
            f"{who} to play. In the capturing race between the Black stones at {format_gtp(black)} "
            f"and the White stones at {format_gtp(white)}: {verb}?"
        )
        return q2, yes_no(result == claim)
    options = [*_RACE_ANSWER.values(), "whoever plays first"]
    return choice(rng, q, _RACE_ANSWER[result], options)


def make_race_position(rng: random.Random) -> tuple[Board, int, int, RaceResult] | None:
    """A closed race whose answer is a random target, so answers are balanced."""
    target = rng.choice(list(RaceResult))
    fallback = None
    for _ in range(30):
        found = race_position(rng)
        if found is None:
            continue
        board, a, b = found
        result = race(board, a, b)
        if result is None:
            continue
        if result == target:
            return board, a, b, result
        fallback = (board, a, b, result)
    return fallback


TASKS: dict[str, Task] = {"ladder": ladder_task}
