"""Stage 1 tasks: static facts about the position now. Answers read the board."""

import random

from goqueen.board import BLACK, EMPTY, NUM_POINTS, WHITE, Board, format_gtp
from goqueen.data.curriculum.core import (
    MAX_LIST,
    QA,
    Format,
    Task,
    choice,
    color_name,
    number_distractors,
    plural,
    point_distractors,
    points_text,
    random_stone,
    yes_no,
)


def stone_at(board: Board, rng: random.Random, fmt: Format) -> QA | None:
    # Ask about stones more often than empty points: an empty board is dull.
    p = random_stone(board, rng) if rng.random() < 0.7 else None
    if p is None:
        p = rng.randrange(NUM_POINTS)
    name, actual = format_gtp(p), board[p]
    if fmt is Format.OPEN:
        return f"What is at {name}?", color_name(actual)
    if fmt is Format.YESNO:
        claim = (
            actual
            if rng.random() < 0.5
            else rng.choice([c for c in (BLACK, WHITE, EMPTY) if c != actual])
        )
        if claim == EMPTY:
            return f"Is {name} empty?", yes_no(actual == EMPTY)
        return f"Is there a {color_name(claim)} stone at {name}?", yes_no(
            actual == claim
        )
    # Three real options only; add "off the board" as a fixed fourth.
    options = ["Black", "White", "empty", "not a point on the board"]
    return choice(rng, f"What is at {name}?", color_name(actual), options)


def stones_of_color(board: Board, rng: random.Random, fmt: Format) -> QA | None:
    color = rng.choice((BLACK, WHITE))
    stones = board.stones(color)
    who = color_name(color)
    if fmt is Format.OPEN:
        if len(stones) > MAX_LIST:
            return None
        return f"List all {who} stones.", points_text(stones)
    if fmt is Format.YESNO:
        if not stones:
            return None
        pick = rng.sample(stones, min(len(stones), rng.randint(1, 4)))
        truth = rng.random() < 0.5
        if not truth:
            # Swap one point for a point without a stone of this colour.
            others = [p for p in range(NUM_POINTS) if board[p] != color]
            pick[rng.randrange(len(pick))] = rng.choice(others)
        return f"Are all of {points_text(pick)} {who} stones?", yes_no(truth)
    if not stones:
        return None
    correct = format_gtp(rng.choice(stones))
    wrong = point_distractors(rng, set(stones))
    return choice(rng, f"Which of these points has a {who} stone?", correct, wrong)


def stone_count(board: Board, rng: random.Random, fmt: Format) -> QA | None:
    color = rng.choice((BLACK, WHITE))
    n = board.count(color)
    q = f"How many {color_name(color)} stones are on the board?"
    if fmt is Format.OPEN:
        return q, str(n)
    if fmt is Format.YESNO:
        claim = n if rng.random() < 0.5 else max(0, n + rng.choice((-2, -1, 1, 2)))
        return (
            f"{'Is' if claim == 1 else 'Are'} there exactly {plural(claim, color_name(color) + ' stone', color_name(color) + ' stones')} on the board?",
            yes_no(claim == n),
        )
    return choice(rng, q, str(n), number_distractors(n))


def captured_count(board: Board, rng: random.Random, fmt: Format) -> QA | None:
    color = rng.choice((BLACK, WHITE))
    n = board.captures[color]
    q = f"How many stones has {color_name(color)} captured so far?"
    if fmt is Format.OPEN:
        return q, str(n)
    if fmt is Format.YESNO:
        claim = n if rng.random() < 0.5 else max(0, n + rng.choice((-2, -1, 1, 2)))
        return (
            f"Has {color_name(color)} captured exactly {plural(claim, 'stone', 'stones')} so far?",
            yes_no(claim == n),
        )
    return choice(rng, q, str(n), number_distractors(n))


def whose_turn(board: Board, rng: random.Random, fmt: Format) -> QA | None:
    who = color_name(board.to_move)
    if fmt is Format.OPEN:
        return "Whose turn is it?", who
    if fmt is Format.YESNO:
        claim = rng.choice((BLACK, WHITE))
        return f"Is it {color_name(claim)}'s turn?", yes_no(claim == board.to_move)
    options = ["Black", "White", "either player", "nobody: the game is over"]
    return choice(rng, "Whose turn is it?", who, options)


def group_size(board: Board, rng: random.Random, fmt: Format) -> QA | None:
    p = random_stone(board, rng)
    if p is None:
        return None
    n = len(board.group(p).stones)
    q = f"How many stones are in the group at {format_gtp(p)}?"
    if fmt is Format.OPEN:
        return q, str(n)
    if fmt is Format.YESNO:
        claim = n if rng.random() < 0.5 else max(1, n + rng.choice((-2, -1, 1, 2)))
        return (
            f"Does the group at {format_gtp(p)} have exactly {plural(claim, 'stone', 'stones')}?",
            yes_no(claim == n),
        )
    return choice(rng, q, str(n), number_distractors(n, lo=1))


def group_stones(board: Board, rng: random.Random, fmt: Format) -> QA | None:
    p = random_stone(board, rng)
    if p is None:
        return None
    stones = board.group(p).stones
    if fmt is Format.OPEN:
        if len(stones) > MAX_LIST:
            return None
        return f"List the stones in the group at {format_gtp(p)}.", points_text(stones)
    if fmt is Format.YESNO:
        mates = sorted(stones - {p})
        if mates and rng.random() < 0.5:
            other = rng.choice(mates)
        else:
            others = sorted(set(board.stones(board[p])) - stones)
            if not others:
                return None
            other = rng.choice(others)
        return (
            f"Are {format_gtp(p)} and {format_gtp(other)} in the same group?",
            yes_no(other in stones),
        )
    group_mates = stones - {p}
    if not group_mates:
        return None
    correct = format_gtp(rng.choice(sorted(group_mates)))
    wrong = point_distractors(rng, set(stones))
    return choice(
        rng,
        f"Which of these stones is in the same group as {format_gtp(p)}?",
        correct,
        wrong,
    )


def group_count(board: Board, rng: random.Random, fmt: Format) -> QA | None:
    color = rng.choice((BLACK, WHITE))
    n = len(board.groups(color))
    q = f"How many {color_name(color)} groups are on the board?"
    if fmt is Format.OPEN:
        return q, str(n)
    if fmt is Format.YESNO:
        claim = n if rng.random() < 0.5 else max(0, n + rng.choice((-2, -1, 1, 2)))
        return (
            f"Does {color_name(color)} have exactly {plural(claim, 'group', 'groups')} on the board?",
            yes_no(claim == n),
        )
    return choice(rng, q, str(n), number_distractors(n))


TASKS: dict[str, Task] = {
    "stone_at": stone_at,
    "stones_of_color": stones_of_color,
    "stone_count": stone_count,
    "captured_count": captured_count,
    "whose_turn": whose_turn,
    "group_size": group_size,
    "group_stones": group_stones,
    "group_count": group_count,
}
