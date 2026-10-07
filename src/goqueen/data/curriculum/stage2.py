"""Stage 2 tasks: dynamic facts about the position now. Answers use the rules engine."""

import random

from goqueen.board import BLACK, EMPTY, NUM_POINTS, WHITE, Board, Color, format_gtp
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

# ----------------------------------------------------------------------
# Facts the tasks ask about


def atari_stones(board: Board, color: Color) -> set[int]:
    """All ``color`` stones whose group has exactly one liberty."""
    out: set[int] = set()
    for g in board.groups(color):
        if len(g.liberties) == 1:
            out |= g.stones
    return out


def move_result(board: Board, p: int) -> str:
    """Why the player to move may or may not play ``p``.

    One of "legal", "occupied", "ko", "suicide".
    """
    if board[p] != EMPTY:
        return "occupied"
    if board.ko == p:
        return "ko"
    if not board.is_legal(p):
        return "suicide"
    return "legal"


def captures_of(board: Board, p: int) -> list[int]:
    """Stones the player to move captures by playing ``p`` (which must be legal)."""
    captured = board.play(p)
    board.undo()
    return captured


def atari_moves(board: Board) -> set[int]:
    """Legal moves for the player to move that put an opponent group in atari.

    A move does this if, after it, some opponent group next to it has exactly
    one liberty and had more than one before.
    """
    me = board.to_move
    them = me.opponent
    out: set[int] = set()
    for p in board.legal_moves():
        before = {}
        for g in _adjacent_groups(board, p, them):
            before[min(g.stones)] = len(g.liberties)
        if not any(n == 2 for n in before.values()):
            continue  # a move removes at most one liberty from each group
        board.play(p)
        for root, n in before.items():
            if n > 1 and board[root] == them and len(board.liberties(root)) == 1:
                out.add(p)
                break
        board.undo()
    return out


def _adjacent_groups(board: Board, p: int, color: Color):
    col, row = p % 19, p // 19
    seen: set[int] = set()
    for c, r in ((col - 1, row), (col + 1, row), (col, row - 1), (col, row + 1)):
        if 0 <= c < 19 and 0 <= r < 19:
            n = r * 19 + c
            if board[n] == color and n not in seen:
                g = board.group(n)
                seen |= g.stones
                yield g


def _pick_empty(board: Board, rng: random.Random, prefer: set[int] | list[int]) -> int:
    """A point from ``prefer`` half the time (if any), else a random legal move."""
    prefer = sorted(prefer)
    if prefer and rng.random() < 0.5:
        return rng.choice(prefer)
    legal = board.legal_moves()
    return rng.choice(legal) if legal else rng.choice(prefer or range(NUM_POINTS))


# ----------------------------------------------------------------------
# Tasks


def liberty_count(board: Board, rng: random.Random, fmt: Format) -> QA | None:
    p = random_stone(board, rng)
    if p is None:
        return None
    n = len(board.liberties(p))
    where = format_gtp(p)
    q = f"How many liberties does the group at {where} have?"
    if fmt is Format.OPEN:
        return q, str(n)
    if fmt is Format.YESNO:
        claim = n if rng.random() < 0.5 else max(1, n + rng.choice((-2, -1, 1, 2)))
        return (
            f"Does the group at {where} have exactly {plural(claim, 'liberty', 'liberties')}?",
            yes_no(claim == n),
        )
    return choice(rng, q, str(n), number_distractors(n, lo=1))


def liberty_points(board: Board, rng: random.Random, fmt: Format) -> QA | None:
    p = random_stone(board, rng)
    if p is None:
        return None
    libs = board.liberties(p)
    where = format_gtp(p)
    if fmt is Format.OPEN:
        if len(libs) > MAX_LIST:
            return None
        return f"List the liberties of the group at {where}.", points_text(libs)
    if fmt is Format.YESNO:
        empties = [x for x in range(NUM_POINTS) if board[x] == EMPTY and x not in libs]
        pool = sorted(libs) if rng.random() < 0.5 or not empties else empties
        x = rng.choice(pool)
        return f"Is {format_gtp(x)} a liberty of the group at {where}?", yes_no(
            x in libs
        )
    if not libs:
        return None
    correct = format_gtp(rng.choice(sorted(libs)))
    wrong = point_distractors(rng, set(libs))
    return choice(
        rng,
        f"Which of these points is a liberty of the group at {where}?",
        correct,
        wrong,
    )


def groups_in_atari(board: Board, rng: random.Random, fmt: Format) -> QA | None:
    color = rng.choice((BLACK, WHITE))
    in_atari = atari_stones(board, color)
    who = color_name(color)
    if fmt is Format.OPEN:
        if len(in_atari) > MAX_LIST:
            return None
        return f"Which {who} stones are in atari?", points_text(in_atari)
    if fmt is Format.YESNO:
        if in_atari and rng.random() < 0.5:
            p = rng.choice(sorted(in_atari))
        else:
            p = random_stone(board, rng, color)
            if p is None:
                return None
        return f"Is the {who} group at {format_gtp(p)} in atari?", yes_no(p in in_atari)
    if not in_atari:
        return None
    correct = format_gtp(rng.choice(sorted(in_atari)))
    safe = set(board.stones(color)) - in_atari
    if len(safe) < 3:
        return None
    wrong = [format_gtp(x) for x in rng.sample(sorted(safe), 3)]
    return choice(rng, f"Which of these {who} stones is in atari?", correct, wrong)


def legal_moves(board: Board, rng: random.Random, fmt: Format) -> QA | None:
    who = color_name(board.to_move)
    legal = board.legal_moves()
    if fmt is Format.OPEN:
        return f"How many legal moves does {who} have, not counting pass?", str(
            len(legal)
        )
    illegal = [p for p in range(NUM_POINTS) if not board.is_legal(p)]
    if fmt is Format.YESNO:
        # Ask about illegal empty points often: those are the hard cases.
        hard = [p for p in illegal if board[p] == EMPTY]
        pool = (
            hard
            if hard and rng.random() < 0.3
            else (illegal if rng.random() < 0.4 else legal)
        )
        if not pool:
            return None
        p = rng.choice(pool)
        return f"Is {format_gtp(p)} a legal move for {who}?", yes_no(board.is_legal(p))
    if not legal or len(illegal) < 3:
        return None
    correct = format_gtp(rng.choice(legal))
    wrong = [format_gtp(p) for p in rng.sample(illegal, min(len(illegal), 6))]
    return choice(rng, f"Which of these is a legal move for {who}?", correct, wrong)


def move_captures(board: Board, rng: random.Random, fmt: Format) -> QA | None:
    who = color_name(board.to_move)
    legal = board.legal_moves()
    if not legal:
        return None
    capturing = [p for p in legal if captures_of(board, p)]
    p = rng.choice(capturing) if capturing and rng.random() < 0.6 else rng.choice(legal)
    captured = captures_of(board, p)
    where = format_gtp(p)
    if fmt is Format.OPEN:
        if len(captured) > MAX_LIST:
            return None
        return f"If {who} plays {where}, which stones are captured?", points_text(
            captured
        )
    if fmt is Format.YESNO:
        return f"Does {who} capture any stones by playing {where}?", yes_no(
            bool(captured)
        )
    n = len(captured)
    return choice(
        rng,
        f"If {who} plays {where}, how many stones are captured?",
        str(n),
        number_distractors(n),
    )


_REASON_TEXT = {
    "legal": "Yes",
    "occupied": "No: the point is occupied",
    "ko": "No: it retakes a ko",
    "suicide": "No: it is suicide",
}


def ko_and_suicide(board: Board, rng: random.Random, fmt: Format) -> QA | None:
    who = color_name(board.to_move)
    ko = board.ko
    suicides = [
        p
        for p in range(NUM_POINTS)
        if board[p] == EMPTY and p != ko and not board.is_legal(p)
    ]
    if fmt is Format.YESNO:
        # Ask about a real ko or suicide point half the time, so "Yes" is not rare.
        kinds = [
            k
            for k, found in (("ko", ko is not None), ("suicide", bool(suicides)))
            if found
        ]
        if not kinds:
            return None
        kind = rng.choice(kinds)
        if rng.random() < 0.5:
            p = ko if kind == "ko" else rng.choice(suicides)
        else:
            p = _pick_empty(board, rng, [])
        reason, where = move_result(board, p), format_gtp(p)
        if kind == "ko":
            return f"Is {where} a ko point that {who} may not take back now?", yes_no(
                reason == "ko"
            )
        return f"Would {who} at {where} be suicide?", yes_no(reason == "suicide")

    p = _pick_empty(board, rng, suicides + ([] if ko is None else [ko]))
    if rng.random() < 0.1:
        stone = random_stone(board, rng)  # sometimes ask about an occupied point
        p = p if stone is None else stone
    reason, where = move_result(board, p), format_gtp(p)
    question = f"Can {who} play {where}? If not, why not?"
    if fmt is Format.OPEN:
        return question, _REASON_TEXT[reason]
    return choice(rng, question, _REASON_TEXT[reason], list(_REASON_TEXT.values()))


def atari_move(board: Board, rng: random.Random, fmt: Format) -> QA | None:
    who = color_name(board.to_move)
    them = color_name(board.to_move.opponent)
    moves = atari_moves(board)
    if fmt is Format.OPEN:
        if len(moves) > MAX_LIST:
            return None
        return f"Which moves by {who} put {them} stones in atari?", points_text(moves)
    if fmt is Format.YESNO:
        p = _pick_empty(board, rng, moves)
        return f"Does {who} at {format_gtp(p)} put any {them} stones in atari?", yes_no(
            p in moves
        )
    if not moves:
        return None
    correct = format_gtp(rng.choice(sorted(moves)))
    others = [p for p in board.legal_moves() if p not in moves]
    if len(others) < 3:
        return None
    wrong = [format_gtp(p) for p in rng.sample(others, 3)]
    return choice(
        rng, f"Which move by {who} puts {them} stones in atari?", correct, wrong
    )


TASKS: dict[str, Task] = {
    "liberty_count": liberty_count,
    "liberty_points": liberty_points,
    "groups_in_atari": groups_in_atari,
    "legal_moves": legal_moves,
    "move_captures": move_captures,
    "ko_and_suicide": ko_and_suicide,
    "atari_move": atari_move,
}
