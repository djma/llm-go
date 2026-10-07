"""Make curriculum examples from a source of positions.

Stages 1 and 2 ask about the position now. Stages 3 and 4 first play a line
of 1-12 random legal moves from the position, then ask a stage 1 or stage 2
question about the result; the example keeps the position before the line.
Stage 4 also has ladder and capturing-race tasks with no line.

Example ``i`` of a run with seed ``s`` uses its own ``random.Random(f"{s}:{i}")``,
so any one example can be made again without making the ones before it.
"""

import random
from collections.abc import Callable

from goqueen.board import PASS, Board, format_gtp
from goqueen.data.curriculum import stage1, stage2, stage4
from goqueen.data.curriculum.core import (
    FORMATS,
    Example,
    Format,
    Task,
    color_name,
    encode_position,
)
from goqueen.data.curriculum.positions import random_move

PositionSource = Callable[[random.Random], Board]

MAX_TRIES = 20
MAX_LINE = 12

# Tasks on the position as it is, by stage.
STAGE_TASKS: dict[int, dict[str, Task]] = {
    1: stage1.TASKS,
    2: stage2.TASKS,
    4: stage4.TASKS,
}

# Tasks asked after a line of moves, by stage.
LINE_TASKS: dict[int, dict[str, Task]] = {3: stage1.TASKS, 4: stage2.TASKS}

# Stage 4 race questions make their own position.
RACE = "race"

TASK_NAMES: dict[int, list[str]] = {
    1: sorted(stage1.TASKS),
    2: sorted(stage2.TASKS),
    3: sorted(stage1.TASKS),
    4: sorted([*stage2.TASKS, *stage4.TASKS, RACE]),
}


def line_text(board: Board, line: list[int]) -> str:
    """``"Black Q16, White D4, Black pass"`` for ``line`` played from ``board``."""
    color = board.to_move
    parts = []
    for p in line:
        parts.append(f"{color_name(color)} {format_gtp(p)}")
        color = color.opponent
    return ", ".join(parts)


def random_line(board: Board, rng: random.Random, length: int) -> list[int]:
    """Up to ``length`` random legal moves from ``board`` (which is not changed)."""
    b = board.copy()
    line = []
    for _ in range(length):
        p = random_move(b, rng)
        b.play(p)
        line.append(p)
        if p == PASS and b.consecutive_passes >= 2:
            break
    return line


def make_example(
    board: Board,
    stage: int,
    rng: random.Random,
    task: str | None = None,
    fmt: str | None = None,
) -> Example | None:
    """One example about ``board`` from a random (or given) task and format."""
    names = TASK_NAMES[stage]
    for _ in range(MAX_TRIES):
        name = task or rng.choice(names)
        f = Format(fmt) if fmt else rng.choice(FORMATS)
        ex = _try(board, stage, name, f, rng)
        if ex is not None:
            return ex
    return None


def _try(
    board: Board, stage: int, name: str, fmt: Format, rng: random.Random
) -> Example | None:
    if stage == 4 and name == RACE:
        found = stage4.make_race_position(rng)
        if found is None:
            return None
        race_board, black, white, _ = found
        qa = stage4.race_question(race_board, black, white, rng, fmt)
        return (
            None
            if qa is None
            else Example(stage, name, str(fmt), *qa, encode_position(race_board), [])
        )

    if stage in LINE_TASKS and name in LINE_TASKS[stage]:
        line = random_line(board, rng, rng.randint(1, MAX_LINE))
        after = board.copy()
        for p in line:
            after.play(p)
        qa = LINE_TASKS[stage][name](after, rng, fmt)
        if qa is None:
            return None
        question = f"After {line_text(board, line)}: {qa[0]}"
        return Example(
            stage,
            name,
            str(fmt),
            question,
            qa[1],
            encode_position(board),
            [format_gtp(p) for p in line],
        )

    qa = STAGE_TASKS[stage][name](board, rng, fmt)
    if qa is None:
        return None
    return Example(stage, name, str(fmt), *qa, encode_position(board), [])


def generate(
    source: PositionSource, n: int, seed: int, stages: tuple[int, ...] = (1, 2)
) -> list[Example]:
    """``n`` examples, stages and tasks chosen uniformly, positions from ``source``."""
    out = []
    i = 0
    while len(out) < n:
        rng = random.Random(f"{seed}:{i}")
        i += 1
        board = source(rng)
        ex = make_example(board, rng.choice(stages), rng)
        if ex is not None:
            out.append(ex)
    return out
