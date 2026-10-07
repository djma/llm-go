"""Make curriculum examples from a source of positions.

Example ``i`` of a run with seed ``s`` uses its own ``random.Random(f"{s}:{i}")``,
so any one example can be made again without making the ones before it.
"""

import random
from collections.abc import Callable

from goqueen.board import Board
from goqueen.data.curriculum import stage1, stage2
from goqueen.data.curriculum.core import FORMATS, Example, Task, encode_position

STAGE_TASKS: dict[int, dict[str, Task]] = {1: stage1.TASKS, 2: stage2.TASKS}

PositionSource = Callable[[random.Random], Board]

MAX_TRIES = 20


def make_example(
    board: Board,
    stage: int,
    rng: random.Random,
    task: str | None = None,
    fmt: str | None = None,
) -> Example | None:
    """One example about ``board`` from a random (or given) task and format."""
    tasks = STAGE_TASKS[stage]
    for _ in range(MAX_TRIES):
        name = task or rng.choice(sorted(tasks))
        f = next(x for x in FORMATS if x == fmt) if fmt else rng.choice(FORMATS)
        qa = tasks[name](board, rng, f)
        if qa is not None:
            question, answer = qa
            return Example(
                stage, name, str(f), question, answer, encode_position(board), []
            )
    return None


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
