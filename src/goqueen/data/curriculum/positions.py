"""Position sources for the curriculum: SGF game records and random play."""

import random
from collections.abc import Iterator
from pathlib import Path

from goqueen.board import BLACK, EMPTY, NUM_POINTS, PASS, SIZE, WHITE, Board, Color
from goqueen.data.sgf import SGFError, read_file, replay


def own_eye(board: Board, p: int, color: Color) -> bool:
    """True if every on-board neighbour of ``p`` is a ``color`` stone."""
    col, row = p % SIZE, p // SIZE
    for c, r in ((col - 1, row), (col + 1, row), (col, row - 1), (col, row + 1)):
        if 0 <= c < SIZE and 0 <= r < SIZE and board[r * SIZE + c] != color:
            return False
    return True


def random_move(board: Board, rng: random.Random) -> int:
    """A uniform random legal move that does not fill an own eye, or PASS if none."""
    color = board.to_move
    # Try random points first; fall back to the full list on crowded boards.
    for _ in range(20):
        p = rng.randrange(NUM_POINTS)
        if board[p] == EMPTY and board.is_legal(p) and not own_eye(board, p, color):
            return p
    candidates = [p for p in board.legal_moves() if not own_eye(board, p, color)]
    return rng.choice(candidates) if candidates else PASS


def random_play(board: Board, rng: random.Random, moves: int) -> list[int]:
    """Play up to ``moves`` random moves (see ``random_move``).

    Stops early after two passes in a row. Returns the moves played.
    """
    played = []
    for _ in range(moves):
        if board.consecutive_passes >= 2:
            break
        p = random_move(board, rng)
        board.play(p)
        played.append(p)
    return played


def random_position(
    rng: random.Random, min_moves: int = 10, max_moves: int = 250
) -> Board:
    """A position after a random number of random moves from the empty board."""
    board = Board()
    random_play(board, rng, rng.randint(min_moves, max_moves))
    return board


def sgf_files(root: str | Path) -> list[Path]:
    """All ``.sgf`` files under ``root``, sorted so the order is reproducible."""
    return sorted(Path(root).rglob("*.sgf"))


def sgf_position(path: Path, rng: random.Random) -> Board | None:
    """A random position from the main line of one game, or None if it cannot be read."""
    try:
        positions = list(replay(read_file(str(path))))
    except (SGFError, OSError, UnicodeError):
        return None
    if not positions:
        return None
    return rng.choice(positions).board


def sgf_positions(files: list[Path], rng: random.Random) -> Iterator[Board]:
    """Random positions from random games, forever. Unreadable games are skipped."""
    if not files:
        raise ValueError("no SGF files")
    while True:
        board = sgf_position(rng.choice(files), rng)
        if board is not None:
            yield board


# ----------------------------------------------------------------------
# Capturing races


def _symmetry(p: int, k: int) -> int:
    """One of the 8 symmetries of the board (k in 0..7)."""
    c, r = p % SIZE, p // SIZE
    if k & 1:
        c = SIZE - 1 - c
    if k & 2:
        r = SIZE - 1 - r
    if k & 4:
        c, r = r, c
    return r * SIZE + c


def race_position(
    rng: random.Random, max_tries: int = 200
) -> tuple[Board, int, int] | None:
    """A closed capturing race on the edge: (board, Black race stone, White race stone).

    The two groups may not touch; ``race`` returns None for those.

    Layout before a random symmetry, columns from x0 on the first lines:
    White wall | Black race | middle | White race | Black wall. Middle cells
    are shared liberties or stones of either race group; random gaps in the
    walls give outside liberties. The player to move is random. Only
    positions that pass ``race_is_closed`` are returned.
    """
    from goqueen.board.tactics import race_is_closed

    for _ in range(max_tries):
        h = rng.randint(2, 4)
        x0 = rng.randint(1, SIZE - 8)
        cells: dict[tuple[int, int], Color] = {}
        black_race = [(x0 + 1, y) for y in range(h)]
        white_race = [(x0 + 3, y) for y in range(h)]
        for xy in black_race:
            cells[xy] = BLACK
        for xy in white_race:
            cells[xy] = WHITE
        middle = [(x0 + 2, y) for y in range(h)]
        for i, xy in enumerate(middle):
            top = i == h - 1
            pick = rng.random()
            if top or pick < 0.25:
                cells[xy] = BLACK if rng.random() < 0.5 else WHITE
            elif pick < 0.4:
                cells[xy] = BLACK
            elif pick < 0.55:
                cells[xy] = WHITE
            # else: empty, a shared liberty
        if not any(xy in cells for xy in middle):
            continue
        # Walls: each empty cell next to exactly one colour of race stone gets the other colour.
        race_cells = dict(cells)
        for x in range(x0 - 1, x0 + 6):
            for y in range(h + 2):
                if (x, y) in cells or not (0 <= x < SIZE):
                    continue
                touching = {
                    race_cells[n]
                    for n in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))
                    if n in race_cells
                }
                if len(touching) == 1 and rng.random() > 0.2:
                    cells[(x, y)] = touching.pop().opponent
        board = Board()
        k = rng.randrange(8)
        for (x, y), color in cells.items():
            board.set_stone(_symmetry(y * SIZE + x, k), color)
        board.to_move = rng.choice((BLACK, WHITE))
        a = _symmetry(black_race[0][1] * SIZE + black_race[0][0], k)
        b = _symmetry(white_race[0][1] * SIZE + white_race[0][0], k)
        if board[a] != BLACK or board[b] != WHITE:
            continue
        if race_is_closed(board, a, b):
            return board, a, b
    return None
