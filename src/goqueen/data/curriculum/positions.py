"""Position sources for the curriculum: SGF game records and random play."""

import random
from collections.abc import Iterator
from pathlib import Path

from goqueen.board import EMPTY, NUM_POINTS, PASS, SIZE, Board, Color
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
