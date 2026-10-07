"""Benchmark the rules engine with random self-play.

Run: ``uv run python -m goqueen.board.bench``
"""

import argparse
import random
import time

from goqueen.board.board import EMPTY, Board, Color
from goqueen.board.coords import NUM_POINTS, PASS, SIZE


def _own_eye(board: Board, p: int, color: Color) -> bool:
    """True if every on-board neighbour of ``p`` is a ``color`` stone."""
    col, row = p % SIZE, p // SIZE
    for c, r in ((col - 1, row), (col + 1, row), (col, row - 1), (col, row + 1)):
        if 0 <= c < SIZE and 0 <= r < SIZE and board[r * SIZE + c] != color:
            return False
    return True


def random_game(rng: random.Random, max_moves: int = 700) -> tuple[Board, list[int]]:
    """Play one random game; return the final board and the moves played.

    Each move picks uniformly among legal moves that do not fill an own eye.
    """
    board = Board()
    moves: list[int] = []
    while len(moves) < max_moves and board.consecutive_passes < 2:
        color = board.to_move
        candidates = [p for p in board.legal_moves() if not _own_eye(board, p, color)]
        move = rng.choice(candidates) if candidates else PASS
        board.play(move)
        moves.append(move)
    return board, moves


def bench_play(games: list[list[int]]) -> float:
    """Moves/sec for ``play`` alone, replaying recorded games."""
    total = sum(len(moves) for moves in games)
    start = time.perf_counter()
    for moves in games:
        b = Board()
        for p in moves:
            b.play(p)
    return total / (time.perf_counter() - start)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    start = time.perf_counter()
    games = [random_game(rng)[1] for _ in range(args.games)]
    total_moves = sum(len(moves) for moves in games)
    elapsed = time.perf_counter() - start
    print(f"random self-play: {args.games} games, {total_moves} moves, {elapsed:.2f}s")
    print(f"  {total_moves / elapsed:,.0f} moves/sec (legal_moves + play per move)")

    rate = bench_play(games)
    print(f"  {rate:,.0f} moves/sec (play only, replay of recorded games)")

    board, _ = random_game(random.Random(args.seed))
    n = 2000
    start = time.perf_counter()
    for _ in range(n):
        board.legal_moves()
    print(
        f"  {n / (time.perf_counter() - start):,.0f} legal_moves() calls/sec (end position)"
    )
    start = time.perf_counter()
    for _ in range(n):
        board.groups()
    print(
        f"  {n / (time.perf_counter() - start):,.0f} groups() calls/sec (end position)"
    )
    start = time.perf_counter()
    for _ in range(n):
        board.score()
    print(
        f"  {n / (time.perf_counter() - start):,.0f} score() calls/sec (end position)"
    )
    empty = sum(1 for p in range(NUM_POINTS) if board[p] == EMPTY)
    print(f"  end position: {NUM_POINTS - empty} stones, score {board.score():+.1f}")


if __name__ == "__main__":
    main()
