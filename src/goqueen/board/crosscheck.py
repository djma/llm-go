"""Cross-check ``goqueen.board`` against sgfmill, a second Go implementation.

This is the Phase 0 gate: the two must agree on every check over 10K positions.

    uv run python -m goqueen.board.crosscheck --positions 10000
    uv run python -m goqueen.board.crosscheck --sgf-dir /data/sgf

For each position it compares:

- the stones on the board;
- for every empty point: is it legal for the player to move, and if so,
  which stones it captures (sgfmill allows suicide and reports the simple-ko
  point, so a move is legal there if it is not suicide and not the ko point);
- the liberties of every group, from a separate flood fill on sgfmill's grid;
- the area score.

With ``--sgf-dir`` it also replays each game with sgfmill's own SGF parser and
compares every position on the main line. sgfmill is a dev dependency (MIT).
"""

import argparse
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from sgfmill import boards, sgf

from goqueen.board import BLACK, EMPTY, NUM_POINTS, PASS, SIZE, WHITE, Board, format_gtp

_TO_SGFMILL = {BLACK: "b", WHITE: "w"}
_FROM_SGFMILL = {None: EMPTY, "b": BLACK, "w": WHITE}


@dataclass
class Report:
    positions: int = 0
    legality_checks: int = 0
    capture_checks: int = 0
    liberty_checks: int = 0
    score_checks: int = 0
    mismatches: list[str] = field(default_factory=list)

    def fail(self, board: Board, what: str) -> None:
        if len(self.mismatches) < 20:
            self.mismatches.append(f"{what}\n{board}\nto move: {board.to_move.name}")
        else:
            self.mismatches.append(what)


def _rc(p: int) -> tuple[int, int]:
    return p // SIZE, p % SIZE  # sgfmill uses (row, col) with row 0 at the bottom


def _stones(mill: boards.Board) -> dict[int, int]:
    return {
        r * SIZE + c: _FROM_SGFMILL[colour]
        for (colour, (r, c)) in mill.list_occupied_points()
    }


def _mill_liberties(mill: boards.Board) -> dict[int, frozenset[int]]:
    """Liberties of each stone's group, by a plain flood fill on sgfmill's grid."""
    grid = mill.board
    out: dict[int, frozenset[int]] = {}
    for r in range(SIZE):
        for c in range(SIZE):
            colour = grid[r][c]
            if colour is None or (r * SIZE + c) in out:
                continue
            group, libs, todo = {(r, c)}, set(), [(r, c)]
            while todo:
                a, b = todo.pop()
                for na, nb in ((a + 1, b), (a - 1, b), (a, b + 1), (a, b - 1)):
                    if not (0 <= na < SIZE and 0 <= nb < SIZE):
                        continue
                    v = grid[na][nb]
                    if v is None:
                        libs.add(na * SIZE + nb)
                    elif v == colour and (na, nb) not in group:
                        group.add((na, nb))
                        todo.append((na, nb))
            frozen = frozenset(libs)
            for a, b in group:
                out[a * SIZE + b] = frozen
    return out


def check_position(
    board: Board, mill: boards.Board, mill_ko: int | None, report: Report
) -> None:
    """Compare one position. ``mill_ko`` is sgfmill's simple-ko point for the player to move."""
    report.positions += 1
    ours = {p: board[p] for p in range(NUM_POINTS) if board[p] != EMPTY}
    theirs = _stones(mill)
    if ours != theirs:
        report.fail(board, "stones differ")
        return

    me = _TO_SGFMILL[board.to_move]
    legal = set(board.legal_moves())
    for p in range(NUM_POINTS):
        if p in ours:
            continue
        trial = mill.copy()
        r, c = _rc(p)
        trial.play(r, c, me)
        suicide = trial.get(r, c) is None
        mill_legal = not suicide and p != mill_ko
        report.legality_checks += 1
        if (p in legal) != mill_legal or board.is_legal(p) != mill_legal:
            report.fail(
                board,
                f"legality of {format_gtp(p)}: ours {p in legal}, sgfmill {mill_legal}",
            )
            continue
        if not mill_legal:
            continue
        mill_captured = {q for q, v in theirs.items() if trial.get(*_rc(q)) is None}
        captured = set(board.play(p))
        board.undo()
        report.capture_checks += 1
        if captured != mill_captured:
            report.fail(
                board,
                f"captures of {format_gtp(p)}: ours {sorted(captured)}, sgfmill {sorted(mill_captured)}",
            )

    mill_libs = _mill_liberties(mill)
    for g in board.groups():
        report.liberty_checks += 1
        if g.liberties != mill_libs[min(g.stones)]:
            report.fail(board, f"liberties of group at {format_gtp(min(g.stones))}")

    black, white = board.area()
    report.score_checks += 1
    if black - white != mill.area_score():
        report.fail(board, f"area: ours {black - white}, sgfmill {mill.area_score()}")


def random_games(
    n_positions: int,
    seed: int,
    report: Report,
    sample: float = 0.1,
    pass_rate: float = 0.01,
    max_moves: int = 500,
) -> int:
    """Check positions of random games until ``n_positions`` are checked. Returns games played.

    Each position is checked with probability ``sample`` (0.5 when a ko is on
    the board), so the checks spread over many games of at most ``max_moves``.
    """
    from goqueen.data.curriculum.positions import random_move

    rng = random.Random(seed)
    games = 0
    while report.positions < n_positions:
        games += 1
        board = Board()
        mill = boards.Board(SIZE)
        mill_ko = None
        # Random games can cycle through kos for ever, so cap their length.
        while (
            board.consecutive_passes < 2
            and board.move_count < max_moves
            and report.positions < n_positions
        ):
            if rng.random() < (0.5 if mill_ko is not None else sample):
                check_position(board, mill, mill_ko, report)
            p = PASS if rng.random() < pass_rate else random_move(board, rng)
            if p == PASS:
                mill_ko = None
            else:
                mill_ko_point = mill.play(*_rc(p), _TO_SGFMILL[board.to_move])
                mill_ko = (
                    None
                    if mill_ko_point is None
                    else mill_ko_point[0] * SIZE + mill_ko_point[1]
                )
            board.play(p)
    return games


def sgf_games(root: str, report: Report, limit: int | None) -> int:
    """Replay games with both SGF readers and check every position. Returns games read."""
    from goqueen.data.sgf import SGFError, read_file, replay

    files = sorted(Path(root).rglob("*.sgf"))[:limit]
    games = 0
    for path in files:
        try:
            positions = list(replay(read_file(str(path))))
            mill_game = sgf.Sgf_game.from_bytes(path.read_bytes())
        except (SGFError, ValueError) as e:
            print(f"skip {path}: {e}", file=sys.stderr)
            continue
        if mill_game.get_size() != SIZE:
            continue
        mill_board = boards.Board(SIZE)
        mill_board.apply_setup(*mill_game.get_root().get_setup_stones())
        plays = [node.get_move() for node in mill_game.get_main_sequence()]
        moves = [(pos.color, pos.move) for pos in positions]
        mill_moves = []
        for colour, move in plays:
            if colour is None:
                continue
            mill_moves.append(
                (
                    BLACK if colour == "b" else WHITE,
                    PASS if move is None else move[0] * SIZE + move[1],
                )
            )
        if moves != mill_moves:
            report.mismatches.append(f"{path}: move lists differ")
            continue
        mill_ko = None
        for pos in positions:
            check_position(pos.board, mill_board, mill_ko, report)
            if pos.move == PASS:
                mill_ko = None
            else:
                k = mill_board.play(*_rc(pos.move), _TO_SGFMILL[pos.color])
                mill_ko = None if k is None else k[0] * SIZE + k[1]
        games += 1
    return games


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--positions", type=int, default=10_000, help="random-game positions to check"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--sample", type=float, default=0.1, help="share of positions checked per game"
    )
    parser.add_argument("--sgf-dir", help="also check every position of these games")
    parser.add_argument("--sgf-limit", type=int, help="at most this many SGF files")
    args = parser.parse_args()

    report = Report()
    start = time.perf_counter()
    games = random_games(args.positions, args.seed, report, args.sample)
    print(f"random games:     {games:,}")
    if args.sgf_dir:
        n = sgf_games(args.sgf_dir, report, args.sgf_limit)
        print(f"SGF games checked: {n}")
    elapsed = time.perf_counter() - start

    print(f"positions:        {report.positions:,}")
    print(f"legality checks:  {report.legality_checks:,}")
    print(f"capture checks:   {report.capture_checks:,}")
    print(f"liberty checks:   {report.liberty_checks:,}")
    print(f"score checks:     {report.score_checks:,}")
    print(f"mismatches:       {len(report.mismatches):,}")
    print(f"time:             {elapsed:.0f}s")
    for m in report.mismatches[:5]:
        print("\n" + m)
    sys.exit(1 if report.mismatches else 0)


if __name__ == "__main__":
    main()
