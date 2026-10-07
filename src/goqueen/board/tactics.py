"""Small tactical solvers: ladders and capturing races.

Both search with ``Board.play`` and ``Board.undo``, so they follow the same
rules as everything else, and both leave the board as they found it.
"""

from dataclasses import dataclass
from enum import Enum

from goqueen.board.board import BLACK, EMPTY, WHITE, Board, Color
from goqueen.board.coords import PASS, SIZE

# A ladder runs at most ~2 * 19 moves across the board; this is a safe cap.
LADDER_MAX_DEPTH = 120
LADDER_MAX_NODES = 20_000


@dataclass(frozen=True)
class LadderResult:
    captured: bool  # True: the attacker captures the prey with ataris alone
    line: list[
        int
    ]  # main line from the start (attacker's winning ataris, or a prey escape)


class _Budget(Exception):
    pass


def _neighbours(p: int) -> list[int]:
    col, row = p % SIZE, p // SIZE
    out = []
    if col > 0:
        out.append(p - 1)
    if col < SIZE - 1:
        out.append(p + 1)
    if row > 0:
        out.append(p - SIZE)
    if row < SIZE - 1:
        out.append(p + SIZE)
    return out


def _adjacent_groups_in_atari(
    board: Board, stones: frozenset[int], color: Color
) -> list[int]:
    """Liberties of ``color`` groups in atari that touch ``stones`` (capturing moves)."""
    seen: set[int] = set()
    out: list[int] = []
    for s in stones:
        for n in _neighbours(s):
            if board[n] == color and n not in seen:
                g = board.group(n)
                seen |= g.stones
                if len(g.liberties) == 1:
                    out.extend(g.liberties)
    return sorted(set(out))


def ladder(board: Board, prey: int) -> LadderResult | None:
    """Can the player to move capture the group at ``prey`` with a ladder?

    The player to move must be the prey's opponent (the attacker). The
    attacker may only play ataris; the prey may extend or capture an
    attacking group in atari. The prey escapes if it reaches three or more
    liberties. Returns None if the prey is not the opponent's group, or the
    search is too big.
    """
    color = board[prey]
    if color == EMPTY or color == board.to_move:
        return None
    nodes = [0]
    try:
        captured, line = _attack(board, prey, 0, nodes)
    except _Budget:
        return None
    return LadderResult(captured, line)


def ladder_starts(board: Board, prey: int) -> list[int] | None:
    """The ataris that start a working ladder against the group at ``prey``.

    The prey must belong to the opponent of the player to move and have
    exactly two liberties. Returns None for other input or a search too big.
    """
    color = board[prey]
    if color == EMPTY or color == board.to_move or len(board.liberties(prey)) != 2:
        return None
    works = []
    for p in sorted(board.liberties(prey)):
        if not board.is_legal(p):
            continue
        board.play(p)
        try:
            if len(board.liberties(prey)) != 1:
                continue
            escaped, _ = _defend(board, prey, 1, [0])
        except _Budget:
            return None
        finally:
            board.undo()
        if not escaped:
            works.append(p)
    return works


def _attack(
    board: Board, prey: int, depth: int, nodes: list[int]
) -> tuple[bool, list[int]]:
    """Attacker to move. True if some atari sequence captures the prey."""
    nodes[0] += 1
    if nodes[0] > LADDER_MAX_NODES or depth > LADDER_MAX_DEPTH:
        raise _Budget
    libs = sorted(board.liberties(prey))
    if len(libs) == 1:
        if board.is_legal(libs[0]):
            return True, [libs[0]]
        return False, []
    if len(libs) != 2:
        return False, []
    escape: list[int] = []
    for p in libs:
        if not board.is_legal(p):
            continue
        board.play(p)
        try:
            if board[prey] == EMPTY:  # cannot happen with two liberties, but be safe
                return True, [p]
            if len(board.liberties(prey)) != 1:
                continue
            escaped, line = _defend(board, prey, depth + 1, nodes)
        finally:
            board.undo()
        if not escaped:
            return True, [p, *line]
        if not escape:
            escape = [p, *line]
    return False, escape


def _defend(
    board: Board, prey: int, depth: int, nodes: list[int]
) -> tuple[bool, list[int]]:
    """Prey to move, in atari. True if the prey escapes."""
    nodes[0] += 1
    if nodes[0] > LADDER_MAX_NODES or depth > LADDER_MAX_DEPTH:
        raise _Budget
    group = board.group(prey)
    attacker = group.color.opponent
    moves = sorted(
        set(group.liberties)
        | set(_adjacent_groups_in_atari(board, group.stones, attacker))
    )
    lost: list[int] = []
    for p in moves:
        if not board.is_legal(p):
            continue
        board.play(p)
        try:
            n = len(board.liberties(prey))
            if n >= 3:
                return True, [p]
            captured, line = _attack(board, prey, depth + 1, nodes)
        finally:
            board.undo()
        if not captured:
            return True, [p, *line]
        if not lost:
            lost = [p, *line]
    return False, lost


# ----------------------------------------------------------------------
# Capturing races


class RaceResult(Enum):
    BLACK = "Black"  # Black captures White's group
    WHITE = "White"  # White captures Black's group
    NEITHER = "neither"  # seki, or no capture within the search


RACE_MAX_NODES = 50_000
RACE_MAX_REGION = 10


def race(
    board: Board, a: int, b: int, require_closed: bool = True
) -> RaceResult | None:
    """Who wins the capturing race between the groups at ``a`` and ``b``?

    The race is a closed problem: moves may only be played on the points that
    are liberties of either group now (the region), or pass. The player to
    move moves first. A side wins when the other race group is captured; two
    passes in a row, or a repeated position, is NEITHER (seki or no result).

    With ``require_closed`` (the default) the race must also pass
    ``race_is_closed``, so the answer is the real local result.

    Returns None if the groups are not of opposite colours and touching, the
    region has more than ``RACE_MAX_REGION`` points or is not closed, or the
    search is too big.
    """
    ca, cb = board[a], board[b]
    if ca == EMPTY or cb == EMPTY or ca == cb:
        return None
    ga, gb = board.group(a), board.group(b)
    if not any(n in gb.stones for s in ga.stones for n in _neighbours(s)):
        return None
    region = sorted(ga.liberties | gb.liberties)
    if len(region) > RACE_MAX_REGION:
        return None
    if require_closed and not race_is_closed(board, a, b):
        return None
    black, white = (a, b) if ca == BLACK else (b, a)
    search = _RaceSearch(board, black, white, region)
    try:
        return search.solve()
    except _Budget:
        return None


def race_is_closed(board: Board, a: int, b: int) -> bool:
    """True if the race at ``a`` and ``b`` cannot leak out of its region.

    Closed means: every empty neighbour of a region point is in the region
    (so filling the region gives no new liberties), no region point touches
    both a race group and an outside group of its colour (so filling it
    joins nothing new), and every other group
    touching the race groups or the region has a liberty outside the region
    (so no capture inside the race opens new points).
    """
    ga, gb = board.group(a), board.group(b)
    region = ga.liberties | gb.liberties
    for r in region:
        for n in _neighbours(r):
            if board[n] == EMPTY and n not in region:
                return False
    race_stones = ga.stones | gb.stones
    # Playing a region point next to a race group and an outside group of the
    # same colour would join them, and the race group would gain their liberties.
    for r in region:
        race_colors = {board[n] for n in _neighbours(r) if n in race_stones}
        other_colors = {
            board[n]
            for n in _neighbours(r)
            if board[n] != EMPTY and n not in race_stones
        }
        if race_colors & other_colors:
            return False
    seen = set(race_stones)
    for p in [*race_stones, *region]:
        for n in _neighbours(p):
            if board[n] != EMPTY and n not in seen:
                g = board.group(n)
                seen |= g.stones
                if g.liberties <= region:  # could be captured inside the race
                    return False
    return True


_PREFERENCE = {
    BLACK: (RaceResult.BLACK, RaceResult.NEITHER, RaceResult.WHITE),
    WHITE: (RaceResult.WHITE, RaceResult.NEITHER, RaceResult.BLACK),
}


class _RaceSearch:
    def __init__(self, board: Board, black: int, white: int, region: list[int]) -> None:
        self.board = board
        self.black = black
        self.white = white
        self.moves = [*region, PASS]
        self.memo: dict[tuple, RaceResult] = {}
        self.path: set[tuple] = set()
        self.nodes = 0

    def solve(self) -> RaceResult:
        b = self.board
        if b[self.white] != WHITE:
            return RaceResult.BLACK
        if b[self.black] != BLACK:
            return RaceResult.WHITE
        if b.consecutive_passes >= 2:
            return RaceResult.NEITHER
        key = (
            b.to_array().tobytes(),
            int(b.to_move),
            b.ko,
            min(b.consecutive_passes, 1),
        )
        if key in self.memo:
            return self.memo[key]
        if key in self.path:
            return RaceResult.NEITHER  # a cycle (kos and passes): nobody captures
        self.nodes += 1
        if self.nodes > RACE_MAX_NODES:
            raise _Budget

        order = _PREFERENCE[b.to_move]
        best = order[-1]
        self.path.add(key)
        try:
            for p in self.moves:
                if not b.is_legal(p):
                    continue
                b.play(p)
                try:
                    result = self.solve()
                finally:
                    b.undo()
                if order.index(result) < order.index(best):
                    best = result
                    if best == order[0]:
                        break
        finally:
            self.path.discard(key)
        self.memo[key] = best
        return best
