"""19x19 Go board: play and undo, captures, suicide, simple ko, groups, scoring.

The board is a flat list with a one-point border of sentinel cells, so
neighbour lookups need no bounds checks. Groups are found by flood fill on
demand; this is fast enough in pure Python (see ``goqueen.board.bench``).

Public methods take and return points as ints in 0..360 (see ``coords``).
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import Final, NamedTuple

import numpy as np

from goqueen.board.coords import NUM_POINTS, PASS, SIZE, format_gtp, parse_gtp

KOMI: Final = 7.5


class Color(IntEnum):
    EMPTY = 0
    BLACK = 1
    WHITE = 2

    @property
    def opponent(self) -> "Color":
        if self is Color.EMPTY:
            raise ValueError("EMPTY has no opponent")
        return Color(3 - self)


EMPTY: Final = Color.EMPTY
BLACK: Final = Color.BLACK
WHITE: Final = Color.WHITE
_BORDER: Final = 3
_COLORS: Final = (EMPTY, BLACK, WHITE)

# Padded layout: (SIZE + 2) columns, one border cell on each side.
_W: Final = SIZE + 2
_OFFSETS: Final = (1, -1, _W, -_W)
_TO_PAD: Final = [(p // SIZE + 1) * _W + (p % SIZE + 1) for p in range(NUM_POINTS)]
_FROM_PAD: Final = [-1] * (_W * _W)
for _p, _q in enumerate(_TO_PAD):
    _FROM_PAD[_q] = _p
del _p, _q


class IllegalMoveError(ValueError):
    """Raised by ``Board.play`` for an occupied point, ko retake or suicide."""


@dataclass(frozen=True, slots=True)
class Group:
    color: Color
    stones: frozenset[int]
    liberties: frozenset[int]


class _Undo(NamedTuple):
    q: int  # padded point, or PASS
    color: int
    captured: list[int]  # padded points
    ko: int
    ko_color: int
    to_move: int
    passes: int


class Board:
    """A 19x19 Go position under Chinese rules with simple ko.

    ``captures[c]`` counts the stones that colour ``c`` has captured.
    """

    __slots__ = (
        "_cells",
        "_history",
        "_ko",
        "_ko_color",
        "_passes",
        "captures",
        "to_move",
    )

    def __init__(self) -> None:
        self._cells = [_BORDER] * (_W * _W)
        for q in _TO_PAD:
            self._cells[q] = EMPTY
        self.to_move: Color = BLACK
        self._ko = -1  # padded point the ko-bound player may not play, or -1
        self._ko_color = EMPTY
        self._passes = 0
        self.captures: dict[Color, int] = {BLACK: 0, WHITE: 0}
        self._history: list[_Undo] = []

    # ------------------------------------------------------------------
    # Setup and copies

    @classmethod
    def from_moves(cls, moves: list[str]) -> "Board":
        """Play GTP moves in turn from an empty board, Black first."""
        b = cls()
        for m in moves:
            b.play(parse_gtp(m))
        return b

    @classmethod
    def from_diagram(cls, diagram: str, to_move: Color = BLACK) -> "Board":
        """Build a position from 19 rows of ``X`` (Black), ``O`` (White), ``.``.

        The first row is rank 19. Spaces are ignored. No captures are made.
        """
        rows = [r.replace(" ", "") for r in diagram.strip().splitlines()]
        if len(rows) != SIZE or any(len(r) != SIZE for r in rows):
            raise ValueError("diagram must be 19 rows of 19 points")
        b = cls()
        symbols = {"X": BLACK, "O": WHITE, ".": EMPTY, "+": EMPTY}
        for i, r in enumerate(rows):
            row = SIZE - 1 - i
            for col, ch in enumerate(r):
                if ch not in symbols:
                    raise ValueError(f"bad diagram symbol: {ch!r}")
                b._cells[_TO_PAD[row * SIZE + col]] = symbols[ch]
        b.to_move = to_move
        return b

    def set_stone(self, p: int, color: Color) -> None:
        """Put ``color`` (or EMPTY) at ``p`` with no rules applied.

        For setup stones (SGF ``AB``/``AW``/``AE``). Clears ko and history.
        """
        self._cells[_TO_PAD[p]] = color
        self._ko = -1
        self._ko_color = EMPTY
        self._history.clear()

    def copy(self) -> "Board":
        b = Board.__new__(Board)
        b._cells = self._cells.copy()
        b.to_move = self.to_move
        b._ko = self._ko
        b._ko_color = self._ko_color
        b._passes = self._passes
        b.captures = self.captures.copy()
        b._history = self._history.copy()
        return b

    # ------------------------------------------------------------------
    # Reading the position

    def __getitem__(self, p: int) -> Color:
        return _COLORS[self._cells[_TO_PAD[p]]]

    def stones(self, color: Color) -> list[int]:
        """All points with a stone of ``color``, in point order."""
        cells = self._cells
        return [p for p, q in enumerate(_TO_PAD) if cells[q] == color]

    def count(self, color: Color) -> int:
        cells = self._cells
        return sum(1 for q in _TO_PAD if cells[q] == color)

    @property
    def ko(self) -> int | None:
        """The point the player to move may not play because of ko, if any."""
        if self._ko < 0 or self._ko_color != self.to_move:
            return None
        return _FROM_PAD[self._ko]

    @property
    def consecutive_passes(self) -> int:
        return self._passes

    @property
    def move_count(self) -> int:
        """Moves (including passes) played since the start or last setup."""
        return len(self._history)

    @property
    def last_move(self) -> int | None:
        """The last move (a point or PASS), or None if no move was played."""
        if not self._history:
            return None
        q = self._history[-1].q
        return PASS if q == PASS else _FROM_PAD[q]

    def to_array(self) -> np.ndarray:
        """The position as a ``(19, 19)`` int8 array indexed ``[row, col]``."""
        cells = self._cells
        return np.array([cells[q] for q in _TO_PAD], dtype=np.int8).reshape(SIZE, SIZE)

    def __str__(self) -> str:
        sym = {EMPTY: ".", BLACK: "X", WHITE: "O"}
        files = "   " + " ".join("ABCDEFGHJKLMNOPQRST")
        lines = [files]
        for row in range(SIZE - 1, -1, -1):
            pts = " ".join(
                sym[self._cells[_TO_PAD[row * SIZE + c]]] for c in range(SIZE)
            )
            lines.append(f"{row + 1:2d} {pts} {row + 1:2d}")
        lines.append(files)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Groups and liberties

    def _flood(self, q: int) -> tuple[list[int], set[int]]:
        """Stones and liberties (padded) of the group at padded point ``q``."""
        cells = self._cells
        color = cells[q]
        stones = [q]
        seen = {q}
        libs: set[int] = set()
        i = 0
        while i < len(stones):
            s = stones[i]
            i += 1
            for d in _OFFSETS:
                n = s + d
                v = cells[n]
                if v == EMPTY:
                    libs.add(n)
                elif v == color and n not in seen:
                    seen.add(n)
                    stones.append(n)
        return stones, libs

    def _captured_if_no_liberty(self, q: int) -> list[int] | None:
        """Stones of the group at ``q`` if it has no liberty, else None.

        Stops at the first liberty it finds.
        """
        cells = self._cells
        color = cells[q]
        stones = [q]
        seen = {q}
        i = 0
        while i < len(stones):
            s = stones[i]
            i += 1
            for d in _OFFSETS:
                n = s + d
                v = cells[n]
                if v == EMPTY:
                    return None
                if v == color and n not in seen:
                    seen.add(n)
                    stones.append(n)
        return stones

    def group(self, p: int) -> Group:
        """The group with a stone at ``p``."""
        q = _TO_PAD[p]
        color = self._cells[q]
        if color == EMPTY:
            raise ValueError(f"no stone at {format_gtp(p)}")
        stones, libs = self._flood(q)
        return Group(
            _COLORS[color],
            frozenset(_FROM_PAD[s] for s in stones),
            frozenset(_FROM_PAD[n] for n in libs),
        )

    def liberties(self, p: int) -> frozenset[int]:
        """Liberties of the group with a stone at ``p``."""
        return self.group(p).liberties

    def groups(self, color: Color | None = None) -> list[Group]:
        """All groups (of ``color``, if given), ordered by their lowest point."""
        cells = self._cells
        seen: set[int] = set()
        out: list[Group] = []
        for q in _TO_PAD:
            v = cells[q]
            if v == EMPTY or q in seen or (color is not None and v != color):
                continue
            stones, libs = self._flood(q)
            seen.update(stones)
            out.append(
                Group(
                    _COLORS[v],
                    frozenset(_FROM_PAD[s] for s in stones),
                    frozenset(_FROM_PAD[n] for n in libs),
                )
            )
        return out

    # ------------------------------------------------------------------
    # Moves

    def is_legal(self, p: int, color: Color | None = None) -> bool:
        """True if ``color`` (default: the player to move) may play ``p``."""
        if p == PASS:
            return True
        if color is None:
            color = self.to_move
        cells = self._cells
        q = _TO_PAD[p]
        if cells[q] != EMPTY:
            return False
        if q == self._ko and color == self._ko_color:
            return False
        return self._not_suicide(q, color)

    def _not_suicide(self, q: int, color: int) -> bool:
        cells = self._cells
        # Fast path: an empty neighbour is a liberty for the new stone.
        for d in _OFFSETS:
            if cells[q + d] == EMPTY:
                return True
        for d in _OFFSETS:
            n = q + d
            v = cells[n]
            if v == _BORDER:
                continue
            _, libs = self._flood(n)
            if v == color:
                if len(libs) > 1:  # keeps a liberty other than q
                    return True
            elif len(libs) == 1:  # the only liberty is q: we capture
                return True
        return False

    def _liberty_counts(self) -> list[int]:
        """Liberty count of each stone's group, indexed by padded point (0 off stones)."""
        cells = self._cells
        counts = [0] * len(cells)
        for q in _TO_PAD:
            if cells[q] in (BLACK, WHITE) and counts[q] == 0:
                stones, libs = self._flood(q)
                n = len(libs)
                for s in stones:
                    counts[s] = n
        return counts

    def legal_moves(self, color: Color | None = None) -> list[int]:
        """All legal board points for ``color``, in point order. Pass is not listed."""
        if color is None:
            color = self.to_move
        cells = self._cells
        ko = self._ko if color == self._ko_color else -1
        counts: list[int] | None = None
        out = []
        for p, q in enumerate(_TO_PAD):
            if cells[q] != EMPTY or q == ko:
                continue
            if (
                cells[q + 1] == EMPTY
                or cells[q - 1] == EMPTY
                or cells[q + _W] == EMPTY
                or cells[q - _W] == EMPTY
            ):
                out.append(p)
                continue
            if counts is None:
                counts = self._liberty_counts()
            for d in _OFFSETS:
                n = q + d
                v = cells[n]
                if v == _BORDER:
                    continue
                # Own group keeps another liberty, or enemy group's last liberty is q.
                if (counts[n] > 1) if v == color else (counts[n] == 1):
                    out.append(p)
                    break
        return out

    def play(self, p: int, color: Color | None = None) -> list[int]:
        """Play ``p`` (or PASS) and return the points captured.

        ``color`` defaults to the player to move. After the move the other
        colour is to move. Raises ``IllegalMoveError`` for an occupied point,
        a simple-ko retake or suicide; the board is then unchanged.
        """
        if color is None:
            color = self.to_move
        if p == PASS:
            self._history.append(
                _Undo(
                    PASS,
                    color,
                    [],
                    self._ko,
                    self._ko_color,
                    self.to_move,
                    self._passes,
                )
            )
            self._ko = -1
            self._ko_color = EMPTY
            self._passes += 1
            self.to_move = color.opponent
            return []

        cells = self._cells
        q = _TO_PAD[p]
        if cells[q] != EMPTY:
            raise IllegalMoveError(f"{format_gtp(p)} is occupied")
        if q == self._ko and color == self._ko_color:
            raise IllegalMoveError(f"{format_gtp(p)} retakes a ko")

        other = 3 - color
        cells[q] = color
        captured: list[int] = []
        for d in _OFFSETS:
            n = q + d
            if cells[n] == other:
                dead = self._captured_if_no_liberty(n)
                if dead:
                    for s in dead:
                        cells[s] = EMPTY
                    captured.extend(dead)

        if not captured and self._captured_if_no_liberty(q) is not None:
            cells[q] = EMPTY
            raise IllegalMoveError(f"{format_gtp(p)} is suicide")

        self._history.append(
            _Undo(
                q, color, captured, self._ko, self._ko_color, self.to_move, self._passes
            )
        )

        # Simple ko: one stone captured by a lone stone left in atari.
        self._ko = -1
        self._ko_color = EMPTY
        if len(captured) == 1:
            libs = 0
            friends = False
            for d in _OFFSETS:
                v = cells[q + d]
                if v == EMPTY:
                    libs += 1
                elif v == color:
                    friends = True
            if libs == 1 and not friends:
                self._ko = captured[0]
                self._ko_color = other

        self.captures[color] += len(captured)
        self._passes = 0
        self.to_move = _COLORS[other]
        return [_FROM_PAD[s] for s in captured]

    def undo(self) -> None:
        """Take back the last move played with ``play``."""
        if not self._history:
            raise IndexError("no move to undo")
        u = self._history.pop()
        cells = self._cells
        if u.q != PASS:
            cells[u.q] = EMPTY
            other = 3 - u.color
            for s in u.captured:
                cells[s] = other
            self.captures[_COLORS[u.color]] -= len(u.captured)
        self._ko = u.ko
        self._ko_color = u.ko_color
        self.to_move = _COLORS[u.to_move]
        self._passes = u.passes

    # ------------------------------------------------------------------
    # Scoring

    def area(self) -> tuple[int, int]:
        """Chinese area for ``(Black, White)``: stones plus surrounded empty points.

        An empty region counts for a colour only if it touches stones of that
        colour alone. Dead stones are not removed: remove them first.
        """
        cells = self._cells
        area = [0, 0, 0, 0]
        for q in _TO_PAD:
            area[cells[q]] += 1
        seen: set[int] = set()
        for q in _TO_PAD:
            if cells[q] != EMPTY or q in seen:
                continue
            region = [q]
            seen.add(q)
            borders = 0  # bit 1: Black, bit 2: White
            i = 0
            while i < len(region):
                s = region[i]
                i += 1
                for d in _OFFSETS:
                    n = s + d
                    v = cells[n]
                    if v == EMPTY:
                        if n not in seen:
                            seen.add(n)
                            region.append(n)
                    elif v != _BORDER:
                        borders |= v
            if borders == BLACK or borders == WHITE:
                area[borders] += len(region)
        return area[BLACK], area[WHITE]

    def score(self, komi: float = KOMI) -> float:
        """Black's area minus White's area minus ``komi``. Positive: Black wins."""
        black, white = self.area()
        return black - white - komi
