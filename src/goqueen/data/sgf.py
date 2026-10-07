"""SGF reader: parse a game record and replay its main line on ``goqueen.board``.

Only what we need from SGF FF[4] is read: the first game tree of a file and the
main line (the first child at each branch). Setup stones (``AB``, ``AW``,
``AE``) are applied with ``Board.set_stone``. Moves (``B``, ``W``) go through
``Board.play``, so an illegal move in a record raises ``SGFError``.

SGF points are two letters, column then row, with ``aa`` at the top left.
An empty value or ``tt`` is a pass.
"""

from collections.abc import Iterator
from dataclasses import dataclass, field

from goqueen.board import (
    BLACK,
    EMPTY,
    PASS,
    SIZE,
    WHITE,
    Board,
    Color,
    IllegalMoveError,
)
from goqueen.board.coords import format_gtp


class SGFError(ValueError):
    """Bad SGF syntax, an unsupported game, or an illegal move in the record."""


Node = dict[str, list[str]]


@dataclass
class Game:
    """The main line of one SGF game."""

    properties: Node  # root node properties
    nodes: list[Node] = field(default_factory=list)  # main line, root first

    def prop(self, key: str, default: str = "") -> str:
        values = self.properties.get(key)
        return values[0] if values else default

    @property
    def komi(self) -> float | None:
        km = self.prop("KM")
        try:
            return float(km) if km else None
        except ValueError:
            return None

    @property
    def result(self) -> str:
        return self.prop("RE")

    @property
    def handicap(self) -> int:
        ha = self.prop("HA")
        return int(ha) if ha.isdigit() else 0

    @property
    def size(self) -> int:
        sz = self.prop("SZ", str(SIZE))
        try:
            return int(sz.split(":")[0])
        except ValueError as e:
            raise SGFError(f"bad SZ: {sz!r}") from e


@dataclass(frozen=True)
class Position:
    """A position from a game and the move played from it."""

    board: Board
    move: int  # a point or PASS
    color: Color
    move_number: int  # 1 for the first move


# ----------------------------------------------------------------------
# Parsing


def _parse_tree(text: str, i: int) -> tuple[list[Node], int]:
    """Parse the game tree that starts at ``text[i] == "("``; keep the main line."""
    assert text[i] == "("
    i += 1
    nodes: list[Node] = []
    n = len(text)
    first_child_taken = False
    while i < n:
        c = text[i]
        if c == ";":
            if first_child_taken:
                raise SGFError(f"node after a variation at offset {i}")
            node: Node = {}
            i += 1
            while True:
                while i < n and text[i].isspace():
                    i += 1
                j = i
                while j < n and text[j].isalpha():
                    j += 1
                if j == i:
                    break
                # FF[3] allows lower-case letters in property names; keep upper case.
                key = "".join(ch for ch in text[i:j] if ch.isupper())
                i = j
                values: list[str] = []
                while True:
                    while i < n and text[i].isspace():
                        i += 1
                    if i >= n or text[i] != "[":
                        break
                    value, i = _parse_value(text, i)
                    values.append(value)
                if not values:
                    raise SGFError(f"property {key} has no value")
                node.setdefault(key, []).extend(values)
            nodes.append(node)
        elif c == "(":
            child, i = _parse_tree(text, i)
            if not first_child_taken:
                nodes.extend(child)
                first_child_taken = True
        elif c == ")":
            return nodes, i + 1
        elif c.isspace():
            i += 1
        else:
            raise SGFError(f"unexpected {c!r} at offset {i}")
    raise SGFError("unterminated game tree")


def _parse_value(text: str, i: int) -> tuple[str, int]:
    """Parse ``[value]`` at ``text[i] == "["``; handle backslash escapes."""
    i += 1
    out = []
    n = len(text)
    while i < n:
        c = text[i]
        if c == "\\":
            i += 1
            if i < n and text[i] not in "\r\n":  # escaped newline is a soft break
                out.append(text[i])
            i += 1
        elif c == "]":
            return "".join(out), i + 1
        else:
            out.append(c)
            i += 1
    raise SGFError("unterminated property value")


def parse(text: str) -> Game:
    """Parse the first game in SGF ``text``."""
    start = text.find("(")
    if start < 0 or ";" not in text[start:]:
        raise SGFError("no game tree")
    nodes, _ = _parse_tree(text, start)
    if not nodes:
        raise SGFError("empty game tree")
    return Game(properties=nodes[0], nodes=nodes)


# ----------------------------------------------------------------------
# Points


def sgf_point(value: str) -> int:
    """Convert an SGF point (``"pd"``) to a board point. ``""`` and ``"tt"`` are PASS."""
    if value == "" or value == "tt":
        return PASS
    if len(value) != 2:
        raise SGFError(f"bad SGF point: {value!r}")
    col = ord(value[0]) - ord("a")
    row = SIZE - 1 - (ord(value[1]) - ord("a"))
    if not (0 <= col < SIZE and 0 <= row < SIZE):
        raise SGFError(f"SGF point off the board: {value!r}")
    return row * SIZE + col


def _point_list(values: list[str]) -> list[int]:
    """Expand a list of points, including compressed ``"aa:cc"`` rectangles."""
    out = []
    for v in values:
        if ":" in v:
            a, b = v.split(":")
            pa, pb = sgf_point(a), sgf_point(b)
            c0, c1 = sorted((pa % SIZE, pb % SIZE))
            r0, r1 = sorted((pa // SIZE, pb // SIZE))
            out.extend(
                r * SIZE + c for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)
            )
        else:
            p = sgf_point(v)
            if p == PASS:
                raise SGFError(f"bad setup point: {v!r}")
            out.append(p)
    return out


# ----------------------------------------------------------------------
# Replay


def _apply_setup(board: Board, node: Node) -> None:
    for key, color in (("AE", EMPTY), ("AB", BLACK), ("AW", WHITE)):
        for p in _point_list(node.get(key, [])):
            board.set_stone(p, color)
    if "PL" in node:
        pl = node["PL"][0].upper()
        board.to_move = BLACK if pl == "B" else WHITE


def replay(game: Game) -> Iterator[Position]:
    """Yield each position on the main line with the move played from it.

    Each ``Position.board`` is a copy, so callers may keep or change it.
    """
    if game.size != SIZE:
        raise SGFError(f"only {SIZE}x{SIZE} is supported, not {game.size}")
    board = Board()
    number = 0
    for node in game.nodes:
        if any(k in node for k in ("AB", "AW", "AE", "PL")):
            _apply_setup(board, node)
        for key, color in (("B", BLACK), ("W", WHITE)):
            if key not in node:
                continue
            move = sgf_point(node[key][0])
            number += 1
            board.to_move = color
            yield Position(board.copy(), move, color, number)
            try:
                board.play(move, color)
            except IllegalMoveError as e:
                raise SGFError(f"move {number} ({format_gtp(move)}): {e}") from e


def final_board(game: Game) -> Board:
    """The board after the last move of the main line."""
    board = Board()
    last = None
    for last in replay(game):
        pass
    if last is not None:
        board = last.board
        board.play(last.move, last.color)
    elif any(k in game.properties for k in ("AB", "AW", "AE")):
        _apply_setup(board, game.properties)
    return board


def read_file(path: str) -> Game:
    """Read and parse an SGF file. Unknown encodings fall back to Latin-1."""
    with open(path, "rb") as f:
        raw = f.read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    return parse(text)
