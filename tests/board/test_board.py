import random

import numpy as np
import pytest

from goqueen.board import (
    BLACK,
    EMPTY,
    NUM_POINTS,
    PASS,
    WHITE,
    Board,
    Color,
    IllegalMoveError,
    format_gtp,
    parse_gtp,
)


def g(*names: str) -> list[int]:
    return [parse_gtp(n) for n in names]


def pts(*names: str) -> set[int]:
    return set(g(*names))


def setup(black: str = "", white: str = "", to_move: Color = BLACK) -> Board:
    b = Board()
    for n in black.split():
        b.set_stone(parse_gtp(n), BLACK)
    for n in white.split():
        b.set_stone(parse_gtp(n), WHITE)
    b.to_move = to_move
    return b


# ----------------------------------------------------------------------
# Basics


def test_empty_board():
    b = Board()
    assert b.to_move == BLACK
    assert b.count(BLACK) == b.count(WHITE) == 0
    assert len(b.legal_moves()) == NUM_POINTS
    assert b.ko is None
    assert b.last_move is None


def test_turns_alternate():
    b = Board()
    b.play(parse_gtp("Q16"))
    assert b.to_move == WHITE
    assert b[parse_gtp("Q16")] == BLACK
    b.play(parse_gtp("D4"))
    assert b.to_move == BLACK
    assert b[parse_gtp("D4")] == WHITE
    assert b.last_move == parse_gtp("D4")
    assert b.move_count == 2


def test_explicit_color():
    b = Board()
    b.play(parse_gtp("D4"), WHITE)
    assert b[parse_gtp("D4")] == WHITE
    assert b.to_move == BLACK


def test_occupied_is_illegal():
    b = Board.from_moves(["D4"])
    assert not b.is_legal(parse_gtp("D4"))
    with pytest.raises(IllegalMoveError):
        b.play(parse_gtp("D4"))
    assert b.to_move == WHITE


def test_color_opponent():
    assert BLACK.opponent == WHITE
    assert WHITE.opponent == BLACK
    with pytest.raises(ValueError):
        _ = EMPTY.opponent


# ----------------------------------------------------------------------
# Liberties


@pytest.mark.parametrize(
    ("name", "libs"),
    [
        ("A1", 2),
        ("T1", 2),
        ("A19", 2),
        ("T19", 2),
        ("A10", 3),
        ("K1", 3),
        ("T10", 3),
        ("K19", 3),
        ("K10", 4),
    ],
)
def test_single_stone_liberties(name, libs):
    b = setup(black=name)
    assert len(b.liberties(parse_gtp(name))) == libs


def test_corner_liberty_points():
    b = setup(black="A1")
    assert b.liberties(parse_gtp("A1")) == pts("A2", "B1")


def test_group_liberties_shared_counted_once():
    # An L of three stones: B2 C2 C3. B3 is next to both B2 and C3.
    b = setup(black="B2 C2 C3")
    grp = b.group(parse_gtp("B2"))
    assert grp.stones == pts("B2", "C2", "C3")
    assert grp.liberties == pts("A2", "B1", "B3", "C1", "D2", "D3", "C4")


def test_enemy_stones_take_liberties():
    b = setup(black="K10", white="K11 J10")
    assert b.liberties(parse_gtp("K10")) == pts("L10", "K9")


def test_groups_listing():
    b = setup(black="A1 B1 K10", white="A2")
    groups = b.groups()
    assert [(gr.color, gr.stones) for gr in groups] == [
        (BLACK, pts("A1", "B1")),
        (WHITE, pts("A2")),
        (BLACK, pts("K10")),
    ]
    assert len(b.groups(WHITE)) == 1
    with pytest.raises(ValueError):
        b.group(parse_gtp("C3"))


# ----------------------------------------------------------------------
# Captures


def test_single_capture_center():
    b = setup(black="J10 L10 K11", white="K10")
    captured = b.play(parse_gtp("K9"))
    assert captured == g("K10")
    assert b[parse_gtp("K10")] == EMPTY
    assert b.captures[BLACK] == 1
    assert b.captures[WHITE] == 0


def test_corner_capture():
    b = setup(black="A2", white="A1")
    assert b.play(parse_gtp("B1")) == g("A1")


def test_multi_stone_edge_capture():
    # White A1 B1 C1 on the first line; Black surrounds them.
    b = setup(black="A2 B2 C2", white="A1 B1 C1")
    captured = b.play(parse_gtp("D1"))
    assert set(captured) == pts("A1", "B1", "C1")
    assert b.count(WHITE) == 0
    assert b.captures[BLACK] == 3


def test_capture_two_groups_at_once():
    # White A1 and C1 are separate stones; Black B1 takes both.
    b = setup(black="A2 C2 D1", white="A1 C1")
    assert set(b.play(parse_gtp("B1"))) == pts("A1", "C1")
    assert b.captures[BLACK] == 2


def test_no_capture_with_liberty_left():
    b = setup(black="J10 L10", white="K10")
    assert b.play(parse_gtp("K11")) == []
    assert b[parse_gtp("K10")] == WHITE


def test_capture_before_suicide_check():
    # Black B1 has no empty neighbour, but it takes White A1, whose last liberty is B1.
    b = setup(black="A2", white="A1 B2 C1")
    assert b.play(parse_gtp("B1")) == g("A1")
    assert b.liberties(parse_gtp("B1")) == pts("A1")
    assert b.ko == parse_gtp("A1")  # a corner ko


def test_snapback():
    # White A2 B2 C2 C1 has two liberties: A1 and B1. Black throws in at B1.
    b = setup(black="A3 B3 C3 D2 D1", white="A2 B2 C2 C1")
    assert b.play(parse_gtp("B1")) == []
    assert b.liberties(parse_gtp("B2")) == pts("A1")
    # White takes the stone at A1, but now has only B1.
    assert b.play(parse_gtp("A1")) == g("B1")
    assert b.ko is None  # not a ko: the capturing stone has friends
    assert b.liberties(parse_gtp("A1")) == pts("B1")
    # Black snaps back and takes five stones.
    assert set(b.play(parse_gtp("B1"))) == pts("A1", "A2", "B2", "C2", "C1")
    assert b.captures == {BLACK: 5, WHITE: 1}


# ----------------------------------------------------------------------
# Ko

KO_BLACK = "E6 D5 E4"
KO_WHITE = "F6 E5 G5 F4"


def test_ko_capture_and_ban():
    b = setup(black=KO_BLACK, white=KO_WHITE)
    assert b.play(parse_gtp("F5")) == g("E5")
    assert b.to_move == WHITE
    assert b.ko == parse_gtp("E5")
    assert not b.is_legal(parse_gtp("E5"))
    assert parse_gtp("E5") not in b.legal_moves()
    with pytest.raises(IllegalMoveError, match="ko"):
        b.play(parse_gtp("E5"))
    # Black may play there (not that it would want to).
    assert b.is_legal(parse_gtp("E5"), BLACK)


def test_ko_threat_then_retake():
    b = setup(black=KO_BLACK, white=KO_WHITE)
    b.play(parse_gtp("F5"))
    b.play(parse_gtp("Q16"))  # White plays elsewhere
    assert b.ko is None
    b.play(parse_gtp("Q4"))  # Black answers elsewhere
    assert b.play(parse_gtp("E5")) == g("F5")  # White retakes
    assert b.ko == parse_gtp("F5")
    assert not b.is_legal(parse_gtp("F5"))


def test_pass_clears_ko():
    b = setup(black=KO_BLACK, white=KO_WHITE)
    b.play(parse_gtp("F5"))
    b.play(PASS)
    b.play(PASS)
    assert b.to_move == WHITE
    assert b.is_legal(parse_gtp("E5"))


def test_two_stone_capture_is_not_ko():
    # Black F5 takes White D5 E5 and is left as a lone stone in atari. Not a ko.
    b = setup(black="C5 D6 D4 E6 E4", white="D5 E5 F6 G5 F4")
    assert set(b.play(parse_gtp("F5"))) == pts("D5", "E5")
    assert b.liberties(parse_gtp("F5")) == pts("E5")
    assert b.ko is None
    assert b.is_legal(parse_gtp("E5"))


def test_ko_cleared_by_undo_and_restored():
    b = setup(black=KO_BLACK, white=KO_WHITE)
    b.play(parse_gtp("F5"))
    b.play(parse_gtp("Q16"))
    assert b.ko is None
    b.undo()
    assert b.ko == parse_gtp("E5")


# ----------------------------------------------------------------------
# Suicide


def test_single_stone_suicide():
    b = setup(black="A2 B1", to_move=WHITE)
    assert not b.is_legal(parse_gtp("A1"))
    with pytest.raises(IllegalMoveError, match="suicide"):
        b.play(parse_gtp("A1"))
    assert b[parse_gtp("A1")] == EMPTY
    assert b.to_move == WHITE
    assert b.move_count == 0


def test_multi_stone_suicide():
    # White A1 B1 would fill its own last liberty at C1.
    b = setup(black="A2 B2 C2 D1", white="A1 B1", to_move=WHITE)
    assert not b.is_legal(parse_gtp("C1"))
    with pytest.raises(IllegalMoveError):
        b.play(parse_gtp("C1"))
    assert b.group(parse_gtp("A1")).stones == pts("A1", "B1")


def test_center_eye_suicide():
    b = setup(black="J10 L10 K11 K9", to_move=WHITE)
    assert not b.is_legal(parse_gtp("K10"))
    assert parse_gtp("K10") not in b.legal_moves()
    assert b.is_legal(parse_gtp("K10"), BLACK)


def test_own_eye_fill_with_liberty_is_legal():
    b = setup(black="J10 L10 K11 K9")
    assert b.is_legal(parse_gtp("K10"))


# ----------------------------------------------------------------------
# Undo


def test_undo_restores_capture():
    b = setup(black="J10 L10 K11", white="K10")
    before = b.to_array()
    b.play(parse_gtp("K9"))
    b.undo()
    assert np.array_equal(b.to_array(), before)
    assert b.captures == {BLACK: 0, WHITE: 0}
    assert b.to_move == BLACK


def test_undo_pass():
    b = Board()
    b.play(PASS)
    assert b.consecutive_passes == 1
    assert b.last_move == PASS
    b.undo()
    assert b.consecutive_passes == 0
    assert b.to_move == BLACK
    with pytest.raises(IndexError):
        b.undo()


def test_copy_is_independent():
    b = Board.from_moves(["D4", "Q16"])
    c = b.copy()
    c.play(parse_gtp("K10"))
    assert b[parse_gtp("K10")] == EMPTY
    assert b.move_count == 2 and c.move_count == 3


# ----------------------------------------------------------------------
# Scoring


def test_empty_board_score():
    b = Board()
    assert b.area() == (0, 0)
    assert b.score() == -7.5


def test_split_board_score():
    # Black wall on file J, White wall on file K: Black owns A-J (9 files), White K-T (10).
    b = Board()
    for row in range(1, 20):
        b.set_stone(parse_gtp(f"J{row}"), BLACK)
        b.set_stone(parse_gtp(f"K{row}"), WHITE)
    assert b.area() == (9 * 19, 10 * 19)
    assert b.score() == 9 * 19 - 10 * 19 - 7.5
    assert b.score(komi=0) == -19


def test_neutral_point_counts_for_nobody():
    # Black wall on J, White wall on L; the K file touches both, so it is neutral.
    b = Board()
    for row in range(1, 20):
        b.set_stone(parse_gtp(f"J{row}"), BLACK)
        b.set_stone(parse_gtp(f"L{row}"), WHITE)
    assert b.area() == (9 * 19, 9 * 19)


def test_lone_stone_owns_board():
    b = setup(black="K10")
    assert b.area() == (NUM_POINTS, 0)


# ----------------------------------------------------------------------
# Rendering and arrays


def test_diagram_round_trip():
    b = setup(black="A1 K10", white="T19")
    text = str(b)
    rows = [line[3:-3] for line in text.splitlines()[1:-1]]
    c = Board.from_diagram("\n".join(rows))
    assert np.array_equal(b.to_array(), c.to_array())
    assert text.splitlines()[-2].startswith(" 1 X")


def test_from_diagram_bad_input():
    with pytest.raises(ValueError):
        Board.from_diagram("X . O")


def test_to_array_orientation():
    b = setup(black="A1", white="T19")
    a = b.to_array()
    assert a.shape == (19, 19)
    assert a[0, 0] == BLACK and a[18, 18] == WHITE
    assert a.sum() == BLACK + WHITE


# ----------------------------------------------------------------------
# Random games against a slow, simple reference engine


class Reference:
    """Dict-of-points Go with no shortcuts.

    Simple ko is checked as "no move may recreate the position from before the
    opponent's last move", which is the same rule stated another way.
    """

    def __init__(self) -> None:
        self.stones: dict[int, Color] = {}
        self.to_move = BLACK
        self.positions = [frozenset()]  # position before each move

    @staticmethod
    def neighbours(p: int) -> list[int]:
        col, row = p % 19, p // 19
        out = []
        for c, r in ((col - 1, row), (col + 1, row), (col, row - 1), (col, row + 1)):
            if 0 <= c < 19 and 0 <= r < 19:
                out.append(r * 19 + c)
        return out

    def group(self, stones: dict[int, Color], p: int) -> tuple[set[int], set[int]]:
        color = stones[p]
        grp, libs, todo = {p}, set(), [p]
        while todo:
            s = todo.pop()
            for n in self.neighbours(s):
                if n not in stones:
                    libs.add(n)
                elif stones[n] == color and n not in grp:
                    grp.add(n)
                    todo.append(n)
        return grp, libs

    def try_play(self, p: int) -> dict[int, Color] | None:
        if p in self.stones:
            return None
        color = self.to_move
        stones = dict(self.stones)
        stones[p] = color
        for n in self.neighbours(p):
            if n in stones and stones[n] != color:
                grp, libs = self.group(stones, n)
                if not libs:
                    for s in grp:
                        del stones[s]
        if not self.group(stones, p)[1]:
            return None
        key = frozenset(stones.items())
        if len(self.positions) >= 2 and key == self.positions[-1]:
            return None
        return stones

    def legal(self) -> list[int]:
        return [p for p in range(NUM_POINTS) if self.try_play(p) is not None]

    def play(self, p: int) -> None:
        new = PASS if p == PASS else self.try_play(p)
        assert new is not None
        self.positions.append(frozenset(self.stones.items()))
        if p != PASS:
            self.stones = new
        self.to_move = self.to_move.opponent


def board_dict(b: Board) -> dict[int, Color]:
    return {p: b[p] for p in range(NUM_POINTS) if b[p] != EMPTY}


@pytest.mark.parametrize("seed", range(3))
def test_random_games_match_reference(seed):
    rng = random.Random(seed)
    b = Board()
    ref = Reference()
    for move_no in range(500):
        legal = b.legal_moves()
        if move_no % 25 == 0 or b.ko is not None:
            assert legal == ref.legal(), f"move {move_no}"
        assert legal == [p for p in range(NUM_POINTS) if b.is_legal(p)]
        p = rng.choice(legal) if legal and rng.random() > 0.02 else PASS
        b.play(p)
        ref.play(p)
        assert board_dict(b) == ref.stones, f"move {move_no}: {format_gtp(p)}"
        assert b.to_move == ref.to_move


def test_illegal_moves_never_change_board():
    rng = random.Random(7)
    b = Board()
    for _ in range(400):
        legal = set(b.legal_moves())
        for p in range(NUM_POINTS):
            if p in legal:
                continue
            before = b.to_array()
            with pytest.raises(IllegalMoveError):
                b.play(p)
            assert np.array_equal(b.to_array(), before)
        b.play(rng.choice(sorted(legal)) if legal else PASS)


@pytest.mark.parametrize("seed", range(3))
def test_undo_round_trip_random_game(seed):
    rng = random.Random(seed)
    b = Board()
    snapshots = []
    for _ in range(400):
        snapshots.append(
            (b.to_array(), dict(b.captures), b.to_move, b.ko, b.consecutive_passes)
        )
        legal = b.legal_moves()
        b.play(rng.choice(legal) if legal and rng.random() > 0.05 else PASS)
    for snap in reversed(snapshots):
        b.undo()
        arr, caps, to_move, ko, passes = snap
        assert np.array_equal(b.to_array(), arr)
        assert b.captures == caps
        assert (b.to_move, b.ko, b.consecutive_passes) == (to_move, ko, passes)
    assert b.move_count == 0


def test_capture_counts_match_stones():
    rng = random.Random(3)
    b = Board()
    for _ in range(600):
        legal = b.legal_moves()
        b.play(rng.choice(legal) if legal else PASS)
    played = {BLACK: 0, WHITE: 0}
    for h in b._history:
        if h.q != PASS:
            played[Color(h.color)] += 1
    assert b.count(BLACK) == played[BLACK] - b.captures[WHITE]
    assert b.count(WHITE) == played[WHITE] - b.captures[BLACK]
