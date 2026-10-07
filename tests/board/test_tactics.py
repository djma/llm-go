import random

import pytest

from goqueen.board import BLACK, WHITE, Board, parse_gtp
from goqueen.board.tactics import (
    RaceResult,
    ladder,
    ladder_starts,
    race,
    race_is_closed,
)


def setup(black: str = "", white: str = "", to_move=BLACK) -> Board:
    b = Board()
    for n in black.split():
        b.set_stone(parse_gtp(n), BLACK)
    for n in white.split():
        b.set_stone(parse_gtp(n), WHITE)
    b.to_move = to_move
    return b


# A White stone at D4 with Black stones below and left. White's liberties: E4, D5.
# Black ataris at E4 or D5 and the ladder runs to the upper right.
LADDER_BLACK = "C4 D3 E3"
LADDER_WHITE = "D4"


def test_ladder_works_on_open_board():
    b = setup(LADDER_BLACK, LADDER_WHITE)
    result = ladder(b, parse_gtp("D4"))
    assert result is not None and result.captured
    assert result.line[0] in {parse_gtp("D5"), parse_gtp("E4")}
    assert len(result.line) > 20  # runs to the edge


def test_ladder_starts():
    b = setup(LADDER_BLACK, LADDER_WHITE)
    # D5 drives White along the open board; E4 lets White out at D5.
    assert ladder_starts(b, parse_gtp("D4")) == [parse_gtp("D5")]
    b = setup(LADDER_BLACK, LADDER_WHITE + " Q16")
    assert ladder_starts(b, parse_gtp("D4")) == []
    b = setup("", "K10")
    assert ladder_starts(b, parse_gtp("K10")) is None  # three liberties


def test_ladder_breaker_saves_prey():
    b = setup(LADDER_BLACK, LADDER_WHITE + " Q16")
    result = ladder(b, parse_gtp("D4"))
    assert result is not None and not result.captured


def test_attacker_breaker_does_not_help_prey():
    # A Black stone on the path only makes the ladder end sooner.
    b = setup(LADDER_BLACK + " Q16", LADDER_WHITE)
    result = ladder(b, parse_gtp("D4"))
    assert result is not None and result.captured


def test_prey_captures_attacker_to_escape():
    # The Black stone at E3 is in atari (White F3, E2): after Black's atari the prey can take it.
    b = setup(LADDER_BLACK, LADDER_WHITE + " F3 E2 D2")
    result = ladder(b, parse_gtp("D4"))
    assert result is not None and not result.captured


def test_ladder_leaves_board_unchanged():
    b = setup(LADDER_BLACK, LADDER_WHITE + " Q16")
    before = b.to_array()
    ladder(b, parse_gtp("D4"))
    assert (b.to_array() == before).all()
    assert b.move_count == 0


def test_ladder_bad_input():
    b = setup(LADDER_BLACK, LADDER_WHITE)
    assert ladder(b, parse_gtp("C4")) is None  # Black's own stone, Black to move
    assert ladder(b, parse_gtp("K10")) is None  # empty


def test_prey_with_three_liberties_is_not_captured():
    b = setup("", "K10")
    result = ladder(b, parse_gtp("K10"))
    assert result is not None and not result.captured


def test_prey_in_atari_is_captured_at_once():
    b = setup("J10 L10 K11", "K10")
    result = ladder(b, parse_gtp("K10"))
    assert result == result.__class__(True, [parse_gtp("K9")])


# ----------------------------------------------------------------------
# Races


def test_race_more_liberties_wins():
    # Black A1 A2 has one liberty (A3); White B1 B2 has three (B3 C1 C2).
    # White A4 and the Black wall on column D close the region.
    b = setup("A1 A2 D1 D2 D3 C3 C4 B4", "B1 B2 A4 A5 B5")
    assert race_is_closed(b, parse_gtp("A1"), parse_gtp("B1"))
    for to_move in (BLACK, WHITE):
        b.to_move = to_move
        assert race(b, parse_gtp("A1"), parse_gtp("B1")) == RaceResult.WHITE


def test_race_equal_liberties_first_move_wins():
    # Black A1 B1: liberties A2 B2. White C1 D1: liberties C2 D2.
    # Row 3 walls (White over A-B, Black over C-D) and E1 E2 close the region.
    b = setup("A1 B1 C3 D3 E1 E2 E3", "C1 D1 A3 B3 A4 B4")
    assert race_is_closed(b, parse_gtp("A1"), parse_gtp("C1"))
    b.to_move = BLACK
    assert race(b, parse_gtp("A1"), parse_gtp("C1")) == RaceResult.BLACK
    b.to_move = WHITE
    assert race(b, parse_gtp("A1"), parse_gtp("C1")) == RaceResult.WHITE


def test_open_race_is_rejected():
    # Black can extend from A3 to A4 and beyond: not a closed race.
    b = setup("A1 A2 A3 C4", "B1 B2 B3")
    assert not race_is_closed(b, parse_gtp("A1"), parse_gtp("B1"))
    assert race(b, parse_gtp("A1"), parse_gtp("B1")) is None
    assert race(b, parse_gtp("A1"), parse_gtp("B1"), require_closed=False) is not None


def test_race_seki():
    # Black B1-B4 C2 and White D1-D4 C4 share their only liberties, C1 and C3.
    # Whoever fills one is captured, so both pass: seki. Walls close the rest.
    b = setup(
        "B1 B2 B3 B4 C2 C5 D5 E1 E2 E3 E4 E5",
        "D1 D2 D3 D4 C4 A1 A2 A3 A4 A5 B5",
    )
    assert b.liberties(parse_gtp("B1")) == {parse_gtp("C1"), parse_gtp("C3")}
    assert b.liberties(parse_gtp("D1")) == {parse_gtp("C1"), parse_gtp("C3")}
    assert race_is_closed(b, parse_gtp("B1"), parse_gtp("D1"))
    for to_move in (BLACK, WHITE):
        b.to_move = to_move
        assert race(b, parse_gtp("B1"), parse_gtp("D1")) == RaceResult.NEITHER


def test_race_leaks_through_friendly_wall():
    # Black C3 would join the race group to the Black wall at C4: not closed.
    b = setup("B1 B2 B3 C2 E1 E2 E3 D4 C4", "D1 D2 D3 A1 A2 A3 A4 B4")
    assert not race_is_closed(b, parse_gtp("B1"), parse_gtp("D1"))


def test_race_order_of_arguments_does_not_matter():
    b = setup("A1 B1 C3 D3 E1 E2 E3", "C1 D1 A3 B3 A4 B4")
    assert race(b, parse_gtp("C1"), parse_gtp("A1")) == race(
        b, parse_gtp("A1"), parse_gtp("C1")
    )


def test_race_bad_input():
    b = setup("K10", "Q16")
    assert race(b, parse_gtp("K10"), parse_gtp("Q16")) is None  # not touching
    assert race(b, parse_gtp("K10"), parse_gtp("K10")) is None


@pytest.mark.parametrize("seed", range(5))
def test_race_leaves_board_unchanged(seed):
    from goqueen.data.curriculum.positions import random_position

    b = random_position(random.Random(seed), 60, 120)
    before = b.to_array()
    moves_before = b.move_count
    for g in b.groups(BLACK)[:5]:
        for s in g.stones:
            for n in (s + 1, s - 1, s + 19, s - 19):
                if 0 <= n < 361 and b[n] == WHITE:
                    race(b, s, n, require_closed=False)
    assert (b.to_array() == before).all()
    assert b.move_count == moves_before


# ----------------------------------------------------------------------
# Symmetry: the answers must not depend on colour or board orientation


def transform(p: int, k: int) -> int:
    """One of the 8 symmetries of the board."""
    c, r = p % 19, p // 19
    if k & 1:
        c = 18 - c
    if k & 2:
        r = 18 - r
    if k & 4:
        c, r = r, c
    return r * 19 + c


def transformed(b: Board, k: int, swap: bool = False) -> Board:
    out = Board()
    for color in (BLACK, WHITE):
        for p in b.stones(color):
            out.set_stone(transform(p, k), color.opponent if swap else color)
    out.to_move = b.to_move.opponent if swap else b.to_move
    return out


SWAP = {
    RaceResult.BLACK: RaceResult.WHITE,
    RaceResult.WHITE: RaceResult.BLACK,
    RaceResult.NEITHER: RaceResult.NEITHER,
}


def _race_cases(n: int):
    from goqueen.data.curriculum.positions import race_position

    rng = random.Random(42)
    for _ in range(n):
        found = race_position(rng)
        if found is not None:
            yield found


def test_race_symmetry_and_colour_swap():
    cases = 0
    for b, s, n in _race_cases(40):
        expected = race(b, s, n)
        if expected is None:
            continue
        cases += 1
        for k in (1, 2, 4, 7):
            assert race(transformed(b, k), transform(s, k), transform(n, k)) == expected
        assert race(transformed(b, 0, swap=True), s, n) == SWAP[expected]
    assert cases >= 20


def test_ladder_symmetry():
    from goqueen.data.curriculum.positions import random_position

    rng = random.Random(7)
    seen = {True: 0, False: 0}
    for _ in range(10):
        b = transformed(random_position(rng, 100, 250), 0)
        for g in b.groups(b.to_move.opponent):
            if len(g.liberties) != 2:
                continue
            p = min(g.stones)
            expected = ladder(b, p)
            if expected is None:
                continue
            seen[expected.captured] += 1
            for k in (3, 5, 6):
                got = ladder(transformed(b, k), transform(p, k))
                assert got is not None and got.captured == expected.captured
    assert seen[True] >= 5 and seen[False] >= 5
