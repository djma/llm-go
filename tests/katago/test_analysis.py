import sys
import time
from pathlib import Path

import numpy as np
import pytest

from goqueen.board import BLACK, PASS, WHITE, Board, parse_gtp
from goqueen.katago.analysis import (
    KataGo,
    KataGoError,
    board_query,
    build_query,
    is_mistake,
    move_loss,
)

FAKE = str(Path(__file__).with_name("fake_katago.py"))


def engine(perspective="BLACK") -> KataGo:
    return KataGo([sys.executable, FAKE, perspective], perspective=perspective)


def test_build_query():
    q = build_query(
        [(BLACK, parse_gtp("Q16")), (WHITE, PASS)],
        max_visits=50,
        allow_moves=[parse_gtp("D4")],
    )
    assert q["moves"] == [["B", "Q16"], ["W", "pass"]]
    assert q["rules"] == "chinese"
    assert q["komi"] == 7.5
    assert q["maxVisits"] == 50
    assert q["includeOwnership"] is True
    assert q["allowMoves"] == [{"player": "B", "moves": ["D4"], "untilDepth": 1}]
    assert "id" not in q


def test_board_query():
    b = Board.from_moves(["Q16", "D4", "Q4"])
    q = board_query(b)
    assert q["moves"] == []
    assert sorted(q["initialStones"]) == [["B", "Q16"], ["B", "Q4"], ["W", "D4"]]
    assert q["initialPlayer"] == "W"


@pytest.mark.parametrize("perspective", ["BLACK", "WHITE", "SIDETOMOVE"])
@pytest.mark.parametrize("to_move", [BLACK, WHITE])
def test_values_are_from_blacks_view(perspective, to_move):
    moves = [] if to_move == BLACK else [(BLACK, parse_gtp("K10"))]
    with engine(perspective) as kg:
        a = kg.analyze(moves)
    assert a.to_move == to_move
    assert a.winrate == pytest.approx(0.70)
    assert a.score_lead == pytest.approx(5.0)
    assert a.best.move == parse_gtp("Q16")
    assert a.best.winrate == pytest.approx(0.72)
    assert a.best.pv == [parse_gtp("Q16"), parse_gtp("D4"), PASS]
    assert a.winrate_for(WHITE) == pytest.approx(0.30)
    assert a.score_for(WHITE) == pytest.approx(-5.0)
    # Ownership comes from A19 first; ours is [row, col] from A1.
    assert a.ownership is not None
    assert a.ownership.shape == (19, 19)
    assert a.ownership[18, 0] == pytest.approx(-1.0)  # A19 is KataGo index 0
    assert a.ownership[0, 18] == pytest.approx(1.0)  # T1 is KataGo index 360
    assert a.ownership[0, 0] == pytest.approx((342 / 360) * 2 - 1)  # A1


def test_mistake_test():
    with engine() as kg:
        a = kg.analyze()  # Black to move
    # D4 loses 0.17 win rate and 3.5 points against Q16.
    wr, score = move_loss(a, parse_gtp("D4"))
    assert wr == pytest.approx(0.17)
    assert score == pytest.approx(3.5)
    assert is_mistake(a, parse_gtp("D4")) is True
    assert is_mistake(a, parse_gtp("Q16")) is False
    assert is_mistake(a, parse_gtp("K10")) is None  # not searched
    assert (
        is_mistake(a, parse_gtp("D4"), max_winrate_drop=0.2, max_score_drop=4) is False
    )


def test_mistake_test_for_white():
    with engine() as kg:
        a = kg.analyze([(BLACK, parse_gtp("K10"))])
    # From White's view D4 is better than Q16 by 0.17 and 3.5 points.
    assert is_mistake(a, parse_gtp("Q16")) is False
    wr, _ = move_loss(a, parse_gtp("D4"))
    assert wr == pytest.approx(-0.17)


def test_allow_moves():
    with engine() as kg:
        a = kg.analyze(allow_moves=[parse_gtp("K10")])
    assert [m.move for m in a.moves] == [parse_gtp("K10")]


def test_analyze_turns():
    moves = [(BLACK, parse_gtp("Q16")), (WHITE, parse_gtp("D4"))]
    with engine() as kg:
        results = kg.analyze_query(build_query(moves, analyze_turns=[0, 1, 2]))
    assert [a.turn for a in results] == [0, 1, 2]
    assert [a.to_move for a in results] == [BLACK, WHITE, BLACK]


def test_concurrent_queries_out_of_order():
    with engine() as kg:
        slow = kg.submit(build_query(extra={"_delay": 0.3}))
        fast = kg.submit(build_query(extra={"_delay": 0.0}))
        assert fast.result(5)[0]["id"] != slow.result(5)[0]["id"]
        start = time.perf_counter()
        futures = [kg.submit(build_query(extra={"_delay": 0.2})) for _ in range(20)]
        assert all(len(f.result(5)) == 1 for f in futures)
        assert time.perf_counter() - start < 2.0  # they ran at once


def test_error_for_one_query():
    with engine() as kg:
        with pytest.raises(KataGoError, match="bad board size"):
            kg.analyze_query({**build_query(), "boardXSize": 9}, timeout=5)
        assert kg.analyze(timeout=5).visits == 200  # the engine is still usable


def test_warning_is_not_fatal(caplog):
    q = build_query()
    del q["rules"]
    with engine() as kg:
        (a,) = kg.analyze_query(q, timeout=5)
    assert a.visits == 200
    assert "no rules" in caplog.text


def test_engine_exit_fails_pending():
    kg = engine()
    pending = kg.submit(build_query(extra={"_delay": 5}))
    with pytest.raises(KataGoError, match="asked to exit"):
        kg.submit(build_query(extra={"_exit": True})).result(5)
    with pytest.raises(KataGoError):
        pending.result(5)
    kg.close()
    with pytest.raises(KataGoError):
        kg.submit(build_query())


def test_duplicate_id_rejected():
    with engine() as kg:
        kg.submit({**build_query(), "id": "x", "_delay": 0.2})
        with pytest.raises(ValueError):
            kg.submit({**build_query(), "id": "x"})


def test_ownership_array_matches_board_orientation():
    # Board.to_array and ownership share [row, col]; check one corner of each.
    b = Board()
    b.set_stone(parse_gtp("A19"), BLACK)
    with engine() as kg:
        a = kg.analyze_board(b)
    arr = b.to_array()
    (row,), (col,) = np.nonzero(arr)
    assert (row, col) == (18, 0)
    assert a.ownership[row, col] == pytest.approx(-1.0)
