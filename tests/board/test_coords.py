import pytest

from goqueen.board import FILES, NUM_POINTS, PASS, col_row, format_gtp, parse_gtp, point


def test_corners():
    assert parse_gtp("A1") == 0
    assert parse_gtp("T1") == 18
    assert parse_gtp("A19") == 342
    assert parse_gtp("T19") == 360
    assert col_row(parse_gtp("A1")) == (0, 0)  # A1 is bottom-left


def test_no_letter_i():
    assert "I" not in FILES
    assert len(FILES) == 19
    assert parse_gtp("J1") == 8
    assert format_gtp(8) == "J1"
    with pytest.raises(ValueError):
        parse_gtp("I5")


def test_round_trip_all_points():
    for p in range(NUM_POINTS):
        assert parse_gtp(format_gtp(p)) == p
    names = {format_gtp(p) for p in range(NUM_POINTS)}
    assert len(names) == NUM_POINTS


def test_point_and_col_row_round_trip():
    for p in range(NUM_POINTS):
        assert point(*col_row(p)) == p


def test_case_and_whitespace():
    assert parse_gtp(" q16 ") == parse_gtp("Q16") == point(15, 15)


def test_pass():
    assert parse_gtp("pass") == PASS
    assert parse_gtp("PASS") == PASS
    assert format_gtp(PASS) == "pass"


@pytest.mark.parametrize(
    "bad", ["", "A", "A0", "A20", "U1", "Z9", "1A", "A1.5", "AA1", "A-1"]
)
def test_bad_vertices(bad):
    with pytest.raises(ValueError):
        parse_gtp(bad)


def test_bad_points():
    for bad in (-2, NUM_POINTS):
        with pytest.raises(ValueError):
            format_gtp(bad)
    with pytest.raises(ValueError):
        point(19, 0)
