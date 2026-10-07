import pytest

from goqueen.board import BLACK, EMPTY, PASS, WHITE, Board, parse_gtp
from goqueen.data.sgf import SGFError, final_board, parse, read_file, replay, sgf_point


def test_sgf_point_corners():
    assert sgf_point("aa") == parse_gtp("A19")  # top left
    assert sgf_point("as") == parse_gtp("A1")
    assert sgf_point("sa") == parse_gtp("T19")
    assert sgf_point("ss") == parse_gtp("T1")
    assert sgf_point("pd") == parse_gtp("Q16")
    assert sgf_point("") == PASS
    assert sgf_point("tt") == PASS
    for bad in ("a", "zz", "abc"):
        with pytest.raises(SGFError):
            sgf_point(bad)


def test_root_properties():
    game = parse("(;GM[1]FF[4]SZ[19]KM[7.5]RE[B+R]HA[0]PB[Alice]PW[Bob];B[pd];W[dp])")
    assert game.komi == 7.5
    assert game.result == "B+R"
    assert game.size == 19
    assert game.handicap == 0
    assert game.prop("PB") == "Alice"
    assert len(game.nodes) == 3


def test_replay_moves():
    game = parse("(;SZ[19];B[pd];W[dp];B[])")
    positions = list(replay(game))
    assert [(p.move, p.color, p.move_number) for p in positions] == [
        (parse_gtp("Q16"), BLACK, 1),
        (parse_gtp("D4"), WHITE, 2),
        (PASS, BLACK, 3),
    ]
    assert positions[0].board.count(BLACK) == 0
    assert positions[1].board[parse_gtp("Q16")] == BLACK
    assert positions[1].board.to_move == WHITE
    board = final_board(game)
    assert board.count(BLACK) == 1 and board.count(WHITE) == 1
    assert board.consecutive_passes == 1


def test_positions_are_independent_copies():
    positions = list(replay(parse("(;B[pd];W[dp])")))
    positions[0].board.play(parse_gtp("K10"))
    assert positions[1].board[parse_gtp("K10")] == EMPTY


def test_main_line_only():
    game = parse("(;SZ[19];B[pd](;W[dp];B[pp])(;W[dd]))")
    moves = [p.move for p in replay(game)]
    assert moves == [parse_gtp("Q16"), parse_gtp("D4"), parse_gtp("Q4")]


def test_nested_variations():
    game = parse("(;B[aa](;W[bb](;B[cc])(;B[dd]))(;W[ee]))")
    assert [p.move for p in replay(game)] == [sgf_point(v) for v in ("aa", "bb", "cc")]


def test_handicap_setup_and_white_first():
    game = parse("(;SZ[19]HA[2]AB[pd][dp]PL[W];W[dd];B[pp])")
    positions = list(replay(game))
    first = positions[0]
    assert first.color == WHITE
    assert first.board.to_move == WHITE
    assert first.board[parse_gtp("Q16")] == BLACK
    assert first.board[parse_gtp("D4")] == BLACK
    assert game.handicap == 2


def test_handicap_without_pl():
    positions = list(replay(parse("(;HA[2]AB[pd][dp];W[dd])")))
    assert positions[0].color == WHITE


def test_compressed_point_list():
    board = final_board(parse("(;AB[aa:cb])"))
    assert board.count(BLACK) == 6
    assert board[parse_gtp("C18")] == BLACK


def test_add_empty():
    board = final_board(parse("(;AB[aa][bb];AE[aa];B[cc])"))
    assert board[sgf_point("aa")] == EMPTY
    assert board[sgf_point("bb")] == BLACK


def test_capture_in_record():
    # Black captures White A1 (SGF "as").
    board = final_board(parse("(;B[ar];W[as];B[bs])"))
    assert board[parse_gtp("A1")] == EMPTY
    assert board.captures[BLACK] == 1


def test_escapes_and_whitespace():
    text = "(\n ; C[a \\] b \\\\ c\\\nd] GM[1]\n ;\tB [pd]\n)"
    game = parse(text)
    assert game.prop("C") == "a ] b \\ cd"
    assert [p.move for p in replay(game)] == [parse_gtp("Q16")]


def test_lowercase_property_names_ff3():
    game = parse("(;GaMe[1]SiZe[19];B[pd])")
    assert game.prop("GM") == "1"
    assert game.size == 19


def test_illegal_move_raises():
    with pytest.raises(SGFError, match="move 2"):
        list(replay(parse("(;B[pd];W[pd])")))


def test_other_board_size_rejected():
    with pytest.raises(SGFError, match="19x19"):
        list(replay(parse("(;SZ[9];B[ee])")))


@pytest.mark.parametrize(
    "bad", ["", "no tree", "(;B[pd]", "(;B[pd", "(;B)", "(;B[pd]x)"]
)
def test_bad_syntax(bad):
    with pytest.raises(SGFError):
        list(replay(parse(bad)))


def test_read_file_latin1(tmp_path):
    path = tmp_path / "g.sgf"
    path.write_bytes("(;PB[Jos\xe9];B[pd])".encode("latin-1"))
    game = read_file(str(path))
    assert game.prop("PB") == "José"


def test_replay_matches_board_from_moves():
    # SGF letters do not skip I, GTP letters do: J10 is "jj" in SGF.
    moves = ["Q16", "D4", "Q4", "D16", "J10", "K9"]
    sgf_letters = "abcdefghijklmnopqrs"
    nodes = []
    for i, m in enumerate(moves):
        p = parse_gtp(m)
        col, row = p % 19, p // 19
        nodes.append(f";{'BW'[i % 2]}[{sgf_letters[col]}{sgf_letters[18 - row]}]")
    assert nodes[4] == ";B[ij]"
    board = final_board(parse("(" + "".join(nodes) + ")"))
    assert (board.to_array() == Board.from_moves(moves).to_array()).all()
