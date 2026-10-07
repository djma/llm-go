from goqueen.board.crosscheck import Report, random_games, sgf_games


def test_random_positions_agree_with_sgfmill():
    report = Report()
    games = random_games(150, seed=1, report=report, sample=0.2)
    assert games >= 1
    assert report.positions == 150
    assert report.legality_checks > 10_000
    assert report.mismatches == []


def test_sgf_games_agree_with_sgfmill(tmp_path):
    # A short game with a capture, a pass and a handicap setup.
    (tmp_path / "a.sgf").write_text("(;SZ[19]KM[7.5];B[ar];W[as];B[bs];W[];B[pd])")
    (tmp_path / "b.sgf").write_text(
        "(;SZ[19]HA[2]AB[pd][dp];W[dd];B[pp](;W[qq])(;W[cc]))"
    )
    report = Report()
    assert sgf_games(str(tmp_path), report, None) == 2
    assert report.positions == 8  # 5 + 3 moves
    assert report.mismatches == []
