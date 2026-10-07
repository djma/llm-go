import random
import re
from collections import Counter

import pytest

from goqueen.board import BLACK, PASS, WHITE, Board, parse_gtp
from goqueen.data.curriculum import (
    FORMATS,
    STAGE_TASKS,
    TASK_NAMES,
    Format,
    decode_position,
    encode_position,
    generate,
    make_example,
)
from goqueen.data.curriculum.generate import line_text, random_line
from goqueen.data.curriculum.positions import own_eye, random_play, random_position
from goqueen.data.curriculum.stage2 import (
    atari_moves,
    atari_stones,
    captures_of,
    ko_and_suicide,
    move_result,
)


def setup(black: str = "", white: str = "", to_move=BLACK) -> Board:
    b = Board()
    for n in black.split():
        b.set_stone(parse_gtp(n), BLACK)
    for n in white.split():
        b.set_stone(parse_gtp(n), WHITE)
    b.to_move = to_move
    return b


def pts(*names: str) -> set[int]:
    return {parse_gtp(n) for n in names}


@pytest.fixture(scope="module")
def positions() -> list[Board]:
    rng = random.Random(0)
    return [random_position(rng, 0, 300) for _ in range(60)]


# ----------------------------------------------------------------------
# Positions


def test_encode_decode_round_trip(positions):
    for b in positions:
        c = decode_position(encode_position(b))
        assert (c.to_array() == b.to_array()).all()
        assert c.to_move == b.to_move
        assert c.ko == b.ko
        assert c.captures == b.captures
        assert c.legal_moves() == b.legal_moves()


def test_encode_keeps_ko():
    b = setup(black="E6 D5 E4", white="F6 E5 G5 F4")
    b.play(parse_gtp("F5"))
    record = encode_position(b)
    assert record["ko"] == "E5"
    c = decode_position(record)
    assert not c.is_legal(parse_gtp("E5"))


def test_random_play_never_fills_own_eye():
    rng = random.Random(1)
    b = Board()
    for p in random_play(b, rng, 300):
        if p >= 0:
            b.undo() if False else None
    # Replay and check each move against the position before it.
    rng = random.Random(1)
    b = Board()
    for _ in range(300):
        before = b.copy()
        moved = random_play(b, rng, 1)
        if not moved or moved[0] < 0:
            continue
        assert not own_eye(before, moved[0], before.to_move)


# ----------------------------------------------------------------------
# Stage 2 facts on known positions


def test_move_result_reasons():
    b = setup(black="E6 D5 E4 A2 B1", white="F6 E5 G5 F4")
    b.play(parse_gtp("F5"))  # Black takes the ko; White to move
    assert move_result(b, parse_gtp("E5")) == "ko"
    assert move_result(b, parse_gtp("A1")) == "suicide"
    assert move_result(b, parse_gtp("F5")) == "occupied"
    assert move_result(b, parse_gtp("K10")) == "legal"


def test_ko_and_suicide_open_answers():
    b = setup(black="E6 D5 E4", white="F6 E5 G5 F4")
    b.play(parse_gtp("F5"))
    answers = set()
    for seed in range(200):
        qa = ko_and_suicide(b, random.Random(seed), Format.OPEN)
        assert qa is not None
        answers.add(qa)
    assert ("Can White play E5? If not, why not?", "No: it retakes a ko") in answers


def test_captures_of_leaves_board_unchanged():
    b = setup(black="A3 B3 C3 D2 D1 B1", white="A2 B2 C2 C1", to_move=WHITE)
    before = b.to_array()
    assert captures_of(b, parse_gtp("A1")) == [parse_gtp("B1")]
    assert (b.to_array() == before).all()
    assert b.to_move == WHITE


def test_atari_stones():
    b = setup(black="K10 A1", white="K11 J10 L10 B1")
    assert atari_stones(b, BLACK) == pts("K10", "A1")
    assert atari_stones(b, WHITE) == set()


def test_atari_moves():
    # White K10 has two liberties after Black J10 L10: K11 and K9.
    b = setup(black="J10 L10", white="K10")
    assert atari_moves(b) == pts("K11", "K9")
    # A White stone with three liberties cannot be put in atari in one move.
    b = setup(black="J10", white="K10")
    assert atari_moves(b) == set()


def test_atari_moves_ignores_groups_already_in_atari():
    # White K10 is already in atari (one liberty, K9); filling it captures, not ataris.
    b = setup(black="J10 L10 K11", white="K10")
    assert parse_gtp("K9") not in atari_moves(b)


# ----------------------------------------------------------------------
# Every task and format


ALL = [(stage, name) for stage, tasks in STAGE_TASKS.items() for name in sorted(tasks)]


@pytest.mark.parametrize(("stage", "name"), ALL)
def test_every_task_makes_every_format(positions, stage, name):
    task = STAGE_TASKS[stage][name]
    made = Counter()
    for i, b in enumerate(positions):
        for fmt in FORMATS:
            qa = task(b, random.Random(i), fmt)
            if qa is None:
                continue
            q, a = qa
            assert q.strip() and a.strip()
            made[fmt] += 1
            if fmt is Format.YESNO:
                assert a in ("Yes", "No")
            if fmt is Format.CHOICE:
                options = re.findall(r"([ABCD])\) (.*?)(?= [ABCD]\) |$)", q)
                assert [label for label, _ in options] == list("ABCD")
                assert len({text for _, text in options}) == 4
                assert a in {f"{label}) {text}" for label, text in options}
    for fmt in FORMATS:
        assert made[fmt] > 0, f"{name} never made {fmt}"


@pytest.mark.parametrize(("stage", "name"), ALL)
def test_yes_no_is_balanced(positions, stage, name):
    task = STAGE_TASKS[stage][name]
    answers = Counter()
    for seed in range(400):
        b = positions[seed % len(positions)]
        qa = task(b, random.Random(seed), Format.YESNO)
        if qa is not None:
            answers[qa[1]] += 1
    total = answers["Yes"] + answers["No"]
    assert total >= 100
    assert 0.15 <= answers["Yes"] / total <= 0.85, answers


def test_tasks_are_deterministic(positions):
    for stage, name in ALL:
        task = STAGE_TASKS[stage][name]
        for fmt in FORMATS:
            b = positions[7]
            assert task(b, random.Random(5), fmt) == task(b, random.Random(5), fmt)


def test_tasks_do_not_change_the_board(positions):
    for stage, name in ALL:
        for fmt in FORMATS:
            b = positions[11]
            before = encode_position(b)
            STAGE_TASKS[stage][name](b, random.Random(3), fmt)
            assert encode_position(b) == before


# ----------------------------------------------------------------------
# Generation


def test_generate_is_deterministic():
    a = generate(random_position, 30, seed=3)
    b = generate(random_position, 30, seed=3)
    c = generate(random_position, 30, seed=4)
    assert [x.to_json() for x in a] == [x.to_json() for x in b]
    assert [x.to_json() for x in a] != [x.to_json() for x in c]


def test_generate_covers_stages_tasks_formats():
    examples = generate(random_position, 1500, seed=0, stages=(1, 2, 3, 4))
    assert {e.stage for e in examples} == {1, 2, 3, 4}
    for stage in (1, 2, 3, 4):
        assert {e.task for e in examples if e.stage == stage} == set(TASK_NAMES[stage])
    assert {e.format for e in examples} == {str(f) for f in FORMATS}


def test_line_examples_replay_to_the_answer():
    # Stage 3: replay the stored line on the stored position and recompute the answer.
    for ex in generate(random_position, 300, seed=5, stages=(3,)):
        assert 1 <= len(ex.line) <= 12
        assert ex.question.startswith("After ")
        if ex.task == "stone_count" and ex.format == "open":
            b = decode_position(ex.position)
            for m in ex.line:
                b.play(parse_gtp(m))
            color = BLACK if "Black stones" in ex.question.split(": ", 1)[1] else WHITE
            assert ex.answer == str(b.count(color))


def test_line_text():
    b = Board()
    assert (
        line_text(b, [parse_gtp("Q16"), PASS, parse_gtp("D4")])
        == "Black Q16, White pass, Black D4"
    )


def test_random_line_does_not_change_board():
    b = random_position(random.Random(2), 50, 60)
    before = encode_position(b)
    line = random_line(b, random.Random(3), 12)
    assert len(line) == 12
    assert encode_position(b) == before


def test_stage4_no_line_tasks():
    examples = generate(random_position, 400, seed=1, stages=(4,))
    ladder = [e for e in examples if e.task == "ladder"]
    race = [e for e in examples if e.task == "race"]
    assert ladder and race
    assert all(e.line == [] for e in ladder + race)
    assert {e.answer for e in race if e.format == "open"} <= {
        "Black",
        "White",
        "neither (seki)",
    }


def test_race_answers_are_balanced():
    from goqueen.data.curriculum.stage4 import make_race_position

    rng = random.Random(0)
    results = Counter(make_race_position(rng)[3] for _ in range(300))
    assert len(results) == 3
    assert min(results.values()) >= 60, results


def test_race_question_matches_solver():
    from goqueen.board.tactics import race
    from goqueen.data.curriculum.stage4 import make_race_position, race_question

    rng = random.Random(1)
    for _ in range(30):
        board, black, white, result = make_race_position(rng)
        assert race(board, black, white) == result
        _, a = race_question(board, black, white, rng, Format.OPEN)
        assert (
            a
            == {"BLACK": "Black", "WHITE": "White", "NEITHER": "neither (seki)"}[
                result.name
            ]
        )


def test_answers_match_decoded_position():
    # The stored position is enough to recompute the answer.
    for ex in generate(random_position, 50, seed=9):
        if ex.task == "stone_count" and ex.format == "open":
            b = decode_position(ex.position)
            color = BLACK if "Black" in ex.question else WHITE
            assert ex.answer == str(b.count(color))


def test_make_example_fixed_task():
    b = setup(black="K10")
    ex = make_example(b, 1, random.Random(0), task="stone_count", fmt="open")
    assert ex is not None
    assert ex.answer in ("0", "1")
    assert ex.position["stones"].count("X") == 1
