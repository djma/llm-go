# goqueen: agent notes

Read `docs/design.md` first. It is the plan of record.

## Tooling

- Python via `uv` only: `uv run`, `uv add`, `uv remove`. Do not edit `pyproject.toml` by hand when a `uv` command can do it.
- Format and lint with `uv run ruff format` and `uv run ruff check`. Fix all warnings before you commit.
- Tests: `uv run pytest`. Every module in `board/` and `data/` needs tests.
- Python 3.12. Type hints on public functions.

## Fixed decisions

- Encoder: KataGo **b18** net, frozen. It is already superhuman.
- Decoder: SmolLM3-3B.
- Rules: Chinese (area) scoring, komi 7.5, 19x19 only.
- Coordinates: GTP (A1-T19, no letter I). A1 is bottom-left.
- Oracle: KataGo analysis engine. "Mistake" = win-rate drop >= 10% OR score-lead drop >= 2 points (tunable).

## Current phase: 0 (tooling, CPU only)

Work items, in order. Each one is a separate PR.

1. `board/`: Go rules engine. Board state, play/undo, captures, suicide, simple ko (positional superko optional), liberties, groups, legal moves, GTP coordinate parse/format. Must be fast enough to generate millions of QA pairs (numpy or pure Python with flood fill is fine; measure first).
2. `data/sgf.py`: SGF reader (main line only is fine) that yields positions for the rules engine.
3. `data/curriculum/`: QA generators for stages 1 and 2 (see the Curriculum table in the design doc). Each task: three question/answer formats; answer computed by `board/`; deterministic given a seed.
4. Cross-check: compare `board/` against a second Go implementation on 10K random positions. This is the Phase 0 gate.
5. `katago/analysis.py`: client for the KataGo analysis engine (JSON over stdin/stdout). Returns top moves, win rate, score lead, PV and ownership.
6. `data/curriculum/`: stages 3 and 4 (play a line, then ask). Ladder solver and simple capturing-race solver.

## Rules for agents

- Cloud sessions have no GPU. Do not start training code until Phase 1.
- Do not commit data, model weights or SGF archives. Use `/data`, `/models`, `/runs` (ignored by git).
- Check licences before you add any dataset or net download script, and note the licence in the PR.
- Keep PRs small and focused on one work item.
