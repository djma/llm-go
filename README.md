# goqueen

Language models that play Go and explain their moves. This is a port of
[Queen](https://queen-project.github.io/) ([paper](https://arxiv.org/abs/2610.03695),
[code](https://github.com/queen-project/queen)) from chess to Go.

Queen joins a silent engine net (Lc0) to a small instruction-tuned LM (SmolLM3-3B)
with gated cross-attention. It teaches the LM to read the engine with a four-stage QA
curriculum. Then seven rounds of self-distillation (a "natural-language Bellman
update") make it stronger. Here we swap Lc0 for a frozen KataGo b18 net and Stockfish
for KataGo search.

Read the full plan in [docs/design.md](docs/design.md).

## Status

Phase 0 (tooling). No model code yet.

| Phase | What | Budget |
| --- | --- | --- |
| 0 | Rules engine, QA generator, KataGo wrapper, data pools | ~$100 |
| 1 | Bridge + curriculum SFT (key go/no-go gate) | ~$1,400 |
| 2 | Seed data + round 0 | ~$1,500 |
| 3 | Distillation rounds 1-3 | ~$1,500 |
| 4 | Rounds 4-7 + final eval | ~$2,000 |

## Setup

```bash
uv sync
```

```bash
uv run pytest
```

## Layout

```
src/goqueen/
  board/    Go rules engine (exact; checks every curriculum answer)
  katago/   KataGo analysis client (oracle) and net loader (encoder)
  data/     SGF parsing, position pools, curriculum QA generators
  model/    Encoder + decoder + gated cross-attention bridge
  distill/  Distillation loop
  evals/    Elo ladder, NMR/FNMR, claim accuracy, coherence
docs/design.md
```
