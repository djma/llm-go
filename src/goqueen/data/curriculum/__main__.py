"""Write curriculum examples as JSON lines.

    uv run python -m goqueen.data.curriculum --n 1000 --out /data/qa/stage12.jsonl
    uv run python -m goqueen.data.curriculum --sgf-dir /data/sgf --n 100000 --out ...

Without ``--sgf-dir``, positions come from random play (for smoke tests only).
"""

import argparse
import json
import sys
import time

from goqueen.data.curriculum.generate import generate
from goqueen.data.curriculum.positions import random_position, sgf_files, sgf_position


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stages", default="1,2", help="comma-separated, e.g. 1,2")
    parser.add_argument(
        "--sgf-dir", help="directory of .sgf files (searched recursively)"
    )
    parser.add_argument("--out", help="output .jsonl (default: stdout)")
    args = parser.parse_args()

    stages = tuple(int(s) for s in args.stages.split(","))
    if args.sgf_dir:
        files = sgf_files(args.sgf_dir)
        if not files:
            sys.exit(f"no .sgf files under {args.sgf_dir}")

        def source(rng):
            while (board := sgf_position(rng.choice(files), rng)) is None:
                pass
            return board
    else:
        source = random_position

    start = time.perf_counter()
    examples = generate(source, args.n, args.seed, stages)
    elapsed = time.perf_counter() - start
    lines = "".join(json.dumps(ex.to_json()) + "\n" for ex in examples)
    if args.out:
        with open(args.out, "w") as f:
            f.write(lines)
    else:
        sys.stdout.write(lines)
    print(
        f"{len(examples)} examples in {elapsed:.1f}s ({len(examples) / elapsed:,.0f}/s)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
