"""A stand-in for ``katago analysis`` that speaks the same JSON-lines protocol.

Usage: python fake_katago.py [BLACK|WHITE|SIDETOMOVE]

Black's true values: root win rate 0.70, lead 5.0; Q16 0.72 / 5.5; D4 0.55 / 2.0.
Ownership of row-major index i (from A19) is (i / 360) * 2 - 1 for Black.
Extra query fields for tests: "_delay" (seconds before replying), "_exit" (quit now).
"""

import json
import sys
import threading
import time

PERSPECTIVE = sys.argv[1] if len(sys.argv) > 1 else "BLACK"
lock = threading.Lock()


def send(obj):
    with lock:
        sys.stdout.write(json.dumps(obj) + "\n")
        sys.stdout.flush()


def reply(q):
    time.sleep(q.get("_delay", 0))
    moves = q.get("moves", [])
    turns = q.get("analyzeTurns", [len(moves)])
    for t in turns:
        if t < len(moves):
            player = moves[t][0]
        elif t > 0:
            player = "W" if moves[t - 1][0] == "B" else "B"
        else:
            player = q.get("initialPlayer", "B")
        flip = PERSPECTIVE == "WHITE" or (PERSPECTIVE == "SIDETOMOVE" and player == "W")

        def wr(x, flip=flip):
            return 1 - x if flip else x

        def sc(x, flip=flip):
            return -x if flip else x

        infos = [("Q16", 0.72, 5.5, ["Q16", "D4", "pass"]), ("D4", 0.55, 2.0, ["D4"])]
        allowed = q.get("allowMoves")
        if allowed:
            infos = [(m, 0.40, -3.0, [m]) for m in allowed[0]["moves"]]
        resp = {
            "id": q["id"],
            "isDuringSearch": False,
            "turnNumber": t,
            "moveInfos": [
                {
                    "move": m,
                    "order": i,
                    "visits": 100 - i,
                    "winrate": wr(w),
                    "scoreLead": sc(s),
                    "prior": 0.5,
                    "pv": pv,
                }
                for i, (m, w, s, pv) in enumerate(infos)
            ],
            "rootInfo": {
                "currentPlayer": player,
                "winrate": wr(0.70),
                "scoreLead": sc(5.0),
                "visits": 200,
            },
        }
        if q.get("includeOwnership"):
            resp["ownership"] = [sc((i / 360) * 2 - 1) for i in range(361)]
        send(
            {"id": q["id"], "isDuringSearch": True, "turnNumber": t}
        )  # a progress report to ignore
        send(resp)


def main():
    for line in sys.stdin:
        q = json.loads(line)
        if q.get("_exit"):
            sys.stderr.write("fake: asked to exit\n")
            sys.stderr.flush()
            sys.exit(3)
        if q.get("boardXSize") != 19:
            send({"error": "bad board size", "field": "boardXSize", "id": q.get("id")})
            continue
        if "rules" not in q:
            send({"warning": "no rules", "field": "rules", "id": q["id"]})
        threading.Thread(target=reply, args=(q,), daemon=True).start()


if __name__ == "__main__":
    main()
