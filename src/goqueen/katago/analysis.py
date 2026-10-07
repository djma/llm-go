"""Client for the KataGo analysis engine (JSON lines over stdin and stdout).

Start KataGo yourself, or let ``KataGo`` start it:

    with KataGo(["katago", "analysis", "-config", "analysis.cfg", "-model", "b18.bin.gz"]) as kg:
        a = kg.analyze([(BLACK, parse_gtp("Q16")), (WHITE, parse_gtp("D4"))], max_visits=1000)
        a.best.move, a.winrate, a.score_lead, a.ownership

All values in ``Analysis`` are from Black's point of view, whatever the
engine's ``reportAnalysisWinratesAs`` setting is (tell the client which one
it uses). ``ownership`` is a ``(19, 19)`` array indexed ``[row, col]`` like
``Board.to_array``, +1 for Black and -1 for White.

Protocol: https://github.com/lightvector/KataGo/blob/master/docs/Analysis_Engine.md
"""

import itertools
import json
import logging
import subprocess
import threading
from collections import deque
from collections.abc import Iterable, Sequence
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, Literal, Self

import numpy as np

from goqueen.board import (
    BLACK,
    KOMI,
    PASS,
    SIZE,
    WHITE,
    Board,
    Color,
    format_gtp,
    parse_gtp,
)

log = logging.getLogger(__name__)

Perspective = Literal["BLACK", "WHITE", "SIDETOMOVE"]
Move = tuple[Color, int]  # (colour, point or PASS)

# Default mistake thresholds from the design doc (tunable).
MAX_WINRATE_DROP = 0.10
MAX_SCORE_DROP = 2.0


class KataGoError(RuntimeError):
    """The engine reported an error, or exited."""


@dataclass(frozen=True)
class MoveInfo:
    move: int  # point or PASS
    visits: int
    winrate: float  # Black's win rate after this move
    score_lead: float  # Black's lead in points after this move
    prior: float
    order: int  # KataGo's rank; 0 is the move it would play
    pv: list[int]  # principal variation, starting with ``move``


@dataclass(frozen=True)
class Analysis:
    turn: int
    to_move: Color
    winrate: float  # Black's win rate at the root
    score_lead: float  # Black's lead in points at the root
    visits: int
    moves: list[MoveInfo]  # in KataGo's order (best first)
    ownership: np.ndarray | None  # (19, 19) float32, [row, col], +1 = Black

    @property
    def best(self) -> MoveInfo:
        return self.moves[0]

    def move_info(self, move: int) -> MoveInfo | None:
        return next((m for m in self.moves if m.move == move), None)

    def winrate_for(self, color: Color) -> float:
        return self.winrate if color == BLACK else 1.0 - self.winrate

    def score_for(self, color: Color) -> float:
        return self.score_lead if color == BLACK else -self.score_lead


def move_loss(analysis: Analysis, move: int) -> tuple[float, float] | None:
    """Win-rate and score lost by ``move`` against KataGo's best move, for the mover.

    None if KataGo did not search ``move`` (ask again with ``allow_moves``).
    """
    info = analysis.move_info(move)
    if info is None:
        return None
    best = analysis.best
    sign = 1.0 if analysis.to_move == BLACK else -1.0
    return sign * (best.winrate - info.winrate), sign * (
        best.score_lead - info.score_lead
    )


def is_mistake(
    analysis: Analysis,
    move: int,
    max_winrate_drop: float = MAX_WINRATE_DROP,
    max_score_drop: float = MAX_SCORE_DROP,
) -> bool | None:
    """True if ``move`` loses at least ``max_winrate_drop`` win rate or ``max_score_drop`` points.

    None if KataGo did not search ``move``.
    """
    loss = move_loss(analysis, move)
    if loss is None:
        return None
    wr, score = loss
    return wr >= max_winrate_drop or score >= max_score_drop


# ----------------------------------------------------------------------
# Queries


def _player(color: Color) -> str:
    return "B" if color == BLACK else "W"


def build_query(
    moves: Sequence[Move] = (),
    *,
    initial_stones: Iterable[Move] = (),
    initial_player: Color | None = None,
    komi: float = KOMI,
    max_visits: int | None = None,
    include_ownership: bool = True,
    analyze_turns: Sequence[int] | None = None,
    allow_moves: Sequence[int] | None = None,
    pv_len: int | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """An analysis query without an ``id`` (the client adds one).

    ``allow_moves`` limits the root to these moves for the player to move
    at the analysed turn, so you can score a move KataGo would not search.
    """
    q: dict[str, Any] = {
        "moves": [[_player(c), format_gtp(p)] for c, p in moves],
        "rules": "chinese",
        "komi": komi,
        "boardXSize": SIZE,
        "boardYSize": SIZE,
        "includeOwnership": include_ownership,
    }
    stones = [[_player(c), format_gtp(p)] for c, p in initial_stones]
    if stones:
        q["initialStones"] = stones
    if initial_player is not None:
        q["initialPlayer"] = _player(initial_player)
    if max_visits is not None:
        q["maxVisits"] = max_visits
    if analyze_turns is not None:
        q["analyzeTurns"] = list(analyze_turns)
    if pv_len is not None:
        q["analysisPVLen"] = pv_len
    if allow_moves is not None:
        if moves:
            mover = moves[-1][0].opponent
        else:
            mover = initial_player if initial_player is not None else BLACK
        q["allowMoves"] = [
            {
                "player": _player(mover),
                "moves": [format_gtp(p) for p in allow_moves],
                "untilDepth": 1,
            }
        ]
    if extra:
        q.update(extra)
    return q


def board_query(board: Board, **kwargs: Any) -> dict[str, Any]:
    """A query for ``board`` as setup stones, with no move history.

    KataGo then sees no ko and no history; use ``build_query`` with the
    game's moves when those matter.
    """
    stones = [(c, p) for c in (BLACK, WHITE) for p in board.stones(c)]
    return build_query(initial_stones=stones, initial_player=board.to_move, **kwargs)


# ----------------------------------------------------------------------
# Responses


def _point(gtp: str) -> int:
    return PASS if gtp.lower() == "pass" else parse_gtp(gtp)


def parse_response(
    resp: dict[str, Any], perspective: Perspective = "BLACK"
) -> Analysis:
    """Turn one final KataGo response into an ``Analysis`` from Black's view."""
    if resp.get("noResults"):
        raise KataGoError(f"query {resp.get('id')} was terminated before any search")
    root = resp["rootInfo"]
    to_move = BLACK if root["currentPlayer"] == "B" else WHITE
    flip = perspective == "WHITE" or (perspective == "SIDETOMOVE" and to_move == WHITE)

    def wr(x: float) -> float:
        return 1.0 - x if flip else x

    def sc(x: float) -> float:
        return -x if flip else x

    moves = [
        MoveInfo(
            move=_point(m["move"]),
            visits=m["visits"],
            winrate=wr(m["winrate"]),
            score_lead=sc(m["scoreLead"]),
            prior=m.get("prior", 0.0),
            order=m["order"],
            pv=[_point(x) for x in m.get("pv", [])],
        )
        for m in sorted(resp.get("moveInfos", []), key=lambda m: m["order"])
    ]
    ownership = None
    if "ownership" in resp:
        # KataGo lists rows from the top (A19 first); we index rows from the bottom.
        own = np.asarray(resp["ownership"], dtype=np.float32).reshape(SIZE, SIZE)[::-1]
        ownership = np.ascontiguousarray(-own if flip else own)
    return Analysis(
        turn=resp["turnNumber"],
        to_move=to_move,
        winrate=wr(root["winrate"]),
        score_lead=sc(root["scoreLead"]),
        visits=root["visits"],
        moves=moves,
        ownership=ownership,
    )


# ----------------------------------------------------------------------
# Client


class _Pending:
    def __init__(self, expected: int) -> None:
        self.expected = expected
        self.responses: list[dict[str, Any]] = []
        self.future: Future[list[dict[str, Any]]] = Future()


class KataGo:
    """A running KataGo analysis engine. Thread-safe; queries may run at once.

    ``perspective`` must match ``reportAnalysisWinratesAs`` in the engine's config.
    """

    def __init__(
        self, command: Sequence[str], perspective: Perspective = "BLACK"
    ) -> None:
        self.perspective = perspective
        self._proc = subprocess.Popen(
            list(command),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._lock = threading.Lock()
        self._pending: dict[str, _Pending] = {}
        self._ids = itertools.count()
        self._stderr: deque[str] = deque(maxlen=50)
        self._closed = False
        self._stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
        self._stderr_thread.start()
        threading.Thread(target=self._read_stdout, daemon=True).start()

    # -- lifecycle

    def close(self, timeout: float = 10.0) -> None:
        """Close stdin; KataGo finishes queued queries and exits."""
        if self._closed:
            return
        self._closed = True
        try:
            assert self._proc.stdin is not None
            self._proc.stdin.close()
        except OSError:
            pass
        try:
            self._proc.wait(timeout)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- queries

    def submit(self, query: dict[str, Any]) -> Future[list[dict[str, Any]]]:
        """Send a raw query; the future gets its final responses, one per analysed turn."""
        if self._closed:
            raise KataGoError("client is closed")
        query = dict(query)
        qid = str(query.setdefault("id", f"q{next(self._ids)}"))
        pending = _Pending(len(query.get("analyzeTurns", [None])))
        with self._lock:
            if qid in self._pending:
                raise ValueError(f"duplicate query id {qid!r}")
            self._pending[qid] = pending
        try:
            assert self._proc.stdin is not None
            self._proc.stdin.write(json.dumps(query) + "\n")
            self._proc.stdin.flush()
        except OSError as e:
            with self._lock:
                self._pending.pop(qid, None)
            raise KataGoError(
                f"cannot write to KataGo: {e}{self._stderr_tail()}"
            ) from e
        return pending.future

    def analyze_query(
        self, query: dict[str, Any], timeout: float | None = None
    ) -> list[Analysis]:
        responses = self.submit(query).result(timeout)
        return [parse_response(r, self.perspective) for r in responses]

    def analyze(
        self, moves: Sequence[Move] = (), timeout: float | None = None, **kwargs: Any
    ) -> Analysis:
        """Analyse the position after ``moves`` (see ``build_query`` for options)."""
        (a,) = self.analyze_query(build_query(moves, **kwargs), timeout)
        return a

    def analyze_board(
        self, board: Board, timeout: float | None = None, **kwargs: Any
    ) -> Analysis:
        """Analyse ``board`` as setup stones (no history or ko; see ``board_query``)."""
        (a,) = self.analyze_query(board_query(board, **kwargs), timeout)
        return a

    # -- reader threads

    def _read_stdout(self) -> None:
        assert self._proc.stdout is not None
        for line in self._proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                log.warning("KataGo sent a non-JSON line: %s", line[:200])
                continue
            self._dispatch(msg)
        self._stderr_thread.join(timeout=1.0)  # so the error shows KataGo's last words
        self._fail_all(KataGoError(f"KataGo exited{self._stderr_tail()}"))

    def _read_stderr(self) -> None:
        assert self._proc.stderr is not None
        for line in self._proc.stderr:
            self._stderr.append(line.rstrip())

    def _dispatch(self, msg: dict[str, Any]) -> None:
        qid = msg.get("id")
        if "warning" in msg:
            log.warning(
                "KataGo warning for %s: %s (%s)", qid, msg["warning"], msg.get("field")
            )
            return
        if "error" in msg:
            err = KataGoError(
                f"KataGo error: {msg['error']} (field {msg.get('field')})"
            )
            if qid is None:
                self._fail_all(err)
                return
            with self._lock:
                pending = self._pending.pop(str(qid), None)
            if pending is not None:
                pending.future.set_exception(err)
            return
        if msg.get("isDuringSearch") or "action" in msg:
            return
        with self._lock:
            pending = self._pending.get(str(qid))
            if pending is None:
                log.warning("KataGo response for unknown id %s", qid)
                return
            pending.responses.append(msg)
            done = len(pending.responses) >= pending.expected
            if done:
                del self._pending[str(qid)]
        if done:
            pending.responses.sort(key=lambda r: r.get("turnNumber", 0))
            pending.future.set_result(pending.responses)

    def _fail_all(self, err: Exception) -> None:
        with self._lock:
            pending = list(self._pending.values())
            self._pending.clear()
        for p in pending:
            if not p.future.done():
                p.future.set_exception(err)

    def _stderr_tail(self) -> str:
        tail = "\n".join(list(self._stderr)[-10:])
        return f"; stderr:\n{tail}" if tail else ""
