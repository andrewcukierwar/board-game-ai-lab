"""Bounded local/staging performance measurements of the packaged AlphaZero path (Phase 4D.4K-A).

``inprocess``: in a FRESH interpreter, time the torch import and strict artifact
load (cold), then warm network inference and warm full decisions, guarded and
unguarded, on seeded legal positions in five game phases for both actors.
``http``: against an isolated staging server only (never the public backend):
end-to-end move latency, simultaneous AlphaZero requests (bounded concurrency,
no partial commits) and repeated complete games.

    python scripts/alphazero/benchmark_backend.py inprocess --output result.json
    python scripts/alphazero/benchmark_backend.py http --base http://127.0.0.1:18103 --output result.json
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import platform
import random
import resource
import statistics
import sys
import time
import urllib.error
import urllib.request

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
PHASES = (("opening", 0, 6), ("early_middle", 7, 13), ("midgame", 14, 24), ("late", 25, 34),
          ("near_terminal", 35, 41))


def stats(values_ms):
    ordered = sorted(values_ms)
    return dict(n=len(ordered), mean=statistics.fmean(ordered), median=statistics.median(ordered),
                p95=ordered[max(0, round(0.95 * len(ordered)) - 1)], worst=ordered[-1])


def peak_rss_mib():
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage / (1024 ** 2 if sys.platform == "darwin" else 1024)


def positions(per_phase_and_actor, seed=4410300):
    """Seeded legal nonterminal boards: (phase, actor, moves)."""
    from games.connect4.alphazero_v2.bitboard import BitboardState
    rows, attempt = [], 0
    for name, low, high in PHASES:
        for actor in (0, 1):
            found = 0
            while found < per_phase_and_actor:
                rng = random.Random(seed + attempt)
                attempt += 1
                target = rng.randrange(low, high + 1)
                if target % 2 != actor:
                    continue
                state, moves = BitboardState((0, 0), 0, 0), []
                while len(moves) < target:
                    options = [m for m in sorted(state.get_valid_moves()) if not state.drop(m).is_game_over()]
                    if not options:
                        break
                    move = rng.choice(options)
                    state, moves = state.drop(move), moves + [move]
                if len(moves) == target:
                    rows.append((name, actor, moves))
                    found += 1
    return rows


def inprocess(args):
    started_wall, started_cpu = time.perf_counter(), time.process_time()
    import torch  # noqa: F401  (timed: the cold-start import cost is part of enabling AlphaZero)
    imported = time.perf_counter()
    from games.connect4.alphazero_v2.bitboard import BitboardState
    from games.connect4.alphazero_v2.production import load_production_model
    from games.connect4.connect4 import Connect4
    model = load_production_model(REPO / "models/alphazero/az2-hc-g6-cnn-s42-v1.json")
    loaded = time.perf_counter()
    rss_after_load = peak_rss_mib()

    def game(moves):
        g = Connect4()
        for m in moves:
            g.make_move(m)
        return g
    rows = positions(args.per_phase)
    for _, _, moves in rows[:5]:  # warm-up, excluded
        model.decide(game(moves), tactical_guard=True)
    inference, decisions = [], {}
    cpu0, wall0 = time.process_time(), time.perf_counter()
    for phase, actor, moves in rows:
        g = game(moves)
        state = BitboardState.from_game(g)
        for _ in range(args.repeats):
            t = time.perf_counter()
            model.inference.predict(state)
            inference.append((time.perf_counter() - t) * 1000)
            for guarded in (False, True):
                d = model.decide(g, tactical_guard=guarded)
                key = ("guarded" if guarded else "unguarded")
                decisions.setdefault(key, []).append((phase, actor, d.total_seconds * 1000,
                                                      (d.guard or {}).get("guard_seconds", 0.0) * 1000))
    cpu, wall = time.process_time() - cpu0, time.perf_counter() - wall0
    result = dict(
        mode="inprocess", runtime=dict(python=platform.python_version(), torch=str(torch.__version__),
                                       platform=platform.platform(), machine=platform.machine(),
                                       torch_threads=torch.get_num_threads()),
        positions=len(rows), repeats=args.repeats,
        cold=dict(torch_import_ms=(imported - started_wall) * 1000, artifact_load_and_self_check_ms=(loaded - imported) * 1000,
                  total_ms=(loaded - started_wall) * 1000, rss_after_load_mib=rss_after_load),
        warm_inference_ms=stats(inference),
        moves={k: dict(all=stats([r[2] for r in v]),
                       by_phase={p: stats([r[2] for r in v if r[0] == p]) for p, _, _ in PHASES},
                       by_actor={str(a): stats([r[2] for r in v if r[1] == a]) for a in (0, 1)})
               for k, v in decisions.items()},
        guard_only_ms=stats([r[3] for r in decisions["guarded"]]),
        measured_cpu_utilization=cpu / wall, peak_rss_mib=peak_rss_mib(),
        process_cpu_seconds=time.process_time() - started_cpu)
    return result


def request(base, method, path, body=None, timeout=30):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data=data, method=method, headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status, json.loads(response.read() or b"null"), (time.perf_counter() - started) * 1000
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read() or b"null"), (time.perf_counter() - started) * 1000


def http(args):
    base = args.base.rstrip("/")
    if not (base.startswith("http://127.0.0.1:") or base.startswith("http://localhost:")) or base.endswith(":8000"):
        raise SystemExit("HTTP measurements are restricted to isolated local staging ports")
    status, info, _ = request(base, "GET", "/v1/connect4/agents/alphazero")
    if not info.get("available"):
        raise SystemExit("Staging server does not report AlphaZero available")

    def start(p1, p2):
        status, game, _ = request(base, "POST", "/v1/connect4/start_game", dict(player1=p1, player2=p2))
        assert status == 201, game
        return game
    latencies = {"unguarded": [], "guarded": []}
    for guarded in (False, True):
        for _ in range(args.games):
            game = start({"type": "alphazero", "tactical_guard": guarded}, {"type": "alphazero", "tactical_guard": guarded})
            while not game["gameOver"]:
                status, game, ms = request(base, "POST", "/v1/connect4/make_move",
                                           dict(game_id=game["game_id"], revision=game["revision"]))
                assert status == 200, game
                latencies["guarded" if guarded else "unguarded"].append(ms)
    # Simultaneous AlphaZero requests on distinct games: at most one runs, the rest are refused, none commit.
    bursts = []
    for _ in range(args.bursts):
        games = [start({"type": "alphazero"}, {"type": "human"}) for _ in range(args.concurrency)]
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            results = list(pool.map(lambda g: request(base, "POST", "/v1/connect4/make_move",
                                                      dict(game_id=g["game_id"], revision=0)), games))
        codes = [r[0] for r in results]
        revisions = [request(base, "GET", f"/v1/connect4/games/{g['game_id']}")[1]["revision"] for g in games]
        bursts.append(dict(statuses=codes, revisions=revisions,
                           busy_codes=sorted({r[1].get("code") for r in results if r[0] != 200}),
                           consistent=all((c == 200) == (v == 1) for c, v in zip(codes, revisions))))
    return dict(mode="http", base="isolated local staging", games_per_variant=args.games,
                move_latency_ms={k: stats(v) for k, v in latencies.items()}, bursts=bursts,
                burst_summary=dict(requests=sum(len(b["statuses"]) for b in bursts),
                                   succeeded=sum(b["statuses"].count(200) for b in bursts),
                                   refused_503=sum(b["statuses"].count(503) for b in bursts),
                                   partial_commits=sum(not b["consistent"] for b in bursts)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("inprocess", "http"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-phase", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--base")
    parser.add_argument("--games", type=int, default=3)
    parser.add_argument("--bursts", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Refusing to overwrite measurements")
    result = inprocess(args) if args.mode == "inprocess" else http(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(json.dumps(result, indent=1, sort_keys=True)[:3000])


if __name__ == "__main__":
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    main()
