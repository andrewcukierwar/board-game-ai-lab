"""Production-versus-research AlphaZero equivalence (Phase 4D.4K-A).

Uses ONLY this checkout's production path: the packaged inference artifact
through ``load_production_model`` and ``ProductionModel.decide`` (the same call
the API agent makes, with an injected research RNG stream). Research inputs
are read as data, never imported:

* ``research-reference``: research-runtime outputs from scripts/alphazero/
  research_reference.py (encodings, logits, priors, values, visits, actions,
  guard diagnostics; mirrors, both actors, full columns, terminal rejection).
* ``historical``: the committed seed-42 4D.4J-A tactical200/solved300/deep128
  records and every seed-42 match decision of 4D.4J-A and 4D.4J-B, replayed
  move by move with each game's own agent RNG stream.

The prospective rule (benchmarks/alphazero/phase4d4k-a/tolerance-declaration.json):
encodings, visits, actions, legal sets and guard outcomes must match exactly;
float outputs must match exactly in the research runtime and within 1e-4 in a
different runtime. The script reports both and never relaxes the integer rule.

    python scripts/alphazero/verify_equivalence.py --research-root <phase4d4j-b worktree> \\
        --reference benchmarks/alphazero/phase4d4k-a/research-reference.json --output result.json
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import platform
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
# One OpenMP thread, as in the API (api/connect4/alphazero.py); read when torch first loads.
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

MANIFEST = REPO / "models/alphazero/az2-hc-g6-cnn-s42-v1.json"
TOLERANCE = 1e-4
TIMING = ("classification_seconds", "guard_seconds")
JA_TASKS = "research/alphazero_tactical_guard/results/tasks/42"
JB_TASKS = "research/alphazero_tactical_guard_confirmation/results/tasks"
FIXTURES = "research/alphazero_resnet_selfplay/fixtures.json"
FIXTURES_SHA256 = "80f6851838b18172d6f818ebd306ce0cd36e4796cac1e3a71df29694feb3ca2c"
ARENA_DOMAIN = "connect4-alphazero-v2-arena"
MATCH_SOURCES = (  # (label, relative file, namespace, seed, guarded)
    ("4D.4J-A control matches", f"{JA_TASKS}/control-matches.json", "phase4d4h-c-strength", 4404902, False),
    ("4D.4J-A guarded matches", f"{JA_TASKS}/guarded-matches.json", "phase4d4h-c-strength", 4404902, True),
    *((f"4D.4J-B {variant} matches vs negamax{d}", f"{JB_TASKS}/42-negamax{d}/{variant}-matches.json",
       f"phase4d4j-b-confirmation:negamax{d}", 4420003, variant == "guarded")
      for d in (3, 5) for variant in ("control", "guarded")),
)

_model = None


def model():
    global _model
    if _model is None:
        from games.connect4.alphazero_v2.production import load_production_model
        _model = load_production_model(MANIFEST)
    return _model


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Tally:
    def __init__(self, label):
        self.label, self.decisions, self.changed, self.mismatches = label, 0, 0, []
        self.max_abs = dict(logits=0.0, prior=0.0, value=0.0)
        self.float_exact = True

    def floats(self, kind, actual, expected, where):
        actual, expected = list(actual), list(expected)
        if len(actual) != len(expected):
            self.mismatches.append(dict(where=where, field=kind, reason="length"))
            return
        for a, e in zip(actual, expected):
            if not (math.isfinite(a) and math.isfinite(e)):
                self.mismatches.append(dict(where=where, field=kind, reason="nonfinite"))
                return
            self.max_abs[kind] = max(self.max_abs[kind], abs(a - e))
            self.float_exact &= a == e

    def exact(self, field, actual, expected, where):
        if actual != expected:
            self.mismatches.append(dict(where=where, field=field, actual=actual, expected=expected))

    def summary(self):
        return dict(label=self.label, decisions=self.decisions, guard_changed=self.changed,
                    integer_and_set_mismatches=len(self.mismatches), mismatches=self.mismatches[:20],
                    float_exact=self.float_exact, max_abs=self.max_abs,
                    within_tolerance=all(v <= TOLERANCE for v in self.max_abs.values()))


def compare_decision(tally, decision, record, guarded, where):
    """``record`` holds research fields: original_action, final_action, visits, root_* and guard."""
    tally.decisions += 1
    tally.exact("original_action", decision.original_action, record["original_action"], where)
    tally.exact("final_action", decision.final_action, record["final_action"], where)
    tally.exact("visits", list(decision.visits), list(record["visits"]), where)
    tally.floats("logits", decision.root_logits, record["root_logits"], where)
    tally.floats("prior", decision.root_prior, record["root_prior"], where)
    tally.floats("value", [decision.root_value], [record["root_value"]], where)
    expected_guard = record["guard"]
    if guarded:
        actual = {k: v for k, v in decision.guard.items() if k not in TIMING}
        expected = {k: v for k, v in expected_guard.items() if k not in TIMING}
        tally.exact("guard", actual, expected, where)
        tally.changed += int(decision.guard["changed"])
    else:
        tally.exact("guard", decision.guard, None, where)
        tally.exact("guard_record", expected_guard, None, where)


def engine(moves):
    from games.connect4.connect4 import Connect4
    game = Connect4()
    for move in moves:
        if game.is_game_over() or not game.make_move(move):
            raise ValueError(f"Illegal history {moves}")
    return game


def reference_unit(path):
    from games.connect4.alphazero_v2.data import seeded_rng
    from games.connect4.alphazero_v2.network import encode_bitboard_current_player
    from games.connect4.alphazero_v2.bitboard import BitboardState
    from games.connect4.neural_mcts import encode_current_player
    data = json.loads(Path(path).read_text())
    rng_rule, m = data["rng"], model()
    tally, terminal = Tally("research reference (mirrors, both actors, full columns, terminal)"), 0
    encodings = dict(engine_exact=0, bitboard_exact=0, compared=0)
    for row in data["rows"]:
        game = engine(row["moves"])
        where = row["id"]
        tally.exact("legal", sorted(game.get_valid_moves()), row["legal"], where)
        tally.exact("terminal", game.is_game_over(), row["terminal"], where)
        if row["terminal"]:
            terminal += 1
            for guarded in (False, True):
                try:
                    m.decide(game, tactical_guard=guarded)
                except ValueError:
                    continue
                tally.mismatches.append(dict(where=where, field="terminal", reason="production accepted"))
            continue
        expected = [repr(v) for v in row["encoding"]]
        engine_input = [repr(v) for v in encode_current_player(game).flatten().tolist()]
        bitboard_input = [repr(v) for v in encode_bitboard_current_player(BitboardState.from_game(game)).flatten().tolist()]
        encodings["compared"] += 1
        encodings["engine_exact"] += engine_input == expected
        encodings["bitboard_exact"] += bitboard_input == expected
        tally.exact("encoding", bitboard_input, expected, where)
        prediction = m.inference.predict(BitboardState.from_game(game))
        tally.floats("logits", prediction.logits, row["prediction"]["logits"], where)
        tally.floats("prior", prediction.policy, row["prediction"]["policy"], where)
        tally.floats("value", [prediction.value], [row["prediction"]["value"]], where)
        for variant, guarded in (("control", False), ("guarded", True)):
            rng = seeded_rng(f"{rng_rule['domain']}:{row['id']}", rng_rule["seed"])
            compare_decision(tally, m.decide(game, tactical_guard=guarded, rng=rng), row[variant], guarded,
                             f"{where}:{variant}")
    result = tally.summary()
    result.update(terminal_rejected=terminal, encodings=encodings, rows=len(data["rows"]))
    return result


def suite_unit(research, suite):
    from games.connect4.alphazero_v2.data import seeded_rng
    fixtures = json.loads((research / FIXTURES).read_text())
    rows = {r["id"]: r for r in fixtures[suite]}
    records = json.loads((research / JA_TASKS / f"{suite}.json").read_text())
    tally, m = Tally(f"4D.4J-A seed-42 {suite} records"), model()
    for record in records["rows"]:
        game = engine(rows[record["id"]]["moves"])
        for variant, guarded in (("control", False), ("guarded", True)):
            rng = seeded_rng(f"phase4d4h-c-evaluation:{suite}:{record['id']}", 4404901)
            compare_decision(tally, m.decide(game, tactical_guard=guarded, rng=rng), record[variant], guarded,
                             f"{record['id']}:{variant}")
    return tally.summary()


def match_unit(research, label, relative, namespace, seed, guarded):
    from games.connect4.alphazero_v2.data import seeded_rng
    from games.connect4.connect4 import Connect4
    data = json.loads((research / relative).read_text())
    tally, m, games = Tally(label), model(), 0
    for record in data["records"]:
        games += 1
        rng = seeded_rng(f"{ARENA_DOMAIN}:{namespace}:{record['opening_id']}:{record['game_index']}:agent", seed)
        game = engine(record["moves"][:record["prefix_length"]])
        decisions = iter(record["decisions"])
        for ply in range(record["prefix_length"], len(record["moves"])):
            entry = next(decisions)
            where = f"{record['opening_id']}:{record['game_index']}:ply{ply}"
            tally.exact("color", entry["color"], game.current_player, where)
            if game.current_player == record["agent_color"]:
                decision = m.decide(Connect4(game.board, game.current_player), tactical_guard=guarded, rng=rng)
                compare_decision(tally, decision, entry, guarded, where)
                tally.exact("played_move", decision.final_action, record["moves"][ply], where)
            if not game.make_move(record["moves"][ply]):
                raise ValueError("Recorded move is illegal")
        tally.exact("winner", game.check_winner(), record["winner"], f"{record['opening_id']}:{record['game_index']}")
    result = tally.summary()
    result["games"] = games
    return result


def run(unit):
    kind, *args = unit
    started = time.perf_counter()
    result = dict(reference=reference_unit, suite=suite_unit, match=match_unit)[kind](*args)
    result["seconds"] = time.perf_counter() - started
    return result


def runtime():
    import numpy
    import torch
    return dict(python=platform.python_version(), torch=str(torch.__version__), numpy=str(numpy.__version__),
                platform=platform.platform(), machine=platform.machine(),
                cpu_capability=str(torch.backends.cpu.get_cpu_capability()),
                intra_op_threads=torch.get_num_threads(),
                thread_environment={k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--research-root", help="read-only research data (omit to skip historical replay)")
    parser.add_argument("--reference", help="research-reference.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Refusing to overwrite evidence")
    units = []
    if args.reference:
        units.append(("reference", str(Path(args.reference).resolve())))
    if args.research_root:
        research = Path(args.research_root).resolve()
        if sha256(research / FIXTURES) != FIXTURES_SHA256:
            raise SystemExit("Development fixtures differ from the frozen research fixtures")
        units += [("suite", research, s) for s in ("tactical", "solved", "deep")]
        units += [("match", research, *source) for source in MATCH_SOURCES]
    if not units:
        raise SystemExit("Nothing to verify")
    started = time.perf_counter()
    m = model()
    # spawn, not fork: forking after torch has started its thread pools can deadlock (Linux default).
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        results = list(pool.map(run, units))
    manifest = m.manifest
    current = runtime()
    export = manifest["export"]["runtime"]
    same_runtime = all(current[k] == export.get(k) for k in ("python", "torch", "numpy", "platform", "machine"))
    integer_ok = all(r["integer_and_set_mismatches"] == 0 for r in results)
    float_exact = all(r["float_exact"] for r in results)
    within = all(r["within_tolerance"] for r in results)
    verdict = ("PASS_EXACT" if integer_ok and float_exact else
               "PASS_WITHIN_TOLERANCE" if integer_ok and within and not same_runtime else "FAIL")
    summary = dict(format="connect4-alphazero-equivalence-result", format_version=1, verdict=verdict,
                   rule="integers/sets/guard exact always; floats exact in the research runtime, <=1e-4 otherwise",
                   same_runtime_as_export=same_runtime, runtime=current, export_runtime_sha256=manifest["export"]["runtime_sha256"],
                   model_id=m.model_id, artifact_sha256=m.artifact_sha256,
                   decisions=sum(r["decisions"] for r in results),
                   guard_changed=sum(r["guard_changed"] for r in results),
                   max_abs={k: max(r["max_abs"][k] for r in results) for k in ("logits", "prior", "value")},
                   seconds=time.perf_counter() - started, units=results)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(summary, stream, indent=1, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: summary[k] for k in ("verdict", "same_runtime_as_export", "decisions", "guard_changed",
                                              "max_abs", "seconds")}, indent=1))
    for r in results:
        print(f"  {r['label']}: decisions={r['decisions']} changed={r['guard_changed']} "
              f"mismatches={r['integer_and_set_mismatches']} float_exact={r['float_exact']}")
    if verdict == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
