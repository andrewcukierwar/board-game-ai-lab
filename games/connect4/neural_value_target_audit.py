"""Read-only retained-event audit. No neural inference, training, or loss imports.

Proofs concern optimal play; recorded outcomes concern actual self-play behavior.
Only legal engine successors supply proofs. Stored tactical annotations are ignored.
"""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random

from .connect4 import Connect4
from .tactical_value import immediate_wins, tactical_proof

ENCODING = "connect4-current-player-1x6x7-v1"
POLICY_TARGET = "root-visits-temperature-v1"
RUNS = {
    "4D.2b": "phase4d2b-neural-scaled-replacement1-seed42-20261004",
    "4D.2c": "phase4d2c-neural-symmetry-seed42-20261005",
    "4D.2d": "phase4d2d-neural-root-noise-seed42-20261005",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_manifest(path):
    path = Path(path)
    manifest = json.loads(path.read_text())
    require(manifest["format_version"] == 1, f"Unsupported manifest: {path}")
    for name, digest in manifest["files_sha256"].items():
        require(sha256(path.parent / name) == digest, f"Manifest mismatch: {path}: {name}")
    # Supplemental manifests can bind the report and preflight outside their directory.
    for key in ("handoff", "preflight_manifest"):
        if key + "_path" in manifest:
            require(sha256(manifest[key + "_path"]) == manifest[key + "_sha256"],
                    f"External manifest mismatch: {key}")
    return {"path": str(path), "sha256": sha256(path),
            "verified_entries": len(manifest["files_sha256"])}


def canonical_board(game):
    own = "X" if game.current_player == 0 else "O"
    return [[0 if cell == " " else 1 if cell == own else -1 for cell in row]
            for row in game.board]



def counts(rows):
    counter = Counter(str(r["outcome"]) for r in rows)
    return {str(v): counter[str(v)] for v in (1, 0, -1)}


def cohort(game):
    return ("1-2" if game <= 2 else "3-20" if game <= 20 else "21-50"
            if game <= 50 else "51-100" if game <= 100 else "101-150"
            if game <= 150 else "151-200")


def replay_events(events, config, report, run):
    """Replay in event order; reject corruption instead of repairing evidence."""
    collection, pending, updates = [], [], []
    game = Connect4()
    game_number, update_number = 1, 0
    sample_rng = random.Random(config["training_sampling_seed"])
    for event in events:
        kind = event["event"]
        if kind == "ply":
            require(not game.is_game_over(), "Ply after terminal state")
            require(event["game"] == game_number and event["game_ply"] == len(pending) + 1,
                    "Wrong game/ply identity")
            require(event["actual_ply"] == len(collection) + len(pending) + 1,
                    "Wrong global ply identity")
            require(event["updates"] == update_number, "Wrong pre-search update identity")
            actor = game.current_player
            observation = canonical_board(game)
            require(event["actor"] == actor and event["observation"] == observation,
                    "Pre-move board/actor mismatch")
            legal = game.get_valid_moves()
            action, visits, policy = event["action"], event["visits"], event["policy"]
            require(type(action) is int and action in legal, "Illegal recorded action")
            require(len(visits) == len(policy) == 7 and
                    all(type(v) is int and v >= 0 for v in visits) and
                    sum(visits) == config["simulations"], "Invalid visit accounting")
            require(event["temperature"] == config["temperature"] == 1.0 and
                    event["tactical_guard"] is False and event["guard_applied"] == "disabled",
                    "Changed collection search contract")
            require(all(math.isfinite(p) and p >= 0 and
                        math.isclose(p, v / sum(visits), rel_tol=0, abs_tol=1e-15)
                        for p, v in zip(policy, visits)), "Policy/visit mismatch")
            require(all(visits[a] == policy[a] == 0 for a in range(7) if a not in legal)
                    and policy[action] > 0, "Invalid legal/action support")
            if "root_noise_epsilon" in event:
                require(event["root_noise_epsilon"] == config["root_noise_epsilon"],
                        "Noise configuration mismatch")
            example = dict(observation=observation, acting_player=actor, policy=policy,
                           temperature=1.0, tactical_guard=False, outcome=None,
                           encoding=ENCODING, policy_target=POLICY_TARGET)
            digest = hashlib.sha256(json.dumps(example, sort_keys=True,
                                               allow_nan=False).encode()).hexdigest()
            require(digest == event["example_sha256"], "Pending example fingerprint mismatch")
            prediction = event["predicted_wdl"]
            require(len(prediction) == 3 and all(math.isfinite(p) and 0 <= p <= 1
                    for p in prediction) and math.isclose(sum(prediction), 1, abs_tol=1e-6),
                    "Invalid contemporaneous WDL")
            proof = tactical_proof(game)
            winning_visits = sum(visits[a] for a in proof["winning_actions"])
            pending.append(dict(run=run, game=game_number, ply=len(pending) + 1,
                                actor=actor, board=[list(row) for row in game.board],
                                example=example, action=action, visits=visits,
                                policy=policy, predicted_wdl=prediction,
                                pre_search_updates=update_number, proof=proof,
                                winning_visits=winning_visits,
                                visited_win=winning_visits > 0,
                                took_win=action in proof["winning_actions"],
                                cohort=cohort(game_number),
                                stage="early" if len(pending) < 14 else "middle"
                                if len(pending) < 28 else "late"))
            require(game.make_move(action), "Replay move failed")
        elif kind == "completed_game":
            require(game.is_game_over() and bool(pending), "Incomplete completed game")
            require(event["index"] == game_number and event["plies"] == len(pending)
                    and event["moves"] == [r["action"] for r in pending], "Game history mismatch")
            winner = game.check_winner()
            require(event["winner"] == winner and len(event["examples"]) == len(pending),
                    "Winner/example count mismatch")
            for row, stored in zip(pending, event["examples"]):
                label = 0 if winner == -1 else 1 if winner == row["actor"] else -1
                expected = dict(row["example"], outcome=label)
                require(stored == expected, "Stored completed target/contract mismatch")
                row.update(example=expected, outcome=label, winner=winner,
                           collection_index=len(collection))
                collection.append(row)
            require(event["labels"] == counts(pending), "Completed label counts mismatch")
            require(event["labels_by_actor"] == {str(a): counts([r for r in pending
                    if r["actor"] == a]) for a in (0, 1)}, "Actor label counts mismatch")
            saved = report["games"][game_number - 1]
            require(all(saved[k] == event[k] for k in
                        ("index", "moves", "winner", "plies", "labels", "update_start")),
                    "Report/completed event mismatch")
            require(saved["update_end"] == (game_number - 2) * 10
                    if 3 <= game_number <= 199 else saved["update_end"] == update_number,
                    "Report update endpoint mismatch")
            pending, game = [], Connect4()
            game_number += 1
        elif kind == "update":
            require(not pending and event["after_game"] == game_number - 1,
                    "Update does not follow completed collection")
            indices = event["sampled_indices"]
            require(len(indices) == config["batch_size"] == 32 and len(set(indices)) == 32
                    and all(type(i) is int and 0 <= i < len(collection) for i in indices),
                    "Invalid recorded minibatch indices")
            require(indices == sample_rng.sample(range(len(collection)), 32),
                    "Minibatch sampling replay mismatch")
            update_number += 1
            require(event["update"] == update_number and 3 <= event["after_game"] <= 199,
                    "Update sequence mismatch")
            if "horizontal_reflected" in event:
                flags = event["horizontal_reflected"]
                require(len(flags) == 32 and all(type(f) is bool for f in flags),
                        "Invalid augmentation flags")
                require(event["transformed_samples"] == sum(flags) and
                        event["untransformed_samples"] == 32 - sum(flags), "Reflection count mismatch")
            updates.append(dict(update=update_number, after_game=event["after_game"],
                                cohort=cohort(event["after_game"]), sampled_indices=indices))
        elif kind == "smoke_gate":
            require(game_number == 3 and not pending and update_number == 0 and
                    event["result"] == "passed", "Invalid smoke gate")
        else:
            raise ValueError(f"Unexpected/quarantined event: {kind}")
    require(not pending and game_number == report["completed_games"] + 1
            and update_number == report["updates"],
            "Incomplete retained run")
    require(report["updates"] == len(updates)
            and report["actual_plies"] == report["collected_plies"] == len(collection)
            and report["labels"] == counts(collection), "Report totals mismatch")
    require(Counter(u["after_game"] for u in updates) ==
            {g: 10 for g in range(3, report["completed_games"])},
            "Update schedule mismatch")
    return collection, updates


def scores(rows, target):
    if not rows:
        return {"n": 0, "brier": None, "nll": None, "scalar_mae": None, "scalar_mse": None}
    brier, nll, absolute, squared = [], [], [], []
    zero_probabilities = 0
    for row in rows:
        value = target(row)
        index = {1: 0, 0: 1, -1: 2}[value]
        p = row["predicted_wdl"]
        brier.append(sum((v - int(i == index)) ** 2 for i, v in enumerate(p)))
        if p[index] == 0:
            zero_probabilities += 1
        else:
            nll.append(-math.log(p[index]))
        error = p[0] - p[2] - value
        absolute.append(abs(error))
        squared.append(error ** 2)
    return dict(n=len(rows), brier=sum(brier) / len(rows),
                nll=None if zero_probabilities else sum(nll) / len(rows),
                nll_infinite=bool(zero_probabilities), zero_target_probability=zero_probabilities,
                scalar_mae=sum(absolute) / len(rows), scalar_mse=sum(squared) / len(rows))


def summary(rows):
    proven = [r for r in rows if r["proof"]["value"] is not None]
    result = dict(examples=len(rows), outcomes=counts(rows),
                  outcome_prediction=scores(rows, lambda r: r["outcome"]),
                  exact_prediction=scores(proven, lambda r: r["proof"]["value"]))
    for value, key in ((1, "proven_win"), (-1, "proven_loss")):
        selected = [r for r in proven if r["proof"]["value"] == value]
        contradictions = [r for r in selected if r["outcome"] != value]
        result[key] = dict(n=len(selected), fraction=len(selected) / len(rows) if rows else None,
                           outcomes=counts(selected), contradictions=len(contradictions),
                           contradiction_rate=len(contradictions) / len(selected) if selected else None,
                           outcome_prediction=scores(selected, lambda r: r["outcome"]),
                           exact_prediction=scores(selected, lambda r: r["proof"]["value"]))
        if value == 1:
            result[key].update(visited_win=sum(r["visited_win"] for r in selected),
                               winning_visits=sum(r["winning_visits"] for r in selected),
                               winning_actions=sum(len(r["proof"]["winning_actions"]) for r in selected),
                               zero_visit_winning_actions=sum(sum(r["visits"][a] == 0
                                   for a in r["proof"]["winning_actions"]) for r in selected),
                               took_win=sum(r["took_win"] for r in selected),
                               decision_outcomes=[dict(visited_win=visited, took_win=took,
                                   outcomes=counts([r for r in selected if r["visited_win"] == visited
                                                   and r["took_win"] == took]))
                                   for visited, took in ((False, False), (True, False), (True, True))])
    result["contradictions"] = sum(r["outcome"] != r["proof"]["value"] for r in proven)
    result["proven"] = len(proven)
    result["contradiction_rate"] = result["contradictions"] / len(proven) if proven else None
    return result


def grouped(rows, field):
    return {str(v): summary([r for r in rows if r[field] == v])
            for v in sorted(set(r[field] for r in rows))}


def sampling_summary(rows, updates):
    appearances = [rows[i] for u in updates for i in u["sampled_indices"]]
    draw_updates = sum(any(rows[i]["outcome"] == 0 for i in u["sampled_indices"]) for u in updates)
    return dict(updates=len(updates), appearances=len(appearances), outcomes=counts(appearances),
                draw_fraction=counts(appearances)["0"] / len(appearances) if appearances else None,
                updates_with_draw=draw_updates, unique_draw_examples_seen=len({r["collection_index"]
                    for r in appearances if r["outcome"] == 0}),
                contradictions=sum(r["proof"]["value"] is not None and
                    r["outcome"] != r["proof"]["value"] for r in appearances),
                by_source_cohort={c: counts([r for r in appearances if r["cohort"] == c])
                                  for c in sorted(set(r["cohort"] for r in rows))})


def audit_run(directory, run):
    directory = Path(directory)
    manifests = [verify_manifest(directory / name) for name in ("manifest.json", "audit-manifest.json")]
    config_envelope = json.loads((directory / "config.json").read_text())
    config = config_envelope["config"]
    report = json.loads((directory / "report.json").read_text())
    require(report["status"] == "bounded_stop" and report["stop_reason"] == "max_games"
            and report.get("partial_game") is None, "Run did not complete cleanly")
    require(report["completed_games"] == 200 and report["updates"] == 1970,
            "Unexpected retained scaled schedule")
    events = [json.loads(line) for line in (directory / "events.jsonl").open()]
    rows, updates = replay_events(events, config, report, run)
    return dict(run=run, directory=str(directory), manifests=manifests,
                checkpoint_sha256=sha256(directory / "candidate.pt"),
                config=config, source=config_envelope["source"], summary=summary(rows),
                by_actor=grouped(rows, "actor"), by_cohort=grouped(rows, "cohort"),
                by_stage=grouped(rows, "stage"),
                by_cohort_actor={c: grouped([r for r in rows if r["cohort"] == c], "actor")
                                 for c in sorted(set(r["cohort"] for r in rows))},
                sampling=sampling_summary(rows, updates),
                sampling_by_update_cohort={c: sampling_summary(rows, [u for u in updates if u["cohort"] == c])
                                          for c in sorted(set(u["cohort"] for u in updates))},
                draw_games=sorted({r["game"] for r in rows if r["outcome"] == 0}),
                rows=rows), rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, default=Path("experiment-output"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    # Never overwrite or place output inside any historical input directory.
    for directory in RUNS.values():
        source = (args.evidence_root / directory).resolve()
        require(args.output.resolve() != source and source not in args.output.resolve().parents,
                "Output overlaps historical evidence")
    require(not args.output.exists(), "Audit output must be new")
    all_rows, runs = [], {}
    for run, directory in RUNS.items():
        runs[run], rows = audit_run(args.evidence_root / directory, run)
        all_rows.extend(rows)
    result = dict(format_version=1, runs=runs, pooled=summary(all_rows),
                  pooled_by_actor=grouped(all_rows, "actor"),
                  pooled_by_cohort=grouped(all_rows, "cohort"),
                  pooled_by_stage=grouped(all_rows, "stage"))
    serialized = json.dumps(result, indent=2, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(serialized)
    print(json.dumps({r: v["summary"] for r, v in runs.items()}, allow_nan=False))


if __name__ == "__main__":
    main()
