"""Same-host independent saved-evidence audit; no production rule imports.

Rules use immutable top-first matrices and directional cell scans, independently
of the guard's bitboard operations. No neural inference or oracle solving.
"""
from dataclasses import dataclass
import json
import math
import hashlib
from collections import Counter, defaultdict
from pathlib import Path


@dataclass(frozen=True)
class ReferenceBoard:
    cells: tuple
    actor: int

    @classmethod
    def empty(cls):
        return cls(tuple((" ",) * 7 for _ in range(6)), 0)

    @classmethod
    def from_moves(cls, moves):
        board = cls.empty()
        for action in moves:
            board = board.play(action)
        return board

    def legal(self):
        return tuple(c for c in range(7) if self.cells[0][c] == " ")

    def winner(self):
        winners = set()
        for row in range(6):
            for col in range(7):
                cell = self.cells[row][col]
                if cell == " ":
                    continue
                for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                    rr, cc = row + 3 * dr, col + 3 * dc
                    if 0 <= rr < 6 and 0 <= cc < 7 and all(
                        self.cells[row + k * dr][col + k * dc] == cell
                        for k in (1, 2, 3)
                    ):
                        winners.add("XO".index(cell))
        if len(winners) > 1:
            raise ValueError("Two winners")
        return next(iter(winners), -1)

    def terminal(self):
        return self.winner() != -1 or not self.legal()

    def play(self, action):
        if self.terminal() or type(action) is not int or action not in self.legal():
            raise ValueError("Illegal or post-terminal action")
        rows = [list(row) for row in self.cells]
        row = max(r for r in range(6) if rows[r][action] == " ")
        rows[row][action] = "XO"[self.actor]
        return ReferenceBoard(tuple(tuple(row) for row in rows), 1 - self.actor)

    def classify(self):
        if self.terminal():
            raise ValueError("Terminal classification")
        wins, safe, unsafe, replies = [], [], [], {}
        for action in self.legal():
            after = self.play(action)
            if after.winner() == self.actor:
                wins.append(action)
            opponent_wins = [] if after.terminal() else [
                reply for reply in after.legal()
                if after.play(reply).winner() == after.actor
            ]
            (unsafe if opponent_wins else safe).append(action)
            if opponent_wins:
                replies[str(action)] = opponent_wins
        return {"legal_actions": list(self.legal()), "immediate_winning_actions": wins,
                "safe_actions": safe, "unsafe_actions": unsafe,
                "opponent_winning_replies": replies}


def reference_selection(board, visits, original):
    """Lowest physical column resolves restricted visit-count ties."""
    classification = board.classify()
    assert len(visits) == 7 and all(type(n) is int and n >= 0 for n in visits)
    assert original in classification["legal_actions"]
    wins, safe = classification["immediate_winning_actions"], classification["safe_actions"]
    if wins:
        chosen, reason = min(wins, key=lambda a: (-visits[a], a)), "IMMEDIATE_WIN"
    elif not safe:
        chosen, reason = original, "NO_SAFE_MOVE"
    elif original in safe:
        chosen, reason = original, "UNCHANGED"
    else:
        chosen, reason = min(safe, key=lambda a: (-visits[a], a)), "AVOID_IMMEDIATE_LOSS"
    return chosen, reason, classification


def cluster_bootstrap(cluster_rows, *, seed, replicates=10000):
    """Ratio-of-sums paired cluster bootstrap using a separate RNG stream.

    Input maps cluster identity to its paired row differences. Entire clusters
    retain all members; varying cluster sizes retain row weighting.
    """
    import numpy as np
    clusters = [cluster_rows[k] for k in sorted(cluster_rows)]
    assert clusters and all(len(rows) > 0 for rows in clusters)
    sums = np.array([sum(rows) for rows in clusters], dtype=float)
    counts = np.array([len(rows) for rows in clusters], dtype=int)
    generator = np.random.Generator(np.random.PCG64(seed))
    samples = generator.integers(0, len(clusters), size=(replicates, len(clusters)))
    distribution = sums[samples].sum(axis=1) / counts[samples].sum(axis=1)
    return {"difference": float(sums.sum() / counts.sum()),
            "ci95": np.quantile(distribution, [0.025, 0.975]).tolist(),
            "clusters": len(clusters), "rows": int(counts.sum()),
            "replicates": replicates, "seed": seed}


def self_check():
    board = ReferenceBoard.from_moves([0, 1, 0, 1, 0, 2])
    assert board.classify()["immediate_winning_actions"] == [0]
    assert reference_selection(board, [0, 256, 0, 0, 0, 0, 0], 1)[:2] == (0, "IMMEDIATE_WIN")
    clone = ReferenceBoard(tuple(tuple(row) for row in board.cells), board.actor)
    assert board == clone
    bootstrap = cluster_bootstrap({"a": [0.125, 0.125], "b": [0.125]}, seed=20261013)
    assert bootstrap["difference"] == 0.125 and bootstrap["ci95"] == [0.125, 0.125]
    return {"matrix_rules": "PASS", "bootstrap_constant_check": "PASS"}


def audit_decision(board, record, enabled):
    assert record["contract"] == "connect4-guarded-decision-v1"
    visits = record["visits"]
    original, final = record["original_action"], record["final_action"]
    legal = board.legal()
    assert len(visits) == 7 and all(type(n) is int and n >= 0 for n in visits)
    assert sum(visits) == 256 and all(visits[a] == 0 for a in range(7) if a not in legal)
    assert original in legal and visits[original] == max(visits[a] for a in legal)
    assert final in legal
    assert record["occupancy"] == sum(cell != " " for row in board.cells for cell in row)
    assert math.isfinite(record["root_value"]) and -1 <= record["root_value"] <= 1
    prior, logits = record["root_prior"], record["root_logits"]
    assert len(prior) == len(logits) == 7 and all(math.isfinite(v) for v in prior + logits)
    assert all(v >= 0 for v in prior) and abs(sum(prior) - 1) < 1e-6
    assert all(prior[a] == 0 for a in range(7) if a not in legal)
    assert record["search_seconds"] >= 0 and record["total_seconds"] >= record["search_seconds"]
    chosen, reason, classification = reference_selection(board, visits, original)
    if not enabled:
        assert record["guard"] is None and final == original
        return classification
    diagnostic = record["guard"]
    assert diagnostic["contract"] == "connect4-tactical-root-guard-v1"
    assert (final, diagnostic["reason"]) == (chosen, reason)
    fields = dict(classification)
    fields["immediate_wins"] = fields.pop("immediate_winning_actions")
    for key, value in fields.items():
        assert diagnostic[key] == value, (key, diagnostic[key], value)
    assert diagnostic["original_action"] == original and diagnostic["final_action"] == final
    assert diagnostic["original_visits"] == visits
    assert diagnostic["changed"] == (final != original)
    assert diagnostic["selected_visits"] == visits[final]
    assert diagnostic["selected_zero_visits"] == (visits[final] == 0)
    assert diagnostic["no_safe_action"] == (not classification["safe_actions"])
    assert diagnostic["candidate_count"] == len(legal)
    assert diagnostic["occupancy"] == record["occupancy"]
    replies_inspected = sum(len(board.play(a).legal()) for a in legal if not board.play(a).terminal())
    assert diagnostic["opponent_replies_inspected"] == replies_inspected
    assert 0 <= diagnostic["guard_seconds"] <= record["total_seconds"]
    assert 0 <= diagnostic["classification_seconds"] <= diagnostic["guard_seconds"]
    return classification


def audit_saved(root, bootstrap_seed=20261013):
    """Audit exact task membership, decisions and outcomes from frozen raw rows."""
    root = Path(root)
    repository = root.parents[1]
    results = root / "results"
    design = json.loads((results / "design.json").read_text())
    freeze = json.loads((results / "freeze.json").read_text())
    def file_hash(path):
        digest = hashlib.sha256()
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    for relative, digest in freeze["files"].items():
        assert file_hash(repository / relative) == digest, relative
    for checkpoint in freeze["checkpoints"]:
        assert file_hash(checkpoint["path"]) == checkpoint["sha256"], checkpoint["path"]
    assert design["seeds"] == [42, 314159] and design["simulations"] == 256
    assert design["matches"] == 320 and design["fixture_searches"] == 2512
    assert design["root_noise"] is False and design["training"] is False and design["selfplay"] is False
    fixtures = json.loads((repository / "research/alphazero_resnet_selfplay/fixtures.json").read_text())
    opening_package = json.loads((repository / "games/connect4/alphazero_v2/frozen/openings-development.json").read_text())
    assert opening_package["split"] == "development"
    openings = {row["id"]: row for row in opening_package["rows"] if row["id"].endswith("-base")}
    assert len(openings) == 40 and len({o["family"] for o in openings.values()}) == 40
    evidence = {"contract": "independent-tactical-guard-audit-v1", "same_host": True,
                "new_model_queries": 0, "reference_rules": self_check(),
                "positions": {}, "matches": {}, "match_differences": {}, "failures": []}
    evidence["source_files_rehashed"] = len(freeze["files"])
    evidence["checkpoints_rehashed"] = len(freeze["checkpoints"])
    decisions = changed = 0
    match_arrays = []
    pooled_position_clusters = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    guarded_timing = []
    for seed in (42, 314159):
        task = results / "tasks" / str(seed)
        identity = json.loads((task / "identity.json").read_text())
        assert identity["complete"] is True and identity["seed"] == seed
        assert identity["generation"] == 6 and identity["architecture"] == design["architecture"]
        assert identity["fixture_searches"] == 1256 and identity["matches"] == 160
        assert identity["before"] == identity["after"]
        assert identity["checkpoint"] in design["checkpoints"]
        for suite in ("tactical", "solved", "deep"):
            raw = json.loads((task / f"{suite}.json").read_text())
            assert raw["complete"] is True and raw["seed"] == seed
            expected = {r["id"]: r for r in fixtures[suite]}
            rows = raw["rows"]
            assert len(rows) == len(expected) and {r["id"] for r in rows} == set(expected)
            summaries = {variant: Counter() for variant in ("control", "guarded")}
            contrasts = defaultdict(lambda: defaultdict(list))
            changed_rows = []
            for row in rows:
                fixture = expected[row["id"]]
                assert row["cluster"] == fixture["cluster"] and row["category"] == fixture.get("category")
                board = ReferenceBoard.from_moves(fixture["moves"])
                computed = {}
                for variant in ("control", "guarded"):
                    record = row[variant]
                    cls = audit_decision(board, record, variant == "guarded")
                    decisions += 1
                    action = record["final_action"]
                    if suite == "tactical":
                        assert cls["legal_actions"] == sorted(fixture["legal"])
                        assert cls["immediate_winning_actions"] == sorted(fixture["immediate_wins"])
                        assert cls["safe_actions"] == sorted(fixture["safe_actions"])
                        assert board.actor == fixture["actor"]
                        metrics = {"correct": int(action == fixture["expected_action"]),
                                   "avoidable": int(bool(cls["safe_actions"]) and action not in cls["safe_actions"]),
                                   "missed_win": int(bool(cls["immediate_winning_actions"]) and action not in cls["immediate_winning_actions"])}
                    else:
                        values = {int(a): v for a, v in fixture["action_values"].items()}
                        assert sorted(values) == cls["legal_actions"] and max(values.values()) == fixture["value"]
                        assert sorted(a for a, v in values.items() if v == fixture["value"]) == sorted(fixture["optimal_actions"])
                        assert all(values[a] == 1 for a in cls["immediate_winning_actions"])
                        value, chosen = fixture["value"], values[action]
                        metrics = dict(correct=int(chosen == value), exact=value, chosen_value=chosen,
                                       mistake=int(chosen < value), win_draw=int(value == 1 and chosen == 0),
                                       win_loss=int(value == 1 and chosen == -1), draw_loss=int(value == 0 and chosen == -1))
                    for key, value in metrics.items():
                        assert record[key] == value, (seed, suite, row["id"], variant, key)
                    summaries[variant].update(metrics)
                    computed[variant] = metrics
                for field in ("original_action", "visits", "root_value", "root_prior", "root_logits"):
                    assert row["control"][field] == row["guarded"][field]
                diagnostic = row["guarded"]["guard"]
                guarded_timing.append(row["guarded"])
                if diagnostic["changed"]:
                    changed += 1
                    change = {"id": row["id"], "reason": diagnostic["reason"],
                              "original_action": row["control"]["final_action"],
                              "guarded_action": row["guarded"]["final_action"]}
                    if suite != "tactical":
                        change["exact_difference"] = computed["guarded"]["chosen_value"] - computed["control"]["chosen_value"]
                        assert change["exact_difference"] >= 0, change
                    changed_rows.append(change)
                for key in computed["control"]:
                    difference = computed["guarded"][key] - computed["control"][key]
                    contrasts[key][str(row["cluster"])].append(difference)
                    pooled_position_clusters[suite][key][str(row["cluster"])].append(difference)
            evidence["positions"][f"{seed}:{suite}"] = dict(rows=len(rows), counts=summaries,
                contrasts={key: cluster_bootstrap(value, seed=bootstrap_seed) for key, value in contrasts.items()},
                changed_rows=changed_rows)
        paired = {}
        for variant in ("control", "guarded"):
            raw = json.loads((task / f"{variant}-matches.json").read_text())
            assert raw["complete"] is True and raw["seed"] == seed and raw["variant"] == variant
            records = raw["records"]
            assert len(records) == 80
            seen, scores, by_color, by_family, reasons = set(), [], defaultdict(list), defaultdict(list), Counter()
            own_decisions = own_changes = 0
            for record in records:
                key = (record["opening_id"], record["game_index"])
                assert key not in seen and key[0] in openings and key[1] in (0, 1)
                seen.add(key)
                opening = openings[key[0]]
                assert record["agent_color"] == key[1] and record["agent"] == variant
                assert record["opponent"] == "negamax3" and not record["abandoned"]
                assert record["family"] == opening["family"] and record["stratum"] == opening["stratum"]
                prefix = record["prefix_length"]
                assert prefix == len(opening["moves"]) and record["moves"][:prefix] == opening["moves"]
                board = ReferenceBoard.from_moves(opening["moves"])
                assert len(record["decisions"]) == len(record["moves"]) - prefix
                for action, decision in zip(record["moves"][prefix:], record["decisions"]):
                    assert decision["color"] == board.actor and decision["move"] == action
                    if board.actor == record["agent_color"]:
                        audit_decision(board, decision, variant == "guarded")
                        assert decision["final_action"] == action
                        own_decisions += 1
                        decisions += 1
                        if variant == "guarded":
                            guarded_timing.append(decision)
                            reasons[decision["guard"]["reason"]] += 1
                            own_changes += decision["guard"]["changed"]
                            changed += decision["guard"]["changed"]
                    else:
                        scores_by_action = {int(a): v for a, v in decision["root_scores"].items()}
                        assert sorted(scores_by_action) == list(board.legal())
                        assert scores_by_action[action] == max(scores_by_action.values())
                    board = board.play(action)
                assert board.terminal() and board.winner() == record["winner"]
                result = "draw" if board.winner() == -1 else "win" if board.winner() == record["agent_color"] else "loss"
                assert record["result"] == result
                score = {"win": 1.0, "draw": 0.5, "loss": 0.0}[result]
                assert record["score"] == score
                scores.append(score)
                by_color[record["agent_color"]].append(score)
                by_family[record["family"]].append(score)
                paired.setdefault(key, {})[variant] = score
            assert seen == {(key, color) for key in openings for color in (0, 1)}
            evidence["matches"][f"{seed}:{variant}"] = dict(games=80,
                results=dict(Counter(r["result"] for r in records)), score=sum(scores) / 80,
                by_color={str(k): sum(v) / len(v) for k, v in by_color.items()},
                by_family={k: sum(v) / len(v) for k, v in by_family.items()},
                own_decisions=own_decisions, changed=own_changes, reasons=reasons)
        clusters = defaultdict(list)
        for key, pair in paired.items():
            assert set(pair) == {"control", "guarded"}
            clusters[openings[key[0]]["family"]].append(pair["guarded"] - pair["control"])
        evidence["match_differences"][str(seed)] = cluster_bootstrap(clusters, seed=bootstrap_seed)
        match_arrays.append(clusters)
    # Both fixed networks share the same opening families: preserve the full
    # family across seeds in the equal-weight pooled resampling.
    pooled = {family: match_arrays[0][family] + match_arrays[1][family] for family in match_arrays[0]}
    evidence["match_differences"]["pooled"] = cluster_bootstrap(pooled, seed=bootstrap_seed)
    evidence["pooled_positions"] = {
        suite: {field: cluster_bootstrap(clusters, seed=bootstrap_seed) for field, clusters in fields.items()}
        for suite, fields in pooled_position_clusters.items()}
    import numpy as np
    def timing_summary(values):
        values = sorted(values)
        return dict(n=len(values), mean=sum(values)/len(values), median=float(np.quantile(values, .5)),
                    p95=float(np.quantile(values, .95)), worst=values[-1])
    evidence["latency"] = dict(
        search_seconds=timing_summary([r["search_seconds"] for r in guarded_timing]),
        classification_seconds=timing_summary([r["guard"]["classification_seconds"] for r in guarded_timing]),
        guard_seconds=timing_summary([r["guard"]["guard_seconds"] for r in guarded_timing]),
        total_seconds=timing_summary([r["total_seconds"] for r in guarded_timing]),
        relative_overhead=timing_summary([r["guard"]["guard_seconds"]/r["search_seconds"] for r in guarded_timing]))
    match = evidence["match_differences"]["pooled"]
    differences = [evidence["match_differences"][str(seed)]["difference"] for seed in (42, 314159)]
    evidence["strength_verdict"] = "PASS" if match["difference"] >= .05 and min(differences) >= 0 and match["ci95"][0] > 0 else (
        "FAIL" if match["difference"] <= -.05 and match["ci95"][1] < 0 else "INCONCLUSIVE")
    evidence["deep_verdict"] = "PASS_NO_OBSERVED_WDL_REGRESSION"
    evidence["tactical_verdict"] = "PASS" if all(evidence["positions"][f"{seed}:tactical"]["counts"]["guarded"]["correct"] == 200 for seed in (42,314159)) else "FAIL"
    latency = evidence["latency"]
    evidence["latency_verdict"] = "PASS" if latency["guard_seconds"]["p95"] <= .001 and latency["guard_seconds"]["worst"] <= .005 and latency["relative_overhead"]["mean"] <= .05 else "FAIL_BUDGET_EXCEEDED"
    evidence.update(decisions_audited=decisions, changed_actions_audited=changed, status="PASS")
    return evidence


def compare_analysis(evidence, analysis):
    """Compare separately recomputed counters, headline CIs and all decisions."""
    comparisons = 0
    def equal(a, b):
        nonlocal comparisons
        if isinstance(a, list):
            assert len(a) == len(b)
            for x, y in zip(a, b): equal(x, y)
        elif isinstance(a, (int, float)):
            assert abs(a-b) < 1e-12, (a, b)
            comparisons += 1
        else:
            assert a == b, (a, b)
            comparisons += 1
    for seed in (42,314159):
        endpoint = analysis["seeds"][str(seed)]
        for suite in ("tactical", "solved", "deep"):
            own = evidence["positions"][f"{seed}:{suite}"]
            for variant in ("control", "guarded"):
                for metric, item in endpoint[suite]["summary"][variant].items():
                    equal(own["counts"][variant][metric], item["count"])
                    equal(own["counts"][variant][metric]/own["rows"], item["mean"])
            for metric, item in endpoint[suite]["contrasts"].items():
                equal(own["contrasts"][metric]["difference"], item["difference"])
                equal(own["contrasts"][metric]["ci95"], item["ci95"])
        for variant in ("control","guarded"):
            own = evidence["matches"][f"{seed}:{variant}"]
            summary = endpoint["matches"]["summary"][variant]
            equal(own["score"], summary["score"])
            equal(own["results"], summary["wdl"])
            equal(own["by_color"], summary["colors"])
            equal(own["by_family"], summary["families"])
            equal(own["own_decisions"], summary["decisions"])
        for key in ("difference", "ci95"):
            equal(evidence["match_differences"][str(seed)][key], endpoint["matches"]["contrasts"]["score"][key])
    for suite, fields in analysis["pooled"].items():
        for metric, item in fields.items():
            own = evidence["match_differences"]["pooled"] if suite == "matches" else evidence["pooled_positions"][suite][metric]
            for key in ("difference", "ci95"): equal(own[key], item[key])
    for metric, fields in evidence["latency"].items():
        for key, value in fields.items(): equal(value, analysis["latency"][metric][key])
    for key in ("strength_verdict", "tactical_verdict", "deep_verdict", "latency_verdict"):
        equal(evidence[key], analysis[key])
    return dict(status="PASS", comparisons=comparisons)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    evidence = audit_saved(root) if args.results else self_check()
    if args.results:
        frozen = audit_saved(root, bootstrap_seed=4410903)
        frozen["analysis_comparison"] = compare_analysis(frozen, json.loads((root/"results/analysis.json").read_text()))
        evidence["frozen_stream_reproduction"] = frozen
    text = json.dumps(evidence, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(text + "\n")
    else:
        print(text)
