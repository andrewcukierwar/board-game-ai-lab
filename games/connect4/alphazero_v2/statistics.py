"""Torch-free statistical summaries, gates and frozen acceptance thresholds (review §9).

Units of resampling follow the review: arena games are clustered by opening
family (a prefix and its mirror form one family; every empty-board pair is one
shared family), positions by base situation (a row and its mirror), and
behavioral-calibration positions by game. Repeated search seeds are averaged
within a row before any aggregation. All bootstraps are percentile intervals
with a fixed analysis seed.
"""
from collections import defaultdict
import math
import random

BOOTSTRAP_RESAMPLES = 10_000
ANALYSIS_SEED = 4_303_009
RESULT_POINTS = {"win": 1.0, "draw": 0.5, "loss": 0.0}

# Frozen before training (review §11 "Acceptance criteria"); thresholds are project gates.
ACCEPTANCE_THRESHOLDS = dict(
    tactical=dict(immediate_win=0.99, safe_response=0.95, per_actor_immediate_win=0.97, per_actor_safe_response=0.90),
    solved=dict(optimal_preserving=0.80, avoidable_loss_max=0.05),
    value=dict(class_balanced_mse_max=0.60, decisive_sign_min=0.85, draw_mae_max=0.35,
               wrong_sign_saturated_max=0.05, saturation=0.95),
    calibration=dict(min_games_per_bin=30, max_bin_gap=0.15, bins=5),
    arena={
        "random": dict(score=0.95, lower_bound=0.90, lower_bound_strict=False),
        "negamax1": dict(score=0.80, lower_bound=0.70, lower_bound_strict=True),
        "negamax2": dict(score=0.65, lower_bound=0.55, lower_bound_strict=True, min_color_score=0.50),
        "initial_v2_512": dict(score=0.60, lower_bound=0.50, lower_bound_strict=True),
        "phase4d2f_512": dict(score=0.60, lower_bound=0.50, lower_bound_strict=True),
        "nn_only_vs_random": dict(score=0.85, lower_bound=0.75, lower_bound_strict=True),
    },
    reported_only=("negamax4", "guarded_uct_800"),
    readiness=dict(p95_ms=500.0, min_positions=500),
)
CHAMPION_GATE = dict(score=0.55, lower_bound=0.50, max_tactical_regression=0.02, schedule=(5, 10, 15, 20))


def percentile(sorted_values, q):
    """Linear-interpolated percentile of an already sorted list (q in [0,1])."""
    if not sorted_values:
        raise ValueError("percentile of an empty sample")
    position = q * (len(sorted_values) - 1)
    low, high = math.floor(position), math.ceil(position)
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (position - low)


def cluster_bootstrap(clusters, statistic, *, resamples=BOOTSTRAP_RESAMPLES, seed=ANALYSIS_SEED):
    """Percentile 95% interval of ``statistic(list of cluster records)`` resampling clusters."""
    clusters = list(clusters)
    if not clusters:
        raise ValueError("No clusters to resample")
    rng = random.Random(seed)
    n = len(clusters)
    values = sorted(statistic([clusters[rng.randrange(n)] for _ in range(n)]) for _ in range(resamples))
    return percentile(values, 0.025), percentile(values, 0.975)


# Arena ------------------------------------------------------------------------------------

def _score(totals):
    points = sum(p for p, _ in totals)
    games = sum(g for _, g in totals)
    return points / games


def arena_summary(games, *, resamples=BOOTSTRAP_RESAMPLES, seed=ANALYSIS_SEED, planned_games=None):
    """Score, W/D/L, colors, strata and an opening-family bootstrap for the agent under test.

    Each game record needs ``family``, ``stratum``, ``agent_color`` (0=X,1=O) and
    ``result`` ('win'/'draw'/'loss' for the agent under test).
    """
    if not games:
        raise ValueError("No completed games")
    by_family = defaultdict(lambda: [0.0, 0])
    counts = defaultdict(int)
    colors = defaultdict(lambda: [0.0, 0])
    strata = defaultdict(lambda: [0.0, 0])
    for game in games:
        points = RESULT_POINTS[game["result"]]
        counts[game["result"]] += 1
        for table, key in ((by_family, game["family"]), (colors, "XO"[game["agent_color"]]),
                           (strata, game["stratum"])):
            table[key][0] += points
            table[key][1] += 1
    clusters = [tuple(v) for v in by_family.values()]
    low, high = cluster_bootstrap(clusters, _score, resamples=resamples, seed=seed)
    complete = planned_games is None or len(games) == planned_games
    return dict(games=len(games), planned_games=planned_games, complete=complete,
                wins=counts["win"], draws=counts["draw"], losses=counts["loss"],
                score=_score(clusters), interval95=[low, high], clusters=len(clusters),
                by_color={k: dict(games=v[1], score=v[0] / v[1]) for k, v in sorted(colors.items())},
                by_stratum={k: dict(games=v[1], score=v[0] / v[1]) for k, v in sorted(strata.items())},
                resamples=resamples, analysis_seed=seed,
                clustering="opening family (prefix+mirror); all empty-board pairs form one family")


def paired_difference(games_a, games_b, *, resamples=BOOTSTRAP_RESAMPLES, seed=ANALYSIS_SEED):
    """Score(A) - score(B) on the same opening families, resampling shared families."""
    def by_family(games):
        table = defaultdict(lambda: [0.0, 0])
        for game in games:
            table[game["family"]][0] += RESULT_POINTS[game["result"]]
            table[game["family"]][1] += 1
        return table
    a, b = by_family(games_a), by_family(games_b)
    if set(a) != set(b):
        raise ValueError("Paired comparison needs identical opening families")
    clusters = [(tuple(a[f]), tuple(b[f])) for f in sorted(a)]

    def difference(sample):
        return _score([x for x, _ in sample]) - _score([y for _, y in sample])
    low, high = cluster_bootstrap(clusters, difference, resamples=resamples, seed=seed)
    return dict(difference=difference(clusters), interval95=[low, high], clusters=len(clusters))


def arena_gate(summary, rule):
    """Apply one frozen arena rule; incomplete arenas never pass."""
    lower = summary["interval95"][0]
    lower_ok = lower > rule["lower_bound"] if rule.get("lower_bound_strict", True) else lower >= rule["lower_bound"]
    color_ok = all(c["score"] >= rule["min_color_score"] for c in summary["by_color"].values()) \
        if "min_color_score" in rule else True
    passed = summary["complete"] and summary["score"] >= rule["score"] and lower_ok and color_ok
    return dict(passed=passed, complete=summary["complete"], score=summary["score"], lower=lower,
                color_ok=color_ok, rule=rule)


# Positions --------------------------------------------------------------------------------

def _family_means(rows, value_of):
    """Average value_of(row) within each family, then return {family: mean}."""
    table = defaultdict(list)
    for row in rows:
        table[row["family"]].append(value_of(row))
    return {family: sum(v) / len(v) for family, v in table.items()}


def _mean(values):
    values = list(values)
    return sum(values) / len(values) if values else None


def tactical_metrics(rows, choices, root_visits=None):
    """Tactical accuracy averaged over seeds within row, rows within family, then families.

    ``choices[row_id]`` is a list of chosen actions (one per declared seed);
    ``root_visits[row_id]`` optional list of 7-visit tuples (one per seed).
    """
    result = {}
    for category, name in (("unique_win", "immediate_win"), ("unique_safe", "safe_response")):
        selected = [r for r in rows if r["category"] == category]

        def correct(row):
            picks = choices[row["id"]]
            return sum(p == row["expected_action"] for p in picks) / len(picks)
        result[name] = _mean(_family_means(selected, correct).values())
        for actor in (0, 1):
            result[f"{name}_{'xo'[actor]}"] = _mean(_family_means(
                [r for r in selected if r["actor"] == actor], correct).values())
        result[f"{name}_families"] = len({r["family"] for r in selected})
    if root_visits is not None:
        wins = [r for r in rows if r["category"] == "unique_win"]
        result["zero_visit_winning_actions"] = sum(
            any(v[r["expected_action"]] == 0 for v in root_visits[r["id"]]) for r in wins)
    return result


def tactical_gate(metrics, thresholds=ACCEPTANCE_THRESHOLDS["tactical"]):
    checks = dict(immediate_win=metrics["immediate_win"] >= thresholds["immediate_win"],
                  safe_response=metrics["safe_response"] >= thresholds["safe_response"])
    for actor in "xo":
        checks[f"immediate_win_{actor}"] = metrics[f"immediate_win_{actor}"] >= thresholds["per_actor_immediate_win"]
        checks[f"safe_response_{actor}"] = metrics[f"safe_response_{actor}"] >= thresholds["per_actor_safe_response"]
    return dict(passed=all(checks.values()), checks=checks)


def solved_decision_metrics(rows, choices):
    """Optimal-outcome preservation and avoidable losses (seed-averaged, family-averaged)."""
    def preserving(row):
        picks = choices[row["id"]]
        return sum(row["action_values"][str(p)] == row["value"] for p in picks) / len(picks)

    def avoidable_loss(row):
        picks = choices[row["id"]]
        return sum(row["action_values"][str(p)] == -1 for p in picks) / len(picks)
    result = dict(optimal_preserving=_mean(_family_means(rows, preserving).values()))
    for value, name in ((1, "win"), (0, "draw"), (-1, "loss")):
        subset = [r for r in rows if r["value"] == value]
        result[f"optimal_preserving_{name}"] = _mean(_family_means(subset, preserving).values())
    recoverable = [r for r in rows if r["value"] >= 0]
    result["avoidable_loss"] = _mean(_family_means(recoverable, avoidable_loss).values())
    return result


def value_metrics(rows, values, saturation=ACCEPTANCE_THRESHOLDS["value"]["saturation"]):
    """Raw scalar value head against exact outcomes (family-averaged per class)."""
    def sq(row):
        return (values[row["id"]] - row["value"]) ** 2
    by_class = {}
    for value, name in ((1, "win"), (0, "draw"), (-1, "loss")):
        subset = [r for r in rows if r["value"] == value]
        if not subset:
            continue
        entry = dict(families=len({r["family"] for r in subset}),
                     mse=_mean(_family_means(subset, sq).values()),
                     mae=_mean(_family_means(subset, lambda r: abs(values[r["id"]] - r["value"])).values()))
        if value:
            entry["correct_sign"] = _mean(_family_means(
                subset, lambda r: float(math.copysign(1, values[r["id"]]) == value and values[r["id"]] != 0)).values())
        else:
            entry["mean_abs"] = _mean(_family_means(subset, lambda r: abs(values[r["id"]])).values())
        by_class[name] = entry
    decisive = [r for r in rows if r["value"] != 0]
    wrong_saturated = sum(1 for r in decisive if values[r["id"]] * r["value"] < 0 and abs(values[r["id"]]) >= saturation)
    result = dict(by_class=by_class,
                  class_balanced_mse=_mean(entry["mse"] for entry in by_class.values()),
                  wrong_sign_saturated_fraction=wrong_saturated / len(decisive) if decisive else None,
                  by_actor={"xo"[a]: _mean(_family_means([r for r in rows if r["actor"] == a], sq).values())
                            for a in (0, 1)})
    return result


def value_gate(metrics, thresholds=ACCEPTANCE_THRESHOLDS["value"]):
    checks = dict(class_balanced_mse=metrics["class_balanced_mse"] < thresholds["class_balanced_mse_max"],
                  win_sign=metrics["by_class"]["win"]["correct_sign"] >= thresholds["decisive_sign_min"],
                  loss_sign=metrics["by_class"]["loss"]["correct_sign"] >= thresholds["decisive_sign_min"],
                  draw_mae=metrics["by_class"]["draw"]["mae"] <= thresholds["draw_mae_max"],
                  wrong_sign_saturated=metrics["wrong_sign_saturated_fraction"] <= thresholds["wrong_sign_saturated_max"])
    return dict(passed=all(checks.values()), checks=checks)


# Behavioral calibration ---------------------------------------------------------------------

def calibration_summary(records, *, constant=None, bins=5, resamples=BOOTSTRAP_RESAMPLES, seed=ANALYSIS_SEED):
    """Held-out pre-search value v vs realized z, clustered by game.

    ``records``: dicts with ``game``, ``value`` (raw v in [-1,1]) and ``outcome`` (z).
    ``constant`` is a development-estimated constant predictor (reported, not tuned here).
    """
    by_game = defaultdict(list)
    for record in records:
        by_game[record["game"]].append((record["value"], record["outcome"]))
    games = list(by_game.values())

    def mse(sample, predictor=None):
        errors = [((v if predictor is None else predictor) - z) ** 2 for game in sample for v, z in game]
        return sum(errors) / len(errors)

    def improvement(sample):
        return mse(sample, 0.0) - mse(sample)
    edges = [-1 + 2 * i / bins for i in range(bins + 1)]
    table = []
    for i in range(bins):
        low, high = edges[i], edges[i + 1]
        inside = [(g, v, z) for g, game in enumerate(games) for v, z in game
                  if low <= v < high or (i == bins - 1 and v == high)]
        contributing = len({g for g, _, _ in inside})
        entry = dict(low=low, high=high, positions=len(inside), games=contributing)
        if inside:
            entry["mean_predicted_score"] = sum((v + 1) / 2 for _, v, _ in inside) / len(inside)
            entry["mean_realized_score"] = sum((z + 1) / 2 for _, _, z in inside) / len(inside)
            entry["gap"] = abs(entry["mean_predicted_score"] - entry["mean_realized_score"])
        table.append(entry)
    low, high = cluster_bootstrap(games, improvement, resamples=resamples, seed=seed)
    return dict(games=len(games), positions=sum(len(g) for g in games), mse=mse(games), mse_zero=mse(games, 0.0),
                mse_constant=None if constant is None else mse(games, constant), constant=constant,
                improvement_over_zero=improvement(games), improvement_interval95=[low, high], bins=table)


def calibration_gate(summary, thresholds=ACCEPTANCE_THRESHOLDS["calibration"]):
    established = [b for b in summary["bins"] if b["games"] >= thresholds["min_games_per_bin"]]
    checks = dict(improves_over_zero=summary["improvement_interval95"][0] > 0,
                  bins=all(b["gap"] <= thresholds["max_bin_gap"] for b in established),
                  any_established_bin=bool(established))
    return dict(passed=all(checks.values()), checks=checks,
                unestablished_bins=[i for i, b in enumerate(summary["bins"]) if b["games"] < thresholds["min_games_per_bin"]])


# Champion selection (development evidence only) ------------------------------------------------

def champion_decision(arena, candidate_tactics, champion_tactics, correctness_passed, gate=CHAMPION_GATE):
    """Promote only on a complete arena with score >= .55, lower bound > .50, passing
    correctness and <= 2 points tactical regression on the fixed development tactics."""
    regressions = {name: champion_tactics[name] - candidate_tactics[name] for name in ("immediate_win", "safe_response")}
    checks = dict(complete=arena["complete"], score=arena["score"] >= gate["score"],
                  lower_bound=arena["interval95"][0] > gate["lower_bound"], correctness=bool(correctness_passed),
                  tactical=all(r <= gate["max_tactical_regression"] + 1e-12 for r in regressions.values()))
    return dict(promote=all(checks.values()), checks=checks, tactical_regression=regressions,
                rule=dict(gate, schedule=list(gate["schedule"])))
