"""Per-generation search, data, training and resource diagnostics (review §11 logging list).

Two kinds of output are kept apart. *Deterministic* diagnostics depend only on
the trajectory (counts, entropies, depths, contradictions, sample ages) and are
stored in the runner history, so they take part in exact-resume identity.
*Resource* measurements (wall time, memory) vary between runs and are returned
beside the summary for campaign logs only; they never enter resume state.
Nothing here consumes an RNG stream.
"""
from collections import Counter, defaultdict
import math
import resource
import sys

import numpy as np

from .oracle import safe_moves_scan, winning_moves_scan


def entropy(probabilities):
    return -sum(p * math.log(p) for p in probabilities if p > 0)


def illegal_raw_mass(logits, legal):
    raw = np.asarray(logits, dtype=np.float64)
    weights = np.exp(raw - raw.max())
    weights /= weights.sum()
    return float(sum(weights[a] for a in range(7) if a not in legal))


def root_search_value(result):
    """Visit-weighted root value for the player to move (children store the opponent's Q)."""
    total = sum(child.visits for child in result.root.children.values())
    return -sum(child.value_sum for child in result.root.children.values()) / total


def quantiles(values, points=(0.5, 0.95, 1.0)):
    if not values:
        return {}
    ordered = sorted(values)
    # Nearest-rank: the smallest value with at least q of the sample at or below it (q=0 -> minimum).
    return {f"p{int(q * 100)}": ordered[min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))]
            for q in points}


def peak_rss_mib():
    """Process peak resident set size (macOS reports bytes, Linux KiB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2 if sys.platform == "darwin" else 1024)


class CollectionObserver:
    """Accumulates per-decision search statistics during collection (no RNG use)."""

    def __init__(self):
        self.decisions = 0
        self.seconds = []
        self.prior_entropy = self.target_entropy = self.illegal_mass = 0.0
        self.raw_value = self.search_value = 0.0
        self.visited_actions = 0
        self.max_depth = 0
        self.leaf_depth = 0
        self.simulations = 0
        self.terminal_leaves = 0
        self.legal_actions = 0

    def __call__(self, index, decision, seconds):
        search = decision.search
        legal = list(search.root.children)
        self.decisions += 1
        self.seconds.append(seconds)
        self.prior_entropy += entropy(search.root_prior)
        self.target_entropy += entropy(search.visit_target)
        self.illegal_mass += illegal_raw_mass(search.root_logits, legal)
        self.raw_value += search.root_value
        self.search_value += root_search_value(search)
        self.visited_actions += sum(v > 0 for v in search.visits)
        self.legal_actions += len(search.root.children)
        self.max_depth = max(self.max_depth, search.max_depth)
        self.leaf_depth += search.leaf_depth_sum
        self.simulations += search.simulations
        self.terminal_leaves += search.terminal_leaves

    def deterministic(self):
        n = self.decisions
        return dict(decisions=n, mean_root_prior_entropy=self.prior_entropy / n,
                    mean_visit_target_entropy=self.target_entropy / n,
                    mean_illegal_raw_policy_mass=self.illegal_mass / n,
                    mean_raw_root_value=self.raw_value / n, mean_root_search_value=self.search_value / n,
                    root_visit_coverage=self.visited_actions / self.legal_actions,
                    max_search_depth=self.max_depth, mean_leaf_depth=self.leaf_depth / self.simulations,
                    terminal_leaf_fraction=self.terminal_leaves / self.simulations)

    def resources(self):
        return dict(search_seconds_total=sum(self.seconds), search_seconds_mean=sum(self.seconds) / len(self.seconds),
                    search_seconds=quantiles(self.seconds))


def family(example):
    mirrored = tuple(row[::-1] for row in example.observation)
    return (example.actor, min(example.observation, mirrored))


def game_diagnostics(games):
    """Lengths, outcomes, unique positions and tactical-contradiction rates for new games."""
    lengths = [len(g.moves) for g in games]
    contradictions = defaultdict(lambda: [0, 0])
    for game in games:
        for example in game.examples:
            prefix = list(game.moves[:example.ply])
            if winning_moves_scan(prefix):
                proven = 1
            elif not safe_moves_scan(prefix):
                proven = -1
            else:
                continue
            key = f"{'xo'[example.actor]}/{'win' if proven == 1 else 'loss'}/ply{min(example.ply // 6 * 6, 36):02d}+"
            contradictions[key][1] += 1
            contradictions[key][0] += example.outcome != proven
    examples = [e for g in games for e in g.examples]
    proven_total = sum(v[1] for v in contradictions.values())
    return dict(
        game_length=dict(mean=sum(lengths) / len(lengths), min=min(lengths), max=max(lengths)),
        draw_games=sum(g.winner == -1 for g in games),
        unique_positions=len({(e.actor, e.observation) for e in examples}),
        unique_reflection_families=len({family(e) for e in examples}),
        proven_positions=proven_total,
        contradictions=sum(v[0] for v in contradictions.values()),
        contradiction_rate_proven=(sum(v[0] for v in contradictions.values()) / proven_total) if proven_total else None,
        contradiction_rate_all=sum(v[0] for v in contradictions.values()) / len(examples),
        contradictions_by_actor_value_ply={k: dict(contradictions=v[0], proven=v[1])
                                           for k, v in sorted(contradictions.items())})


def replay_diagnostics(replay):
    examples = [e for g in replay.iter_games() for e in g.examples]
    return dict(unique_positions=len({(e.actor, e.observation) for e in examples}),
                unique_reflection_families=len({family(e) for e in examples}))


def sampling_diagnostics(addresses, generation):
    """Sample appearances in this generation's updates, by source generation and age."""
    by_source = Counter(source for source, _, _ in addresses)
    return dict(samples_by_source_generation={str(g): n for g, n in sorted(by_source.items())},
                samples_by_age={str(generation - g): n for g, n in sorted(by_source.items())},
                unique_sampled_positions=len(set(addresses)))


def gradient_diagnostics(metrics):
    norms = [m["gradient_norm"] for m in metrics]
    return dict(gradient_norm=quantiles(norms), finite_updates=len(metrics),
                clipped_fraction=sum(m["clipped"] for m in metrics) / len(metrics))
