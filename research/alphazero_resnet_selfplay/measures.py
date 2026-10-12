"""Training-dynamics measures computed read-only from published generation boundaries.

Nothing here trains or mutates an evaluated model: train-mode batch-norm
probes run on disposable deep copies. Losses use the training definitions
(seven raw logits against normalized visits; squared error against the game
outcome), so "training" and "held-out" numbers are the same quantity on
different positions.

Held-out here is prequential: the network published at generation g is scored
on the games of generation g+1, which that same network generated and has not
been trained on. It measures generalization to the network's own next data,
not to an independent position distribution.
"""
from collections import Counter
from copy import deepcopy
import math
import random

import torch
from torch.nn import functional as F

EVAL_BATCH = 512
PROBE_BATCH = 128      # the training minibatch size, so train-mode batch statistics are realistic
PROBE_POSITIONS = 2048
BINS = 10


def tensors(examples):
    from games.connect4.alphazero_v2.training import batch_tensors
    parts = [batch_tensors(examples[i:i + 4096]) for i in range(0, len(examples), 4096)]
    return tuple(torch.cat([p[k] for p in parts]) for k in range(3))


def forward(model, states):
    if model.training:
        raise ValueError('Evaluation requires an eval-mode model')
    logits, values = [], []
    with torch.inference_mode():
        for i in range(0, states.shape[0], EVAL_BATCH):
            p, v = model(states[i:i + EVAL_BATCH])
            logits.append(p)
            values.append(v)
    return torch.cat(logits).double(), torch.cat(values).double().squeeze(1)


def scores(logits, values, states, policies, outcomes):
    policies, outcomes = policies.double(), outcomes.double().squeeze(1)
    ce = -(policies * F.log_softmax(logits, dim=1)).sum(1)
    entropy = -(policies * torch.log(policies.clamp_min(1e-300))).sum(1)
    legal = states[:, 0, 0, :] == 0
    masked = torch.softmax(logits.masked_fill(~legal, -math.inf), dim=1)
    return dict(ce=ce, kl=ce - entropy, se=(values - outcomes) ** 2,
                top1=(masked.argmax(1) == policies.argmax(1)).double(), value=values, outcome=outcomes)


def value_reliability(value, target):
    """Equal-width predicted-value bins on [-1,1]; error is the count-weighted |mean target - mean prediction|."""
    rows, error, n = [], 0.0, value.numel()
    index = torch.clamp(((value + 1) / 2 * BINS).long(), max=BINS - 1)
    for b in range(BINS):
        mask = index == b
        count = int(mask.sum())
        if count:
            predicted, observed = float(value[mask].mean()), float(target[mask].mean())
            error += count / n * abs(observed - predicted)
            rows.append(dict(bin=b, count=count, predicted=predicted, observed=observed))
    return dict(error=error, bins=rows)


def summary(s, mask=None):
    pick = (lambda t: t) if mask is None else (lambda t: t[mask])
    n = int(pick(s['ce']).numel())
    if not n:
        return dict(positions=0)
    value, outcome = pick(s['value']), pick(s['outcome'])
    return dict(positions=n, policy_ce=float(pick(s['ce']).mean()), policy_kl=float(pick(s['kl']).mean()),
                value_mse=float(pick(s['se']).mean()), top1=float(pick(s['top1']).mean()),
                constant_zero_value_mse=float((outcome ** 2).mean()), mean_abs_value=float(value.abs().mean()),
                saturated_share=float((value.abs() > 0.95).double().mean()),
                value_calibration=value_reliability(value, outcome))


def score_examples(model, examples):
    """(per-position scores, input tensors, eval-mode logits, eval-mode values) for one model on examples."""
    data = tensors(examples)
    logits, values = forward(model, data[0])
    return scores(logits, values, *data), data, logits, values


def batch_norm_probe(model, data, eval_logits, eval_values):
    """Running statistics and train- versus eval-mode predictions on the same positions (None without batch norm)."""
    norms = [(n, m) for n, m in model.named_modules() if isinstance(m, torch.nn.BatchNorm2d)]
    states, policies, outcomes = data
    usable = states.shape[0] // PROBE_BATCH * PROBE_BATCH
    if not norms or not usable:
        return None
    copy = deepcopy(model).train()
    logits, values = [], []
    with torch.no_grad():
        for i in range(0, usable, PROBE_BATCH):
            p, v = copy(states[i:i + PROBE_BATCH])
            logits.append(p)
            values.append(v)
    logits, values = torch.cat(logits).double(), torch.cat(values).double().squeeze(1)
    train = scores(logits, values, states[:usable], policies[:usable], outcomes[:usable])
    evaluated = scores(eval_logits[:usable], eval_values[:usable], states[:usable], policies[:usable], outcomes[:usable])
    log_eval, log_train = F.log_softmax(eval_logits[:usable], dim=1), F.log_softmax(logits, dim=1)
    kl = (log_eval.exp() * (log_eval - log_train)).sum(1)
    reference = deepcopy(model).train()
    for module in reference.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.reset_running_stats()
            module.momentum = None
    with torch.no_grad():
        for i in range(0, usable, PROBE_BATCH):
            reference(states[i:i + PROBE_BATCH])
    probe_modules = dict(reference.named_modules())
    layers, shifts, ratios = {}, [], []
    for name, module in norms:
        probe_mean, probe_var = probe_modules[name].running_mean.double(), probe_modules[name].running_var.double()
        run_mean, run_var = module.running_mean.double(), module.running_var.double()
        shift = float(((run_mean - probe_mean).abs() / torch.sqrt(probe_var + module.eps)).mean())
        ratio = float(torch.log((run_var + module.eps) / (probe_var + module.eps)).abs().mean())
        shifts.append(shift)
        ratios.append(ratio)
        layers[name] = dict(running_mean_abs=float(run_mean.abs().mean()), running_var_mean=float(run_var.mean()),
                            running_var_min=float(run_var.min()), running_var_max=float(run_var.max()),
                            standardized_mean_shift=shift, abs_log_variance_ratio=ratio,
                            scale_mean=float(module.weight.detach().abs().mean()), updates=int(module.num_batches_tracked))
    return dict(positions=usable,
                eval_mode=dict(policy_ce=float(evaluated['ce'].mean()), value_mse=float(evaluated['se'].mean())),
                train_mode=dict(policy_ce=float(train['ce'].mean()), value_mse=float(train['se'].mean())),
                mode_gap=dict(policy_ce=float(evaluated['ce'].mean() - train['ce'].mean()),
                              value_mse=float(evaluated['se'].mean() - train['se'].mean())),
                mean_abs_value_difference=float((eval_values[:usable] - values).abs().mean()),
                max_abs_value_difference=float((eval_values[:usable] - values).abs().max()),
                mean_policy_kl_eval_to_train=float(kl.mean()),
                choice_disagreement=float((eval_logits[:usable].argmax(1) != logits.argmax(1)).double().mean()),
                worst_standardized_mean_shift=max(shifts), worst_abs_log_variance_ratio=max(ratios),
                smallest_running_variance=min(v['running_var_min'] for v in layers.values()),
                counters=sorted({v['updates'] for v in layers.values()}), layers=layers)


def family(example):
    mirrored = tuple(row[::-1] for row in example.observation)
    return (example.actor, min(example.observation, mirrored))


def entropy(probabilities):
    return -sum(p * math.log(p) for p in probabilities if p > 0)


def diversity(games):
    """Self-play diversity, outcomes and one-ply errors for one generation's games (Phase 4D.4G definitions)."""
    from games.connect4.alphazero_v2.oracle import engine_position, safe_moves_scan, winning_moves_scan
    examples = [e for g in games for e in g.examples]
    post = [e for e in examples if e.ply >= 8]
    openings = Counter(min(tuple(g.moves[:8]), tuple(6 - a for a in g.moves[:8])) for g in games)
    mistakes = missed = blocked = 0
    for g in games:
        for e in g.examples:
            prefix = list(g.moves[:e.ply])
            wins, safe = winning_moves_scan(prefix), safe_moves_scan(prefix)
            missed += bool(wins) and e.action not in wins
            blocked += (not wins and bool(safe) and len(safe) < len(engine_position(prefix).get_valid_moves())
                        and e.action not in safe)
            mistakes += bool(safe) and e.action not in safe
    n = len(examples)
    lengths = [len(g.moves) for g in games]
    return dict(games=len(games), positions=n, mean_length=sum(lengths) / len(games), min_length=min(lengths),
                max_length=max(lengths),
                duplicate_fraction=(n - len({(e.actor, e.observation) for e in examples})) / n,
                mirror_duplicate_fraction=(n - len({family(e) for e in examples})) / n,
                post_ply8_positions=len(post),
                post_ply8_duplicate_fraction=((len(post) - len({(e.actor, e.observation) for e in post})) / len(post)
                                              if post else None),
                opening_families=len(openings), opening_concentration=sum((c / len(games)) ** 2 for c in openings.values()),
                distinct_games=len({tuple(g.moves) for g in games}),
                target_entropy=sum(entropy(e.policy_target) for e in examples) / n,
                first_player_wins=sum(g.winner == 0 for g in games), second_player_wins=sum(g.winner == 1 for g in games),
                draws=sum(g.winner == -1 for g in games),
                outcome_labels={str(z): sum(e.outcome == z for e in examples) for z in (-1, 0, 1)},
                avoidable_one_reply_errors=mistakes, missed_immediate_wins=missed, missed_safe_responses=blocked,
                avoidable_per_position=mistakes / n)


def replay_composition(runner):
    examples = [e for g in runner.replay.iter_games() for e in g.examples]
    if not examples:
        return dict(positions=0, games=0, generations=[])
    n = len(examples)
    return dict(positions=n, games=runner.replay.games, generations=list(runner.replay.generations),
                by_generation={str(g): c for g, c in runner.replay.counts().items()},
                unique_positions=len({(e.actor, e.observation) for e in examples}),
                unique_reflection_families=len({family(e) for e in examples}),
                mirror_duplicate_fraction=(n - len({family(e) for e in examples})) / n,
                outcome_labels={str(z): sum(e.outcome == z for e in examples) for z in (-1, 0, 1)},
                mean_ply=sum(e.ply for e in examples) / n,
                plies_before_8_share=sum(e.ply < 8 for e in examples) / n)


def probe_sample(runner, generation):
    """Fixed, seed-determined sample of the retained replay (the positions the learner trains on)."""
    examples = [e for g in runner.replay.iter_games() for e in g.examples]
    rng = random.Random(f'phase4d4h-c-probe:{runner.config.training.seed}:{generation}')
    return rng.sample(examples, min(PROBE_POSITIONS, len(examples)))
