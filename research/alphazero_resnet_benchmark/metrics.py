"""Frozen evaluation procedures: held-out replay, development suites, batch norm and calibration.

Nothing here trains or mutates the evaluated model: train-mode probes run on
a disposable deep copy. Losses are the training definitions (seven raw logits
against normalized visits; squared error against the game outcome). Choices
and calibration use the inference definition (legal-masked softmax).
"""
from copy import deepcopy
import math

import torch
from torch.nn import functional as F

from .common import REPO, read
from .dataset import canonical, ply_bucket

EVAL_BATCH = 512
PROBE_BATCH = 128   # the training minibatch size, so train-mode batch statistics are realistic
BINS = 10


def tensors(examples):
    """Model inputs, visit targets and outcomes through the trainer's own batch_tensors."""
    from games.connect4.alphazero_v2.training import batch_tensors
    parts = [batch_tensors(examples[i:i + 4096]) for i in range(0, len(examples), 4096)]
    return tuple(torch.cat([p[k] for p in parts]) for k in range(3))


def forward(model, states, batch=EVAL_BATCH):
    """Eval-mode logits and values; the model's training flag is asserted, never changed."""
    if model.training:
        raise ValueError('Evaluation requires an eval-mode model')
    logits, values = [], []
    with torch.inference_mode():
        for i in range(0, states.shape[0], batch):
            p, v = model(states[i:i + batch])
            logits.append(p)
            values.append(v)
    return torch.cat(logits).double(), torch.cat(values).double().squeeze(1)


def position_scores(logits, values, states, policies, outcomes):
    """Per-position losses and choices as float64 tensors."""
    policies, outcomes = policies.double(), outcomes.double().squeeze(1)
    ce = -(policies * F.log_softmax(logits, dim=1)).sum(1)
    entropy = -(policies * torch.log(policies.clamp_min(1e-300))).sum(1)
    legal = states[:, 0, 0, :] == 0
    masked = torch.softmax(logits.masked_fill(~legal, -math.inf), dim=1)
    confidence, choice = masked.max(1)
    target_choice = policies.argmax(1)
    return dict(ce=ce, kl=ce - entropy, se=(values - outcomes) ** 2, top1=(choice == target_choice).double(),
                confidence=confidence, value=values, outcome=outcomes, target_mass=masked.gather(1, target_choice[:, None]).squeeze(1))


def mean(tensor):
    return float(tensor.mean()) if tensor.numel() else None


def reliability(confidence, correct, bins=BINS):
    """Equal-width confidence bins on [0,1]; expected calibration error is count-weighted |accuracy - confidence|."""
    rows, ece, n = [], 0.0, confidence.numel()
    index = torch.clamp((confidence * bins).long(), max=bins - 1)
    for b in range(bins):
        mask = index == b
        count = int(mask.sum())
        if count:
            conf, acc = float(confidence[mask].mean()), float(correct[mask].mean())
            ece += count / n * abs(acc - conf)
            rows.append(dict(bin=b, count=count, confidence=conf, accuracy=acc))
    return dict(ece=ece, bins=rows)


def value_reliability(value, target, bins=BINS):
    """Equal-width predicted-value bins on [-1,1]; error is count-weighted |mean target - mean prediction|."""
    rows, error, n = [], 0.0, value.numel()
    index = torch.clamp(((value + 1) / 2 * bins).long(), max=bins - 1)
    for b in range(bins):
        mask = index == b
        count = int(mask.sum())
        if count:
            predicted, observed = float(value[mask].mean()), float(target[mask].mean())
            error += count / n * abs(observed - predicted)
            rows.append(dict(bin=b, count=count, predicted=predicted, observed=observed))
    return dict(error=error, bins=rows, mean_abs_value=float(value.abs().mean()),
                saturated_share=float((value.abs() > 0.95).double().mean()))


class HeldOut:
    """One seed's frozen held-out and training-probe tensors with family and novelty labels."""

    def __init__(self, data):
        held = data['held']
        self.states, self.policies, self.outcomes = tensors([e for e, _, _ in held])
        self.novel = torch.tensor(data['novel'])
        self.family = torch.tensor([family for _, _, family in held])
        self.bucket = [ply_bucket(e.ply) for e, _, _ in held]
        probe = [data['train'][i][0] for i in data['probe']]
        self.probe = tensors(probe)

    def evaluate(self, model, *, detail=False):
        logits, values = forward(model, self.states)
        scores = position_scores(logits, values, self.states, self.policies, self.outcomes)
        novel = self.novel
        result = {}
        for name, mask in (('novel', novel), ('all', torch.ones_like(novel)), ('seen', ~novel)):
            result[name] = dict(positions=int(mask.sum()), policy_ce=mean(scores['ce'][mask]),
                                policy_kl=mean(scores['kl'][mask]), value_mse=mean(scores['se'][mask]),
                                top1=mean(scores['top1'][mask]))
        result['novel_by_ply'] = {}
        for bucket in sorted(set(self.bucket)):
            mask = novel & torch.tensor([b == bucket for b in self.bucket])
            result['novel_by_ply'][bucket] = dict(positions=int(mask.sum()), policy_ce=mean(scores['ce'][mask]),
                                                  value_mse=mean(scores['se'][mask]), top1=mean(scores['top1'][mask]))
        # Per-family sums over novel positions: everything a paired family bootstrap needs.
        families = {}
        for family in sorted(set(self.family[novel].tolist())):
            mask = novel & (self.family == family)
            families[str(family)] = [int(mask.sum()), float(scores['ce'][mask].sum()), float(scores['se'][mask].sum()),
                                     float(scores['top1'][mask].sum())]
        result['novel_families'] = families
        result['calibration'] = dict(
            policy=reliability(scores['confidence'][novel], scores['top1'][novel]),
            value=value_reliability(scores['value'][novel], scores['outcome'][novel]))
        p_logits, p_values = forward(model, self.probe[0])
        probe = position_scores(p_logits, p_values, *self.probe)
        result['train_probe'] = dict(positions=int(self.probe[0].shape[0]), policy_ce=mean(probe['ce']),
                                     policy_kl=mean(probe['kl']), value_mse=mean(probe['se']), top1=mean(probe['top1']))
        result['generalization_gap'] = dict(
            policy_ce=result['novel']['policy_ce'] - result['train_probe']['policy_ce'],
            value_mse=result['novel']['value_mse'] - result['train_probe']['value_mse'])
        result['batch_norm'] = batch_norm_probe(model, self.probe, p_logits, p_values)
        if detail:
            result['positions'] = dict(ce=scores['ce'].float(), se=scores['se'].float(), top1=scores['top1'].bool(),
                                       value=scores['value'].float(), confidence=scores['confidence'].float())
        return result


def batch_norm_probe(model, probe, eval_logits, eval_values):
    """Running statistics, their distance from probe-batch statistics, and train- versus eval-mode outputs."""
    norms = [(n, m) for n, m in model.named_modules() if isinstance(m, torch.nn.BatchNorm2d)]
    if not norms:
        return None
    states, policies, outcomes = probe
    usable = states.shape[0] // PROBE_BATCH * PROBE_BATCH
    # Train-mode forward on a disposable copy: batch statistics per 128-position minibatch.
    copy = deepcopy(model).train()
    logits, values = [], []
    with torch.no_grad():
        for i in range(0, usable, PROBE_BATCH):
            p, v = copy(states[i:i + PROBE_BATCH])
            logits.append(p)
            values.append(v)
    logits, values = torch.cat(logits).double(), torch.cat(values).double().squeeze(1)
    train = position_scores(logits, values, states[:usable], policies[:usable], outcomes[:usable])
    evaluated = position_scores(eval_logits[:usable], eval_values[:usable], states[:usable], policies[:usable], outcomes[:usable])
    log_eval, log_train = F.log_softmax(eval_logits[:usable], dim=1), F.log_softmax(logits, dim=1)
    kl = (log_eval.exp() * (log_eval - log_train)).sum(1)
    # Cumulative probe statistics (momentum=None averages batches) on a second disposable copy.
    reference = deepcopy(model).train()
    for module in reference.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.reset_running_stats()
            module.momentum = None
    with torch.no_grad():
        for i in range(0, usable, PROBE_BATCH):
            reference(states[i:i + PROBE_BATCH])
    probe_modules = dict(reference.named_modules())
    layers, mean_shift, var_ratio = {}, [], []
    for name, module in norms:
        probe_mean, probe_var = probe_modules[name].running_mean.double(), probe_modules[name].running_var.double()
        run_mean, run_var = module.running_mean.double(), module.running_var.double()
        shift = ((run_mean - probe_mean).abs() / torch.sqrt(probe_var + module.eps)).mean()
        ratio = (torch.log((run_var + module.eps) / (probe_var + module.eps))).abs().mean()
        mean_shift.append(float(shift))
        var_ratio.append(float(ratio))
        layers[name] = dict(running_mean_abs=float(run_mean.abs().mean()), running_var_mean=float(run_var.mean()),
                            running_var_min=float(run_var.min()), running_var_max=float(run_var.max()),
                            standardized_mean_shift=float(shift), abs_log_variance_ratio=float(ratio),
                            scale_mean=float(module.weight.detach().abs().mean()), updates=int(module.num_batches_tracked))
    return dict(positions=usable,
                eval_mode=dict(policy_ce=mean(evaluated['ce']), value_mse=mean(evaluated['se'])),
                train_mode=dict(policy_ce=mean(train['ce']), value_mse=mean(train['se'])),
                mode_gap=dict(policy_ce=mean(evaluated['ce']) - mean(train['ce']),
                              value_mse=mean(evaluated['se']) - mean(train['se'])),
                mean_abs_value_difference=float((eval_values[:usable] - values).abs().mean()),
                max_abs_value_difference=float((eval_values[:usable] - values).abs().max()),
                mean_policy_kl_eval_to_train=float(kl.mean()),
                choice_disagreement=float((eval_logits[:usable].argmax(1) != logits.argmax(1)).double().mean()),
                standardized_mean_shift=sum(mean_shift) / len(mean_shift), abs_log_variance_ratio=sum(var_ratio) / len(var_ratio),
                worst_standardized_mean_shift=max(mean_shift), worst_abs_log_variance_ratio=max(var_ratio), layers=layers)


# Development suites -------------------------------------------------------------------------

def suite_clusters(rows):
    """Phase 4D.4G's rule: union mirror-position families and reflection-equivalent prefixes through ply 8."""
    parents = list(range(len(rows)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    prefixes = [tuple(r['moves'][:8]) for r in rows]
    for i, a in enumerate(rows):
        for j in range(i):
            p, q = prefixes[i], prefixes[j]
            n = min(len(p), len(q))
            if a['family'] == rows[j]['family'] or p[:n] == q[:n] or p[:n] == tuple(6 - c for c in q[:n]):
                parents[find(i)] = find(j)
    names = {v: k for k, v in enumerate(sorted({find(i) for i in range(len(rows))}))}
    return [names[find(i)] for i in range(len(rows))]


class Suites:
    """Development solved300 and tactical200 (never the sealed packages), with replay-overlap flags."""

    def __init__(self, train_keys=frozenset()):
        from games.connect4.alphazero_v2.bitboard import BitboardState
        from games.connect4.alphazero_v2.data import encode_board
        from games.connect4.alphazero_v2.oracle import engine_position
        self.rows, self.states, self.cluster, self.in_train = {}, {}, {}, {}
        for suite in ('solved', 'tactical'):
            package = read(REPO / f'games/connect4/alphazero_v2/frozen/{suite}-development.json')
            if package['split'] != 'development':
                raise ValueError('Only development suites may be evaluated')
            rows = package['rows']
            games = [engine_position(r['moves']) for r in rows]
            self.rows[suite] = rows
            self.states[suite] = [BitboardState.from_game(g) for g in games]
            self.cluster[suite] = suite_clusters(rows)
            self.in_train[suite] = [canonical(encode_board(g)) in train_keys for g in games]

    def evaluate(self, inference):
        """Raw network only: argmax of the legal-masked policy (lowest column on an exact tie) and the raw value."""
        result = {}
        for suite in ('solved', 'tactical'):
            predictions = []
            states = self.states[suite]
            for i in range(0, len(states), 64):
                predictions.extend(inference.predict_batch(states[i:i + 64]))
            records, ties = [], 0
            for row, prediction, cluster, seen in zip(self.rows[suite], predictions, self.cluster[suite], self.in_train[suite]):
                best = max(prediction.policy)
                ties += sum(p == best for p in prediction.policy) > 1
                action = prediction.policy.index(best)
                record = dict(id=row['id'], cluster=cluster, in_train=seen, action=action, confidence=best,
                              value=prediction.value)
                if suite == 'solved':
                    mass = sum(prediction.policy[a] for a in row['optimal_actions'])
                    record.update(correct=int(action in row['optimal_actions']), optimal_mass=mass,
                                  ce=-math.log(max(mass, 1e-300)), se=(prediction.value - row['value']) ** 2,
                                  exact=row['value'], mistake=int(row['action_values'][str(action)] < row['value']))
                else:
                    record.update(category=row['category'], correct=int(action == row['expected_action']),
                                  avoidable=int(action not in row['safe_actions']),
                                  missed_win=int(bool(row['immediate_wins']) and action not in row['immediate_wins']))
                records.append(record)
            result[suite] = dict(rows=records, ties=int(ties), **suite_summary(suite, records))
        return result


def suite_summary(suite, records):
    def rate(key, subset=None):
        chosen = [r for r in records if subset is None or subset(r)]
        return sum(r[key] for r in chosen) / len(chosen) if chosen else None
    if suite == 'solved':
        confidence = torch.tensor([r['confidence'] for r in records], dtype=torch.float64)
        correct = torch.tensor([float(r['correct']) for r in records], dtype=torch.float64)
        value = torch.tensor([r['value'] for r in records], dtype=torch.float64)
        exact = torch.tensor([float(r['exact']) for r in records], dtype=torch.float64)
        return dict(accuracy=rate('correct'), optimal_set_ce=rate('ce'), optimal_mass=rate('optimal_mass'),
                    value_mse=rate('se'), mistake_rate=rate('mistake'),
                    accuracy_excluding_train_overlap=rate('correct', lambda r: not r['in_train']),
                    train_overlap_rows=sum(r['in_train'] for r in records),
                    calibration=dict(policy=reliability(confidence, correct), value=value_reliability(value, exact)))
    return dict(accuracy=rate('correct'), unique_win_accuracy=rate('correct', lambda r: r['category'] == 'unique_win'),
                unique_safe_accuracy=rate('correct', lambda r: r['category'] == 'unique_safe'),
                avoidable_rate=rate('avoidable'), missed_win_rate=rate('missed_win', lambda r: r['category'] == 'unique_win'),
                train_overlap_rows=sum(r['in_train'] for r in records))
