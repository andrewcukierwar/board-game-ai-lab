"""Phase 4D.4H-B tooling: split, leakage, calibration and uncertainty logic on small synthetic inputs.

No historical replay, official artifact or trained model is read here.
"""
import random

import numpy as np
import torch

from games.connect4.alphazero_v2.batching import TemperatureSchedule
from games.connect4.alphazero_v2.experimental import rebuild_scheduled_game
from research.alphazero_resnet_benchmark import analyze, dataset, metrics

SCHEDULE = TemperatureSchedule('legacy_verified', ((0, 1), (8, 0)))


def game(index, moves):
    """A finished game whose stored visits make every greedy move the maximum-visit move."""
    visits = [[10 if a == m else 0 for a in range(7)] for m in moves]
    record = dict(index=index, moves=list(moves), winner=None, visits=visits,
                  action_temperatures=[SCHEDULE.temperature(p) for p in range(len(moves))])
    from games.connect4.alphazero_v2.oracle import engine_position
    record['winner'] = engine_position(moves).check_winner()
    return rebuild_scheduled_game(13, record, SCHEDULE, 10)


def random_game(index, rng, prefix=()):
    from games.connect4.connect4 import Connect4
    board, moves = Connect4(), []
    for move in prefix:
        board.make_move(move)
        moves.append(move)
    while not board.is_game_over():
        move = rng.choice(board.get_valid_moves())
        board.make_move(move)
        moves.append(move)
    return game(index, moves)


def games(count=60, seed=5):
    rng = random.Random(seed)
    shared = (0, 0, 1, 1, 2, 2, 6, 6, 5, 5)   # ten moves, no win: positions at plies 8 and 9 are shared
    return [random_game(i, rng, shared if i % 6 == 0 else ()) for i in range(count)]


def test_canonical_key_is_mirror_invariant_and_actor_specific():
    a, b = game(0, [0, 1, 0, 1, 0, 1, 0]), game(1, [6, 5, 6, 5, 6, 5, 6])
    assert [dataset.canonical(e.observation) for e in a.examples] == [dataset.canonical(e.observation) for e in b.examples]
    assert len({dataset.canonical(e.observation) for e in a.examples}) == len(a.examples)


def test_families_link_only_from_the_declared_ply_and_never_straddle_the_split():
    played = games()
    everything = dataset.components(played, 0)
    assert len(everything) == 1                                   # every game shares the empty board
    families = dataset.components(played, 8)
    linked = next(f for f in families if 0 in f)
    assert set(linked) >= {i for i in range(len(played)) if i % 6 == 0}
    train, held, family_of = dataset.split(42, played, 8)
    assert sorted(train + held) == list(range(len(played))) and held
    assert not {family_of[i] for i in train} & {family_of[i] for i in held}
    train_deep = set().union(*(dataset.game_keys(played[i], 8) for i in train))
    assert not any(dataset.game_keys(played[i], 8) & train_deep for i in held)
    assert dataset.split(42, played, 8)[1] == held and dataset.split(43, played, 8)[1] != held


def test_oversized_families_stay_in_training(monkeypatch):
    played = games()
    monkeypatch.setattr(dataset, 'MAX_FAMILY_FRACTION', 0.05)
    _, held, family_of = dataset.split(42, played, 8)
    big = family_of[0]
    assert sum(f == big for f in family_of.values()) > 0.05 * len(played)
    assert all(family_of[i] != big for i in held)


def test_reliability_is_zero_for_perfect_calibration_and_counts_every_row():
    confidence = torch.tensor([0.25] * 4 + [0.75] * 4, dtype=torch.float64)
    correct = torch.tensor([1, 0, 0, 0, 1, 1, 1, 0], dtype=torch.float64)
    result = metrics.reliability(confidence, correct)
    assert abs(result['ece']) < 1e-12 and sum(b['count'] for b in result['bins']) == 8
    assert abs(metrics.reliability(torch.ones(4, dtype=torch.float64), torch.zeros(4, dtype=torch.float64))['ece'] - 1) < 1e-12
    value = metrics.value_reliability(torch.tensor([-1.0, 1.0], dtype=torch.float64), torch.tensor([-1.0, 1.0], dtype=torch.float64))
    assert value['error'] == 0 and value['saturated_share'] == 1


def test_position_scores_match_the_training_loss():
    from games.connect4.alphazero_v2.training import training_loss
    played = games(6)
    examples = [e for g in played for e in g.examples]
    states, policies, outcomes = metrics.tensors(examples)
    generator = torch.Generator().manual_seed(1)
    logits = torch.randn(len(examples), 7, generator=generator)
    values = torch.tanh(torch.randn(len(examples), 1, generator=generator))
    scores = metrics.position_scores(logits.double(), values.double().squeeze(1), states, policies, outcomes)
    combined, policy, value = training_loss(logits, values, policies, outcomes)
    assert abs(float(scores['ce'].mean()) - float(policy)) < 1e-5 and abs(float(scores['se'].mean()) - float(value)) < 1e-5
    assert bool((scores['kl'] > -1e-9).all()) and bool((scores['confidence'] <= 1).all())


def view(families, **novel):
    return dict(families=families, novel=novel)


def test_family_bootstrap_contrast_and_verdict():
    rng = np.random.default_rng(0)
    families = {str(i): [10, float(rng.uniform(5, 15)), float(rng.uniform(2, 8)), float(rng.integers(0, 10))] for i in range(40)}
    better = {k: [v[0], v[1] - 1.0, v[2] - 0.5, v[3]] for k, v in families.items()}
    same = analyze.held_out_contrast({42: (view(families), view(families)), 7: (view(families), view(families))}, 400, 3)
    assert all(same['pooled'][m]['difference'] == 0 and same['pooled'][m]['low'] == 0 for m in analyze.FAMILY_METRICS)
    assert analyze.verdict({k: same[k]['combined'] for k in same}, 'combined', (42, 7)) == 'INCONCLUSIVE'
    improved = analyze.held_out_contrast({42: (view(better), view(families)), 7: (view(better), view(families))}, 400, 3)
    assert abs(improved['42']['policy_ce']['difference'] + 0.1) < 1e-9 and abs(improved['42']['combined']['difference'] + 0.15) < 1e-9
    assert improved['pooled']['combined']['high'] < 0
    assert analyze.verdict({k: improved[k]['combined'] for k in improved}, 'combined', (42, 7)) == 'BETTER'
    worse = analyze.held_out_contrast({42: (view(families), view(better)), 7: (view(families), view(better))}, 400, 3)
    assert analyze.verdict({k: worse[k]['combined'] for k in worse}, 'combined', (42, 7)) == 'WORSE'
    split = analyze.held_out_contrast({42: (view(better), view(families)), 7: (view(families), view(better))}, 400, 3)
    assert analyze.verdict({k: split[k]['combined'] for k in split}, 'combined', (42, 7)) == 'INCONCLUSIVE'


def test_accuracy_verdict_direction_and_cluster_resampling():
    def rows(correct):
        return [dict(id=str(i), cluster=i // 2, correct=c, category='unique_win' if i % 2 else 'unique_safe',
                     avoidable=1 - c) for i, c in enumerate(correct)]
    base, improved = rows([0] * 40), rows([1] * 30 + [0] * 10)
    result = analyze.suite_contrast({42: (improved, base), 7: (improved, base)}, 'tactical', 400, 3)
    assert result['accuracy']['pooled']['difference'] == 0.75 and result['accuracy']['pooled']['clusters'] == 20
    assert analyze.verdict(result['accuracy'], 'accuracy', (42, 7)) == 'BETTER'
    assert analyze.verdict(result['avoidable_rate'], 'avoidable_rate', (42, 7)) == 'BETTER'
    assert result['unique_win_accuracy']['42']['rows'] == 20


def test_time_interpolation_is_linear_between_milestones():
    def row(updates, seconds, ce):
        return dict(updates=updates, train_seconds=seconds, held_out=dict(
            novel=dict(policy_ce=ce, value_mse=0.5, top1=0.4), novel_families={'0': [2, 2 * ce, 1.0, 1.0]}),
            suites=dict(solved=dict(accuracy=0.5, optimal_set_ce=1.0, value_mse=0.5, mistake_rate=0.5),
                        tactical=dict(accuracy=0.5, unique_win_accuracy=0.5, unique_safe_accuracy=0.5)))
    run = dict(milestones=[row(0, 0.0, 2.0), row(100, 10.0, 1.0), row(200, 20.0, 0.5)])
    middle = analyze.at_seconds(run, 15.0)
    assert abs(middle['novel']['policy_ce'] - 0.75) < 1e-12 and middle['updates'] == 150
    assert abs(middle['families']['0'][1] - 1.5) < 1e-12 and analyze.at_seconds(run, 21.0) is None
