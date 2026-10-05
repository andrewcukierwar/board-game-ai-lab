"""Completed-example augmentation correctness; synthetic discarded updates only."""
from dataclasses import asdict, replace
import json
import random

import pytest

torch = pytest.importorskip('torch')

from games.connect4 import neural_self_play as driver
from games.connect4.train_mcts_nn import (TrainingExample, reflect_completed_example,
                                         capture_example, finalize_examples, batch_tensors)
from games.connect4.neural_mcts import encode_current_player
from games.connect4.neural_symmetry_diagnostics import freeze, validate_rows, board_key, reflected_key
from games.connect4.neural_evaluation import frozen_fixtures
from test_neural_self_play import setup, threads, DRAW, X_WIN, O_WIN
from test_neural_scaled import fake_updates


@pytest.mark.parametrize('moves', [DRAW,X_WIN,O_WIN])
def test_reflection_complete_episodes_both_actors_labels_full_columns(moves):
    trainer, search = setup(moves)
    game = driver.position(())
    pending = []
    for move in moves:
        pending.append(capture_example(search.search(game)))
        game.make_move(move)
    complete = finalize_examples(pending, game)
    assert {e.acting_player for e in complete} == {0,1}
    for i, original in enumerate(complete):
        original_fields = asdict(original)
        mirrored = reflect_completed_example(original)
        assert mirrored is not original
        assert reflect_completed_example(mirrored) == original
        assert asdict(original) == original_fields
        physical = driver.position([6-m for m in moves[:i]])
        expected = tuple(tuple(int(c) for c in row) for row in encode_current_player(physical)[0,0].tolist())
        assert mirrored.observation == expected
        assert all(mirrored.policy[c] == original.policy[6-c] for c in range(7))
        assert all(mirrored.policy[c] == 0 for c in range(7) if c not in physical.get_valid_moves())
        for key in ('acting_player','outcome','temperature','encoding','policy_target','tactical_guard'):
            assert getattr(mirrored,key) == getattr(original,key)
        states, policies, labels = batch_tensors([mirrored])
        assert labels.item() == {1:0,0:1,-1:2}[original.outcome]
    if moves == DRAW:
        assert any(e.observation[0][0] != 0 for e in complete)


@pytest.mark.parametrize('guard', [True,False])
@pytest.mark.parametrize('temperature', [0.,1.,2.5])
def test_metadata_and_owned_immutable_inputs(guard, temperature):
    board = [[0]*7 for _ in range(6)]
    board[-1][1] = 1
    policy = [0,.1,.2,.3,.1,.1,.2]
    original = TrainingExample(board,1,policy,temperature,guard,-1)
    mirror = reflect_completed_example(original)
    board[-1][1] = -1
    policy[1] = 0
    assert original.observation[-1][1] == mirror.observation[-1][5] == 1
    assert reflect_completed_example(mirror) == original
    with pytest.raises(ValueError):
        reflect_completed_example(replace(original,outcome=None))
    with pytest.raises(ValueError):
        reflect_completed_example(None)


@pytest.mark.parametrize('probability', [-.01,1.01,float('nan'),float('inf'),True,'0.5'])
def test_invalid_probability(probability):
    with pytest.raises(ValueError):
        driver.SymmetryAugmenter(probability)


def examples():
    board = [[0]*7 for _ in range(6)]
    board[-1][1] = 1
    return [TrainingExample(board,p,(.1,.1,.2,.2,.1,.1,.2),1.,False,o)
            for p in (0,1) for o in (-1,0,1)]


def test_determinism_domain_separation_and_rng_ownership():
    before = driver.rng_states(random.Random(42),random.Random(42))
    search, sampling = random.Random(42),random.Random(42)
    a,b = driver.SymmetryAugmenter(.5),driver.SymmetryAugmenter(.5)
    c = driver.SymmetryAugmenter(.5,seed=43)
    assert a.rng_state() != search.getstate() and a.seed_digest != c.seed_digest
    flags = []
    for _ in range(20):
        batch, transformed = a.batch(examples())
        assert (batch,transformed) == b.batch(examples())
        flags.extend(transformed)
        # Unrelated RNG use cannot steer augmentation, and conversely.
        c.batch(examples())
    assert any(flags) and not all(flags)
    assert a.record()['transformed'] == sum(flags)
    assert a.record()['untransformed'] == len(flags)-sum(flags)
    assert driver.rng_states(search,sampling) == before
    assert type(json.loads(json.dumps(a.record()))['transformed']) is int


def test_disabled_exact_identity_no_rng_draws_and_baseline_numerical_updates():
    a = driver.SymmetryAugmenter()
    state = a.rng_state()
    batch, flags = a.batch(examples())
    assert not any(flags) and a.rng_state() == state
    source = examples()
    assert all(x is y for x,y in zip(a.batch(source)[0],source))
    first,_ = setup()
    second,_ = setup()
    for _ in range(4):
        first.step(source)  # Original direct baseline path, no augmenter.
        second.step(a.batch(source)[0])
        assert first.last_metrics == second.last_metrics
        assert all(torch.equal(v,second.model.state_dict()[k]) for k,v in first.model.state_dict().items())
        for p,q in zip(first.optimizer.state.values(),second.optimizer.state.values()):
            assert p.keys() == q.keys()
            assert all(torch.equal(p[k],q[k]) for k in p)
    assert first.steps == second.steps == 4
    assert a.rng_state() == state


def test_collection_episode_optimizer_sampling_accounting_unchanged_and_serializable():
    reports, event_sets = [], []
    for probability in (0.,.5):
        trainer, search = setup()
        fake_updates(trainer)
        events = []
        report = driver.run_training(trainer, search, driver.Bounds.scaled(), emit=events.append,
                                     horizontal_symmetry_probability=probability)
        reports.append(report)
        event_sets.append(events)
        json.loads(json.dumps(dict(report=report,events=events),allow_nan=False))
        assert report['status'] == 'bounded_stop' and report['updates'] == 1970
        aug = report['augmentation']
        assert aug['transformed']+aug['untransformed'] == 1970*32
        assert all(len(e['sampled_indices']) == len(e['horizontal_reflected']) == 32
                   for e in events if e['event']=='update')
    for key in ('completed_games','actual_plies','collected_plies','labels','labels_by_actor','games','losses'):
        assert reports[0][key] == reports[1][key]
    assert reports[0]['augmentation']['transformed'] == 0
    assert .48 < reports[1]['augmentation']['observed_rate'] < .52
    for name in ('completed_game','update'):
        left = [e for e in event_sets[0] if e['event']==name]
        right = [e for e in event_sets[1] if e['event']==name]
        assert len(left) == len(right)
        for a,b in zip(left,right):
            keys = ('examples','winner','moves') if name=='completed_game' else ('sampled_indices','after_game','update')
            assert all(a[k] == b[k] for k in keys)


def test_blind_fixture_model_free_frozen_labels_exclusions_and_serialization(tmp_path):
    path = tmp_path/'blind.json'
    digest = freeze(path)
    import hashlib
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    payload = json.loads(path.read_text())
    rows = payload['positions']
    assert len(rows) == 96 and payload['model_blind'] and not payload['training_feedback']
    keys = validate_rows(rows)
    for suite in frozen_fixtures().values():
        for row in suite:
            key = board_key(driver.position(row['moves']))
            assert key not in keys and reflected_key(key) not in keys
    with pytest.raises(FileExistsError):
        freeze(path)
    other = tmp_path/'other.json'
    assert freeze(other) == digest
