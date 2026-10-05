"""Target semantics only: no research optimizer or learned model needed."""
from dataclasses import asdict, replace
from copy import deepcopy
import json
from pathlib import Path
import random

import numpy as np
import pytest

torch = pytest.importorskip('torch')
from games.connect4.connect4 import Connect4
from games.connect4.neural_mcts import encode_current_player
from games.connect4.tactical_value import reconstruct_canonical, tactical_proof
from games.connect4.train_mcts_nn import (TrainingExample, ValueTargetAnchorer,
    batch_tensors, reflect_completed_example)
from games.connect4.neural_self_play import SymmetryAugmenter

DRAW = [2,0,0,0,0,0,0,1,1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,3,6,4,4,4,4,4,4,5,5,5,5,5,5,6,6,6,6,6]
CASES = [([0,1,0,1,0,2],1), ([0,1,0,1,2,1,2],1),
         ([6,1,6,2,5,3],-1), ([1,6,2,6,3],-1),
         ([0,1,0,1,2,1],None), ([0,1,0,1,0],None), (DRAW[:-1],None)]


def position(moves):
    game = Connect4()
    for move in moves:
        assert not game.is_game_over() and game.make_move(move)
    return game


def example(game, outcome):
    legal = game.get_valid_moves()
    return TrainingExample(encode_current_player(game)[0,0].tolist(), game.current_player,
                           tuple(1/len(legal) if a in legal else 0 for a in range(7)),
                           1.0, outcome=outcome)


@pytest.mark.parametrize('moves,value', CASES)
@pytest.mark.parametrize('mirror', [False,True])
def test_reconstruct_proofs_and_reflection(moves,value,mirror):
    game = position([6-m if mirror else m for m in moves])
    e = example(game, 0)
    reconstructed = reconstruct_canonical(e.observation, e.acting_player)
    assert reconstructed.__dict__ == game.__dict__
    before = deepcopy(game.__dict__)
    assert tactical_proof(reconstructed)['value'] == value
    batch, records = ValueTargetAnchorer(True).batch([e])
    reflected = reflect_completed_example(batch[0])
    rgame = reconstruct_canonical(reflected.observation, reflected.acting_player)
    assert tactical_proof(rgame)['value'] == value
    after_reflection, _ = ValueTargetAnchorer(True).batch([reflect_completed_example(e)])
    assert after_reflection[0] == reflected
    assert game.__dict__ == before
    assert records[0]['training_outcome'] == (0 if value is None else value)


@pytest.mark.parametrize('moves,value', CASES)
@pytest.mark.parametrize('outcome', [-1,0,1])
def test_changes_only_temporary_value(moves,value,outcome):
    e = example(position(moves), outcome)
    collection = [e]
    before = asdict(e)
    batch, records = ValueTargetAnchorer(True).batch(collection)
    r = records[0]
    assert collection == [e] and asdict(e) == before and e.outcome == outcome
    assert batch[0] == replace(e, outcome=outcome if value is None else value)
    assert {k:v for k,v in asdict(batch[0]).items() if k!='outcome'} == {k:v for k,v in before.items() if k!='outcome'}
    assert r == dict(behavioral_outcome=outcome, proven_value=value,
                     training_outcome=outcome if value is None else value,
                     anchored=value is not None, contradiction=value is not None and value!=outcome)
    assert json.loads(json.dumps(records, allow_nan=False)) == records
    if value == outcome or value is None:
        assert batch[0] is e


def test_disabled_tensors_and_no_rng_consumption():
    examples = [example(position(moves),0) for moves,_ in CASES]
    sampler = random.Random(42)
    aug = SymmetryAugmenter(.5)
    states = (random.getstate(), deepcopy(np.random.get_state()), torch.get_rng_state().clone(),
              sampler.getstate(), aug.rng_state())
    disabled, records = ValueTargetAnchorer().batch(examples)
    assert all(a is b for a,b in zip(disabled,examples))
    assert all(torch.equal(a,b) for a,b in zip(batch_tensors(disabled),batch_tensors(examples)))
    ValueTargetAnchorer(True).batch(examples)
    assert random.getstate() == states[0]
    np_state = np.random.get_state()
    assert np_state[0] == states[1][0] and np.array_equal(np_state[1],states[1][1]) and np_state[2:] == states[1][2:]
    assert torch.equal(torch.get_rng_state(),states[2])
    assert sampler.getstate() == states[3] and aug.rng_state() == states[4]
    a,b = SymmetryAugmenter(.5), SymmetryAugmenter(.5)
    anchored,_ = ValueTargetAnchorer(True).batch(examples)
    assert a.batch(anchored)[1] == b.batch(examples)[1]


def test_invalid_reconstruction_and_pending_rejected():
    e = example(position([]),0)
    with pytest.raises(ValueError):
        reconstruct_canonical(e.observation,1)
    floating = [list(row) for row in e.observation];floating[0][0]=1
    with pytest.raises(ValueError,match='Floating'):
        reconstruct_canonical(floating,1)
    with pytest.raises(ValueError):
        reconstruct_canonical(e.observation[:-1],0)
    with pytest.raises(ValueError):
        ValueTargetAnchorer(True).batch([replace(e,outcome=None)])
    terminal = position([0,1,0,1,0,2,0])
    observation = tuple(tuple(int(c) for c in row) for row in encode_current_player(terminal)[0,0].tolist())
    with pytest.raises(ValueError,match='nonterminal'):
        reconstruct_canonical(observation,terminal.current_player)


def test_complete_retained_audit_classifications_and_witnesses():
    path = Path('experiment-output/phase4d2e-value-target-audit-20261005/analysis.json')
    if not path.exists():
        pytest.skip('Retained research artifacts are local and Git-ignored')
    retained = json.loads(path.read_text())
    n = 0
    for run in retained['runs'].values():
        for row in run['rows']:
            game = reconstruct_canonical(tuple(tuple(r) for r in row['example']['observation']), row['actor'])
            assert [list(r) for r in game.board] == row['board']
            assert tactical_proof(game) == row['proof']
            n += 1
    assert n == 10608


@pytest.mark.parametrize('moves,value', CASES)
def test_independent_exact_holdout_validator(moves,value):
    from games.connect4.neural_exact_value_diagnostics import independent_proof
    game = position(moves)
    assert independent_proof(game) == tactical_proof(game)


def test_freezer_balanced_unique_and_excludes_mirrors(tmp_path):
    from games.connect4.neural_exact_value_diagnostics import freeze, position as replay
    from games.connect4.neural_symmetry_diagnostics import board_key, reflected_key
    path = tmp_path/'blind.json'
    excluded = [dict(moves=moves) for moves,_ in CASES]
    freeze(path, excluded_rows=excluded, bases_per_cell=1, seed=420404)
    rows = json.loads(path.read_text())['positions']
    assert len(rows)==8
    assert {(r['actor'],r['proven_value']) for r in rows} == {(a,v) for a in (0,1) for v in (-1,1)}
    forbidden = {key for row in excluded for key in (board_key(replay(row['moves'])),reflected_key(board_key(replay(row['moves']))))}
    assert not {board_key(replay(row['moves'])) for row in rows} & forbidden
    with pytest.raises(FileExistsError):
        freeze(path, excluded_rows=excluded, bases_per_cell=1)


def test_driver_anchors_before_symmetry_and_preserves_completed_events(monkeypatch):
    from games.connect4 import neural_self_play as driver
    from games.connect4.agents.mcts_nn_agent import Node, SearchResult, RootDirichletNoise
    from games.connect4.neural_mcts import Connect4Net, NeuralInference
    from games.connect4.train_mcts_nn import NeuralTrainer
    scripted_draw = [3,2,3,3,3,4,4,1,2,4,4,1,1,0,2,1,4,0,0,2,5,5,5,2,4,5,1,2,5,3,6,5,1,0,6,0,0,3,6,6,6,6]
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng():
            torch.manual_seed(42)
            trainer = NeuralTrainer(Connect4Net())
        class Scripted:
            simulation_limit, exploration, temperature, tactical_guard = 32,1.41,1.,False
            def __init__(self):
                self.inference = NeuralInference(trainer.model)
                self.calls = 0
            def search(self,game):
                move = scripted_draw[self.calls % len(scripted_draw)];self.calls+=1
                root = Node(deepcopy(game));root.visits=32
                for a in game.get_valid_moves():
                    after=deepcopy(game);assert after.make_move(a)
                    root.children[a]=Node(after,root,a,1/len(game.get_valid_moves()))
                root.children[move].visits=32
                return SearchResult(move,tuple(float(a==move) for a in range(7)),
                                    tuple(32 if a==move else 0 for a in range(7)),1.,False,'disabled',root)
        received=[]
        def step(batch):
            received.extend(batch)
            trainer.steps+=1
            trainer.last_metrics=dict(combined_loss=0.,finite_gradients=True,finite_parameters=True)
        monkeypatch.setattr(trainer,'step',step)  # No optimizer update, scripted draw histories.
        noise=RootDirichletNoise(.25,.30);before=deepcopy(noise.rng_state())
        events=[]
        report=driver.run_training(trainer,Scripted(),driver.Bounds(max_updates=1),
                                   value_anchoring=True,horizontal_symmetry_probability=.5,emit=events.append)
        assert report['status']=='bounded_stop' and report['updates']==1
        completed=[e for e in events if e['event']=='completed_game']
        stored=[TrainingExample(**e) for game in completed for e in game['examples']]
        assert len(stored)==126 and all(e.outcome==0 for e in stored)
        update=next(e for e in events if e['event']=='update')
        base=[stored[i] for i in update['sampled_indices']]
        anchored,records=ValueTargetAnchorer(True).batch(base)
        expected,flags=SymmetryAugmenter(.5).batch(anchored)
        assert received==expected and update['value_anchors']==records and update['horizontal_reflected']==flags
        assert any(r['contradiction'] for r in records)
        assert noise.rng_state()==before
        json.dumps(events,allow_nan=False)
    finally:
        torch.set_num_threads(previous)
