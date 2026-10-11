"""Meaningful guards for instrumentation, evidence isolation and paired analysis."""
import random

import pytest
import torch

from games.connect4.connect4 import Connect4
from games.connect4.alphazero_v2.network import AlphaZeroV2Net, V2Inference
from games.connect4.alphazero_v2.search import PUCTSearch, V2RootNoise
from research.alphazero_baseline.audit import (Timings, clustered_interval, fresh_output,
                                             search_signature, MainOpponent)


@pytest.fixture
def inference():
    with torch.random.fork_rng():
        torch.manual_seed(9)
        return V2Inference(AlphaZeroV2Net().eval())


@pytest.mark.parametrize('moves', [[], [3,2,3,2,3,1], [0,3,1,3,0,3]])
@pytest.mark.parametrize('noisy', [False, True])
def test_instrumentation_preserves_entire_tree_rng_and_input(inference, moves, noisy):
    game = Connect4()
    for move in moves:
        assert game.make_move(move)
    before = tuple(tuple(row) for row in game.board)
    rng = random.Random(17)
    noise = V2RootNoise(seed=23) if noisy else None
    original = PUCTSearch(inference, 12).run(game, rng=rng, root_noise=noise)
    state = rng.getstate()
    rng = random.Random(17)
    second_noise = V2RootNoise(seed=23) if noisy else None
    meter = Timings()
    old_expand = PUCTSearch._expand
    with meter.instrument(inference):
        measured = PUCTSearch(inference, 12).run(game, rng=rng, root_noise=second_noise)
    assert search_signature(original) == search_signature(measured)
    assert state == rng.getstate()
    assert not noisy or noise.get_state() == second_noise.get_state()
    assert tuple(tuple(row) for row in game.board) == before
    assert PUCTSearch._expand is old_expand
    assert meter.calls['neural_forward'] == 13-original.terminal_leaves
    assert all(value >= 0 for value in meter.exclusive.values())


def test_nested_times_are_additive_and_restore_on_error(inference):
    meter = Timings()
    clock = iter([0., 1., 3., 5.])
    from unittest.mock import patch
    child = meter.wrap('child', lambda: 2)
    parent = meter.wrap('parent', child)
    with patch('research.alphazero_baseline.audit.time.perf_counter', side_effect=lambda: next(clock)):
        assert parent() == 2
    assert meter.inclusive == {'child':2., 'parent':5.}
    assert meter.exclusive == {'child':2., 'parent':3.}
    original = inference.model.forward
    with pytest.raises(RuntimeError), meter.instrument(inference):
        raise RuntimeError('probe')
    assert inference.model.forward == original


def test_output_refuses_campaign_overlap_existing_and_symlink(tmp_path):
    campaign = tmp_path/'campaign'
    campaign.mkdir()
    for candidate in (campaign,campaign/'new',tmp_path):
        with pytest.raises(ValueError):
            fresh_output(campaign,candidate)
    alias = tmp_path/'alias'
    alias.symlink_to(campaign,target_is_directory=True)
    with pytest.raises(ValueError):
        fresh_output(campaign,alias/'child')
    destination = fresh_output(campaign,tmp_path/'fresh')
    assert destination.is_dir()
    with pytest.raises(FileExistsError):
        fresh_output(campaign,destination)


def test_bootstrap_retains_color_pairs_and_ignores_abandoned_games():
    records = [dict(family='a',agent_color=0,result='win',abandoned=False),
               dict(family='a',agent_color=1,result='loss',abandoned=False),
               dict(family='b',agent_color=0,result='draw',abandoned=False),
               dict(family='b',agent_color=1,result='draw',abandoned=False),
               dict(family='c',agent_color=0,abandoned=True)]
    summary = clustered_interval(records)
    assert summary['games'] == 4
    assert summary['clusters'] == 2
    assert summary['score'] == .5
    assert summary['interval95'] == [.5,.5]
    assert summary['by_color']['X']['score'] == .75
    assert summary['by_color']['O']['score'] == .25


def test_archive_extracts_development_records_and_replay_without_runner(tmp_path, monkeypatch):
    import json
    from research.alphazero_baseline import audit
    monkeypatch.setattr(audit, 'SEEDS', (42,))
    campaign, output = tmp_path/'campaign', tmp_path/'output'
    output.mkdir()
    moves = [0,1,0,1,0,1,0]
    for generation in range(1,21):
        directory = audit.checkpoint(campaign,42,generation).parent
        directory.mkdir(parents=True)
        (directory/'summary.json').write_text(json.dumps(dict(resources={},summary={})))
        (directory/f'generation-{generation:04d}.games.jsonl').write_text(
            json.dumps(dict(index=0,moves=moves,winner=0))+'\n')
    for generation in (5,10,15,20):
        (campaign/f'runs/seed-42/champion-check-{generation:04d}.json').write_text(
            json.dumps(dict(seed=42,generation=generation,promote=False,records={'private':'omitted'})))
    torch.save(dict(replay=dict(generations=[dict(generation=20,games=[dict(
        moves=moves,visits=[[256,0,0,0,0,0,0]]*7,action_temperatures=[1.]*7)])])),
        audit.checkpoint(campaign,42,20).with_name('generation-0020.resume.pt'))
    audit.archives(campaign,output,lambda:None)
    checks = audit.read(output/'official_development.json')
    assert [x['generation'] for x in checks] == [5,10,15,20]
    assert all('records' not in x for x in checks)
    assert len(audit.read(output/'archives.json')) == 20
    assert audit.read(output/'replay_exploration.json')[0]['stages']['before8']['positions'] == 7


def test_both_main_mcts_opponents_declare_their_root_guards():
    assert MainOpponent({},'classic',1000).describe()['root_guard']
    assert MainOpponent({},'tactical',1000).describe()['root_guard']
    assert not MainOpponent({},'negamax',8).describe()['root_guard']
    assert MainOpponent({},'negamax',8).describe()['ties'] == 'deterministic center order'


@pytest.mark.parametrize('seconds',['nan','inf','0','-1'])
def test_cli_refuses_unbounded_or_invalid_duration_before_reading_evidence(monkeypatch, seconds):
    from research.alphazero_baseline.audit import main
    monkeypatch.setattr('sys.argv',['audit','--mode','profile','--campaign','missing',
                                  '--output','must-not-be-created','--seconds',seconds])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2


def test_summary_cli_cannot_write_inside_the_campaign(tmp_path, monkeypatch):
    import json
    from research.alphazero_baseline.summarize import main
    campaign, profile = tmp_path/'campaign', tmp_path/'profile'
    campaign.mkdir()
    profile.mkdir()
    (profile/'provenance.json').write_text(json.dumps(dict(parameters=dict(campaign=str(campaign)))))
    argv=['summary','--profile',str(profile),'--output',str(campaign/'new')]
    for flag in ('positions','ladder','archive','state'):
        argv += ['--'+flag,'missing']
    monkeypatch.setattr('sys.argv',argv)
    with pytest.raises(ValueError,match='disjoint'):
        main()
    assert not (campaign/'new').exists()
