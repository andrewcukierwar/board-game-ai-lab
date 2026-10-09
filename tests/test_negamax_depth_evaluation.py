"""Independent evidence integrity, paired inference and diagnostic isolation."""
import copy
import json
import random

import pytest

from scripts import evaluate_negamax_depths as evaluation
from scripts.analyze_negamax_depths import interval, summarize
from scripts.profile_negamax_depths import decision, memory_run, profile_run, line_run, assert_same
from scripts.benchmark_public_agents import DRAW, position
from games.connect4.agents.negamax_agent import SearchTable
from games.connect4.agents import negamax_agent as search
from games.connect4.connect4 import Connect4


@pytest.fixture
def tiny(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, 'check_source', lambda _: None)
    opening=dict(id='o',history=DRAW[:38],cohort='agent',family='f',length=38,
                 challenger_seed=101,opponent_seed=202,**evaluation.labels(DRAW[:38]))
    config=dict(source_commit='test',max_seconds=20,preflight_seconds=20,
                matchups=[dict(id='n2-vs-m4',challenger=dict(type='negamax',depth=2),
                               opponent=dict(type='mcts',simulations=4),primary=True,pairs=1)],
                openings=[opening],preflight_openings=[opening],
                analysis=dict(seed=42,replicates=200))
    evaluation.atomic_json(tmp_path/'experiment.json',config)
    return tmp_path,config


def without_timings(row):
    row=copy.deepcopy(row)
    row.pop('elapsed_seconds')
    for move in row['moves']:
        move.pop('wall_seconds')
        move.pop('cpu_seconds')
    return row


def test_real_checkpoint_resume_and_seeded_repeat(tiny):
    directory,config=tiny
    state=random.getstate()
    first=evaluation.run(directory,max_games=1)
    assert first['status']=='checkpoint requested' and first['completed']==1
    second=evaluation.run(directory)
    assert second['completed']==2 and second['status']=='complete'
    rows=evaluation.load_rows(directory/'results.jsonl')
    for row,plan in zip(rows,evaluation.schedule(config)):
        assert without_timings(row)==without_timings(evaluation.play(config,*plan))
    assert random.getstate()==state
    assert evaluation.validate(config,rows)=={r['id'] for r in rows}
    assert summarize(config,rows)['matchups'][0]['complete_pairs']==1


@pytest.mark.parametrize('field', ['duplicate','hash','winner','history','role','timing','stats','seed'])
def test_reject_corrupt_evidence(tiny,field):
    _,config=tiny
    row=evaluation.play(config,*next(evaluation.schedule(config)))
    rows=[row]
    if field=='duplicate': rows.append(copy.deepcopy(row))
    elif field=='hash': row['experiment_sha256']='other'
    elif field=='winner': row['winner']=99
    elif field=='history': row['complete_history']=[0]
    elif field=='role': row['moves'][0]['role']='bad'
    elif field=='timing': row['moves'][0]['wall_seconds']=float('nan')
    elif field=='stats':
        move=next(m for m in row['moves'] if m['role']=='challenger')
        move['negamax_stats']['nodes']=-1
    elif field=='seed': row['challenger_seed']+=1
    with pytest.raises((ValueError,KeyError)):
        evaluation.validate(config,rows)


def test_errors_are_explicit_unscored_and_deadline_is_enforced(tiny,monkeypatch):
    directory,config=tiny
    class Failure:
        def choose_move(self,game):
            raise RuntimeError('injected')
    monkeypatch.setattr(evaluation,'make_agent',lambda *args:Failure())
    status=evaluation.run(directory)
    assert status['completed']==0
    row=evaluation.load_rows(directory/'results-attempts.jsonl')[0]
    assert row['status']=='incomplete' and 'challenger_score' not in row
    evaluation.validate(config,[row],allow_incomplete=True)
    with pytest.raises(ValueError): evaluation.validate(config,[row])
    config['max_seconds']=.02
    evaluation.atomic_json(directory/'experiment.json',config)
    (directory/'results-attempts.jsonl').unlink()
    (directory/'results-status.json').unlink()
    class Slow:
        def choose_move(self,game):
            while True:
                pass
    monkeypatch.setattr(evaluation,'make_agent',lambda *args:Slow())
    assert 'BudgetExpired' in evaluation.run(directory)['status']


def test_source_drift_fails_closed(tiny,monkeypatch):
    directory,config=tiny
    # Exercise the actual source checker, rather than the fixture stub.
    monkeypatch.undo()
    with pytest.raises((ValueError,KeyError)):
        evaluation.run(directory)


def test_generator_deduplicates_boards_and_keeps_related_snapshots():
    seen=set()
    rows=evaluation.openings(2026100904,'test',1,seen)
    assert len(rows)==len({r['board_sha256'] for r in rows})==8
    assert {r['stage'] for r in rows}=={'early','midgame','late'}
    agent=[r for r in rows if r['cohort']=='agent']
    assert len({r['family'] for r in agent})==1
    assert [r['length'] for r in agent]==[5,14,23,32]
    for a,b in zip(agent,agent[1:]):
        assert b['history'][:len(a['history'])]==a['history']
    for row in rows:
        game=position(row['history'])
        assert not game.is_game_over() and evaluation.board_key(game)==row['board_sha256']


def test_cluster_bootstrap_preserves_related_snapshots():
    openings={str(i):dict(cohort='agent',family='a' if i<4 else 'b') for i in range(8)}
    values={str(i):float(i>=4) for i in range(8)}
    ci=interval(values,openings,123,1000)
    assert ci['clusters']==2 and ci['openings']==8
    assert ci['ci95']==[0,1] and ci['mean']==.5
    assert ci==interval(values,openings,123,1000)
    degenerate=interval({k:1 for k in values},openings,123,100)
    assert degenerate['conservative97_5'][0]<.5


def test_profiling_does_not_change_search_and_restores_hooks():
    game=position([3,2,4,3])
    expected=decision(game,3)
    for measured in (memory_run(game,3),profile_run(game,3),line_run(game,3)):
        assert_same(expected,measured)
    assert search.SearchTable is SearchTable
    assert profile_run(game,3)['leaf_evaluations']>0


def test_freeze_timing_rule_and_no_overwrite(tmp_path,monkeypatch):
    monkeypatch.setattr(evaluation,'check_source',lambda _:None)
    config=dict(source_commit='test',matchups=evaluation.matchups(),openings=[{'id':str(i)} for i in range(64)],
                preflight_openings=[{'id':str(i)} for i in range(8)])
    evaluation.atomic_json(tmp_path/'candidate.json',config)
    rows=[dict(elapsed_seconds=4,matchup=m['id']) for m in config['matchups'] for _ in range(16)]
    evaluation.atomic_json(tmp_path/'preflight-memory.json',dict(candidate_sha256=evaluation.digest(config)))
    for row in rows: evaluation.append_json(tmp_path/'preflight.jsonl',row)
    monkeypatch.setattr(evaluation,'validate',lambda *args:None)
    frozen=evaluation.freeze(tmp_path)
    assert frozen['freeze']['primary_pairs']==16
    assert frozen['freeze']['secondary_pairs']==8
    with pytest.raises(ValueError,match='already frozen'): evaluation.freeze(tmp_path)


@pytest.mark.parametrize('depth',[6,8,10])
def test_deeper_endgame_scores_against_unpruned_array_oracle(depth):
    def oracle(game,remaining):
        winner=game.check_winner()
        if winner!=-1:
            return (1 if winner==game.current_player else -1)*(1_000_000+remaining)
        if game.is_board_full():
            return 0
        # At most four empty cells: every branch reaches exact terminals.
        assert remaining>0
        values=[]
        for col in game.get_valid_moves():
            child=Connect4(game.board,game.current_player)
            assert child.make_move(col)
            values.append(-oracle(child,remaining-1))
        return max(values)
    game=position(DRAW[:38])
    scores={}
    for col in game.get_valid_moves():
        child=Connect4(game.board,game.current_player)
        assert child.make_move(col)
        scores[col]=-oracle(child,depth-1)
    agent=search.NegamaxAgent(depth)
    assert agent.score_moves(game)==scores
    assert agent.choose_move(game)==max(scores,key=scores.get)
