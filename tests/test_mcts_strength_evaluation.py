"""Exercise experiment integrity, deterministic resume, and cluster inference."""
import copy
import json
import random

import pytest

from scripts import evaluate_mcts_strength as evaluation
from scripts.analyze_mcts_strength import bootstrap, summarize
from scripts.benchmark_public_agents import position
from scripts.profile_mcts_strength import memory_run


@pytest.fixture
def tiny(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, 'BUDGETS', (4, 8))
    config = evaluation.declare(tmp_path, pairs=2, secondary_pairs=1, preflight_pairs=1)
    config['analysis']['replicates'] = 100
    evaluation.atomic_json(tmp_path / 'experiment.json', config)
    return tmp_path, config


def test_opening_declaration_is_deterministic_diverse_legal():
    before = random.getstate()
    rows = evaluation.generate_openings(128, 57, 'test')
    assert rows == evaluation.generate_openings(128, 57, 'test')
    assert random.getstate() == before
    assert len({tuple(o['history']) for o in rows}) == 128
    assert {o['length'] for o in rows} == set(evaluation.LENGTHS)
    assert len({s for o in rows for s in (o['challenger_seed'], o['opponent_seed'])}) == 256
    for opening in rows:
        assert not position(opening['history']).is_game_over()


def stable(row):
    row = copy.deepcopy(row)
    row.pop('elapsed_seconds')
    row.pop('search_totals')
    for move in row['moves']:
        move.pop('wall_seconds')
        move.pop('cpu_seconds')
    return row


def test_resume_matches_fresh_games_and_rejects_corruption(tiny):
    directory, config = tiny
    with pytest.raises(ValueError, match='preflight'):
        evaluation.run(directory)
    evaluation.run(directory, preflight=True)
    evaluation.run(directory, max_games=1)
    assert evaluation.run(directory)['status'] == 'complete'
    rows = evaluation.load_rows(directory / 'results.jsonl')
    assert len(rows) == len(list(evaluation.schedule(config)))
    for row, (matchup, opening, color) in zip(rows, evaluation.schedule(config)):
        assert stable(row) == stable(evaluation.play_game(config, matchup, opening, color))
    assert evaluation.run(directory)['completed'] == len(rows)
    assert len(evaluation.load_rows(directory / 'results.jsonl')) == len(rows)
    with pytest.raises(ValueError, match='Duplicate'):
        evaluation.validate_rows(config, rows + rows[:1])
    damaged = copy.deepcopy(rows)
    damaged[0]['challenger_seed'] += 1
    with pytest.raises(ValueError, match='plan mismatch'):
        evaluation.validate_rows(config, damaged)
    damaged = copy.deepcopy(rows)
    damaged[0]['winner'] = 1 - damaged[0]['winner']
    with pytest.raises(ValueError, match='incorrect recorded result'):
        evaluation.validate_rows(config, damaged)
    damaged = copy.deepcopy(rows)
    damaged[0]['moves'][0]['column'] = 9
    with pytest.raises(ValueError, match='move history'):
        evaluation.validate_rows(config, damaged)
    config['source_hashes']['games/connect4/agents/mcts_agent.py'] = 'foreign'
    evaluation.atomic_json(directory / 'experiment.json', config)
    with pytest.raises(ValueError, match='drift'):
        evaluation.run(directory)


def test_errors_are_incomplete_not_losses(tiny, monkeypatch):
    _, config = tiny
    class Broken:
        def choose_move(self, game):
            raise KeyboardInterrupt()
    monkeypatch.setattr(evaluation, 'make_agent', lambda *_: Broken())
    row = evaluation.play_game(config, config['matchups'][0], config['openings'][0], 0)
    assert row['status'] == 'incomplete'
    assert row['error'].startswith('KeyboardInterrupt')
    assert 'winner' not in row and 'challenger_score' not in row


def test_cluster_bootstrap_and_unpaired_exclusion(tiny):
    _, config = tiny
    matches = list(evaluation.schedule(config))[:4]
    rows = [evaluation.play_game(config, *match) for match in matches]
    partial = summarize(config, rows[:1])['matchups'][0]
    assert partial['complete_pairs'] == 0
    assert partial['unpaired_games'] == 1
    assert partial['paired_score'] is None
    complete = summarize(config, rows[:2])['matchups'][0]
    assert complete['paired_score']['independent_openings'] == 1
    assert complete['completed_games'] == 2
    values = {'a': .5, 'b': .5, 'c': .5}
    result = bootstrap(values, dict(a=2, b=2, c=5), 37, 100)
    assert result['mean'] == .5 and result['ci95'] == [.5, .5]
    assert result == bootstrap(values, dict(a=2, b=2, c=5), 37, 100)
    varied = {'a': 0, 'b': 1, 'c': .5, 'd': 1}
    strata = dict(a=2, b=2, c=5, d=5)
    assert bootstrap(varied, strata, 41, 100) == bootstrap(
        dict(reversed(list(varied.items()))), strata, 41, 100)


def test_memory_diagnostics_count_simulations_and_guard():
    result = memory_run(position([]), 12, 42)
    assert result['executed_simulations'] == 12
    assert result['nodes'] == 13
    assert result['peak_bytes'] >= result['retained_bytes'] > 0
    guard = memory_run(position([0, 1, 0, 1, 0, 2]), 12, 42)
    assert guard['executed_simulations'] == 0
    assert guard['nodes'] == 0


def test_corrupt_jsonl_fails_closed(tmp_path):
    path = tmp_path / 'results.jsonl'
    path.write_text('{"id":"ok"}\n{"id":')
    with pytest.raises(json.JSONDecodeError):
        evaluation.load_rows(path)
