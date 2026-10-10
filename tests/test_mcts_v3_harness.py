"""MCTS v3 study harness: declared openings, resume integrity, and inference."""
import copy
import json
import shutil
from pathlib import Path

import pytest

from scripts.mcts_v3 import analysis, harness
from scripts.mcts_v3.harness import matchup, mcts, negamax


@pytest.fixture(scope='module')
def openings():
    return harness.generate_openings()


def test_committed_openings_match_generator_and_declared_shape(openings):
    committed = json.loads(Path('docs/search-mcts-v3/openings.json').read_text())
    assert committed == json.loads(json.dumps(openings))
    sets = openings['sets']
    assert {name: len(rows) for name, rows in sets.items()} == dict(
        dev=96, holdout=256, preflight=8, empty=64, empty2=64)
    for name in ('dev', 'holdout'):
        counts = {length: sum(o['length'] == length for o in sets[name]) for length in harness.LENGTHS}
        assert set(counts.values()) == {len(sets[name]) // 8}
    seeds = [o[key] for rows in sets.values() for o in rows
             for key in ('challenger_seed', 'opponent_seed')]
    assert len(set(seeds)) == len(seeds)


def test_development_and_held_out_positions_are_disjoint_and_undecided(openings):
    boards = []
    for name in ('dev', 'holdout', 'preflight'):
        for opening in openings['sets'][name]:
            game = harness.position(opening['history'])
            assert len(opening['history']) == opening['length']
            assert not game.is_game_over()
            assert not harness.decided_by_root_guards(game)
            boards.append(tuple(map(tuple, game.board)))
    assert len(set(boards)) == len(boards) == 360


def test_root_guard_decided_positions_are_recognised():
    assert harness.decided_by_root_guards(harness.position([0, 1, 0, 1, 0, 2]))   # mover wins
    assert harness.decided_by_root_guards(harness.position([2, 2, 3, 3, 4]))      # two threats
    assert not harness.decided_by_root_guards(harness.position([0, 1, 0, 1, 0]))  # one block
    assert not harness.decided_by_root_guards(harness.position([]))


@pytest.fixture
def tiny(tmp_path, monkeypatch):
    shutil.copy('docs/search-mcts-v3/openings.json', tmp_path / 'openings.json')
    monkeypatch.setattr(harness, 'ROOT', tmp_path)
    monkeypatch.setattr(analysis, 'ROOT', tmp_path)
    rows = [matchup('solver', mcts(12, solver=True, rollout='safe'), mcts(12), pairs=3),
            matchup('negamax', mcts(12, exploration=0.7), negamax(2), pairs=2)]
    config = harness.declare('tiny', 'preflight', rows, cap_seconds=600)
    config['analysis']['replicates'] = 200
    harness.atomic_json(tmp_path / 'tiny' / 'study.json', config)
    return tmp_path / 'tiny', config


def stable(row):
    row = copy.deepcopy(row)
    for key in ('elapsed_seconds', 'wall_us', 'cpu_us'):
        row.pop(key)
    return row


def test_resume_matches_fresh_games_and_rejects_corruption(tiny):
    directory, config = tiny
    assert len(list(harness.schedule(config))) == 10
    assert harness.run('tiny', max_games=3)['status'] == 'batch limit'
    partial = harness.load_rows(directory / 'results.jsonl')
    summary = harness.run('tiny')
    assert (summary['status'], summary['completed'], summary['planned']) == ('complete', 10, 10)
    rows = harness.load_rows(directory / 'results.jsonl')
    assert [stable(r) for r in rows[:3]] == [stable(r) for r in partial]
    plans = {f'{m["id"]}/{o["id"]}/{c}': (m, o, c) for m, o, c in harness.schedule(config)}
    for row in rows:
        fresh = harness.play_game(config, *plans[row['id']])
        assert stable(fresh) == stable(row)
        assert row['simulations'][0] in (0, 12, None) or 0 < row['simulations'][0] <= 12
    assert harness.run('tiny')['games_this_invocation'] == 0
    for damage in (dict(winner=7), dict(challenger_score=0.25), dict(study_sha256='x')):
        with pytest.raises(ValueError):
            harness.validate_rows(config, [dict(rows[0], **damage)])
    with pytest.raises(ValueError, match='Duplicate'):
        harness.validate_rows(config, [rows[0], rows[0]])
    with pytest.raises(ValueError, match='already declared'):
        harness.declare('tiny', 'preflight', [], cap_seconds=1)


def test_analysis_counts_pairs_and_reports_both_roles(tiny):
    directory, config = tiny
    harness.run('tiny')
    result, pair_scores, strata, _ = analysis.analyze('tiny')
    assert result['complete_games'] == result['planned_games'] == 10
    by_id = {row['matchup']: row for row in result['matchups']}
    assert by_id['solver']['complete_pairs'] == 3 and by_id['negamax']['complete_pairs'] == 2
    for row in result['matchups']:
        assert row['wins'] + row['draws'] + row['losses'] == row['games']
        low, high = row['score']['ci_family']
        assert 0 <= low <= row['score']['ci95'][0] <= row['score']['mean'] <= row['score']['ci95'][1] <= high <= 1
        assert row['timing']['challenger']['decisions'] > 0 and row['time_ratio'] > 0
    assert by_id['negamax']['timing']['opponent']['mean_simulations'] is None
    assert '| solver |' in analysis.table(result)


def test_source_drift_blocks_resume(tiny, monkeypatch):
    monkeypatch.setattr(harness, 'source_hashes', lambda: {'changed': 'x'})
    with pytest.raises(ValueError, match='drift'):
        harness.run('tiny')


def test_cluster_bootstrap_and_sign_flip_behave():
    values = {f'o{i}': 0.5 for i in range(16)}
    strata = {key: i % 2 for i, key in enumerate(values)}
    flat = analysis.bootstrap(values, strata, 1, 200)
    assert flat['ci95'] == flat['ci_family'] == [0.5, 0.5]
    assert analysis.sign_flip_p([0.0] * 16, 1, 200) == 1.0
    assert analysis.sign_flip_p([1.0] * 16, 1, 2000) < 0.01
    mixed = analysis.sign_flip_p([1.0, -1.0] * 8, 1, 2000)
    assert mixed == 1.0


def test_paired_contrast_uses_shared_openings(tiny, monkeypatch):
    from scripts.mcts_v3 import studies
    monkeypatch.setitem(studies.CONTRASTS, 'tiny', [('solver minus negamax', 'solver', 'negamax')])
    harness.run('tiny')
    result, pair_scores, _, _ = analysis.analyze('tiny')
    contrast = result['contrasts'][0]
    shared = pair_scores['solver'].keys() & pair_scores['negamax'].keys()
    assert contrast['difference']['openings'] == len(shared) == 2
    expected = sum(pair_scores['solver'][k] - pair_scores['negamax'][k] for k in shared) / 2
    assert contrast['difference']['mean'] == pytest.approx(expected)
    assert 'Paired difference' in analysis.table(result)


def test_frozen_finalist_budgets_follow_the_declared_rule():
    from scripts.mcts_v3 import studies
    for name, budgets in studies.FROZEN_BUDGETS.items():
        for budget, expected in budgets.items():
            spec = studies.finalist(name, budget, True)
            assert spec['simulations'] == expected < budget
            assert studies.finalist(name, budget, False)['simulations'] == budget
    assert studies.FINALISTS['f1'][0] == dict(rollout='safe', solver=True, exploration=0.5)
    assert studies.FINALISTS['f2'][0] == dict(rollout='decisive', solver=True)


def test_tactical_audit_classifies_proven_blunders_and_wins():
    from games.connect4.agents.negamax_agent import NegamaxAgent
    from scripts.mcts_v3.tactical_audit import classify
    must_block = NegamaxAgent(4).score_moves(harness.position([0, 1, 0, 1, 0]))
    assert classify(must_block, 3, 4) == ('blunder', 1)
    assert classify(must_block, 0, 4)[0] == 'unproven'
    forced = NegamaxAgent(4).score_moves(harness.position([3, 3, 2, 2]))
    assert classify(forced, 4, 4) == ('win_kept', 2)
    assert classify(forced, 6, 4) == ('win_missed', 2)
    immediate = NegamaxAgent(4).score_moves(harness.position([0, 1, 0, 1, 0, 2]))
    assert classify(immediate, 0, 4) == ('win_kept', 0)
    doomed = NegamaxAgent(4).score_moves(harness.position([2, 2, 3, 3, 4]))
    assert classify(doomed, 1, 4) == ('already_lost', None)


def test_strict_latency_budgets_are_below_the_frozen_equal_time_budgets():
    from scripts.mcts_v3 import studies
    assert {b: studies.strict_budget(b) for b in studies.PRIMARY_BUDGETS} == {400: 151, 2000: 821}
