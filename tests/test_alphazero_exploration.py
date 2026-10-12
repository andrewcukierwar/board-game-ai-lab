"""Phase 4D.4D analysis tools: positions, tactics, attribution, uncertainty, replay, decisions."""
import json
import math
import random
import statistics

import pytest

from games.connect4.agents.mcts_nn_agent import visit_policy
from games.connect4.alphazero_v2.bitboard import BitboardState
from games.connect4.alphazero_v2.config import V2Config
from games.connect4.alphazero_v2.data import GenerationReplay
from games.connect4.alphazero_v2.generation import rebuild_game
from games.connect4.alphazero_v2.network import V2Inference, V2Prediction, legal_policy_from_logits
from games.connect4.alphazero_v2.oracle import engine_position
from research.alphazero_baseline import audit
from research.alphazero_exploration import analysis as a, analyze, collect, replay, schedules as s


class FakeInference(V2Inference):
    """Deterministic state-dependent predictions, as in the batching tests."""
    def __init__(self):
        self.model = None

    def predict(self, state):
        n = state.occupied.bit_count()
        logits = tuple((a - 3) * .125 + (n % (a + 1)) * .0625 for a in range(7))
        return V2Prediction(logits, tuple(legal_policy_from_logits(logits, state.get_valid_moves())),
                            ((n % 5) - 2) * .125)

    def predict_batch(self, states):
        return tuple(self.predict(state) for state in states)


BLUNDER = (0, 1, 0, 1, 0, 1, 0)          # O ignores X's column-0 threat at ply 5; X wins at ply 6
MISSED = (0, 1, 0, 1, 0, 1, 2, 1)        # X misses the column-0 win at ply 6; O wins at ply 7


def one_hot(move):
    return [256 if a == move else 0 for a in range(7)]


def rows(moves, schedule=s.ORIGINAL, visits=None):
    visits = visits or {}
    result = []
    for ply, move in enumerate(moves):
        counts = visits.get(ply, one_hot(move))
        tau = schedule.temperature(ply)
        result.append(dict(ply=ply, visits=counts, action=move, temperature=tau,
                           distribution=[float(p) for p in visit_policy(counts, tau)]))
    return result


def winner_of(moves):
    return engine_position(list(moves)).check_winner()


def game(moves, identity=(1, 20, 0, 0), schedule=s.ORIGINAL, visits=None, **changes):
    winner = winner_of(moves)
    record = dict(moves=moves, winner=winner, plies=rows(moves, schedule, visits),
                  outcomes=[0 if winner == -1 else 1 if winner == p % 2 else -1 for p in range(len(moves))])
    record.update(changes)
    return a.build_game(identity, schedule, record['moves'], record['winner'], record['plies'],
                        outcomes=record['outcomes'])


def test_mirror_and_canonical_match_board_reflection():
    rng = random.Random(7)
    for _ in range(300):
        moves, state = [], BitboardState.from_game(engine_position([]))
        for _ in range(rng.randrange(0, 30)):
            if state.is_game_over():
                break
            move = rng.choice(state.get_valid_moves())
            moves.append(move)
            state = state.drop(move)
        mirrored = BitboardState.from_game(engine_position([6 - m for m in moves]))
        assert (a.mirror_bits(state.pieces[0]), a.mirror_bits(state.pieces[1])) == mirrored.pieces
        assert mirrored.board == tuple(row[::-1] for row in state.board)
        assert a.canonical(state.pieces) == a.canonical(mirrored.pieces)
        assert a.mirror_bits(a.mirror_bits(state.pieces[0])) == state.pieces[0]
    center = BitboardState.from_game(engine_position([3, 3, 3])).pieces
    assert a.canonical(center) == center == (a.mirror_bits(center[0]), a.mirror_bits(center[1]))


def test_tactical_counts_blunder_and_missed_win():
    counts = game(BLUNDER)['tactics']
    expected = dict(games=1, positions=7, win_opps=1, missed_wins=0, choice_states=1, unsafe=1, block_states=1,
                    blockable_states=1, missed_blocks=1, avoidable_unsafe_moves=1, game_ending_errors=1,
                    tactical_states=2, errors=1, safe_available=7, lost_states=0)
    assert {k: counts[k] for k in expected} == expected
    counts = game(MISSED)['tactics']
    expected = dict(games=1, positions=8, win_opps=2, missed_wins=1, result_changing_missed_wins=1, choice_states=1,
                    unsafe=1, missed_blocks=1, block_states=1, avoidable_unsafe_moves=2, game_ending_errors=1, errors=2)
    assert {k: counts[k] for k in expected} == expected
    assert counts['missed_wins@0-7'] == 1 and counts['win_opps@tau1'] == 2 and counts['unsafe@tau1'] == 1


@pytest.mark.parametrize('visits,kind,greedy,expected', [
    ([0, 256, 0, 0, 0, 0, 0], 'search', 1., 1.),
    ([200, 56, 0, 0, 0, 0, 0], 'sampling', 0., 56 / 256),
    ([128, 128, 0, 0, 0, 0, 0], 'tie', .5, .5)])
def test_error_attribution_separates_search_preference_from_sampling(visits, kind, greedy, expected):
    counts = game(BLUNDER, visits={5: visits})['tactics']
    assert counts['unsafe'] == 1 and counts['unsafe:' + kind] == 1
    assert counts['game_ending_errors:' + kind] == 1
    assert counts['greedy_unsafe'] == greedy and counts['expected_unsafe'] == pytest.approx(expected)
    assert sum(counts['unsafe:' + k] for k in ('search', 'sampling', 'tie')) == 1


def test_missed_win_attribution_under_sampling():
    counts = game(MISSED, visits={6: [250, 0, 6, 0, 0, 0, 0]})['tactics']
    assert counts['missed_wins:sampling'] == 1 and counts['greedy_missed_wins'] == 0
    assert counts['expected_missed_wins'] == pytest.approx(6 / 256)
    counts = game(MISSED)['tactics']
    assert counts['missed_wins:search'] == 1 and counts['greedy_missed_wins'] == 1


@pytest.mark.parametrize('corrupt', ['temperature', 'distribution', 'visit_sum', 'illegal_visit', 'greedy',
                                     'outcome', 'winner', 'incomplete', 'action', 'continued'])
def test_loader_rejects_contract_violations(corrupt):
    moves = list(MISSED)
    data = rows(moves)
    winner = winner_of(moves)
    outcomes = [1 if winner == p % 2 else -1 for p in range(len(moves))]
    if corrupt == 'temperature': data[3]['temperature'] = 0.
    if corrupt == 'distribution': data[3]['distribution'] = [1 / 7] * 7
    if corrupt == 'visit_sum': data[3]['visits'] = [0, 255, 0, 0, 0, 0, 0]
    if corrupt == 'illegal_visit':   # column 0 is full at ply 12
        moves = [0] * 6 + [1] * 6 + [2, 3, 2, 3, 2, 3]
        data, outcomes = rows(moves, visits={12: [1, 0, 255, 0, 0, 0, 0]}), None
    if corrupt == 'greedy':
        late = (3, 3, 2, 2, 4, 4, 1, 1, 5)
        with pytest.raises(ValueError):
            game(late, visits={8: [0, 0, 0, 0, 0, 100, 156]})
        return
    if corrupt == 'outcome': outcomes[2] = -outcomes[2]
    if corrupt == 'winner': winner = 0
    if corrupt == 'incomplete': moves, data, outcomes = moves[:-1], data[:-1], outcomes[:-1]
    if corrupt == 'action': data[2]['action'] = 5
    if corrupt == 'continued': moves, data, outcomes = moves + [4], data + rows([4]), outcomes + [1]
    with pytest.raises(ValueError):
        a.build_game((1, 20, 0, 0), s.ORIGINAL, moves, winner, data, outcomes=outcomes)


def test_duplicate_collision_prefix_and_family_statistics():
    mirrored = tuple(6 - m for m in BLUNDER)
    games = [game(BLUNDER, (1, 20, 0, 0)), game(BLUNDER, (1, 20, 0, 1)), game(mirrored, (1, 20, 0, 2))]
    assert a.duplicate_fraction(games) == pytest.approx(1 - 13 / 21)
    assert a.duplicate_fraction(games, field='canon') == pytest.approx(1 - 7 / 21)
    assert a.duplicate_fraction(games, 1, 42) == pytest.approx(1 - 12 / 18)
    assert a.collision_probability(games) == pytest.approx(18 / 42)
    assert a.collision_probability(games, field='canon') == 1
    assert a.prefixes(games, 4) == dict(eligible_games=3, unique=2, unique_per_eligible=2 / 3)
    assert a.prefixes(games, 8)['eligible_games'] == 0
    assert a.families(games, 4)['unique'] == 1 and a.families(games, 7)['unique'] == 0
    profile = a.diversity(games)
    assert (profile['unique_trajectories'], profile['unique_mirror_canonical_trajectories']) == (2, 1)
    assert profile['most_repeated_trajectories'][0]['count'] == 2
    assert profile['stages']['0-7']['positions'] == 21 and profile['stages']['8+']['positions'] == 0
    assert profile['unique_tactical_positions'] == 4 and profile['unique_mirror_canonical_tactical_positions'] == 2
    # Collision probability is free of the game count; the duplicate fraction is not.
    doubled = games + [game(g['moves'], (2, 20, 0, i)) for i, g in enumerate(games)]
    assert a.duplicate_fraction(doubled) > a.duplicate_fraction(games)
    assert a.collision_probability(doubled, field='canon') == a.collision_probability(games, field='canon')


def test_jackknife_pairing_and_stratification():
    values = [3., 1., 4., 1., 5., 9., 2., 6.]
    result = a.jackknife(values, statistics.fmean)
    assert result['estimate'] == statistics.fmean(values)
    assert result['se'] == pytest.approx(statistics.stdev(values) / math.sqrt(len(values)))
    assert (result['low'], result['high']) == (pytest.approx(result['estimate'] - 1.96 * result['se']),
                                               pytest.approx(result['estimate'] + 1.96 * result['se']))
    reference = [game(BLUNDER, (1, 20, 0, i)) for i in range(4)]
    candidate = [game(MISSED if i % 2 else BLUNDER, (1, 20, 0, i)) for i in range(4)]
    length = lambda games: statistics.fmean(len(g['moves']) for g in games)
    difference = a.paired_difference(reference, list(reversed(candidate)), length)
    assert difference['estimate'] == pytest.approx(.5) and difference['units'] == 4
    assert a.paired_difference(reference, reference, length)['se'] == 0
    with pytest.raises(ValueError):
        a.paired_difference(reference, candidate[:3], length)
    with pytest.raises(ValueError):
        a.paired_difference(reference, candidate[:3] + [game(MISSED, (1, 20, 0, 9))], length)
    other = [game(MISSED, (2, 20, 0, i)) for i in range(2)]
    assert a.stratified(length)(reference + other) == pytest.approx((7 + 8) / 2)
    assert a.jackknife(reference, lambda games: a.rate(games, 'missed_wins', 'lost_states'))['se'] is None


def test_expected_distinct_and_rarefaction():
    assert replay.expected_distinct([1, 1, 1, 1], 1) == pytest.approx(1)
    assert replay.expected_distinct([2, 2], 2) == pytest.approx(1.5)
    assert replay.expected_distinct([5], 10) == pytest.approx(1)
    mirrored = tuple(6 - m for m in BLUNDER)
    games = [game(BLUNDER, (1, 20, 0, 0)), game(BLUNDER, (1, 20, 0, 1)), game(mirrored, (1, 20, 0, 2))]
    curve = replay.rarefaction(games, [7, 21], permutations=50)
    assert curve['7'] == dict(unique_positions=7, unique_mirror_canonical=7, unique_per_example=1)
    assert curve['21']['unique_positions'] == 13 and curve['21']['unique_mirror_canonical'] == 7
    assert curve == replay.rarefaction(games, [21, 7], permutations=50)
    with pytest.raises(ValueError):
        replay.rarefaction(games, [22], permutations=2)
    profile = replay.profile(games, draws=42, rarefaction_sizes=[21])
    assert profile['examples'] == 21 and profile['unique_mirror_canonical'] == 7 and profile['max_multiplicity'] == 3
    assert profile['duplicated_example_share'] == 1 and profile['multiplicity']['3-4']['states'] == 7
    assert profile['mirror']['both_orientations_present'] == 6
    assert profile['outcome_labels']['win'] == pytest.approx(12 / 21)
    assert profile['equal_budget']['expected_distinct_states'] == pytest.approx(7 * (1 - (1 - 1 / 7) ** 42))


def test_counterfactual_replay_uses_real_buffer_rules():
    def stored(generation, moves, count):
        return tuple(rebuild_game(generation, i, list(moves), [one_hot(m) for m in moves],
                                  [1. if p < 8 else 0. for p in range(len(moves))], winner_of(moves), 256, 8)
                     for i in range(count))
    window = {g: stored(g, BLUNDER, 20) for g in range(13, 21)}
    candidates = [game(MISSED if i % 2 else BLUNDER, (1, 20, c, i), s.SMOOTHED) for c in range(2) for i in range(10)]
    candidate = replay.completed_games(candidates, 21)
    assert [g.index for g in candidate] == list(range(20)) and all(g.generation == 21 for g in candidate)
    assert [e.action_temperature for e in candidate[1].examples] == [1.] * 8
    result = replay.counterfactual(window, candidate, updates=3, seeds=(1, 2, 3))
    assert result['window_generations'] == list(range(14, 22)) and result['evicted_generations'] == [13]
    assert result['window_positions'] == 7 * 20 * 7 + 10 * 7 + 10 * 8
    assert result['archive_unique_states'] == 7 and result['candidate_examples'] == 150
    # MISSED shares the positions before plies 0-6 with BLUNDER; only its ply-7 position is new.
    assert result["candidate_novel_states"] == 1 and result["candidate_novel_examples"] == 10
    assert result['sampling']['draws']['mean'] == 3 * 128 and result['counterfactual'] is True
    assert result == replay.counterfactual(window, candidate, updates=3, seeds=(1, 2, 3))
    with pytest.raises(ValueError):   # the real max-games rule still applies
        replay.counterfactual({g: stored(g, BLUNDER, 256) for g in range(14, 21)},
                              replay.completed_games([game(BLUNDER, (1, 20, 0, i)) for i in range(257)], 21),
                              updates=1, seeds=(1,))
    assert GenerationReplay(8, 2048).window_generations == V2Config().replay_generations


def interval(estimate, low, high):
    return dict(estimate=estimate, low=low, high=high, se=(high - low) / 3.92, units=256)


def test_predeclared_tactical_classification():
    quiet = dict(missed_win_rate=interval(.0, -.02, .02), unsafe_rate=interval(.001, -.005, .007),
                 errors_per_game=interval(.01, -.05, .07), game_ending_error_share=interval(.0, -.02, .02))
    assert analyze.classify_tactics(quiet)['classification'] == 'NO MATERIAL REGRESSION'
    wide = dict(quiet, missed_win_rate=interval(.02, -.04, .08))
    assert analyze.classify_tactics(wide)['classification'] == 'INCONCLUSIVE'
    worse = dict(quiet, unsafe_rate=interval(.03, .01, .05))
    assert analyze.classify_tactics(worse)['classification'] == 'MATERIAL REGRESSION'
    noisy = dict(quiet, unsafe_rate=interval(.03, -.01, .07))   # past the margin but compatible with zero
    assert analyze.classify_tactics(noisy)['classification'] == 'INCONCLUSIVE'
    assert analyze.CRITERIA['tactics'] == dict(missed_win_rate=.05, unsafe_rate=.02, errors_per_game=.15,
                                               game_ending_error_share=.05)
    assert analyze.CRITERIA['diversity'] == dict(material_relative=.25, material_absolute=.05,
                                                 suggestive_relative=.10, larger_change_absolute=.10)


def test_recommendation_prefers_smallest_eligible_change():
    grade = lambda **k: {n: dict(classification=k.get(n, 'NONE')) for n in analyze.ORDER}
    clear = grade(extended='NO MATERIAL REGRESSION', smoothed='INCONCLUSIVE', extended_smoothed='MATERIAL REGRESSION')
    none = analyze.recommend(grade(), clear, {})
    assert none['recommended_schedule'] == 'original' and not none['training_ablation_justified']
    only_d = analyze.recommend(grade(extended_smoothed='MATERIAL'), clear, {})
    assert only_d['recommended_schedule'] == 'original' and only_d['eligible'] == []
    smoothed = analyze.recommend(grade(smoothed='MATERIAL', extended_smoothed='MATERIAL'), clear, {})
    assert smoothed['recommended_schedule'] == 'smoothed' and smoothed['conditional_on_tactical_monitoring']
    assert smoothed['training_ablation_justified']
    extended = analyze.recommend(grade(extended='MATERIAL'), clear, {})
    assert extended['recommended_schedule'] == 'extended' and not extended['conditional_on_tactical_monitoring']


def test_collection_cells_load_match_and_classify(tmp_path, monkeypatch):
    """Synthetic end-to-end: collector cells -> verified games -> matched blocks -> decisions/tables."""
    monkeypatch.setattr(s, 'SIMULATIONS', 16)
    monkeypatch.setattr(collect, 'V2Config', lambda seed: V2Config(seed=seed, self_play_simulations=16))
    monkeypatch.setattr(collect.network, 'weights_sha256', lambda model: 'fake')
    inference = FakeInference()
    cells = []
    for seed, generation in s.PRIMARY:
        for cohort in range(2):
            for schedule in s.SCHEDULES:
                stop = (seed, cohort, schedule.name) == (314159, 1, 'smoothed')
                calls = iter(range(10 ** 9))
                def deadline():
                    if stop and next(calls) > 200:
                        raise collect.StopEvaluation('test stop')
                meta, games = collect.run_cell(inference, schedule, 900 + cohort, range(6), deadline,
                                               tmp_path / f'{seed}-{cohort}-{schedule.name}.jsonl',
                                               seed=seed, generation=generation, cohort=cohort, mode='main')
                assert (games is None) == stop and meta['complete'] != stop
                cells.append(meta)
    audit.write(tmp_path / 'cells.json', cells)
    games, loaded, skipped = a.load_collection(tmp_path)
    assert len(games) == 15 * 6 and len(skipped) == 1 and skipped[0]['completed_games'] < 6
    stored = (tmp_path / skipped[0]['file']).read_text().splitlines()
    assert len(stored) == skipped[0]['completed_games']   # only complete games were written
    blocks = analyze.matched_blocks(games)
    assert sorted(blocks) == [(42, 20, 0), (42, 20, 1), (314159, 20, 0)]   # the stopped cohort is excluded whole
    data = analyze.datasets(blocks)
    assert {k: {n: len(g) for n, g in v.items()} for k, v in data.items()} == {
        (42, 20): dict.fromkeys(s.BY_NAME, 12), (314159, 20): dict.fromkeys(s.BY_NAME, 6)}
    control = analyze.negative_control(data)
    assert len(control) == 8 and all(r['mismatches'] == 0 for r in control)
    results = dict(schedules=[x.to_dict() for x in s.SCHEDULES], conditions={}, paired_vs_original={})
    for checkpoint, by_schedule in data.items():
        key = analyze.label(checkpoint)
        results['conditions'][key] = {n: analyze.condition_summary(g) for n, g in by_schedule.items()}
        results['paired_vs_original'][key] = {
            n: analyze.comparisons(by_schedule['original'], by_schedule[n], {**analyze.ENDPOINTS, **analyze.TACTICAL})
            for n in analyze.ORDER}
        assert set(analyze.factorial(by_schedule)) == {'extend_main_effect', 'smooth_main_effect', 'interaction'}
    pooled = {n: [g for by in data.values() for g in by[n]] for n in s.BY_NAME}
    results['pooled'] = dict(tactics={n: a.tactical_profile(g) for n, g in pooled.items()})
    diversity = {n: analyze.classify_diversity(n, data, blocks, results['paired_vs_original']) for n in analyze.ORDER}
    tactics = {n: analyze.classify_tactics(analyze.comparisons(pooled['original'], pooled[n], analyze.TACTICAL))
               for n in analyze.ORDER}
    assert all(d['classification'] in ('MATERIAL', 'SUGGESTIVE', 'NONE') for d in diversity.values())
    decision = dict(diversity=diversity, tactics=tactics, recommendation=analyze.recommend(diversity, tactics, data))
    text = analyze.tables(analyze.clean(results), analyze.clean(decision),
                          [dict(criterion='x', result='PASS', evidence='y')])
    assert '42/g20: diversity' in text and 'extended_smoothed' in text
    json.dumps(analyze.clean(results), allow_nan=False)
