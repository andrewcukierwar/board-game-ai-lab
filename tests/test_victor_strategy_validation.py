"""Exploratory falsification and response scheduling, never bound acceptance."""
from dataclasses import replace
import json

import pytest

from games.connect4.victor import Position, search_covering_set
from victor_validation.exact_oracle import OracleResult, replay, solve
from victor_validation.report import FIXTURES, build_report, summarize
from victor_validation.strategic_harness import compare_history, examine_strategy, generated_histories
from victor_validation import reference_game as reference

DATA = json.loads(FIXTURES.read_text())
CASES = {case['id']: case['moves'] for case in DATA['inputs']['cases']}


def witness_for(moves):
    p = replay(moves)
    return p, search_covering_set(Position.from_board(p.board, p.turn)).witness


def test_reproducible_bounded_survey_curated_reflections_and_resource_checks():
    actual = json.loads(json.dumps(build_report(DATA['inputs'])))
    # Includes exact boards, replays, every move value, candidate/group identities,
    # status distinctions, execution counts and digests of all 513 sampled cases.
    assert actual == DATA['expected'], json.dumps({
        'potential_counterexamples': actual['generated_summary']['potential_counterexamples']
        + actual['curated_summary']['potential_counterexamples'],
    }, indent=2)
    assert actual['generated_summary']['positions'] == 513
    assert actual['generated_summary']['potential_counterexamples'] == []
    assert actual['curated_summary']['potential_counterexamples'] == []
    for original, mirror in zip(actual['curated'][::2], actual['curated'][1::2]):
        assert original['board'] == [row[::-1] for row in mirror['board']]
        assert original['white_value'] == mirror['white_value']
        assert original['black_value'] == mirror['black_value']
        assert original['cover_status'] == mirror['cover_status']
        assert sorted((6 - c, v) for c, v in original['move_values_for_white']) == [
            tuple(pair) for pair in mirror['move_values_for_white']]
        assert original['interpretation'] == 'exploratory_implication_check_outcome_uncertified'


@pytest.mark.parametrize('name', ['6b1-cover-black-win', 'claim-draw', 'claim-base-draw',
                                 'base-black-win', 'vertical-black-win', 'claim-vertical-black-win'])
def test_bounded_set_valued_execution_all_adversarial_moves(name):
    p, witness = witness_for(CASES[name])
    result = examine_strategy(p, witness)
    assert result.status == 'bounded_execution_checked_outcome_uncertified'
    assert result.white_edges and result.forced_reply_edges and result.spare_reply_edges
    if name in ('base-black-win', 'vertical-black-win', 'claim-vertical-black-win'):
        assert result.proactive_pair_edges > 0
    assert not hasattr(result, 'bound') and not hasattr(result, 'accepted')


def test_strategy_cutoffs_never_report_execution_success():
    p, witness = witness_for(CASES['claim-base-draw'])
    complete = examine_strategy(p, witness)
    for budget in (0, 1, complete.visited_positions - 1):
        result = examine_strategy(p, witness, position_budget=budget)
        assert result.status == 'unknown_position_budget' and result.visited_positions == budget
    assert examine_strategy(p, witness, position_budget=complete.visited_positions).status == complete.status
    assert examine_strategy(p, witness, max_remaining=4).status == 'unknown_remaining_cap'
    with pytest.raises(ValueError):
        examine_strategy(p, witness, position_budget=True)
    with pytest.raises(ValueError):
        examine_strategy(p, witness, max_remaining=11)


def test_strategy_rejects_unverified_witness_and_wrong_exact_position():
    p, w = witness_for(CASES['claim-base-draw'])
    with pytest.raises(ValueError, match='verified'):
        examine_strategy(p, replace(w, evidence=()))
    other = replay(CASES['claim-draw'])
    with pytest.raises(ValueError, match='verified'):
        examine_strategy(other, w)
    with pytest.raises(ValueError, match='White-to-move'):
        examine_strategy(p.drop(p.legal_columns[0]), w)


def test_strategy_execution_does_not_use_exact_oracle_move_evaluation(monkeypatch):
    p, w = witness_for(CASES['claim-base-draw'])
    def forbidden(*args, **kwargs):
        raise AssertionError('strategy reference reused oracle move evaluation')
    monkeypatch.setattr('victor_validation.strategic_harness.solve', forbidden)
    monkeypatch.setattr(type(p), 'drop', forbidden)
    assert examine_strategy(p, w).status == 'bounded_execution_checked_outcome_uncertified'


def test_covered_draw_can_be_lost_when_black_disobeys_response_obligations():
    moves = CASES['claim-base-draw']
    root = replay(moves)
    assert solve(root).value_for(0) == 0
    # White d5 triggers BI b5-d5. Black instead takes protected CL c5.
    # White c6 completes c6-d5-e4-f3. This is a wrong Black response,
    # NOT a White forced win at the root or a counterexample to a valid strategy.
    bad = replay(moves + [3, 2, 2])
    assert bad.winner == reference.winner(bad.board) == 0
    assert solve(replay(moves + [3, 2])).value_for(0) == 1
    correct = replay(moves + [3, 1])
    assert solve(correct).value_for(1) >= 0
    assert compare_history(moves)['potential_counterexample'] is False


def test_immediate_threats_forced_replies_and_parity_labels_are_real():
    p = replay(CASES['no-cover-forced-draw'])
    assert [c for c, v in solve(p).move_values if v >= 0] == [5]
    assert not any(reference.winner(reference.drop(p.board, 0, c)) == 0 for c in p.legal_columns)
    for c in (0, 6):
        q = p.drop(c)
        assert any(q.drop(d).winner == 1 for d in q.legal_columns)
    # This constructed fixture has an unsupported odd diagonal threat through d5.
    odd = replay(CASES['odd-threat-forced-loss'])
    assert odd.board[1][3] == ' ' and odd.board[2][3] == ' '
    assert (6 - 1) % 2 == 1  # d5 is odd, not directly playable.
    assert odd.landing(3) != (1, 3)
    # Independently find the containing White three-stone line by endpoints.
    assert any(all(odd.board[1 + i * dr][3 + i * dc] == 'X' for i in (1, 2, 3))
               for dr, dc in [(1, 0), (1, 1), (1, -1)]
               if 0 <= 1 + 3 * dr < 6 and 0 <= 3 + 3 * dc < 7)
    assert solve(odd).value_for(1) == 1  # An odd threat alone does not decide value.


def test_coverage_failure_and_exhausted_budget_allow_all_exact_values():
    names = ['immediate-white-win', 'no-cover-forced-draw', '6b1-no-cover-black-win']
    reports = [compare_history(CASES[n]) for n in names]
    assert [r['white_value'] for r in reports] == [1, 0, -1]
    assert all(r['cover_status'] == 'no_cover_in_supported_universe' for r in reports)
    exhausted = compare_history(CASES['claim-base-draw'], cover_budget=1)
    assert exhausted['cover_status'] == 'unknown_budget_exhausted'
    assert exhausted['white_value'] == 0 and exhausted['verification'] is None


def test_harness_preserves_potential_counterexample_instead_of_relabelling(monkeypatch):
    # Synthetic oracle contradiction tests reporting only, never mathematical data.
    def contrary(position, **kwargs):
        return OracleResult('exact', 0, 1, (), 1, 0, 8, 100_000)
    monkeypatch.setattr('victor_validation.strategic_harness.solve', contrary)
    record = compare_history(CASES['claim-base-draw'])
    assert record['potential_counterexample'] is True
    assert record['white_value'] == 1 and record['verification'] == 'coverage_verified_outcome_uncertified'
    assert record['moves'] == CASES['claim-base-draw']
    assert summarize([record])['potential_counterexamples'] == [record]


def test_generated_sampler_bounds_determinism_and_legal_replay():
    histories = generated_histories(seed=1988, attempts=4, plies=34)
    assert histories == generated_histories(seed=1988, attempts=4, plies=34)
    assert len(histories) <= 4
    assert all(len(h) == 34 and replay(h).turn == 0 and not replay(h).terminal for h in histories)
    for kwargs in ({'attempts': 257}, {'attempts': True}, {'plies': 0}, {'plies': 35}):
        with pytest.raises(ValueError):
            generated_histories(**kwargs)


def test_source_6_1_is_coverage_only_and_early_exact_search_stays_capped():
    result = compare_history([2, 3, 3, 3, 3, 3, 3, 4])
    assert result['verification'] == 'coverage_verified_outcome_uncertified'
    assert result['oracle_status'] == 'unknown_remaining_cap' and result['white_value'] is None
    assert result['execution']['status'] == 'unknown_remaining_cap'
    with pytest.raises(ValueError):
        compare_history([0])
    with pytest.raises(ValueError):
        compare_history([0, 1, 0, 1, 0, 1, 0])
