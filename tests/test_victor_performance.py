"""Performance work must not change answers: fast paths equal their references.

Also pins the benchmark's ground truth tooling (independent native oracle) and
the new White refutation labels, deadline and session behavior.
"""
import json
from dataclasses import replace
from pathlib import Path
from random import Random
from time import perf_counter

import pytest

import games.connect4.victor.solver as solver_module
from games.connect4.connect4 import Connect4
from games.connect4.victor import Position, SearchBudget, SolverBudget, analyze_position, solve_exact
from games.connect4.victor.nine_rules import analyze_nine_rules, conflict_masks, pairwise_conflicts
from victor_validation.exact_oracle import replay, solve
from victor_validation.native_oracle import OracleUnavailable, solve_histories
from victor_validation import performance_benchmark as bench

SUITE = Path(__file__).resolve().parent.parent / 'docs' / 'victor-performance' / 'suite.json'
EXACT = SearchBudget(nodes=400_000, seconds=None, max_remaining=24, table_entries=400_000)
# Strategic-path tests bypass the opening book (several suite positions are entries).
NINE = SolverBudget(exact=replace(EXACT, nodes=200_000), white_contexts=0, opening_book=False,
                    policy_audit=SearchBudget(nodes=20_000, seconds=None, max_remaining=10))


def random_history(rng, plies):
    """Seeded nonterminal prefix; restarts if every move would end the game."""
    while True:
        game, history = Connect4(), []
        while len(history) < plies:
            options = []
            for c in game.get_valid_moves():
                trial = Connect4(game.board, game.current_player)
                trial.make_move(c)
                if not trial.is_game_over():
                    options.append((c, trial))
            if not options:
                break
            c, game = rng.choice(options)
            history.append(c)
        if len(history) == plies:
            return history, game


@pytest.fixture(scope='module')
def oracle():
    try:
        solve_histories([(3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2)], node_limit=1_000_000)
    except OracleUnavailable as exc:
        pytest.skip(str(exc))
    return solve_histories


@pytest.fixture(scope='module')
def suite():
    return json.loads(SUITE.read_text())


def test_conflict_masks_equal_pairwise_reference_on_random_positions():
    rng = Random(20261008)
    for _ in range(10):
        _, game = random_history(rng, rng.randint(0, 26))
        position = Position.from_board(game.board, game.current_player)
        for defender in (0, 1):
            useful = tuple(e.candidate for e in analyze_nine_rules(position, defender).evidence
                           if e.conditional_solved_groups)[:220]
            fast, count = conflict_masks(useful)
            reference = [1 << i for i in range(len(useful))]
            pairs = pairwise_conflicts(useful)
            for i, j in pairs:
                reference[i] |= 1 << j
                reference[j] |= 1 << i
            assert fast == reference and count == len(pairs)


def test_bitmask_coverage_equals_clause_semantics():
    rng = Random(7)
    for _ in range(12):
        _, game = random_history(rng, rng.randint(0, 30))
        position = Position.from_board(game.board, game.current_player)
        for defender in (0, 1):
            report = analyze_nine_rules(position, defender)
            for e in report.evidence:
                assert e.conditional_solved_groups == tuple(
                    g for g in report.target_groups if any(c.solves(g) for c in e.clauses))


def test_exact_engine_matches_independent_python_oracle():
    rng = Random(11)
    for remaining in (4, 6, 8, 9, 10) * 4:
        history, game = random_history(rng, 42 - remaining)
        p = replay(history)
        expected = solve(p, max_remaining=10, position_budget=1_000_000)
        result = solve_exact(Position.from_board(game.board, game.current_player), EXACT)
        assert result.status == 'exact' and result.value == expected.mover_value
        assert dict(result.move_values) == dict(expected.move_values)


def test_exact_engine_matches_native_oracle_up_to_the_24_cell_ceiling(oracle, suite):
    cases = [p for p in suite['positions'] if 15 <= p['remaining'] <= 24][::4]
    assert cases
    native = oracle([p['history'] for p in cases])
    for p, truth in zip(cases, native):
        game = bench.game_from(p['history'])
        result = solve_exact(Position.from_board(game.board, game.current_player), EXACT)
        assert result.status == 'exact'
        assert truth.status == 'exact' and result.value == truth.value
        assert dict(result.move_values) == truth.move_values
        assert truth.move_values == {int(c): v for c, v in p['move_values'].items()}


def test_exact_values_are_mirror_symmetric():
    rng = Random(3)
    for _ in range(6):
        _, game = random_history(rng, 22)
        mirrored = [list(reversed(row)) for row in game.board]
        a = solve_exact(Position.from_board(game.board, game.current_player), EXACT)
        b = solve_exact(Position.from_board(mirrored, game.current_player), EXACT)
        assert a.status == b.status == 'exact'
        assert {6 - c: v for c, v in a.move_values} == dict(b.move_values)


def test_native_oracle_reports_unknown_instead_of_partial_values(oracle):
    result = oracle([()], node_limit=1_000)[0]
    assert result.status == 'unknown' and result.value is None and result.move_values == {}
    terminal = oracle([(0, 1, 0, 1, 0, 1, 0)])[0]
    assert terminal.status == 'error'


def test_default_exact_cap_is_the_hard_ceiling():
    assert SolverBudget().exact.max_remaining == 24
    with pytest.raises(ValueError):
        SearchBudget(max_remaining=25)


def test_white_refutation_avoidance_is_exploratory_and_pinned():
    # Suite e0-010 (native oracle): columns 2 and 4 win; Negamax-4 prefers 3, which
    # loses and lets Black reach a nine-rule cover. Column 2 is unrefuted.
    game = bench.game_from([3, 2, 0, 4, 1, 1])
    result = analyze_position(game.board, 0, NINE)
    assert result.move == 2 and result.move_kind == 'exploratory_unrefuted'
    assert result.white_refutations == ((3, 'nine_rule'), (2, 'unrefuted'))
    assert result.exact_value is None and result.bound is None and not result.justified_move


def test_e0_006_refutation_avoidance_failure_is_preserved_and_labelled(suite, oracle):
    """Known failure of the default policy, kept because it wins on balance.

    Suite e0-006 is a draw; only column 1 draws. A Black reply to 1 reaches an
    exploratory nine-rule cover (consistent with a draw), so the default policy
    plays unrefuted column 6, which loses. 'certified_only' keeps 1 here but was
    worse overall on the fresh development set (334 vs 362 of 449) and on both
    suites, so it is not the default. The move is never a proof claim.
    """
    position = next(p for p in suite['positions'] if p['id'] == 'e0-006')
    game = bench.game_from(position['history'])
    result = analyze_position(game.board, 0, NINE)
    assert result.move == 6 and result.move_kind == 'exploratory_unrefuted'
    assert result.white_refutations[0] == (1, 'nine_rule')  # exploratory, not certified
    assert result.exact_value is None and result.bound is None and not result.justified_move
    assert position['move_values']['6'] == -1 and position['move_values']['1'] == 0
    certified_only = analyze_position(game.board, 0, replace(NINE, white_refutation='certified_only'))
    assert certified_only.move == 1 and certified_only.move_kind == 'heuristic'
    assert analyze_position(game.board, 0, replace(NINE, white_refutation='off')).move == 1
    with pytest.raises(ValueError):
        replace(NINE, white_refutation='always')
    fresh = oracle([position['history'] + [1]])[0]
    assert fresh.status == 'exact' and fresh.value == 0  # Black-relative: a draw after 1


def test_white_choice_policies_on_a_fixed_scan():
    from games.connect4.victor.solver import white_choice_from_scan
    scan = [(1, 'nine_rule', False), (2, 'certified', False), (6, 'unrefuted', False),
            (4, 'unrefuted', True)]
    assert white_choice_from_scan(scan, 1, 'off') == (1, None)
    assert white_choice_from_scan(scan, 1, 'first_unrefuted') == (4, 'exploratory_white_context')
    assert white_choice_from_scan(scan[:3], 1, 'first_unrefuted') == (6, 'exploratory_unrefuted')
    assert white_choice_from_scan(scan[:3], 1, 'certified_only') == (1, None)
    assert white_choice_from_scan(scan[:3], 1, 'positive_evidence') == (1, None)
    assert white_choice_from_scan(scan, 1, 'positive_evidence') == (4, 'exploratory_white_context')
    assert white_choice_from_scan([(1, 'unchecked', False)], 1, 'first_unrefuted') == (1, None)


def test_certified_refutations_never_mark_a_winning_white_move(suite):
    # Suite e0-016: every White move is refuted by a CL/BI/VE certificate; the
    # native oracle confirms none of them wins (position value is a draw).
    position = next(p for p in suite['positions'] if p['id'] == 'e0-016')
    game = bench.game_from(position['history'])
    result = analyze_position(game.board, 0, NINE)
    certified = [c for c, status in result.white_refutations if status == 'certified']
    assert len(certified) == 7
    assert all(position['move_values'][str(c)] <= 0 for c in certified)


def test_analysis_reuses_a_supplied_exact_result(monkeypatch):
    game = bench.game_from([3, 2, 0, 4, 1, 1])
    exact = solve_exact(Position.from_board(game.board, 0), replace(EXACT, max_remaining=14))
    monkeypatch.setattr(solver_module, 'solve_exact', lambda *a, **k: pytest.fail('repeated'))
    result = analyze_position(game.board, 0, replace(NINE, cover_nodes=0), exact=exact)
    assert result.exact is exact and result.move in game.get_valid_moves()


def test_deadline_bounds_strategic_work_and_still_returns_a_legal_move():
    game = bench.game_from([3, 3])
    budget = replace(NINE, white_contexts=4, deadline=0.05)
    started = perf_counter()
    result = analyze_position(game.board, 0, budget)
    assert perf_counter() - started < 2.0  # Generous: one bounded step may overshoot.
    assert result.move in game.get_valid_moves() and result.deadline_reached
    assert 'deadline reached' in result.reason
    for bad in (0, -1, float('inf'), float('nan'), True, '1'):
        with pytest.raises(ValueError):
            SolverBudget(deadline=bad)


def test_suite_ground_truth_is_well_formed(suite):
    assert suite['oracle_unknown'] == 0 and len(suite['positions']) == 371
    ids = set()
    for p in suite['positions']:
        game = bench.game_from(p['history'])
        values = {int(c): v for c, v in p['move_values'].items()}
        assert sorted(values) == sorted(game.get_valid_moves())
        assert p['value'] == max(values.values()) and p['decisive']
        assert p['optimal'] == [c for c, v in values.items() if v == p['value']]
        assert p['mover'] == game.current_player and p['id'] not in ids
        ids.add(p['id'])
        if p['subset'] == 'negamax_hard':
            assert p['quiet'] and not p['negamax4_optimal']


def test_benchmark_decisions_are_legal_for_every_configuration(suite):
    history = suite['positions'][0]['history']
    legal = bench.game_from(history).get_valid_moves()
    for config in bench.CONFIGS:
        assert bench.decide(config, history)['move'] in legal


def test_equal_value_exact_moves_are_ranked_by_negamax():
    """Lost/drawn positions keep an exact-optimal move but prefer resilient ones."""
    from games.connect4.agents.negamax_agent import NegamaxAgent
    rng, checked = Random(19), 0
    while checked < 12:
        _, game = random_history(rng, rng.randint(26, 32))
        result = analyze_position(game.board, game.current_player, NINE)
        if result.move_kind != 'exact' or not result.exact.move_values:
            continue
        values = dict(result.exact.move_values)
        best = [c for c, v in result.exact.move_values if v == result.exact_value]
        assert values[result.move] == result.exact_value
        if len(best) > 1:
            scores = NegamaxAgent(NINE.fallback_depth).score_moves(game)
            assert result.move == max(best, key=lambda c: scores[c])
            checked += 1


def test_baseclaim_policy_spare_gap_is_preserved_and_not_a_cover_contradiction():
    """Composite benchmark counterexample to the concrete spare policy (not the cover).

    The native oracle values this White-to-move position as a Black win, so the
    Baseclaim+Claimeven cover is not contradicted. After White g4, Black's spare
    g5 and White g6, every playable square is a forbidden rule square. The legacy
    policy (no retiring spares) is preserved here; the current policy retires the
    Baseclaim with b5 (both of its groups contain b5) and the complete adversarial
    replay then verifies the concrete policy non-losing on this board.
    """
    from games.connect4.victor.execution import NineRulePolicy
    from games.connect4.victor.nine_rules import search_nine_rule_cover
    history = [0, 1, 1, 1, 4, 6, 1, 4, 3, 3, 3, 3, 5, 2, 4, 4, 0, 4, 0, 4, 3, 5, 5, 0,
               6, 6, 2, 2, 5, 3]
    game = bench.game_from(history)
    witness = search_nine_rule_cover(Position.from_board(game.board, 0)).witness
    assert {e.candidate.rule.value for e in witness.evidence} == {'baseclaim', 'claimeven'}
    audit_budget = SearchBudget(nodes=100_000, seconds=None, max_remaining=12)
    legacy = NineRulePolicy(witness, retiring_spares=False)
    assert legacy.select((6,)).column == 6
    stuck = legacy.select((6, 6, 6))
    assert stuck.status == 'no_permitted_spare' and stuck.column is None
    audit = legacy.audit(audit_budget)
    assert audit.status == 'unsupported_policy' and audit.detail == 'no_permitted_spare'
    policy = NineRulePolicy(witness)
    assert policy.select((6,)).column == 6
    retiring = policy.select((6, 6, 6))
    assert retiring.status == 'selected' and retiring.kind == 'retiring_spare'
    assert retiring.column == 1  # b5: blocks a6-b5-c4-d3 and a5-b5-c5-d5
    assert policy.audit(audit_budget).status == 'verified_policy_nonloss'
    exact = solve_exact(Position.from_board(game.board, 0), EXACT)
    assert exact.status == 'exact' and exact.value == -1  # Black wins; cover not contradicted.


def test_retiring_spare_requires_every_owning_rule_to_retire():
    """c4 blocks only one Baseclaim group, a5 neither: neither may be a retiring spare."""
    from games.connect4.victor.execution import NineRulePolicy
    from games.connect4.victor.nine_rules import search_nine_rule_cover
    history = [0, 1, 1, 1, 4, 6, 1, 4, 3, 3, 3, 3, 5, 2, 4, 4, 0, 4, 0, 4, 3, 5, 5, 0,
               6, 6, 2, 2, 5, 3]
    witness = search_nine_rule_cover(
        Position.from_board(bench.game_from(history).board, 0)).witness
    policy = NineRulePolicy(witness)
    state = policy.select((6, 6, 6)).state
    allowed = []
    for c in state.board.legal():
        decision_board = state.board.drop(c)
        pruned = policy._prune(type(state)(decision_board, state.rules))
        owners = [i for i, r in enumerate(state.rules)
                  if any(sq.column == c for o in r.obligations for sq in o.squares)]
        if owners and all(pruned.rules[i].phase == 'retired' for i in owners):
            allowed.append(c)
    assert sorted(allowed) == [1, 5]  # b5 (Baseclaim) and f5 (Claimeven f5/f6)
    with pytest.raises(ValueError):
        NineRulePolicy(witness, retiring_spares='yes')
