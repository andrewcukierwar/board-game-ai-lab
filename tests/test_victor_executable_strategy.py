"""Executable sigma_R vs independent obligations, two exact oracles, all spares.

All adversarial failures carry the initial board/rules and full continuation.
The independent audit and bitboard oracle remain untouched test-only models.
"""
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from random import Random

import pytest

from games.connect4.victor import certificate as C
from games.connect4.victor import strategy as S
from test_victor_certificates import (
    ALL_BLOCKED, BI_ONLY, CLAIMODD, CL_LOWER_SPARE, CL_ONLY, FLOATING_BI,
    MIXED, MIXED3, OVERLAP, UNREACHABLE, VALID, cert_for, instances,
    matrix_cert, mirror_rules, moves,
)
from victor_validation import independent_audit as audit
from victor_validation.exact_oracle import EndgamePosition, solve
from victor_validation.spare_policies import explore


def select(fixture, history, **kwargs):
    return S.select_black_move(S.StrategyRequest(cert_for(*fixture), history), **kwargs)


def no_move(result, status):
    assert result.status is status, result
    assert result.column is result.square is result.response is result.triggering_rule is None


@pytest.mark.parametrize('fixture,white,square,kind', [
    (CL_ONLY, 1, 'b6', 'claimeven'),
    (BI_ONLY, 3, 'f6', 'baseinverse'), (BI_ONLY, 5, 'd6', 'baseinverse'),
    (VALID['ve_only'], 3, 'd3', 'vertical'),
    (MIXED3, 1, 'c6', 'baseinverse'), (MIXED3, 2, 'b6', 'baseinverse'),
    (MIXED3, 3, 'd6', 'claimeven'), (MIXED3, 4, 'e6', 'claimeven'),
    (MIXED, 5, 'f2', 'claimeven'),
])
def test_forced_response(fixture, white, square, kind):
    out = select(fixture, (white,))
    assert out.status is S.StrategyStatus.MOVE_SELECTED
    assert (out.square, out.column, out.response) == (
        square, 'abcdefg'.index(square[0]), S.ResponseKind.FORCED)
    assert out.triggering_rule.kind == kind
    assert out.triggering_rule in out.state.active_rules
    assert out.state.player_to_move == 1
    assert out.state.continuation == (white,)


def test_mandatory_response_overrides_an_immediate_black_win():
    board = audit.replay(moves(MIXED3[0]))
    after = board.play(board.landing(3))  # White d5 activates CL d5-d6.
    assert after.wins_at(1, after.landing(1))  # b6 would win immediately.
    assert not after.wins_at(1, after.landing(3))
    assert select(MIXED3, (3,)).square == 'd6'
    no_move(select(MIXED3, (3, 1)), S.StrategyStatus.STRATEGY_VIOLATED)


@pytest.mark.parametrize('fixture,history,square', [
    (MIXED3, (0,), 'a4'), (MIXED3, (6,), 'b6'),
    (VALID['ve_only'], (5,), 'd2'), (ALL_BLOCKED, (5,), 'f6'),
    (MIXED, (5, 5, 5), 'e6'),
])
def test_even_row_spares_with_partially_filled_columns(fixture, history, square):
    out = select(fixture, history)
    assert out.status is S.StrategyStatus.MOVE_SELECTED
    assert (out.square, out.response, out.triggering_rule) == (square, S.ResponseKind.SPARE, None)
    assert int(square[1]) % 2 == 0


def test_deterministic_spare_tie_break_and_rejection_of_other_permitted_spares():
    after = audit.replay(moves(MIXED3[0]))
    after = after.play(after.landing(6))
    assert [audit.name(s) for s in after.landings() if (s[1] + 1) % 2 == 0] == [
        'b6', 'c6', 'g6']
    assert select(MIXED3, (6,)).square == 'b6'
    # c6 is a sound permissive spare, but it is not this deterministic strategy.
    out = select(MIXED3, (6, 2))
    no_move(out, S.StrategyStatus.STRATEGY_VIOLATED)
    assert 'required even_row_spare at b6' in out.findings[0].detail


def test_repeated_vertical_activations_retire_only_the_touched_instance():
    fixture = VALID['ve_only']
    first = select(fixture, (3, 3, 3))
    assert first.square == 'd5'
    assert [r.label for r in first.state.retired_rules] == ['vertical:d2-d3']
    assert [r.label for r in first.state.active_rules] == ['vertical:d4-d5']
    last = select(fixture, (3, 3, 3, 3, 3))
    assert last.square == 'f6' and last.response is S.ResponseKind.SPARE
    assert len(last.state.retired_rules) == 2 and last.state.active_rules == ()
    assert [m.square for m in last.state.trace] == ['d2', 'd3', 'd4', 'd5', 'd6']
    assert [m.newly_retired for m in last.state.trace if m.player == 1] == [
        (instances(fixture[1])[0],), (instances(fixture[1])[1],)]


def test_multi_turn_mixed_rules_and_repeated_cl_retirement():
    # Two CL activations, a proactive BI spare, then VE's mandatory reply.
    history = (2, 2, 5, 5, 5, 4, 5, 5)
    out = select(MIXED, history)
    no_move(out, S.StrategyStatus.UNSUPPORTED)  # Complete, nonterminal White turn.
    assert out.state.active_rules == ()
    assert out.state.retired_rules == instances(MIXED[1])
    black = [m for m in out.state.trace if m.player == 1]
    assert [m.square for m in black] == ['c6', 'f2', 'e6', 'f5']
    assert [m.response for m in black] == [S.ResponseKind.FORCED, S.ResponseKind.FORCED,
                                          S.ResponseKind.SPARE, S.ResponseKind.FORCED]
    assert [r.kind for m in black for r in m.newly_retired] == [
        'claimeven', 'claimeven', 'baseinverse', 'vertical']
    assert len(select(MIXED3, (3,)).state.active_rules) == 4


def test_rule_identity_is_bound_and_not_reenumerated_during_replay():
    cert = cert_for(*MIXED)
    request = S.StrategyRequest(cert, (2, 2, 5))
    original = S.select_black_move(request)
    assert original.state.active_rules + original.state.retired_rules != cert.rules
    assert original.state.retired_rules == (cert.rules[1],)
    assert original.triggering_rule == cert.rules[2]
    reordered = S.select_black_move(replace(request, certificate=replace(
        cert, rules=tuple(reversed(cert.rules)))))
    assert reordered.square == original.square
    assert reordered.certificate_digest != original.certificate_digest
    assert reordered.state.active_rules == tuple(reversed(original.state.active_rules))


@pytest.mark.parametrize('history,label', [
    ((0, 0), 'vertical:a4-a5'), ((6, 1), 'baseinverse:b6-c6'),
])
def test_proactive_retirement_and_black_terminal_stop(history, label):
    out = select(MIXED3, history)
    no_move(out, S.StrategyStatus.TERMINAL)
    assert [r.label for r in out.state.retired_rules] == [label]
    assert out.state.trace[-1].response is S.ResponseKind.SPARE
    assert out.state.trace[-1].newly_retired == out.state.retired_rules
    no_move(select(MIXED3, history + (0,)), S.StrategyStatus.ILLEGAL_CONTINUATION)


def test_full_board_terminal_and_empty_rules():
    for fixture, history in ((VALID['ve_only'], (3, 3, 3, 3, 3, 5)),
                             (ALL_BLOCKED, (5, 5)), (CL_ONLY, (1, 1))):
        out = select(fixture, history)
        no_move(out, S.StrategyStatus.TERMINAL)
        assert not any(' ' in r for r in out.state.board)
        assert not out.state.active_rules
    out = select(ALL_BLOCKED, (5,))
    assert out.state.active_rules == out.state.retired_rules == ()


@pytest.mark.parametrize('history', [(), (3, 3)])
def test_white_turn_endpoint_is_unsupported(history):
    no_move(select(MIXED3, history), S.StrategyStatus.UNSUPPORTED)


@pytest.mark.parametrize('history', [None, '3', [3], (True,), (1.0,), (-1,), (7,),
                                     (None,), (5,), (0,) * 43])
def test_illegal_continuation_data_or_full_column(history):
    no_move(select(MIXED3, history), S.StrategyStatus.ILLEGAL_CONTINUATION)


@pytest.mark.parametrize('field', ['schema', 'theorem', 'ruleset', 'rule_model',
                                  'compatibility_model'])
def test_unsupported_certificate_versions(field):
    cert = replace(cert_for(*MIXED3), **{field: 'future-v2'})
    no_move(S.select_black_move(S.StrategyRequest(cert, (3,))), S.StrategyStatus.UNSUPPORTED)


def test_context_tampering_and_exact_board_binding():
    cert = cert_for(*MIXED3)
    no_move(S.select_black_move(S.StrategyRequest(replace(cert, defender=0), (3,))),
            S.StrategyStatus.UNSUPPORTED)
    for forged in (replace(cert, board_digest='sha256:forged'),
                   replace(cert, player_to_move=True), replace(cert, rules=()),
                   replace(cert, rules=(C.RuleInstance('claimeven', ('d4', 'd5')),)),
                   replace(cert, replay=(0,)), replace(cert, assignments=())):
        no_move(S.select_black_move(S.StrategyRequest(forged, (3,))),
                S.StrategyStatus.INVALID_CERTIFICATE)
    board = [list(r) for r in cert.board]
    # Swap in a board with the same counts but a distinct binding.
    board[5][0], board[5][2] = board[5][2], board[5][0]
    no_move(S.select_black_move(S.StrategyRequest(replace(cert, board=board), (3,))),
            S.StrategyStatus.INVALID_CERTIFICATE)


def test_saved_verdict_and_result_are_never_authority():
    cert = cert_for(*MIXED3)
    verdict = C.verify_certificate(cert)
    no_move(S.select_black_move(S.StrategyRequest(verdict, (3,))),
            S.StrategyStatus.INVALID_CERTIFICATE)
    no_move(S.select_black_move(select(MIXED3, (3,))), S.StrategyStatus.UNSUPPORTED)


@pytest.mark.parametrize('certificate', [None, {}, object(),
    replace(cert_for(*MIXED3), rules=None),
    replace(cert_for(*MIXED3), rules=cert_for(*MIXED3).rules * 6),
    replace(cert_for(*MIXED3), rules=(C.RuleInstance('claimeven', None),)),
    replace(cert_for(*MIXED3), replay=(0,) * 43),
    replace(cert_for(*MIXED3), assignments=((None,),)),
])
def test_malformed_certificate_cannot_escape_as_a_move(certificate):
    no_move(S.select_black_move(S.StrategyRequest(certificate, (3,))),
            S.StrategyStatus.INVALID_CERTIFICATE)


def test_reverification_private_snapshot_and_immutable_outputs(monkeypatch):
    original = cert_for(*MIXED3)
    board = [list(r) for r in original.board]
    squares = [list(r.squares) for r in original.rules]
    cert = replace(original, board=board, rules=[
        C.RuleInstance(r.kind, sq) for r, sq in zip(original.rules, squares)])
    calls = []
    verify = C.verify_certificate

    def verify_then_mutate(snapshot, **kwargs):
        calls.append(snapshot)
        verdict = verify(snapshot, **kwargs)
        board[5][0] = 'X'
        squares[0][0] = 'a1'
        return verdict

    monkeypatch.setattr(C, 'verify_certificate', verify_then_mutate)
    out = S.select_black_move(S.StrategyRequest(cert, (3,)))
    assert out.square == 'd6' and len(calls) == 1
    assert calls[0].board == original.board and calls[0].rules == original.rules
    assert type(out.state.board) is type(out.state.active_rules) is tuple
    with pytest.raises(FrozenInstanceError):
        out.column = 0
    with pytest.raises(FrozenInstanceError):
        out.state.active_rules[0].squares = ('a1', 'a2')
    no_move(S.select_black_move(S.StrategyRequest(cert, (3,))),
            S.StrategyStatus.INVALID_CERTIFICATE)
    assert len(calls) == 2


def test_previous_missed_mandatory_reply_rejected_before_later_terminal():
    out = select(MIXED3, (3, 1, 0))
    no_move(out, S.StrategyStatus.STRATEGY_VIOLATED)
    assert 'ply 1' in out.findings[0].detail and 'claimeven:d5-d6' in out.findings[0].detail


def test_real_cl_lower_violation_preserves_failure_history():
    board = audit.replay(moves(CL_LOWER_SPARE[0]))
    history = (3, 3, 3, 2, 2)  # d3,d4,d5,c5,c6: White wins after forbidden spare.
    for c in history:
        board = board.play(board.landing(c))
    assert board.has_four(0)
    status, detail, _ = explore(audit.replay(moves(CL_LOWER_SPARE[0])),
                                CL_LOWER_SPARE[1], 'unrestricted')
    assert status == 'violated' and detail[1] == history
    no_move(select(CL_LOWER_SPARE, history), S.StrategyStatus.STRATEGY_VIOLATED)
    assert select(CL_LOWER_SPARE, history[:3]).square != 'c5'


@pytest.mark.parametrize('fixture,code', [(OVERLAP, 'H3.overlap'),
                                          (CLAIMODD, 'H2.parity'),
                                          (FLOATING_BI, 'H2.not_playable')])
def test_real_relaxed_hypothesis_failures_never_select(fixture, code):
    out = select(fixture, (3,))
    no_move(out, S.StrategyStatus.INVALID_CERTIFICATE)
    assert code in {f.code for f in out.findings}
    board = audit.replay(moves(fixture[0]))
    assert audit.white_can_force_win(board)[0] is True
    if board.empty_count <= 10:
        oracle = solve(EndgamePosition.from_board(tuple(map(tuple, board.matrix())), 0),
                       max_remaining=10, position_budget=1_000_000)
        assert oracle.value_for(0) == 1


def test_history_provenance_is_separate_from_continuation_verification():
    cert = cert_for(*MIXED3, replay=moves(MIXED3[0]))
    backed = S.select_black_move(S.StrategyRequest(cert, (3,)))
    synthetic = select(MIXED3, (3,))
    assert backed.starting_history_backed is True
    assert synthetic.starting_history_backed is False
    assert backed.state == synthetic.state and backed.square == synthetic.square
    assert backed.certificate_digest != synthetic.certificate_digest
    board = audit.Board.from_matrix(UNREACHABLE)
    assert audit.reachable(board) is False
    rules = audit.find_cover(board)[1]
    unreachable = matrix_cert(UNREACHABLE, instances(rules))
    for square in board.landings():
        out = S.select_black_move(S.StrategyRequest(unreachable, (square[0],)))
        assert out.status is S.StrategyStatus.MOVE_SELECTED
        assert out.starting_history_backed is False


def test_reflection_of_forced_responses_and_physical_spare_tie_break():
    mirrored = (''.join(str(6 - c) for c in moves(MIXED3[0])), mirror_rules(MIXED3[1]))
    for c in (1, 2, 3, 4):
        original, reflected = select(MIXED3, (c,)), select(mirrored, (6 - c,))
        assert reflected.column == 6 - original.column
        assert reflected.response is original.response is S.ResponseKind.FORCED
    original, reflected = select(MIXED3, (6,)), select(mirrored, (0,))
    assert (original.square, reflected.square) == ('b6', 'a6')
    # Physical lowest-column spares intentionally do not commute with reflection.
    assert reflected.column != 6 - original.column


def test_cutoffs_never_supply_a_move_and_bad_budgets_raise():
    no_move(select(MIXED3, (3,), certificate_work_budget=0), S.StrategyStatus.UNKNOWN)
    no_move(select(MIXED3, (3,), replay_budget=0), S.StrategyStatus.UNKNOWN)
    no_move(select(MIXED3, (3, 3, 4), replay_budget=2), S.StrategyStatus.UNKNOWN)
    assert select(MIXED3, (3, 3, 4), replay_budget=3).square == 'e6'
    for key in ('replay_budget', 'certificate_work_budget'):
        for value in (-1, True, 0.5, None):
            with pytest.raises(ValueError):
                select(MIXED3, (3,), **{key: value})


def test_unexpected_response_failure_is_explicit(monkeypatch):
    def broken(*args):
        raise ValueError('mandatory response is not a landing square')
    monkeypatch.setattr(S, '_response', broken)
    out = select(MIXED3, (3,))
    no_move(out, S.StrategyStatus.INVARIANT_FAILURE)
    assert out.findings[0].code == 'invariant.response'


def independent_reply(before, after, raw, white_square):
    """Independent mechanical specification, on audit bitboards BEFORE White."""
    live = []
    for k, a, b in raw:
        black_cells = audit.bit(*b) if k == 'CL' else audit.bit(*a) | audit.bit(*b)
        if not before.black & black_cells:
            assert before.is_empty(a) and before.is_empty(b)
            live.append((k, a, b))
    touched = [r for r in live if white_square in r[1:]]
    assert len(touched) <= 1
    if touched:
        k, a, b = touched[0]
        return (b if white_square == a else a), S.ResponseKind.FORCED
    permitted = [s for s in after.landings() if (s[1] + 1) % 2 == 0]
    assert permitted
    return min(permitted), S.ResponseKind.SPARE


def execute_every_white_continuation(cert, board, raw, *, limit=10_000):
    """Complete bounded tree, no memo pruning: every path replayed independently."""
    stats = {'selections': 0, 'black_wins': 0, 'full_boards': 0}
    oracle_states = set()

    def visit(before, history):
        for white_square in before.landings():
            path = history + (white_square[0],)
            context = (cert.board, raw, cert.replay, path)
            assert not before.wins_at(0, white_square), context
            after = before.play(white_square)
            result = S.select_black_move(S.StrategyRequest(cert, path))
            stats['selections'] += 1
            assert stats['selections'] <= limit, context
            assert result.status is S.StrategyStatus.MOVE_SELECTED, (context, result)
            reply, mode = independent_reply(before, after, raw, white_square)
            assert (result.square, result.response) == (audit.name(reply), mode), context
            assert result.column == reply[0] and reply in after.landings(), context
            assert result.state.board == tuple(map(tuple, after.matrix())), context
            exact = EndgamePosition.from_board(result.state.board, 1)
            assert result.column in exact.legal_columns, context
            if after.key() not in oracle_states:
                oracle_states.add(after.key())
                bitboard = solve(exact, max_remaining=10, position_budget=1_000_000)
                assert bitboard.status == 'exact', (context, bitboard)
                audit_value = audit.exact_value(after)[0]
                assert audit_value == bitboard.value_for(0) and audit_value <= 0, context
                assert dict(bitboard.move_values)[result.column] >= 0, context
            path += (result.column,)
            won = after.wins_at(1, reply)
            completed = after.play(reply)
            # Check state transitions and the invariant independently of production.
            for rule in raw:
                k, a, b = rule
                retired = completed.black & (audit.bit(*b) if k == 'CL' else
                                              audit.bit(*a) | audit.bit(*b))
                if k == 'CL':
                    assert not completed.black & audit.bit(*a), context
                if retired:
                    for line in audit.targets(board):
                        if audit.coverage_mask(rule) & line == audit.coverage_mask(rule):
                            assert completed.black & line, context
                else:
                    assert completed.is_empty(a) and completed.is_empty(b), context
            if won or completed.empty_count == 0:
                ended = S.select_black_move(S.StrategyRequest(cert, path))
                no_move(ended, S.StrategyStatus.TERMINAL)
                assert ended.state.board == tuple(map(tuple, completed.matrix())), context
                stats['black_wins' if won else 'full_boards'] += 1
            else:
                visit(completed, path)
    visit(board, ())
    return stats


def test_complete_execution_against_all_white_continuations_and_independent_oracles():
    cases = []
    for history, raw in (CL_ONLY, BI_ONLY, VALID['ve_only'], MIXED, ALL_BLOCKED, CL_LOWER_SPARE):
        board = audit.replay(moves(history))
        cases.append((cert_for(history, raw, replay=moves(history)), board, raw))
    unreachable = audit.Board.from_matrix(UNREACHABLE)
    raw = audit.find_cover(unreachable)[1]
    cases.append((matrix_cert(UNREACHABLE, instances(raw)), unreachable, raw))
    rng = Random(6442)
    for empties in (4, 6, 8) * 40:
        history = audit.sample_history(rng, 42 - empties, follow_up=0.8)
        if history is None:
            continue
        board = audit.replay(history)
        for raw in audit.enumerate_covers(board, limit=2, budget=20_000):
            cases.append((matrix_cert(board.matrix(), instances(raw), replay=history), board, raw))
    totals = {'selections': 0, 'black_wins': 0, 'full_boards': 0}
    for cert, board, raw in cases:
        context = (cert.board, raw, cert.replay)
        assert not audit.check_hypotheses(board, raw), context
        assert audit.white_can_force_win(board)[0] is False, context
        oracle = solve(EndgamePosition.from_board(cert.board, 0), max_remaining=10,
                       position_budget=1_000_000)
        assert oracle.status == 'exact' and oracle.value_for(0) <= 0, context
        assert audit.exact_value(board)[0] == oracle.value_for(0), context
        assert audit.check_strategy(board, raw)[:2] == ('held', None), context
        for policy in ('permissive', 'even_row'):
            outcome = explore(board, raw, policy)
            assert outcome[:2] == ('held', None), (context, policy, outcome)
            assert outcome[2]['invariant_breaks'] == 0, context
        stats = execute_every_white_continuation(cert, board, raw)
        for key, value in stats.items():
            totals[key] += value
    print({'cases': len(cases), **totals})
    assert len(cases) == 22
    assert totals == {'selections': 299, 'black_wins': 34, 'full_boards': 96}


def test_research_api_has_no_game_value_and_no_public_integration():
    result = select(MIXED3, (3,))
    for forbidden in ('value', 'outcome', 'winner', 'draw', 'best_move', 'solved',
                      'provisional_bound'):
        assert not hasattr(result, forbidden)
    assert result.game_theoretic_certification == 'not_certified_pending_independent_review'
    assert result.theorem_id == C.THEOREM_ID
    assert result.certificate_digest == C.verify_certificate(cert_for(*MIXED3)).certificate_digest
    root = Path(__file__).resolve().parents[1]
    for folder in ('api', 'games/connect4/agents'):
        for path in (root / folder).rglob('*.py'):
            assert 'victor.strategy' not in path.read_text(), path
    assert 'strategy' not in (root / 'games/connect4/victor/__init__.py').read_text()
