"""Phase 6B.5 independent assurance probes; production is intentionally unchanged.

Strict xfails preserve confirmed boundary defects, not theorem counterexamples.
Run with --runxfail to reproduce their original failures.
"""
from dataclasses import replace
from itertools import combinations

import pytest

from games.connect4.victor import Position, enumerate_candidates, search_covering_set
from games.connect4.victor import certificate as C, strategy as S
from games.connect4.victor.certificate_producer import certificate_from_witness, rule_instance
from victor_validation import assurance_model as M
from victor_validation.exact_oracle import EndgamePosition, solve


KINDS = {'CL': 'claimeven', 'BI': 'baseinverse', 'VE': 'vertical'}
EARLY_HISTORY = tuple(map(int, '23333334'))
EARLY = M.replay(EARLY_HISTORY)
# Newly constructed cover: all three rule kinds stacked in column a.
# a1 is a BI endpoint, a2-a3 is VE, a5-a6 is CL; a4 is free.
EARLY_RULES = (('BI', (0, 1), (2, 2)), ('VE', (0, 2), (0, 3))) + tuple(
    r for r in M.candidates(EARLY) if r[0] == 'CL' and r[1] not in ((0, 1), (0, 3)))
# Existing boards, re-encoded and checked by the new reference implementation.
MIXED_HISTORY = tuple(map(int, '33003316106600034431441142662221'))
MIXED_RULES = (('BI', (4, 6), (6, 6)), ('CL', (2, 5), (2, 6)),
               ('CL', (5, 1), (5, 2)), ('VE', (5, 4), (5, 5)))
ODD_HISTORY = tuple(map(int, '511322464155553320540623416612'))
ODD_RULES = (('BI', (1, 6), (2, 6)), ('CL', (3, 5), (3, 6)),
             ('CL', (4, 5), (4, 6)), ('VE', (0, 4), (0, 5)))


def name(s):
    return 'abcdefg'[s[0]] + str(s[1])


def instances(rules):
    return tuple(C.RuleInstance(KINDS[k], (name(a), name(b))) for k, a, b in rules)


def certificate(history=EARLY_HISTORY, rules=EARLY_RULES):
    return C.draft_certificate(M.matrix(M.replay(history)), instances(rules), replay=history)


def raw_instances(rules):
    return tuple((next(k for k, value in KINDS.items() if value == r.kind),
                  *((ord(s[0]) - 97, int(s[1])) for s in r.squares)) for r in rules)


def test_four_subset_geometry_and_all_single_rule_h2_predicates():
    assert len(M.GROUPS) == 69
    assert set(M.GROUPS) == {frozenset(g) for g in C.GROUPS}
    # Exactly 3 * 3 * C(42,2) = 7,749 shape/prerequisite comparisons.
    count = 0
    squares = tuple((c, r) for c in range(7) for r in range(1, 7))
    for board in (M.EMPTY, EARLY, M.replay(MIXED_HISTORY)):
        legal = set(M.candidates(board))
        for a, b in combinations(squares, 2):
            for k in KINDS:
                raw = (k, a, b)
                verdict = C.verify_certificate(C.draft_certificate(
                    M.matrix(board), instances((raw,))))
                assert (dict(verdict.hypotheses)['H2'] == 'verified') == (raw in legal), raw
                if raw in legal:
                    uncovered = {'-'.join(name(s) for s in sorted(g)) for g in M.GROUPS
                                 if all(M.cell(board, s) != 'O' for s in g)
                                 and not M.needs(raw) <= g}
                    actual = {f.detail.split()[2] for f in verdict.findings
                              if f.code == 'H4.uncovered'}
                    assert actual == uncovered
                count += 1
    assert count == 7749


def test_new_early_mixed_cover_and_untrusted_producer_pipeline():
    M.hypotheses(EARLY, EARLY_RULES)
    cert = certificate()
    verdict = C.verify_certificate(cert)
    assert len(cert.rules) == 16 and verdict.history_backed
    position = Position.from_board(cert.board, 0)
    production = {raw_instances((rule_instance(r),))[0] for r in enumerate_candidates(position)}
    assert production == set(M.candidates(EARLY))
    search = search_covering_set(position, node_budget=2000)
    assert search.witness is not None
    generated = certificate_from_witness(search.witness, replay=EARLY_HISTORY)
    M.hypotheses(EARLY, raw_instances(generated.rules))
    assert C.verify_certificate(generated).history_backed
    reordered = replace(cert, rules=tuple(reversed(cert.rules)))
    assert C.verify_certificate(reordered).history_backed
    assert C.verify_certificate(reordered).certificate_digest != verdict.certificate_digest


@pytest.mark.parametrize('history,rules,expected', [
    (MIXED_HISTORY, MIXED_RULES, (38, 88, 0, 18, 0, 8, 10, 0)),
    (ODD_HISTORY, ODD_RULES, (111, 481, 182, 15, 36, 60, 50, 0)),
])
def test_complete_all_white_all_permitted_spares(history, rules, expected):
    status, stats = M.explore(M.replay(history), rules, node_budget=2000)
    assert status == 'complete'
    assert tuple(stats.values()) == expected


def test_early_all_spares_prefix_is_explicitly_unknown_beyond_frontier():
    status, stats = M.explore(EARLY, EARLY_RULES, round_limit=3, node_budget=2000)
    assert status == 'unknown_depth_frontier'
    assert tuple(stats.values()) == (120, 226, 0, 0, 10, 20, 21, 84)
    status, _ = M.explore(EARLY, EARLY_RULES, node_budget=1)
    assert status == 'unknown_node_budget'


@pytest.mark.parametrize('history,rules,round_limit', [
    (EARLY_HISTORY, EARLY_RULES, 3), (MIXED_HISTORY, MIXED_RULES, None),
    (ODD_HISTORY, ODD_RULES, None),
])
def test_executable_refinement_against_new_model(history, rules, round_limit):
    cert = certificate(history, rules)
    board = M.replay(history)
    selections = terminals = oracle_checks = 0

    def visit(before, path):
        nonlocal selections, terminals, oracle_checks
        if round_limit is not None and len(path) // 2 >= round_limit:
            return
        for wc, _ in M.landings(before):
            after, allowed, forced = M.replies(before, rules, wc)
            expected = allowed[0] if forced else min(s for s in allowed if s[1] % 2 == 0)
            continuation = path + (wc,)
            selected = S.select_black_move(S.StrategyRequest(cert, continuation))
            context = (history, rules, continuation, selected)
            assert selected.status is S.StrategyStatus.MOVE_SELECTED, context
            assert selected.square == name(expected) and selected.state.board == M.matrix(after)
            assert selected.response is (S.ResponseKind.FORCED if forced else S.ResponseKind.SPARE)
            assert len(selected.state.trace) == len(continuation)
            selections += 1
            assert selections <= 3000, context  # Explicit path budget; no silent truncation.
            if sum(map(len, after)) >= 32:
                exact = solve(EndgamePosition.from_board(M.matrix(after), 1),
                              max_remaining=10, position_budget=50_000)
                assert exact.status == 'exact', context
                assert dict(exact.move_values)[expected[0]] >= 0, context
                oracle_checks += 1
            completed = M.drop(after, expected[0])
            M.invariant(completed, rules)
            next_path = continuation + (expected[0],)
            if M.terminal(completed):
                end = S.select_black_move(S.StrategyRequest(cert, next_path))
                assert end.status is S.StrategyStatus.TERMINAL and end.column is None
                assert end.state.board == M.matrix(completed)
                # Even a well-typed legal-column number after the terminal is illegal.
                extra = S.select_black_move(S.StrategyRequest(cert, next_path + (0,)))
                assert extra.status is S.StrategyStatus.ILLEGAL_CONTINUATION
                terminals += 1
            else:
                visit(completed, next_path)
    visit(board, ())
    print({'history': history, 'selections': selections, 'terminals': terminals,
           'oracle_checks': oracle_checks, 'round_limit': round_limit})
    assert selections > 0


def test_early_stacked_obligations_and_previous_response_corruption():
    cert = certificate()
    # BI a1-c2; then VE a2-a3; then free a4 -> even spare; then CL a5-a6.
    history = ()
    expected = ('c2', 'a3', 'e2', 'a6')
    for reply in expected:
        history += (0,)
        result = S.select_black_move(S.StrategyRequest(cert, history))
        assert result.square == reply
        wrong = (result.column + 1) % 7
        bad = S.select_black_move(S.StrategyRequest(cert, history + (wrong,)))
        assert bad.status in (S.StrategyStatus.STRATEGY_VIOLATED,
                              S.StrategyStatus.ILLEGAL_CONTINUATION)
        history += (result.column,)
    assert history == (0, 2, 0, 0, 0, 4, 0, 0)


def test_plain_malformed_field_grid_never_verifies():
    cert = certificate()
    bad = (None, True, 1.0, {}, [], (), '', 'forged', ('a1',), [[None]])
    checked = 0
    for field in ('board', 'rules', 'board_digest', 'player_to_move', 'defender',
                  'schema', 'theorem', 'ruleset', 'rule_model', 'compatibility_model'):
        for value in bad:
            result = C.verify_certificate(replace(cert, **{field: value}))
            assert result.status is not C.CertificateStatus.HYPOTHESES_VERIFIED, (field, value)
            assert result.provisional_bound is None
            checked += 1
    assert checked == 100
    mapping = C.certificate_to_mapping(cert)
    for field in ('trace', 'certificate_digest', 'verifier_id', 'history_backed'):
        assert C.verify_certificate_mapping(dict(mapping, **{field: True})).status is C.CertificateStatus.REJECTED


def test_strategy_snapshots_optional_containers_before_verification(monkeypatch):
    original = certificate()
    replay = list(EARLY_HISTORY)
    assignments = [list(a) for a in C.verify_certificate(original).evidence.assignment]
    cert = replace(original, replay=replay, assignments=assignments)
    verify = C.verify_certificate

    def interleave(snapshot, **kwargs):
        replay[:] = [6]
        assignments[0][:] = ['forged', 100]
        return verify(snapshot, **kwargs)

    monkeypatch.setattr(C, 'verify_certificate', interleave)
    out = S.select_black_move(S.StrategyRequest(cert, (0,)))
    assert out.status is S.StrategyStatus.MOVE_SELECTED and out.square == 'c2'
    assert out.starting_history_backed
    assert verify(cert).status is C.CertificateStatus.REJECTED


@pytest.mark.xfail(strict=True, reason='F1: mapping constructs every rule before cardinality/budget checks')
def test_mapping_rejects_impossible_rule_count_before_constructing_rules(monkeypatch):
    data = C.certificate_to_mapping(certificate())
    data['rules'] = [data['rules'][0]] * 22
    built = []
    constructor = C.RuleInstance

    def counted(*args):
        built.append(args)
        return constructor(*args)

    monkeypatch.setattr(C, 'RuleInstance', counted)
    out = C.verify_certificate_mapping(data, work_budget=0)
    assert out.status is not C.CertificateStatus.HYPOTHESES_VERIFIED
    assert not built


@pytest.mark.xfail(strict=True, reason='F1: digest traverses oversized replay before budget/shape rejection')
def test_oversized_optional_data_is_rejected_before_digest_traversal(monkeypatch):
    calls = []
    plain = C._plain

    def counted(value):
        calls.append(type(value))
        return plain(value)

    monkeypatch.setattr(C, '_plain', counted)
    out = C.verify_certificate(replace(certificate(), replay=[0] * 1000), work_budget=69)
    assert out.status is not C.CertificateStatus.HYPOTHESES_VERIFIED
    assert not calls  # Baseline traverses all 1,000 entries before its cutoff.


@pytest.mark.parametrize('field', ['replay', 'assignments'])
@pytest.mark.xfail(strict=True, reason='F2: optional containers are digested and checked at different times')
def test_optional_snapshot_digest_binds_what_was_verified(monkeypatch, field):
    original = certificate()
    if field == 'replay':
        valid = list(EARLY_HISTORY)
        mutable = [6]  # Invalid initial replay, still bound into the first digest.
    else:
        valid = list(C.verify_certificate(original).evidence.assignment)
        mutable = []  # Invalid incomplete assignment.
    cert = replace(original, **{field: mutable})
    honest_digest = C.verify_certificate(replace(original, **{field: valid})).certificate_digest
    real_digest = C._certificate_digest

    def interleave(*args):
        digest = real_digest(*args)
        # Deterministically simulate a caller thread editing its OWN list.
        # No verdict or mathematical predicate is changed by this hook.
        mutable[:] = valid
        return digest

    monkeypatch.setattr(C, '_certificate_digest', interleave)
    result = C.verify_certificate(cert)
    assert result.status is not C.CertificateStatus.HYPOTHESES_VERIFIED or (
        result.certificate_digest == honest_digest)


class ExplodingRepr:
    def __repr__(self):
        raise RuntimeError('6B.5 malformed leaf reached repr')


@pytest.mark.parametrize('entrypoint', ['verifier', 'strategy', 'mapping'])
@pytest.mark.xfail(strict=True, raises=RuntimeError, reason='F3: invalid optional leaf executes repr and escapes')
def test_invalid_object_leaf_returns_rejection_without_running_repr(entrypoint):
    cert = replace(certificate(), replay=(ExplodingRepr(),))
    if entrypoint == 'strategy':
        out = S.select_black_move(S.StrategyRequest(cert, (0,)))
        assert out.status is S.StrategyStatus.INVALID_CERTIFICATE
    elif entrypoint == 'mapping':
        assert C.verify_certificate_mapping(C.certificate_to_mapping(cert)).status is C.CertificateStatus.REJECTED
    else:
        assert C.verify_certificate(cert).status is C.CertificateStatus.REJECTED
