"""Untrusted witness mutations and independent raw-predicate verification."""
from copy import copy
from dataclasses import replace

import pytest

from games.connect4.victor import (
    ALL_GROUPS, CandidateEvidence, CoverageWitness, Position, RuleCandidate, RuleName,
    Square, VerificationStatus, black_evaluation_context, enumerate_candidates,
    search_covering_set, verify_coverage_witness,
)
from games.connect4.victor.contracts import CoverageAssignment
from test_victor_coverage import THESIS_MOVES, play, raw_coverage

CL, BI, VE = RuleName.CLAIMEVEN, RuleName.BASEINVERSE, RuleName.VERTICAL


def pair(rule, *names):
    return RuleCandidate(rule, tuple(Square.from_name(n) for n in names))


def forge(value, **fields):
    """Deliberately bypass frozen dataclass invariants, modelling corrupted input."""
    value = copy(value)
    for name, field in fields.items():
        object.__setattr__(value, name, field)
    return value


@pytest.fixture
def original():
    p = play(THESIS_MOVES)
    return p, search_covering_set(p).witness


def corrupt(w, kind):
    e, a, ctx = w.evidence[0], w.assignments[0], w.context
    if kind == 'defender':
        return replace(w, context=replace(ctx, defender=0))
    if kind == 'boolean_player':
        return replace(w, context=replace(ctx, defender=True))
    if kind == 'opponent':
        return replace(w, context=replace(ctx, opponent=1))
    if kind == 'mode':
        return replace(w, context=replace(ctx, mode='white_odd_threat'))
    if kind == 'context_turn':
        return replace(w, context=replace(ctx, position=forge(ctx.position, player_to_move=1)))
    if kind == 'context_board':
        # Move White's c1 stone to b1, preserving valid counts/gravity/turn.
        board = [list(r) for r in ctx.position.board]
        board[5][1], board[5][2] = board[5][2], board[5][1]
        return replace(w, context=replace(ctx, position=Position.from_board(board, 0)))
    if kind == 'missing_target':
        return replace(w, context=replace(ctx, target_groups=ctx.target_groups[1:]))
    if kind == 'duplicate_target':
        return replace(w, context=replace(ctx, target_groups=ctx.target_groups + ctx.target_groups[:1]))
    if kind == 'blocked_target':
        blocked = next(g for g in ALL_GROUPS if g not in ctx.target_groups)
        return replace(w, context=replace(ctx, target_groups=tuple(sorted(ctx.target_groups + (blocked,)))))
    if kind == 'rule':
        candidate = forge(e.candidate, rule=VE if e.candidate.rule == CL else CL)
    elif kind == 'unsupported_rule':
        candidate = forge(e.candidate, rule=RuleName.AFTEREVEN)
    elif kind == 'untyped_rule':
        candidate = forge(e.candidate, rule=e.candidate.rule.value)
    elif kind == 'occupied_square':
        candidate = pair(CL, 'c1', 'c2')
    elif kind == 'unplayable_square':
        candidate = pair(BI, 'a3', 'b1')
    elif kind == 'square_change':
        candidate = pair(CL, 'g5', 'g6')
    elif kind == 'role_order':
        candidate = forge(e.candidate, squares=tuple(reversed(e.candidate.squares)))
    elif kind == 'out_of_bounds_square':
        square = forge(e.candidate.squares[0], column=7)
        candidate = forge(e.candidate, squares=(square, e.candidate.squares[1]))
    elif kind == 'boolean_coordinate':
        square = forge(e.candidate.squares[0], column=True)
        candidate = forge(e.candidate, squares=(square, e.candidate.squares[1]))
    else:
        candidate = None
    if candidate is not None:
        return replace(w, evidence=(replace(e, candidate=candidate),) + w.evidence[1:])
    if kind == 'missing_coverage':
        return replace(w, evidence=(replace(e, conditional_solved_groups=e.conditional_solved_groups[1:]),) + w.evidence[1:])
    if kind == 'extra_coverage':
        extra = next(g for g in ctx.target_groups if g not in e.conditional_solved_groups)
        return replace(w, evidence=(replace(e, conditional_solved_groups=tuple(sorted(e.conditional_solved_groups + (extra,)))),) + w.evidence[1:])
    if kind == 'duplicate_coverage':
        return replace(w, evidence=(replace(e, conditional_solved_groups=e.conditional_solved_groups + e.conditional_solved_groups[:1]),) + w.evidence[1:])
    if kind == 'duplicate_candidate':
        return replace(w, evidence=w.evidence + (e,))
    if kind == 'missing_candidate':
        return replace(w, evidence=w.evidence[1:])
    if kind == 'missing_assignment':
        return replace(w, assignments=w.assignments[1:])
    if kind == 'duplicate_assignment':
        return replace(w, assignments=w.assignments + (a,))
    if kind == 'wrong_assignment':
        wrong = next(item.candidate for item in w.evidence if a.group not in item.conditional_solved_groups)
        return replace(w, assignments=(replace(a, candidate=wrong),) + w.assignments[1:])
    if kind == 'unselected_assignment':
        unselected = next(c for c in enumerate_candidates(ctx.position) if c not in {item.candidate for item in w.evidence})
        return replace(w, assignments=(replace(a, candidate=unselected),) + w.assignments[1:])
    if kind == 'wrong_group':
        blocked = next(g for g in ALL_GROUPS if g not in ctx.target_groups)
        return replace(w, assignments=(replace(a, group=blocked),) + w.assignments[1:])
    if kind == 'corrupted_group_square':
        square = forge(a.group.squares[0], row_index=-1)
        group = forge(a.group, squares=(square,) + a.group.squares[1:])
        return replace(w, assignments=(replace(a, group=group),) + w.assignments[1:])
    if kind == 'conflicting_candidate':
        used = {item.candidate for item in w.evidence}
        conflict = next(c for c in enumerate_candidates(ctx.position) if c not in used
                        and any(set(c.squares) & set(old.squares) for old in used))
        claim = CandidateEvidence(conflict, tuple(sorted(raw_coverage(ctx.position, conflict, ctx.target_groups))))
        return replace(w, evidence=w.evidence + (claim,))
    if kind == 'schema':
        return replace(w, schema_version='accepted-proof-v1')
    if kind == 'certification':
        return forge(w, outcome_certification='proven_draw')
    if kind == 'malformed_evidence':
        return replace(w, evidence=(None,))
    raise AssertionError(kind)


@pytest.mark.parametrize('kind', [
    'defender', 'boolean_player', 'opponent', 'mode', 'context_turn', 'context_board',
    'missing_target', 'duplicate_target', 'blocked_target', 'rule', 'unsupported_rule',
    'untyped_rule', 'occupied_square', 'unplayable_square', 'square_change', 'role_order',
    'out_of_bounds_square', 'boolean_coordinate', 'missing_coverage', 'extra_coverage',
    'duplicate_coverage', 'duplicate_candidate', 'missing_candidate', 'missing_assignment',
    'duplicate_assignment', 'wrong_assignment', 'unselected_assignment', 'wrong_group',
    'corrupted_group_square', 'conflicting_candidate', 'schema', 'certification', 'malformed_evidence',
])
def test_corrupted_witnesses_rejected(original, kind):
    p, witness = original
    mutated = corrupt(witness, kind)
    # Python equality conflates True/1 and str-enums/their strings. The verifier
    # must still reject those identity/type corruptions independently of equality.
    assert mutated != witness or kind in ('boolean_player', 'untyped_rule')
    result = verify_coverage_witness(p, mutated)
    assert result.status == VerificationStatus.REJECTED and result.rejection_reasons
    assert result.outcome_certification == 'uncertified'


@pytest.mark.parametrize('position', [
    None, play([0]), play([0,1,0,1,0,1,0]), play([0,1,0,1,2,1,2,1]),
    Position.from_board([list(r) for r in ['XXOOXXO','OOXXOOX'] * 3], 0),
])
def test_invalid_side_or_terminal_position_rejected(original, position):
    assert verify_coverage_witness(position, original[1]).status == VerificationStatus.REJECTED


def test_valid_alternative_snapshot_and_malformed_snapshot_rejected(original):
    p, w = original
    changed = corrupt(w, 'context_board').context.position
    assert verify_coverage_witness(changed, w).status == VerificationStatus.REJECTED
    for invalid in (forge(p, player_to_move=1), forge(p, board=()), None):
        assert verify_coverage_witness(invalid, w).status == VerificationStatus.REJECTED
    assert verify_coverage_witness(p, None).status == VerificationStatus.REJECTED


def test_duplicate_and_response_conflict_reported_specifically(original):
    p, w = original
    assert 'duplicate selected candidate' in verify_coverage_witness(p, corrupt(w, 'duplicate_candidate')).rejection_reasons
    assert 'selected candidates conflict' in verify_coverage_witness(p, corrupt(w, 'conflicting_candidate')).rejection_reasons


def test_verifier_independent_of_search_enumeration_context_and_candidate_metadata(original, monkeypatch):
    p, w = original

    def forbidden(*args, **kwargs):
        raise AssertionError('verifier reused a production search/detection assertion')

    for module, names in [
        ('games.connect4.victor.rules', ['enumerate_candidates']),
        ('games.connect4.victor.evidence', ['analyze_candidates', 'enumerate_candidates']),
        ('games.connect4.victor.compatibility', ['compatible', 'required_constraints']),
        ('games.connect4.victor.coverage', ['search_covering_set', 'black_evaluation_context', 'compatible', 'analyze_candidates']),
    ]:
        for name in names:
            monkeypatch.setattr(f'{module}.{name}', forbidden)
    for name in ('prerequisites', 'coverage_squares', 'affected_squares'):
        monkeypatch.setattr(RuleCandidate, name, property(forbidden))
    monkeypatch.setattr(Position, 'potential_groups', forbidden)
    assert verify_coverage_witness(p, w).status == VerificationStatus.VERIFIED_UNCERTIFIED
    assert verify_coverage_witness(p, corrupt(w, 'missing_coverage')).status == VerificationStatus.REJECTED


def test_manually_constructed_thesis_claimeven_witness_without_search():
    p = play(THESIS_MOVES)
    ctx = black_evaluation_context(p)
    # Exact printed diagram 6.1 list, independent of the production enumerator.
    names = ['a1-a2','a3-a4','a5-a6','b1-b2','b3-b4','b5-b6','c3-c4','c5-c6',
             'e3-e4','e5-e6','f1-f2','f3-f4','f5-f6','g1-g2','g3-g4','g5-g6']
    candidates = tuple(pair(CL, *name.split('-')) for name in names)
    evidence = tuple(CandidateEvidence(c, tuple(g for g in ctx.target_groups if c.squares[1] in g.squares)) for c in candidates)
    assignments = tuple(CoverageAssignment(g, next(c for c in candidates if c.squares[1] in g.squares)) for g in ctx.target_groups)
    w = CoverageWitness(ctx, evidence, assignments)
    result = verify_coverage_witness(p, w)
    assert result.status == VerificationStatus.VERIFIED_UNCERTIFIED
    assert result.rejection_reasons == () and result.outcome_certification == 'uncertified'
    # Selection and assignment order are irrelevant; group claims are canonical.
    assert verify_coverage_witness(p, replace(w, evidence=tuple(reversed(evidence)), assignments=tuple(reversed(assignments)))) == result


@pytest.mark.parametrize('rows,rule', [
    ([' XX OXO','OOXXOOX','XXOOXXO','OOXXOOX','XXOOXXO','OOXXOOX'], BI),
    ([' O OXXO',' OXXOOX',' XOOXXO','XOXXOOX','XXOOXXO','OOXXOOX'], VE),
])
def test_constructed_snapshots_verify_nonempty_baseinverse_and_vertical_coverage(rows, rule):
    # Constructed basic-valid snapshots, not thesis diagrams or replay evidence.
    # BI: only a6-d6 remains; VE: a3-a6 and a2-a5 remain.
    p = Position.from_board([list(row) for row in rows], 0)
    w = search_covering_set(p).witness
    assert w is not None and len(w.evidence) == 1
    assert w.evidence[0].candidate.rule == rule
    assert w.evidence[0].conditional_solved_groups == w.context.target_groups
    assert verify_coverage_witness(p, w).status == VerificationStatus.VERIFIED_UNCERTIFIED
    assert verify_coverage_witness(p, replace(w, evidence=(replace(
        w.evidence[0], conditional_solved_groups=()),))).status == VerificationStatus.REJECTED
