"""Nine-rule covering-set search, independent verification and exact cross-checks.

No game values are asserted for covers. Exact-oracle comparisons are falsification
evidence, never a proof of the nine-rule composition theorem.
"""
from copy import copy
from dataclasses import replace
from random import Random
from types import MappingProxyType

import pytest

from games.connect4.connect4 import Connect4
from games.connect4.victor import (
    Position, RuleCandidate, RuleName, SearchStatus, Square, VerificationStatus,
    search_covering_set, verify_coverage_witness,
)
from games.connect4.victor import compatibility as compat_module
from games.connect4.victor import composite as composite_module
from games.connect4.victor.certificate_producer import certificate_from_witness
from games.connect4.victor.composite import Component, CompositeCandidate, _clause, above
from games.connect4.victor.composite import below as below_square
from games.connect4.victor.nine_rule_verification import verify_nine_rule_witness
from games.connect4.victor.nine_rules import (
    NINE_RULE_OBLIGATIONS, RuleEvidence, analyze_nine_rules,
    search_nine_rule_cover,
)
from victor_validation import nine_rule_reference as R
from victor_validation.exact_oracle import replay, solve
from test_victor_coverage import SAT_MOVES, UNSAT_MOVES

S = Square.from_name
CL, BI, VE = RuleName.CLAIMEVEN, RuleName.BASEINVERSE, RuleName.VERTICAL
AE, LI, HI = RuleName.AFTEREVEN, RuleName.LOWINVERSE, RuleName.HIGHINVERSE
BC, BE, SB = RuleName.BASECLAIM, RuleName.BEFORE, RuleName.SPECIALBEFORE
FOUND = SearchStatus.FOUND
NO_COVER = SearchStatus.EXHAUSTIVE_NO_COVER


def play(moves):
    game = Connect4()
    for column in moves:
        assert game.make_move(column)
    return Position.from_board(game.board, game.current_player)


def mirror(p):
    return Position.from_board([list(reversed(row)) for row in p.board], p.player_to_move)


def rules_of(witness):
    return {e.candidate.rule for e in witness.evidence}


# ------------------------------------------------------------ thesis covers

def test_diagram_6_4_two_afterevens_complete_a_cover_claimevens_cannot():
    p = play(R.DIAGRAMS['6.4'])
    assert search_covering_set(p).status == NO_COVER
    result = search_nine_rule_cover(p)
    assert result.status == FOUND and AE in rules_of(result.witness)
    afterevens = {e.candidate.group for e in result.witness.evidence if e.candidate.rule == AE}
    assert {tuple(s.name for s in g.squares) for g in afterevens} == {
        ('b2', 'c2', 'd2', 'e2'), ('c2', 'd2', 'e2', 'f2')}  # "solve all groups" (§6.4).
    assert verify_nine_rule_witness(p, result.witness).status == VerificationStatus.VERIFIED_UNCERTIFIED


def test_diagram_6_10_mixed_family_cover_contains_allis_specialbefore():
    p = play(R.DIAGRAMS['6.10'])
    assert search_covering_set(p).status == NO_COVER
    result = search_nine_rule_cover(p)
    assert result.status == FOUND
    special = CompositeCandidate(SB, _line('d2', 'g2'),
                                 (Component(CL, S('f1'), S('f2')), Component(CL, S('g1'), S('g2'))),
                                 (S('e2'), S('d3')))
    candidates = {e.candidate for e in result.witness.evidence}
    assert special in candidates and RuleCandidate(BI, (S('a1'), S('b1'))) in candidates
    assert len(rules_of(result.witness)) >= 4 and {CL, BI, SB} <= rules_of(result.witness)
    assert verify_nine_rule_witness(p, result.witness).status == VerificationStatus.VERIFIED_UNCERTIFIED
    # Assignments explain each group with a clause of a selected rule.
    for assignment in result.witness.assignments:
        evidence = next(e for e in result.witness.evidence if e.candidate == assignment.candidate)
        assert assignment.group in evidence.conditional_solved_groups
        assert assignment.clause in [c.label for c in evidence.clauses if c.solves(assignment.group)]


def _line(first, last):
    from games.connect4.victor import Group
    a, b = S(first), S(last)
    dr, dc = (b.row_index - a.row_index) // 3, (b.column - a.column) // 3
    return Group(tuple(Square(a.row_index + i * dr, a.column + i * dc) for i in range(4)))


def test_diagram_6_5_exhausts_without_outcome():
    result = search_nine_rule_cover(play(R.DIAGRAMS['6.5']))
    assert result.status == NO_COVER and result.witness is None
    assert result.outcome_certification == 'uncertified'
    assert result.unproven_obligations == NINE_RULE_OBLIGATIONS
    assert not hasattr(result, 'winner') and not hasattr(result, 'value')


# ------------------------------------------------- statuses and boundaries

@pytest.mark.parametrize('moves', [R.DIAGRAMS['6.4'], R.DIAGRAMS['6.10'], R.DIAGRAMS['6.5']])
def test_budget_boundary_root_and_solution_nodes_counted(moves):
    p = play(moves)
    completed = search_nine_rule_cover(p)
    for budget in (0, 1, completed.expanded_nodes - 1):
        if budget >= completed.expanded_nodes:
            continue
        cutoff = search_nine_rule_cover(p, node_budget=budget)
        assert cutoff.status == SearchStatus.BUDGET_EXHAUSTED
        assert cutoff.expanded_nodes == budget and cutoff.witness is None
    exact = search_nine_rule_cover(p, node_budget=completed.expanded_nodes)
    assert replace(exact, node_budget=completed.node_budget) == completed  # Incl. witness.


def test_empty_board_exhausts_and_budget_cutoff_is_unknown():
    result = search_nine_rule_cover(play([]))
    assert result.status == NO_COVER and result.expanded_nodes > 1
    assert dict((r, n) for r, n, _ in result.candidate_counts)[SB] == 224
    assert search_nine_rule_cover(play([]), node_budget=5).status == SearchStatus.BUDGET_EXHAUSTED


@pytest.mark.parametrize('position,defender', [
    (play([0]), 1), (play([]), 0), (play([]), True), (play([0, 1, 0, 1, 0, 1, 0]), 1),
    (Position.from_board([list(r) for r in ['XXOOXXO', 'OOXXOOX'] * 3], 0), 1), (None, 1),
])
def test_unsupported_contexts(position, defender):
    result = search_nine_rule_cover(position, defender=defender)
    assert result.status == SearchStatus.UNSUPPORTED_CONTEXT and result.reason
    assert result.expanded_nodes == 0 and result.witness is None


@pytest.mark.parametrize('kwargs', [
    {'node_budget': -1}, {'node_budget': True}, {'node_budget': 2.0}, {'rules': ()},
    {'rules': ('claimeven',)}, {'rules': CL}, {'rules': None},
])
def test_invalid_arguments(kwargs):
    with pytest.raises(ValueError):
        search_nine_rule_cover(play([]), **kwargs)


@pytest.mark.parametrize('moves', [SAT_MOVES, UNSAT_MOVES, R.DIAGRAMS['6.1'], R.DIAGRAMS['6.4']])
def test_three_rule_subset_agrees_with_the_established_search(moves):
    p = play(moves)
    for position in (p, mirror(p)):
        old = search_covering_set(position, node_budget=100_000)
        new = search_nine_rule_cover(position, rules=(VE, BI, CL))
        assert new.rules == (CL, BI, VE)
        assert old.status == new.status
        if new.witness:
            assert rules_of(new.witness) <= {CL, BI, VE}
            assert verify_nine_rule_witness(position, new.witness).status == \
                VerificationStatus.VERIFIED_UNCERTIFIED


def _endgames(seed, count, empties):
    rng = Random(seed)
    out = []
    while len(out) < count:
        p = replay([])
        moves = []
        while p.remaining > rng.choice(empties) and not p.terminal:
            column = rng.choice(p.legal_columns)
            moves.append(column)
            p = p.drop(column)
        if not p.terminal and p.turn == 0:
            out.append((moves, p))
    return out


def _reference_cover_exists(board):
    """Exhaustive, non-MRV cover search over the INDEPENDENT reference predicates."""
    targets = sorted(R.white_groups(board), key=sorted)
    rules = [i for i in R.reference_identities(board)
             if any(R.solved(i, board, g) for g in targets)]
    covers = {i: {g for g in targets if R.solved(i, board, g)} for i in rules}

    def search(chosen, uncovered):
        if not uncovered:
            return True
        target = uncovered[0]
        for rule in rules:
            if target in covers[rule] and all(R.compatible(rule, c) for c in chosen):
                if search(chosen + [rule], [g for g in uncovered if g not in covers[rule]]):
                    return True
        return False
    return search([], targets)


@pytest.mark.parametrize('moves,p', _endgames(31, 40, (4, 6, 8)))
def test_search_matches_independent_exhaustive_reference_and_mirror(moves, p):
    position = Position.from_board(p.board, 0)
    result = search_nine_rule_cover(position)
    assert result.status in (FOUND, NO_COVER)
    assert (result.status == FOUND) == _reference_cover_exists(p.board)
    assert search_nine_rule_cover(mirror(position)).status == result.status
    if result.witness:
        assert verify_nine_rule_witness(position, result.witness).status == \
            VerificationStatus.VERIFIED_UNCERTIFIED


# --------------------------------------------------------- verification

@pytest.fixture(scope='module')
def original():
    p = play(R.DIAGRAMS['6.10'])
    return p, search_nine_rule_cover(p).witness


def _forge(value, **fields):
    value = copy(value)
    for name, field in fields.items():
        object.__setattr__(value, name, field)
    return value


def _evidence_for(witness, rule):
    return next(i for i, e in enumerate(witness.evidence) if e.candidate.rule == rule)


def _replace_evidence(witness, index, evidence):
    items = list(witness.evidence)
    items[index] = evidence
    return replace(witness, evidence=tuple(items))


def _corrupt(p, w, kind):
    sb = _evidence_for(w, SB)
    special = w.evidence[sb]
    li = _evidence_for(w, LI)
    if kind == 'component_kind':
        changed = (Component(VE, S('f1'), S('f2')),) + special.candidate.components[1:]
        return _replace_evidence(w, sb, replace(special, candidate=_forge(special.candidate, components=changed)))
    if kind == 'role_swap':
        roles = tuple(reversed(special.candidate.roles))
        return _replace_evidence(w, sb, replace(special, candidate=_forge(special.candidate, roles=roles)))
    if kind == 'extra_not_playable':
        return _replace_evidence(w, sb, replace(special, candidate=_forge(
            special.candidate, roles=(S('e2'), S('b2')))))
    if kind == 'clause_label':
        clauses = (replace(special.clauses[0], label='successors'),) + special.clauses[1:]
        return _replace_evidence(w, sb, replace(special, clauses=clauses))
    if kind == 'clause_dropped':
        return _replace_evidence(w, sb, replace(special, clauses=special.clauses[1:]))
    if kind == 'clause_requirement':
        clauses = (replace(special.clauses[0], requirements=special.clauses[0].requirements[:-1]),) \
            + special.clauses[1:]
        return _replace_evidence(w, sb, replace(special, clauses=clauses))
    if kind == 'missing_coverage':
        return _replace_evidence(w, sb, replace(
            special, conditional_solved_groups=special.conditional_solved_groups[1:]))
    if kind == 'extra_coverage':
        extra = next(g for g in w.context.target_groups if g not in special.conditional_solved_groups)
        groups = tuple(sorted(special.conditional_solved_groups + (extra,)))
        return _replace_evidence(w, sb, replace(special, conditional_solved_groups=groups))
    if kind == 'claimeven_crossing_lowinverse':
        # Constraint 2 is the ONLY CL/LI constraint (no disjointness requirement).
        upper_pair = w.evidence[li].candidate.components[1]  # Lowinverse pair d4-d5.
        candidate = RuleCandidate(CL, (below_square(upper_pair.lower), upper_pair.lower))
        clauses = composite_module.solution_clauses(candidate, p)
        claim = RuleEvidence(candidate, clauses, tuple(g for g in w.context.target_groups
                                                       if any(c.solves(g) for c in clauses)))
        return replace(w, evidence=w.evidence + (claim,))
    if kind == 'highinverse_conditional_forged':
        candidate = CompositeCandidate(HI, roles=tuple(S(n) for n in ('a2', 'a3', 'a4', 'c4', 'c5', 'c6')))
        playable_board = play([0])  # a2 directly playable there, not on diagram 6.10.
        clauses = candidate.solution_clauses(playable_board)
        assert 'lower-first+upper-second' in [c.label for c in clauses]
        claim = RuleEvidence(candidate, clauses, tuple(g for g in w.context.target_groups
                                                       if any(c.solves(g) for c in clauses)))
        return replace(w, evidence=w.evidence + (claim,))
    if kind == 'rule_set':
        return replace(w, rules=(CL, BI, VE))
    if kind == 'rule_set_order':
        return replace(w, rules=tuple(reversed(w.rules)))
    if kind == 'missing_assignment':
        return replace(w, assignments=w.assignments[1:])
    if kind == 'duplicate_assignment':
        return replace(w, assignments=w.assignments + w.assignments[:1])
    if kind == 'wrong_clause':
        a = w.assignments[0]
        return replace(w, assignments=(replace(a, clause='playable-pair-forged'),) + w.assignments[1:])
    if kind == 'unselected_candidate':
        a = w.assignments[0]
        return replace(w, assignments=(replace(a, candidate=RuleCandidate(CL, (S('g5'), S('g6')))),)
                       + w.assignments[1:])
    if kind == 'duplicate_candidate':
        return replace(w, evidence=w.evidence + (special,))
    if kind == 'schema':
        return replace(w, schema_version='coverage-6b1-v1')
    if kind == 'certification':
        return _forge(w, outcome_certification='proven_draw')
    if kind == 'obligations_stripped':
        return _forge(w, unproven_obligations=())
    if kind == 'target_missing':
        return replace(w, context=replace(w.context, target_groups=w.context.target_groups[1:]))
    if kind == 'context_defender':
        return replace(w, context=replace(w.context, defender=0))
    if kind == 'malformed_evidence':
        return replace(w, evidence=(None,))
    raise AssertionError(kind)


@pytest.mark.parametrize('kind', [
    'component_kind', 'role_swap', 'extra_not_playable', 'clause_label', 'clause_dropped',
    'clause_requirement', 'missing_coverage', 'extra_coverage', 'claimeven_crossing_lowinverse',
    'highinverse_conditional_forged', 'rule_set', 'rule_set_order', 'missing_assignment',
    'duplicate_assignment', 'wrong_clause', 'unselected_candidate', 'duplicate_candidate',
    'schema', 'certification', 'obligations_stripped', 'target_missing', 'context_defender',
    'malformed_evidence',
])
def test_corrupted_nine_rule_witnesses_rejected(original, kind):
    p, w = original
    assert verify_nine_rule_witness(p, w).status == VerificationStatus.VERIFIED_UNCERTIFIED
    result = verify_nine_rule_witness(p, _corrupt(p, w, kind))
    assert result.status == VerificationStatus.REJECTED and result.rejection_reasons
    assert result.outcome_certification == 'uncertified'


def test_constraint_two_violation_reported_specifically(original):
    p, w = original
    reasons = verify_nine_rule_witness(p, _corrupt(p, w, 'claimeven_crossing_lowinverse')).rejection_reasons
    assert reasons == ('selected candidates conflict: constraint 2',)


def test_wrong_position_and_turn_rejected(original):
    p, w = original
    for position in (play(R.DIAGRAMS['6.4']), play([0]), None, _forge(p, player_to_move=1)):
        assert verify_nine_rule_witness(position, w).status == VerificationStatus.REJECTED


def test_three_rule_certificate_boundary_rejects_nine_rule_witnesses(original):
    p, w = original
    assert verify_coverage_witness(p, w).status == VerificationStatus.REJECTED
    with pytest.raises((AttributeError, TypeError, ValueError, KeyError)):
        certificate_from_witness(w)
    old = search_covering_set(play(R.DIAGRAMS['6.1'])).witness
    assert verify_nine_rule_witness(play(R.DIAGRAMS['6.1']), old).status == VerificationStatus.REJECTED


def test_verifier_independent_of_producer_semantics(original, monkeypatch):
    p, w = original

    def forbidden(*args, **kwargs):
        raise AssertionError('verifier reused producer semantics')

    for module, names in [
        ('games.connect4.victor.composite', [
            'solution_clauses', 'prerequisite_failures', 'enumerate_composites', 'enumerate_afterevens',
            'enumerate_befores', 'enumerate_specialbefores', 'above', 'below']),
        ('games.connect4.victor.compatibility', [
            'compatible', 'failed_constraints', 'footprint', 'required_constraints', 'checked_candidate']),
        ('games.connect4.victor.nine_rules', [
            'analyze_nine_rules', 'search_nine_rule_cover', 'conflict_masks']),
        ('games.connect4.victor.coverage', ['black_evaluation_context', 'backtrack_cover']),
    ]:
        for name in names:
            monkeypatch.setattr(f'{module}.{name}', forbidden)
    for name in ('affected_squares', 'claimeven_parts', 'inverse_columns', 'special_squares',
                 'baseclaim_square', 'depends_on_zugzwang'):
        monkeypatch.setattr(CompositeCandidate, name, property(forbidden))
    monkeypatch.setattr(CompositeCandidate, 'solution_clauses', forbidden)
    for name in ('squares', 'before_square', 'name'):
        monkeypatch.setattr(Component, name, property(forbidden))
    monkeypatch.setattr(Component, 'clause', forbidden)
    for name in ('prerequisites', 'coverage_squares', 'affected_squares'):
        monkeypatch.setattr(RuleCandidate, name, property(forbidden))
    monkeypatch.setattr(Position, 'potential_groups', forbidden)
    assert verify_nine_rule_witness(p, w).status == VerificationStatus.VERIFIED_UNCERTIFIED
    assert verify_nine_rule_witness(p, replace(w, assignments=w.assignments[1:])).status == \
        VerificationStatus.REJECTED


def test_mirrored_witness_verifies_only_on_the_mirrored_board(original):
    p, w = original
    m = mirror(p)
    reflected = search_nine_rule_cover(m).witness
    assert verify_nine_rule_witness(m, reflected).status == VerificationStatus.VERIFIED_UNCERTIFIED
    assert verify_nine_rule_witness(p, reflected).status == VerificationStatus.REJECTED
    assert {e.candidate.reflected() for e in w.evidence} == {e.candidate for e in reflected.evidence}


# ------------------------------------------------ exact-oracle cross-checks

def test_covers_are_never_contradicted_by_exact_endgame_values():
    """Falsification sample: White never wins where a nine-rule cover exists."""
    covers = composite_only = aftereven_wins = 0
    for moves, p in _endgames(1988, 250, (6, 8, 10)):
        position = Position.from_board(p.board, 0)
        result = search_nine_rule_cover(position)
        assert result.status in (FOUND, NO_COVER)
        if result.status != FOUND:
            continue
        exact = solve(p, max_remaining=10, position_budget=1_000_000)
        assert exact.status == 'exact'
        assert exact.value_for(1) >= 0, moves  # Counterexample: preserve these moves.
        covers += 1
        if search_covering_set(position).status != FOUND:
            composite_only += 1
        if AE in rules_of(result.witness):
            aftereven_wins += exact.value_for(1) == 1
    assert covers >= 20 and composite_only >= 5  # The sample is not vacuous.
    assert aftereven_wins >= 1


def _ae_any_column(c, p):
    union = frozenset().union(*(composite_module._squares_above(k.upper) for k in c.components))
    return (composite_module.SolutionClause('aftereven-columns', (union,)),) + tuple(
        k.clause() for k in c.components)


def _before_one_successor(c, p):
    return tuple(_clause('s', above(k.before_square)) for k in c.components) + tuple(
        k.clause() for k in c.components)


def _specialbefore_without_extra(c, p):
    playable, extra = c.roles
    successors = [above(k.before_square) for k in c.components] + [above(playable)]
    return ((_clause('s', *successors), _clause('pp', playable, extra))
            + tuple(k.clause() for k in c.components))


def _lowinverse_both_uppers(c, p):
    a, b = c.components
    return (_clause('u1', a.upper), _clause('u2', b.upper), a.clause(), b.clause())


def _literal_constraint_two(a, b):
    inverse, other = (a, b) if a.rule in compat_module.INVERSES else (b, a)
    for lower, upper in other.claimevens:
        squares = inverse.inverse_columns.get(lower.column)
        if squares is not None and upper.row < min(s.row for s in squares):
            return False
    return True


# Each loosened semantics yields a "cover" on its fixture where White in fact wins.
MUTANTS = [
    ('aftereven-any-column', [3, 5, 5, 5, 5, 3, 4, 3, 2, 6, 3, 6, 6, 6, 3, 1, 4, 2, 1, 4, 6, 1, 5,
                              5, 1, 6, 4, 4, 3, 1, 4, 0], ('clauses', AE, _ae_any_column)),
    ('before-one-successor', [5, 0, 3, 0, 3, 3, 4, 6, 1, 0, 4, 3, 5, 4, 4, 5, 3, 2, 5, 5, 0, 2, 2,
                              6, 6, 3, 6, 6, 6, 5, 2, 0], ('clauses', BE, _before_one_successor)),
    ('specialbefore-without-extra', [6, 4, 4, 1, 3, 0, 4, 0, 0, 0, 1, 3, 1, 5, 1, 1, 5, 5, 5, 6, 4,
                                     3, 5, 3, 5, 6, 6, 4, 4, 0, 0, 2, 1, 2, 2, 6],
     ('clauses', SB, _specialbefore_without_extra)),
    ('lowinverse-both-uppers', [5, 0, 5, 0, 1, 5, 4, 6, 4, 6, 4, 2, 1, 4, 6, 4, 0, 4, 1, 6, 6, 1, 6,
                                2, 0, 5, 5, 5, 2, 3, 1, 1], ('clauses', LI, _lowinverse_both_uppers)),
    ('no-constraint-2', [4, 0, 1, 4, 3, 0, 3, 3, 0, 6, 0, 3, 5, 0, 4, 5, 3, 0, 4, 6, 5, 1, 4, 5, 6,
                         6, 3, 5, 5, 4, 6, 2, 6, 2], ('check', 'C2', lambda a, b: True)),
    ('literal-constraint-2', [2, 4, 2, 0, 1, 2, 5, 2, 2, 6, 2, 6, 1, 1, 6, 5, 3, 3, 1, 5, 5, 6, 5,
                              1, 1, 0, 0, 6, 6, 5, 4, 0], ('check', 'C2', _literal_constraint_two)),
    ('loose-constraint-3', [2, 4, 3, 3, 3, 1, 0, 5, 4, 2, 5, 5, 6, 0, 0, 4, 3, 0, 6, 4, 4, 3, 1, 4,
                            0, 6, 5, 2, 2, 0, 3, 1], ('check', 'C3', lambda a, b: True)),
]


@pytest.mark.parametrize('name,moves,patch', MUTANTS, ids=[m[0] for m in MUTANTS])
def test_each_source_condition_is_load_bearing(name, moves, patch, monkeypatch):
    p = replay(moves)
    assert p.turn == 0 and p.remaining <= 10
    exact = solve(p, max_remaining=10, position_budget=1_000_000)
    assert exact.value_for(0) == 1  # White wins with exact play.
    position = Position.from_board(p.board, 0)
    assert search_nine_rule_cover(position).status == NO_COVER  # Faithful rules: no claim.
    kind, key, function = patch
    if kind == 'clauses':
        monkeypatch.setitem(composite_module._CLAUSES, key, function)
    else:
        checks = dict(compat_module._CHECKS)
        checks[getattr(compat_module, key)] = function
        monkeypatch.setattr(compat_module, '_CHECKS', MappingProxyType(checks))
    assert search_nine_rule_cover(position).status == FOUND  # The loosened rule over-claims.


def test_results_and_witnesses_never_carry_outcomes(original):
    p, w = original
    result = search_nine_rule_cover(p)
    for value in (result, w, verify_nine_rule_witness(p, w)):
        assert value.outcome_certification == 'uncertified'
        assert value.unproven_obligations == NINE_RULE_OBLIGATIONS
        for name in ('winner', 'value', 'safe', 'proven', 'best_move'):
            assert not hasattr(value, name)
    report = analyze_nine_rules(p)
    assert report.status == 'candidates_only'
