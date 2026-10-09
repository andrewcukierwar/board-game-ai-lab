"""Phase 6B.3 adversarial audit checks; NO outcome or certificate acceptance.

The audit model (``victor_validation.independent_audit``) shares no code with
production Victor, the engine, grounding or the Phase 6B.2 oracle/harness. Each
test below pins one step of the CL/BI/VE theorem or demonstrates that relaxing
one hypothesis yields an exact White win, so finite agreement is not vacuous.
"""
from itertools import combinations
from pathlib import Path
from random import Random
import subprocess
import sys

import pytest

from games.connect4.victor import (
    CoverageWitness, Position, RuleCandidate, RuleName, SearchStatus, Square,
    VerificationStatus, analyze_candidates, black_evaluation_context, search_covering_set,
    verify_coverage_witness,
)
from games.connect4.victor.contracts import CoverageAssignment
from victor_validation import independent_audit as audit
from victor_validation.exact_oracle import EndgamePosition, replay as oracle_replay, solve

KINDS = {'CL': RuleName.CLAIMEVEN, 'BI': RuleName.BASEINVERSE, 'VE': RuleName.VERTICAL}


def sq(text):
    return 'abcdefg'.index(text[0]), int(text[1]) - 1


def moves(text):
    return tuple(int(c) for c in text)


def production_witness(board, raw_rules):
    """Wrap a hand-chosen raw rule set as an UNTRUSTED production witness."""
    position = Position.from_board([list(row) for row in board.matrix()], 0)
    context = black_evaluation_context(position)
    wanted = {RuleCandidate(KINDS[k], tuple(Square(5 - h, c) for c, h in (a, b)))
              for k, a, b in raw_rules}
    evidence = tuple(e for e in analyze_candidates(position, 1).evidence if e.candidate in wanted)
    assert {e.candidate for e in evidence} == wanted
    assignments = tuple(CoverageAssignment(g, next(e.candidate for e in evidence
                                                   if g in e.conditional_solved_groups))
                        for g in context.target_groups)
    return position, CoverageWitness(context, evidence, assignments)


def oracle_white_value(board):
    return solve(EndgamePosition.from_board(tuple(map(tuple, board.matrix())), board.turn),
                 max_remaining=10, position_budget=1_000_000).value_for(0)


def test_audit_model_imports_nothing_from_production_or_prior_validation():
    tests_dir = str(Path(__file__).parent.resolve())
    code = f'''
import sys, importlib.abc
sys.path.insert(0, {tests_dir!r})
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split('.')[0]
        if root in ('games', 'torch', 'numpy', 'api') or fullname in (
                'victor_validation.exact_oracle', 'victor_validation.strategic_harness',
                'victor_validation.reference_game'):
            raise AssertionError('forbidden import: ' + fullname)
sys.meta_path.insert(0, Block())
from victor_validation import independent_audit as a
assert len(a.LINES) == 69
assert a.replay((0, 1, 0, 1, 0, 1, 0)).has_four(0)
assert a.white_can_force_win(a.replay((3, 3, 2, 2)), budget=10)[0] is None  # cutoff = unknown
'''
    run = subprocess.run([sys.executable, '-I', '-c', code], capture_output=True, text=True, timeout=20)
    assert run.returncode == 0, run.stderr


def test_spare_move_parity_lemma_over_every_height_vector():
    # Odd empties on Black's turn => a landing square on an EVEN row exists, and
    # an even-row square is never a CL lower (always odd) or an untouched CL upper.
    assert audit.parity_lemma_exhaustive() == 411_771  # Black-to-move vectors of 7**7.


def test_audit_solver_agrees_with_phase6b2_oracle_on_bounded_sample():
    rng = Random(20261008)
    compared = 0
    for plies in (34, 36, 38, 40) * 16:
        history = audit.sample_history(rng, plies, follow_up=0.4)
        if history is None:
            continue
        board = audit.replay(history)
        assert board.matrix() == tuple(''.join(r) for r in oracle_replay(history).board)
        assert audit.exact_value(board)[0] == oracle_white_value(board)
        compared += 1
    assert compared >= 30


# Section 5.3 conflict shape: BI d3-e3 shares e3 with CL e3-e4's lower trigger.
OVERLAP = ('0610216610160066212202555525154343',
           (('CL', sq('d5'), sq('d6')), ('BI', sq('d3'), sq('e3')), ('CL', sq('e3'), sq('e4'))))
# "Claimodd": two Claimeven-like claims on ODD upper squares f3 and f5.
CLAIMODD = ('2024243302333341506611441164162266',
            (('CLODD', sq('f2'), sq('f3')), ('CLODD', sq('f4'), sq('f5'))))


def test_overlapping_cover_is_a_real_false_implication_and_is_rejected():
    board = audit.replay(moves(OVERLAP[0]))
    rules = OVERLAP[1]
    assert board.empty_count == 8 and board.turn == 0
    # Only H3 (disjointness) fails; the overlapping set is otherwise complete.
    assert audit.check_hypotheses(board, rules) == (
        "H3: ('CL', (4, 2), (4, 3)) overlaps another rule",)
    assert audit.find_cover(board, allow_overlap=True)[0] == 'found'
    assert audit.find_cover(board)[0] == 'none'
    position = Position.from_board([list(r) for r in board.matrix()], 0)
    assert search_covering_set(position).status == SearchStatus.EXHAUSTIVE_NO_COVER
    # Two independent exact searches: White wins.
    assert audit.white_can_force_win(board)[0] is True
    assert oracle_white_value(board) == 1
    # White e3 demands BOTH d3 (BI) and e4 (CL); each single reply loses exactly.
    after = board.play(sq('e3'))
    for reply in ('d3', 'e4'):
        assert audit.white_can_force_win(after.play(sq(reply)))[0] is True
    assert audit.check_strategy(board, rules)[0] == 'violated'
    # Production refuses exactly this witness for the right reason.
    p, witness = production_witness(board, rules)
    verdict = verify_coverage_witness(p, witness)
    assert verdict.status == VerificationStatus.REJECTED
    assert verdict.rejection_reasons == ('selected candidates conflict',)


def test_claimodd_cover_is_a_real_false_implication_and_unrepresentable():
    board = audit.replay(moves(CLAIMODD[0]))
    rules = CLAIMODD[1]
    assert audit.find_cover(board, claimodd=True) == ('found', rules)
    assert audit.find_cover(board)[0] == 'none'
    assert audit.white_can_force_win(board)[0] is True
    assert oracle_white_value(board) == 1
    assert audit.check_strategy(board, rules)[0] == 'violated'
    with pytest.raises(ValueError, match='parity'):
        RuleCandidate(RuleName.CLAIMEVEN, (Square.from_name('f2'), Square.from_name('f3')))


# Mixed column f: CL f1-f2, free f3, VE f4-f5, free f6; plus CL c5-c6, BI e6-g6.
MIXED = ('33003316106600034431441142662221',
         (('BI', sq('e6'), sq('g6')), ('CL', sq('c5'), sq('c6')),
          ('CL', sq('f1'), sq('f2')), ('VE', sq('f4'), sq('f5'))))


def test_mixed_cl_bi_ve_cover_survives_every_permitted_spare_schedule():
    board = audit.replay(moves(MIXED[0]))
    rules = MIXED[1]
    assert board.empty_count == 10 and audit.check_hypotheses(board, rules) == ()
    p, witness = production_witness(board, rules)
    assert verify_coverage_witness(p, witness).status == VerificationStatus.VERIFIED_UNCERTIFIED
    status, detail, states = audit.check_strategy(board, rules)
    assert (status, detail) == ('held', None) and states > 1
    assert audit.white_can_force_win(board)[0] is False
    assert oracle_white_value(board) <= 0


# Every column top is White with White to move: the previous mover (Black) has
# no removable top stone, so no legal history reaches this board.
UNREACHABLE = ('X   XX ', 'OXX OO ', 'XOO XX ', 'OXX OO ', 'OOOXOOX', 'XXXOOOX')


def test_theorem_does_not_need_reachability_and_verifier_does_not_check_it():
    board = audit.Board.from_matrix(UNREACHABLE)
    assert board.turn == 0 and audit.reachable(board) is False
    status, rules = audit.find_cover(board)
    assert status == 'found' and audit.check_hypotheses(board, rules) == ()
    assert audit.check_strategy(board, rules)[:2] == ('held', None)
    assert audit.white_can_force_win(board)[0] is False
    assert oracle_white_value(board) <= 0
    # Boundary record: basic validation accepts it and production verifies coverage.
    position = Position.from_board([list(r) for r in UNREACHABLE], 0)
    result = search_covering_set(position)
    assert result.status == SearchStatus.FOUND
    assert verify_coverage_witness(position, result.witness).status == \
        VerificationStatus.VERIFIED_UNCERTIFIED


def test_thesis_diagram_8_1_declared_rules_are_disjoint_despite_d5_d6_attribution():
    # Rendered from PDF p.51 (White X / Black O, Black to move, odd threat a3).
    # Source check only: White evaluation contexts remain unimplemented/unaccepted.
    board = audit.Board.from_matrix(
        ('       ', '       ', '   OX  ', ' XXXO  ', ' XOOO  ', 'XOXXO  '))
    assert board.turn == 1 and board.wins_at(0, sq('a3'))
    declared = [('CL', sq(a), sq(b)) for a, b in (
        ('b5', 'b6'), ('c5', 'c6'), ('f1', 'f2'), ('f3', 'f4'), ('f5', 'f6'),
        ('g3', 'g4'), ('g5', 'g6'))] + [('BI', sq('d5'), sq('e5'))]
    cells = [audit.bit(*r[1]) | audit.bit(*r[2]) for r in declared]
    assert all(not x & y for x, y in combinations(cells, 2))
    above_a2 = sum(audit.bit(0, h) for h in range(2, 6))
    relevant = [m for m in audit.LINES if not m & board.white and not m & above_a2]
    assert len(relevant) == 19
    assert all(any(audit.coverage_mask(r) & m == audit.coverage_mask(r) for r in declared)
               for m in relevant)
    # The text credits b6-e6, c6-f6, d6-g6 to "Claimeven d5-d6", which would
    # overlap BI d5-e5; declared CLs on b6/c6/f6/g6 already cover all three.
    for group in (('b6', 'c6', 'd6', 'e6'), ('c6', 'd6', 'e6', 'f6'), ('d6', 'e6', 'f6', 'g6')):
        mask = sum(audit.bit(*sq(s)) for s in group)
        assert any(r[0] == 'CL' and audit.coverage_mask(r) & mask for r in declared)
    assert cells[-1] & (audit.bit(*sq('d5')) | audit.bit(*sq('d6')))


def test_bounded_campaign_smoke_reports_no_violation_or_disagreement():
    stats = audit.run_campaign(seed=6203, per_setting=4, empties=(8, 12, 16))
    assert stats['positions'] > 20
    for key in ('existence_disagreements', 'theorem_violations', 'strategy_violations',
                'hypothesis_rejections', 'unreachable_violations'):
        assert stats[key] == [], key
    assert stats['unknown_values'] == 0 and stats['strategy_unknown'] == 0
