"""Bounded finite coverage tests; no game values are asserted."""
from copy import copy
from itertools import combinations

import pytest

from games.connect4.connect4 import Connect4
from games.connect4.victor import (
    ALL_GROUPS, Position, RuleName, SearchStatus, VerificationStatus,
    black_evaluation_context, enumerate_candidates, search_covering_set, verify_coverage_witness,
)

# Diagram 6.1's legal reconstruction (not an Allis move list).
THESIS_MOVES = [2, 3, 3, 3, 3, 3, 3, 4]
# Constructed bounded endgames; all moves are replayed, never assumed reachable.
SAT_MOVES = [1,4,5,4,6,6,0,0,2,6,6,5,1,1,5,2,4,2,2,5,1,1,2,5,5,3,4,2,0,4,3,6,1,6]
UNSAT_MOVES = [3,2,0,0,4,1,5,3,0,3,1,0,2,4,5,5,5,3,4,1,5,0,4,4,5,6,2,6,0,6,6,6,3,1,3,1]


def play(moves):
    game = Connect4()
    for column in moves:
        assert game.make_move(column)
    return Position.from_board(game.board, game.current_player)


def raw_coverage(position, candidate, targets):
    needed = {candidate.squares[1]} if candidate.rule == RuleName.CLAIMEVEN else set(candidate.squares)
    return {g for g in targets if needed <= set(g.squares)}


def subset_reference(position):
    """Exhaust subsets, unlike MRV backtracking; bounded to <= 10 candidates."""
    candidates = enumerate_candidates(position)
    assert len(candidates) <= 10
    targets = {g for g in ALL_GROUPS if all(position.board[s.row_index][s.column] != 'O' for s in g.squares)}
    for size in range(len(candidates) + 1):
        for selected in combinations(candidates, size):
            used = [s for c in selected for s in c.squares]
            if len(set(used)) != len(used):
                continue
            covered = set().union(*(raw_coverage(position, c, targets) for c in selected))
            if covered == targets:
                return True
    return False


def test_black_targets_recomputed_with_exact_identities():
    p = play(THESIS_MOVES)
    ctx = black_evaluation_context(p)
    expected = tuple(g for g in ALL_GROUPS if all(p.board[s.row_index][s.column] != 'O' for s in g.squares))
    assert ctx.position == p and ctx.position is not p  # Revalidated detached snapshot.
    assert ctx.defender == 1 and ctx.opponent == 0
    assert ctx.target_groups == expected and len(expected) > 0
    # Potential groups include completely empty lines as well as White stones;
    # this is deliberately broader than an immediate-threat scan.
    assert any(all(p.board[s.row_index][s.column] == ' ' for s in g.squares) for g in expected)
    assert any(any(p.board[s.row_index][s.column] == 'X' for s in g.squares) for g in expected)
    assert black_evaluation_context(play([])).target_groups == ALL_GROUPS


def test_thesis_diagram_6_1_cover_has_exact_conditional_claims_and_full_assignments():
    p = play(THESIS_MOVES)
    result = search_covering_set(p)
    assert result.status == SearchStatus.FOUND and result.witness is not None
    witness = result.witness
    assert len(witness.evidence) == len({e.candidate for e in witness.evidence})
    for evidence in witness.evidence:
        assert set(evidence.conditional_solved_groups) == raw_coverage(p, evidence.candidate, witness.context.target_groups)
    for a, b in combinations(witness.evidence, 2):
        assert set(a.candidate.squares).isdisjoint(b.candidate.squares)
    assert tuple(a.group for a in witness.assignments) == witness.context.target_groups
    assert verify_coverage_witness(p, witness).status == VerificationStatus.VERIFIED_UNCERTIFIED
    assert result.outcome_certification == witness.outcome_certification == 'uncertified'
    assert not hasattr(result, 'winner') and not hasattr(result, 'safe')


@pytest.mark.parametrize('moves', [SAT_MOVES, UNSAT_MOVES])
def test_search_matches_exhaustive_independent_subset_reference_and_mirror(moves):
    p = play(moves)
    mirror = Position.from_board([list(reversed(row)) for row in p.board], p.player_to_move)
    for position in (p, mirror):
        expected = subset_reference(position)
        result = search_covering_set(position, node_budget=1000)
        assert (result.status == SearchStatus.FOUND) == expected
        assert result.status in (SearchStatus.FOUND, SearchStatus.EXHAUSTIVE_NO_COVER)
        assert search_covering_set(position, node_budget=1000) == result
        if result.witness:
            assert verify_coverage_witness(position, result.witness).status == VerificationStatus.VERIFIED_UNCERTIFIED
    assert {g.reflected() for g in black_evaluation_context(p).target_groups} == set(black_evaluation_context(mirror).target_groups)


def test_incompatibility_prevents_cover_even_when_every_group_has_a_candidate():
    p = play(UNSAT_MOVES)
    targets = black_evaluation_context(p).target_groups
    # Every group is individually coverable. It is the combination that fails.
    assert all(any(g in raw_coverage(p, c, targets) for c in enumerate_candidates(p)) for g in targets)
    assert search_covering_set(p).status == SearchStatus.EXHAUSTIVE_NO_COVER


def test_uncovered_group_finishes_exhaustively_without_game_outcome():
    result = search_covering_set(play([]), node_budget=1)
    assert result.status == SearchStatus.EXHAUSTIVE_NO_COVER and result.expanded_nodes == 1
    assert result.witness is None and result.outcome_certification == 'uncertified'


@pytest.mark.parametrize('moves', [THESIS_MOVES, SAT_MOVES, UNSAT_MOVES])
def test_budget_boundary_root_and_solution_nodes_counted(moves):
    p = play(moves)
    completed = search_covering_set(p, node_budget=1000)
    assert completed.expanded_nodes > 1
    for budget in (0, 1, completed.expanded_nodes - 1):
        cutoff = search_covering_set(p, node_budget=budget)
        assert cutoff.status == SearchStatus.BUDGET_EXHAUSTED
        assert cutoff.expanded_nodes == budget and cutoff.witness is None
    exact = search_covering_set(p, node_budget=completed.expanded_nodes)
    assert exact.status == completed.status and exact.witness == completed.witness


@pytest.mark.parametrize('budget', [-1, True, 1.5, None])
def test_invalid_budget_is_rejected(budget):
    with pytest.raises(ValueError, match='node_budget'):
        search_covering_set(play([]), node_budget=budget)


@pytest.mark.parametrize('position,defender', [
    (play([0]), 1), (play([]), 0), (play([]), True), (play([]), 2),
    (play([0,1,0,1,0,1,0]), 1), (play([0,1,0,1,2,1,2,1]), 1),
    (Position.from_board([list(r) for r in ['XXOOXXO','OOXXOOX'] * 3], 0), 1),
    (None, 1),
])
def test_invalid_or_unsupported_evaluation_context(position, defender):
    result = search_covering_set(position, defender=defender)
    assert result.status == SearchStatus.UNSUPPORTED_CONTEXT
    assert result.expanded_nodes == 0 and result.witness is None and result.reason
    with pytest.raises(ValueError):
        black_evaluation_context(position, defender)


def test_bypassed_position_constructor_is_revalidated():
    p = copy(play([]))
    object.__setattr__(p, 'player_to_move', 1)
    assert search_covering_set(p).status == SearchStatus.UNSUPPORTED_CONTEXT
    p = copy(play([]))
    board = [list(r) for r in p.board]
    board[4][0] = 'X'
    object.__setattr__(p, 'board', board)
    assert search_covering_set(p).status == SearchStatus.UNSUPPORTED_CONTEXT


def test_mirrored_witness_transports_all_exact_identities():
    from dataclasses import replace
    from games.connect4.victor import CandidateEvidence
    from games.connect4.victor.contracts import CoverageAssignment
    p = play(THESIS_MOVES)
    mirror = Position.from_board([list(reversed(row)) for row in p.board], 0)
    w = search_covering_set(p).witness
    reflected = replace(w, context=black_evaluation_context(mirror), evidence=tuple(
        CandidateEvidence(e.candidate.reflected(), tuple(sorted(g.reflected() for g in e.conditional_solved_groups)))
        for e in w.evidence), assignments=tuple(CoverageAssignment(a.group.reflected(), a.candidate.reflected()) for a in w.assignments))
    assert verify_coverage_witness(mirror, reflected).status == VerificationStatus.VERIFIED_UNCERTIFIED
    assert verify_coverage_witness(p, reflected).status == VerificationStatus.REJECTED
