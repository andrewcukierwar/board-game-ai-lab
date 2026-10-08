"""Bounded compatible coverage research, never a game-theoretic certificate."""
from dataclasses import dataclass, field
from enum import Enum
from itertools import combinations
from typing import Literal

from .compatibility import compatible
from .contracts import CoverageAssignment
from .evidence import CandidateEvidence, analyze_candidates
from .geometry import Group
from .position import Player, Position, validate_player


class SearchStatus(str, Enum):
    FOUND = 'compatible_covering_set_found'
    EXHAUSTIVE_NO_COVER = 'no_cover_in_supported_universe'
    BUDGET_EXHAUSTED = 'unknown_budget_exhausted'
    UNSUPPORTED_CONTEXT = 'unsupported_or_invalid_context'


@dataclass(frozen=True)
class BlackEvaluationContext:
    """A whole-board, White-to-move target inventory, NOT Chapter 8 acceptance."""
    position: Position
    defender: Player
    opponent: Player
    target_groups: tuple[Group, ...]
    mode: Literal['black_opponent_to_move'] = 'black_opponent_to_move'


def black_evaluation_context(position: Position, defender: Player = 1) -> BlackEvaluationContext:
    """Revalidate the snapshot and derive targets. No supplied target declarations."""
    if type(position) is not Position:
        raise ValueError('position must be a Position snapshot')
    position = Position.from_board(position.board, position.player_to_move)
    validate_player(defender)
    if defender != 1:
        raise ValueError('White evaluation contexts are not implemented')
    if position.player_to_move != 0 or position.terminal:
        raise ValueError('Black coverage requires a nonterminal White-to-move position')
    return BlackEvaluationContext(position, 1, 0, position.potential_groups(0))


@dataclass(frozen=True)
class CoverageWitness:
    """Untrusted explicit identities/claims; only a coverage witness after checking."""
    context: BlackEvaluationContext
    evidence: tuple[CandidateEvidence, ...]
    assignments: tuple[CoverageAssignment, ...]
    schema_version: str = 'coverage-6b1-v1'
    outcome_certification: Literal['uncertified'] = field(default='uncertified', init=False)


@dataclass(frozen=True)
class CoveringSetResult:
    status: SearchStatus
    expanded_nodes: int
    node_budget: int
    witness: CoverageWitness | None = None
    reason: str = ''
    outcome_certification: Literal['uncertified'] = field(default='uncertified', init=False)


def search_covering_set(position: Position, *, defender: Player = 1,
                        node_budget: int = 10_000) -> CoveringSetResult:
    """Complete within CL/BI/VE only if the bounded DFS finishes.

    Budget counts visited recursive states, including root and solution leaves.
    Zero permits no states. Preprocessing is fixed by the 69-group/56-candidate
    standard-board domain. Branch on the uncovered group with fewest currently
    compatible candidates; ties use canonical group order. Branch candidates use
    Phase 6A's CL/BI/VE enumeration order. No caller coverage/universe is accepted.
    """
    if type(node_budget) is not int or node_budget < 0:
        raise ValueError('node_budget must be a nonnegative integer')
    try:
        context = black_evaluation_context(position, defender)
    except (ValueError, TypeError, AttributeError) as exc:
        return CoveringSetResult(SearchStatus.UNSUPPORTED_CONTEXT, 0, node_budget, reason=str(exc))
    report = analyze_candidates(context.position, context.defender)
    # Removing empty coverage is safe: such a candidate cannot help any cover.
    evidence = tuple(e for e in report.evidence if e.conditional_solved_groups)
    groups = context.target_groups
    masks = tuple(sum(1 << i for i, g in enumerate(groups)
                      if g in e.conditional_solved_groups) for e in evidence)
    options = tuple(tuple(j for j, mask in enumerate(masks) if mask & (1 << i))
                    for i in range(len(groups)))
    conflicts = [1 << i for i in range(len(evidence))]
    for i, j in combinations(range(len(evidence)), 2):
        if not compatible(evidence[i].candidate, evidence[j].candidate):
            conflicts[i] |= 1 << j
            conflicts[j] |= 1 << i
    expanded = 0
    exhausted = False

    def visit(uncovered: int, blocked: int, selected: tuple[int, ...]) -> tuple[int, ...] | None:
        nonlocal expanded, exhausted
        if expanded >= node_budget:
            exhausted = True
            return None
        expanded += 1
        if not uncovered:
            return selected
        choices = min((tuple(j for j in options[i] if not blocked & (1 << j))
                       for i in range(len(groups)) if uncovered & (1 << i)), key=len)
        for j in choices:
            found = visit(uncovered & ~masks[j], blocked | conflicts[j], selected + (j,))
            if found is not None:
                return found
            if exhausted:
                return None
        return None

    selected = visit((1 << len(groups)) - 1, 0, ())
    if selected is None:
        status = SearchStatus.BUDGET_EXHAUSTED if exhausted else SearchStatus.EXHAUSTIVE_NO_COVER
        return CoveringSetResult(status, expanded, node_budget)
    chosen = tuple(evidence[i] for i in sorted(selected))
    assignments = tuple(CoverageAssignment(g, next(e.candidate for e in chosen
                        if g in e.conditional_solved_groups)) for g in groups)
    witness = CoverageWitness(context, chosen, assignments)
    return CoveringSetResult(SearchStatus.FOUND, expanded, node_budget, witness)
