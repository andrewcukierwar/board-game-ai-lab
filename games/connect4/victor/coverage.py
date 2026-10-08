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


@dataclass(frozen=True)
class BacktrackOutcome:
    selected: tuple[int, ...] | None
    expanded: int
    exhausted: bool
    memo_hits: int = 0


def backtrack_cover(group_count: int, masks, conflicts, node_budget: int, *,
                    memoize: bool = False) -> BacktrackOutcome:
    """Allis §9.2 FindChosenSet as a bounded, deterministic bitmask DFS.

    ``masks[j]`` is candidate j's solved-group bitmask; ``conflicts[j]`` holds j
    itself and every candidate incompatible with it. The budget counts visited
    recursive states, including root and solution leaves; zero permits none.
    Branch on the uncovered group with fewest compatible candidates (ties: lowest
    group index), trying candidates in ascending index order. With ``memoize``,
    a completely explored failing (uncovered, blocked) state is never revisited:
    the subproblem depends only on that pair, so this cannot change the answer.
    """
    options = [0] * group_count
    for j, mask in enumerate(masks):
        while mask:
            low = mask & -mask
            options[low.bit_length() - 1] |= 1 << j
            mask ^= low
    expanded = hits = 0
    exhausted = False
    failed: set[tuple[int, int]] = set()

    def visit(uncovered: int, blocked: int, selected: tuple[int, ...]) -> tuple[int, ...] | None:
        nonlocal expanded, exhausted, hits
        if memoize and (uncovered, blocked) in failed:
            hits += 1
            return None
        if expanded >= node_budget:
            exhausted = True
            return None
        expanded += 1
        if not uncovered:
            return selected
        best, best_count, rest = 0, None, uncovered
        while rest:
            low = rest & -rest
            rest ^= low
            available = options[low.bit_length() - 1] & ~blocked
            count = available.bit_count()
            if best_count is None or count < best_count:
                best, best_count = available, count
                if not count:
                    break
        while best:
            low = best & -best
            best ^= low
            j = low.bit_length() - 1
            found = visit(uncovered & ~masks[j], blocked | conflicts[j], selected + (j,))
            if found is not None:
                return found
            if exhausted:
                return None
        if memoize:
            failed.add((uncovered, blocked))
        return None

    selected = visit((1 << group_count) - 1, 0, ())
    return BacktrackOutcome(selected, expanded, exhausted, hits)


def search_covering_set(position: Position, *, defender: Player = 1,
                        node_budget: int = 10_000) -> CoveringSetResult:
    """Complete within CL/BI/VE only if the bounded DFS finishes.

    Budget counts visited recursive states, including root and solution leaves.
    Zero permits no states. Preprocessing is fixed by the 69-group/56-candidate
    standard-board domain. Branch on the uncovered group with fewest currently
    compatible candidates; ties use canonical group order. Branch candidates use
    Phase 6A's CL/BI/VE enumeration order. No caller coverage/universe is accepted.
    The nine-rule search is ``nine_rules.search_nine_rule_cover``.
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
    conflicts = [1 << i for i in range(len(evidence))]
    for i, j in combinations(range(len(evidence)), 2):
        if not compatible(evidence[i].candidate, evidence[j].candidate):
            conflicts[i] |= 1 << j
            conflicts[j] |= 1 << i
    outcome = backtrack_cover(len(groups), masks, conflicts, node_budget)
    selected, expanded = outcome.selected, outcome.expanded
    if selected is None:
        status = (SearchStatus.BUDGET_EXHAUSTED if outcome.exhausted
                  else SearchStatus.EXHAUSTIVE_NO_COVER)
        return CoveringSetResult(status, expanded, node_budget)
    chosen = tuple(evidence[i] for i in sorted(selected))
    assignments = tuple(CoverageAssignment(g, next(e.candidate for e in chosen
                        if g in e.conditional_solved_groups)) for g in groups)
    witness = CoverageWitness(context, chosen, assignments)
    return CoveringSetResult(SearchStatus.FOUND, expanded, node_budget, witness)
