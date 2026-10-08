"""Nine-rule Allis research pipeline: candidates, coverage and compatible covers.

This produces honest, inspectable COVERAGE evidence for all nine rules of Allis
chapter 6 under the full §7.4 matrix. It never certifies an outcome:

- a found cover is NOT a proved Black non-loss (see ``NINE_RULE_OBLIGATIONS``);
- an exhaustive failure says nothing about the game value (§5.4: the method
  then "can say nothing about the given position");
- a budget cutoff is unknown.

The CL/BI/VE certificate (``cl-bi-ve-black-nonloss-v1``), its verifier and the
executable strategy remain restricted to those three rules and never accept
anything produced here.
"""
from dataclasses import dataclass, field
from itertools import combinations
from typing import Literal

from .compatibility import INVERSES, failed_constraints, footprint
from .composite import (
    CompositeCandidate, SolutionClause, enumerate_afterevens, enumerate_baseclaims,
    enumerate_befores, enumerate_highinverses, enumerate_lowinverses,
    enumerate_specialbefores, solution_clauses,
)
from .coverage import BlackEvaluationContext, SearchStatus, backtrack_cover, black_evaluation_context
from .geometry import Group
from .position import Player, Position, validate_player
from .rules import (
    RuleCandidate, RuleName, enumerate_baseinverses, enumerate_claimevens, enumerate_verticals,
)

ALL_RULES = tuple(RuleName)
SCHEMA_VERSION = 'allis9-coverage-v1'
RULE_MODEL_ID = 'allis1988-nine-rules-research-v1'
COMPATIBILITY_MODEL_ID = 'allis1988-s7.4-full-matrix-v1'

# What a found cover does NOT establish. Each item is an open proof obligation.
NINE_RULE_OBLIGATIONS = (
    'Zugzwang control: Allis 8.1 argues Black need not check it, but no proof '
    'covers the composite rules',
    'local soundness of each composite rule (Aftereven/Before timing, Lowinverse/'
    'Highinverse parity, Baseclaim and Specialbefore variants) is Allis\'s informal '
    'argument, not a reviewed proof',
    'global composition: pairwise 7.4 compatibility is not proved to imply that '
    'all selected rules can be executed simultaneously',
    'the conditional composite response policy has no general non-loss theorem; '
    'a complete adversarial replay establishes only the particular board and policy',
    'Allis 9.2\'s "an Aftereven in the cover means Black wins" is not inferred',
    'historical reachability of the position is not checked',
)

_ENUMERATORS = (
    (RuleName.CLAIMEVEN, lambda p, c: enumerate_claimevens(p)),
    (RuleName.BASEINVERSE, lambda p, c: enumerate_baseinverses(p)),
    (RuleName.VERTICAL, lambda p, c: enumerate_verticals(p)),
    (RuleName.AFTEREVEN, enumerate_afterevens),
    (RuleName.LOWINVERSE, lambda p, c: enumerate_lowinverses(p)),
    (RuleName.HIGHINVERSE, lambda p, c: enumerate_highinverses(p)),
    (RuleName.BASECLAIM, lambda p, c: enumerate_baseclaims(p)),
    (RuleName.BEFORE, enumerate_befores),
    (RuleName.SPECIALBEFORE, enumerate_specialbefores),
)


@dataclass(frozen=True)
class RuleEvidence:
    """One candidate, its source-defined clauses and the target groups they solve."""

    candidate: RuleCandidate | CompositeCandidate
    clauses: tuple[SolutionClause, ...]
    conditional_solved_groups: tuple[Group, ...]


@dataclass(frozen=True)
class NineRuleAssignment:
    group: Group
    candidate: RuleCandidate | CompositeCandidate
    clause: str  # Label of the first clause of that candidate solving the group.


@dataclass(frozen=True)
class NineRuleReport:
    """Every candidate of the requested rules, with coverage. No outcome."""

    position: Position
    defender: Player
    rules: tuple[RuleName, ...]
    target_groups: tuple[Group, ...]
    evidence: tuple[RuleEvidence, ...]
    status: Literal['candidates_only'] = field(default='candidates_only', init=False)

    def counts(self) -> tuple[tuple[RuleName, int, int], ...]:
        """(rule, enumerated candidates, candidates solving at least one target)."""
        return tuple((rule, sum(e.candidate.rule == rule for e in self.evidence),
                      sum(e.candidate.rule == rule and bool(e.conditional_solved_groups)
                          for e in self.evidence)) for rule in self.rules)


@dataclass(frozen=True)
class NineRuleWitness:
    """Untrusted producer claim; meaningful only after independent verification."""

    context: BlackEvaluationContext
    rules: tuple[RuleName, ...]
    evidence: tuple[RuleEvidence, ...]
    assignments: tuple[NineRuleAssignment, ...]
    schema_version: str = SCHEMA_VERSION
    outcome_certification: Literal['uncertified'] = field(default='uncertified', init=False)
    unproven_obligations: tuple[str, ...] = field(default=NINE_RULE_OBLIGATIONS, init=False)


@dataclass(frozen=True)
class NineRuleSearchResult:
    status: SearchStatus
    rules: tuple[RuleName, ...]
    expanded_nodes: int
    node_budget: int
    memo_hits: int = 0
    candidate_counts: tuple[tuple[RuleName, int, int], ...] = ()
    conflict_pairs: int = 0
    witness: NineRuleWitness | None = None
    reason: str = ''
    outcome_certification: Literal['uncertified'] = field(default='uncertified', init=False)
    unproven_obligations: tuple[str, ...] = field(default=NINE_RULE_OBLIGATIONS, init=False)


def _checked_rules(rules) -> tuple[RuleName, ...]:
    if type(rules) not in (tuple, list, frozenset, set) or not rules:
        raise ValueError('rules must be a nonempty collection of RuleName')
    if any(type(rule) is not RuleName for rule in rules):
        raise ValueError('rules must be RuleName members')
    return tuple(rule for rule in ALL_RULES if rule in set(rules))


def analyze_nine_rules(position: Position, defender: Player = 1, *,
                       rules=ALL_RULES) -> NineRuleReport:
    """Enumerate the requested rules and attach coverage of defender's opponent.

    Whole-board, any turn; like ``analyze_candidates`` this does NOT establish
    the Chapter 8 evaluation context. Zero-coverage candidates are retained.
    """
    if type(position) is not Position:
        raise ValueError('position must be a Position snapshot')
    validate_player(defender)
    rules = _checked_rules(rules)
    opponent: Player = 1 if defender == 0 else 0
    targets = position.potential_groups(opponent)
    evidence = []
    for rule, enumerate_rule in _ENUMERATORS:
        if rule not in rules:
            continue
        for candidate in enumerate_rule(position, defender):
            clauses = solution_clauses(candidate, position)
            evidence.append(RuleEvidence(candidate, clauses, tuple(
                g for g in targets if any(c.solves(g) for c in clauses))))
    return NineRuleReport(position, defender, rules, targets, tuple(evidence))


def conflict_masks(candidates) -> tuple[list[int], int]:
    """Incompatibility bitmasks equal to pairwise ``not compatible`` (tested).

    Pairs sharing no square satisfy constraints 1 and 3 trivially, so they can
    only conflict through constraint 2 or 4, both of which involve an inverse.
    Only overlapping pairs and pairs with an inverse are evaluated in full.
    """
    prints = [footprint(c) for c in candidates]
    conflicts = [1 << i for i in range(len(candidates))]
    users: dict = {}
    for i, fp in enumerate(prints):
        for square in fp.squares:
            users[square] = users.get(square, 0) | (1 << i)
    inverses = [i for i, fp in enumerate(prints) if fp.rule in INVERSES]
    inverse_set = set(inverses)
    pairs = 0

    def check(i: int, j: int) -> None:
        nonlocal pairs
        if candidates[i] == candidates[j] or failed_constraints(candidates[i], candidates[j]):
            if not conflicts[i] >> j & 1:
                pairs += 1
            conflicts[i] |= 1 << j
            conflicts[j] |= 1 << i

    for i, fp in enumerate(prints):
        overlap = 0
        for square in fp.squares:
            overlap |= users[square]
        overlap &= ~((1 << (i + 1)) - 1)  # Each unordered pair once (j > i).
        while overlap:
            low = overlap & -overlap
            overlap ^= low
            check(i, low.bit_length() - 1)
    for i in inverses:
        for j in range(len(candidates)):
            if j != i and not prints[i].squares & prints[j].squares:
                if j < i and j in inverse_set:
                    continue  # Already checked as (j, i).
                check(i, j)
    return conflicts, pairs


def search_nine_rule_cover(position: Position, *, defender: Player = 1,
                           node_budget: int = 100_000, rules=ALL_RULES) -> NineRuleSearchResult:
    """Find a §7.4-compatible collection solving every White potential group.

    Same Black/White-to-move whole-board context and bounded MRV DFS as the
    CL/BI/VE search, over the complete enumeration of the requested rules, with
    memoized fully-explored failures. EXHAUSTIVE_NO_COVER is relative to exactly
    those rules and this research model; nothing here is a game value.
    """
    if type(node_budget) is not int or node_budget < 0:
        raise ValueError('node_budget must be a nonnegative integer')
    rules = _checked_rules(rules)
    try:
        context = black_evaluation_context(position, defender)
    except (ValueError, TypeError, AttributeError) as exc:
        return NineRuleSearchResult(SearchStatus.UNSUPPORTED_CONTEXT, rules, 0, node_budget,
                                    reason=str(exc))
    report = analyze_nine_rules(context.position, context.defender, rules=rules)
    useful = tuple(e for e in report.evidence if e.conditional_solved_groups)
    groups = context.target_groups
    index = {g: i for i, g in enumerate(groups)}
    masks = tuple(sum(1 << index[g] for g in e.conditional_solved_groups) for e in useful)
    conflicts, pairs = conflict_masks(tuple(e.candidate for e in useful))
    outcome = backtrack_cover(len(groups), masks, conflicts, node_budget, memoize=True)
    common = dict(rules=rules, expanded_nodes=outcome.expanded, node_budget=node_budget,
                  memo_hits=outcome.memo_hits, candidate_counts=report.counts(),
                  conflict_pairs=pairs)
    if outcome.selected is None:
        status = (SearchStatus.BUDGET_EXHAUSTED if outcome.exhausted
                  else SearchStatus.EXHAUSTIVE_NO_COVER)
        return NineRuleSearchResult(status, **common)
    chosen = tuple(useful[i] for i in sorted(outcome.selected))
    assignments = []
    for group in groups:
        evidence = next(e for e in chosen if group in e.conditional_solved_groups)
        clause = next(c.label for c in evidence.clauses if c.solves(group))
        assignments.append(NineRuleAssignment(group, evidence.candidate, clause))
    witness = NineRuleWitness(context, rules, chosen, tuple(assignments))
    return NineRuleSearchResult(SearchStatus.FOUND, witness=witness, **common)


def pairwise_conflicts(candidates) -> tuple[tuple[int, int], ...]:
    """Reference O(n^2) conflict list, for tests and small inspections."""
    return tuple((i, j) for i, j in combinations(range(len(candidates)), 2)
                 if candidates[i] == candidates[j]
                 or failed_constraints(candidates[i], candidates[j]))
