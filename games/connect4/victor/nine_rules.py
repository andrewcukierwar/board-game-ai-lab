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

from .compatibility import (
    C1, C2, C3, C4, INVERSES, _COLUMN, compact, compact_conflict, failed_constraints,
    footprint, required_constraints,
)
from . import compatibility as _compatibility
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
_CANONICAL_CHECKS = _compatibility._CHECKS
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
    # Bitmask form of SolutionClause.solves over all targets at once: a clause
    # solves the groups meeting EVERY requirement set (equivalence is tested).
    touching: dict = {}
    for i, g in enumerate(targets):
        for square in g.squares:
            touching[square] = touching.get(square, 0) | (1 << i)
    everything = (1 << len(targets)) - 1
    evidence = []
    for rule, enumerate_rule in _ENUMERATORS:
        if rule not in rules:
            continue
        for candidate in enumerate_rule(position, defender):
            clauses = solution_clauses(candidate, position)
            solved = 0
            for clause in clauses:
                groups = everything
                for required in clause.requirements:
                    met = 0
                    for square in required:
                        met |= touching.get(square, 0)
                    groups &= met
                solved |= groups
            evidence.append(RuleEvidence(candidate, clauses, tuple(
                g for i, g in enumerate(targets) if solved >> i & 1)))
    return NineRuleReport(position, defender, rules, targets, tuple(evidence))


def conflict_masks(candidates) -> tuple[list[int], int]:
    """Incompatibility bitmasks equal to pairwise ``not compatible`` (tested).

    Each §7.4 constraint is evaluated for whole candidate classes with bitmasks:

    - constraint 1 (and the disjointness part of 4) fails exactly for
      overlapping pairs;
    - constraint 3 fails for a pair overlapping in some column whose two column
      parts differ, or are equal but contain a Specialbefore square;
    - constraint 2 fails for Claimeven parts at or below an inverse's top
      square in an inverse column;
    - constraint 4's column-set clause is checked for each disjoint inverse pair.

    Identical instances always conflict. Only rule pairs whose §7.4 entry
    requires a constraint are marked by it. This bitmask form encodes the
    canonical checks; if ``compatibility._CHECKS`` has been substituted (e.g. by a
    mutation test), the pairwise ``failed_constraints`` reference is used instead.
    """
    if _compatibility._CHECKS is not _CANONICAL_CHECKS:
        conflicts = [1 << i for i in range(len(candidates))]
        found = pairwise_conflicts(candidates)
        for i, j in found:
            conflicts[i] |= 1 << j
            conflicts[j] |= 1 << i
        return conflicts, len(found)
    n = len(candidates)
    fast = [compact(footprint(c)) for c in candidates]
    conflicts = [1 << i for i in range(n)]
    identity: dict = {}
    for i, c in enumerate(candidates):
        identity[c] = identity.get(c, 0) | (1 << i)
    users = [0] * 42
    claimeven_users: dict = {}
    by_rule: dict = {}
    parts: list = []
    part_users: dict = {}
    special_users = [0] * 7
    for i, fp in enumerate(fast):
        bit = 1 << i
        rest = fp.squares
        while rest:
            low = rest & -rest
            rest ^= low
            users[low.bit_length() - 1] |= bit
        for part in fp.claimevens:
            claimeven_users[part] = claimeven_users.get(part, 0) | bit
        by_rule[fp.rule] = by_rule.get(fp.rule, 0) | bit
        mine = tuple((c, fp.squares & _COLUMN[c]) for c in range(7) if fp.squares & _COLUMN[c])
        parts.append(mine)
        for c, part in mine:
            part_users[c, part] = part_users.get((c, part), 0) | bit
            if part & fp.special:
                special_users[c] |= bit

    def requiring(code):
        return {rule: sum(members for other, members in by_rule.items()
                          if code in required_constraints(rule, other)) for rule in by_rule}

    needs_c1 = requiring(C1)
    needs_c4 = requiring(C4)
    needs_c3 = requiring(C3)
    needs_c2 = requiring(C2)

    def touching(mask):
        found = 0
        while mask:
            low = mask & -mask
            mask ^= low
            found |= users[low.bit_length() - 1]
        return found

    for i, fp in enumerate(fast):
        row = conflicts[i] | identity[candidates[i]]
        c3 = 0
        for c, part in parts[i]:
            near = touching(part)
            if part & fp.special:
                c3 |= near
            else:
                c3 |= near & ~(part_users[c, part] & ~special_users[c])
        overlap = touching(fp.squares)
        row |= overlap & (needs_c1[fp.rule] | needs_c4[fp.rule])
        row |= c3 & needs_c3[fp.rule]
        conflicts[i] = row

    def mark(i: int, others: int) -> None:
        conflicts[i] |= others
        while others:
            low = others & -others
            others ^= low
            conflicts[low.bit_length() - 1] |= 1 << i

    inverses = [i for i, fp in enumerate(fast) if fp.rule in INVERSES]
    for k, i in enumerate(inverses):
        inverse = fast[i]
        below = 0
        for column, top in inverse.inverse_top.items():
            for row in range(1, top + 1):
                below |= claimeven_users.get((column, row), 0)
        mark(i, below & needs_c2[inverse.rule] & ~sum(1 << j for j in inverses))
        for j in inverses[k + 1:]:
            if not inverse.squares & fast[j].squares and compact_conflict(inverse, fast[j]):
                mark(i, 1 << j)
    pairs = (sum(c.bit_count() for c in conflicts) - n) // 2
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
