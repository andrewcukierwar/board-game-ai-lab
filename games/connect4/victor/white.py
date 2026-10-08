"""Allis §§8.2–8.4 restricted White contexts, never automatic White wins.

Claims below are source-defined conditional exclusions. They require successful
execution on the remainder and the reserved columns. No White theorem is added
to the independent CL/BI/VE certificate boundary.
"""
from dataclasses import dataclass
from itertools import combinations

from .compatibility import footprint
from .composite import SolutionClause
from .coverage import SearchStatus, backtrack_cover
from .geometry import Group, Square
from .nine_rules import ALL_RULES, RuleEvidence, analyze_nine_rules, conflict_masks
from .position import Position


def above_range(s):
    return frozenset(Square(r, s.column) for r in range(s.row_index))


@dataclass(frozen=True)
class WhiteContext:
    kind: str
    groups: tuple[Group, ...]
    crossing: Square
    odd: Square | None
    even: Square | None
    reserved_columns: tuple[int, ...]
    permitted_squares: frozenset[Square]
    claims: tuple[SolutionClause, ...]
    target_groups: tuple[Group, ...]
    excluded_groups: tuple[Group, ...]
    preconditions: tuple[str, ...] = (
        'nonterminal Black to move',
        'reserved threat must be maintained while executing the remaining rules',
        'joint Zugzwang and response executability remains conditional',
    )


def white_evaluation_contexts(position: Position) -> tuple[WhiteContext, ...]:
    p = Position.from_board(position.board, position.player_to_move)
    if p.terminal or p.player_to_move != 1:
        return ()
    groups = p.potential_groups(0)
    black_groups = p.potential_groups(1)
    landings = {s.column: s for s in p.playable_squares}
    contexts = []

    def add(kind, source, cross, odd, even, claims):
        columns = tuple(sorted({cross.column} | ({odd.column} if odd else set())))
        permitted = frozenset(Square(r, c) for r in range(6) for c in range(7)
                              if c not in columns)
        excluded = tuple(g for g in black_groups if any(cl.solves(g) for cl in claims))
        contexts.append(WhiteContext(kind, source, cross, odd, even, columns,
                                     permitted, tuple(claims),
                                     tuple(g for g in black_groups if g not in excluded), excluded))

    def crossing_claims(cross):
        first = landings[cross.column]
        return [SolutionClause('crossing-odd:' + s.name, (frozenset((s,)),))
                for r in range(6) if not (s := Square(r, cross.column)).is_even
                and p.is_empty(s) and s != first]

    # Three White stones and a nonplayable odd hole. Select ONE reserved threat
    # per context; other threats remain ordinary potential groups (§8.2).
    for group in groups:
        holes = tuple(s for s in group.squares if p.is_empty(s))
        if len(holes) != 1 or holes[0].is_even or p.is_playable(holes[0]):
            continue
        cross = holes[0]
        claims = crossing_claims(cross)
        claims.append(SolutionClause('odd-threat-terminal-region',
                                     (above_range(cross) | {cross},)))
        add('odd_threat', (group,), cross, None, None, claims)

    # §8.4: two groups each with exactly two White stones. One needs two
    # odd holes; the other shares a nonplayable odd crossing hole and needs
    # an even hole adjacent to the OTHER odd hole in another column.
    half = [(g, frozenset(s for s in g.squares if p.is_empty(s))) for g in groups
            if sum(p.is_empty(s) for s in g.squares) == 2]
    for (g1, h1), (g2, h2) in combinations(half, 2):
        for go, ho, ge, he in ((g1, h1, g2, h2), (g2, h2, g1, h1)):
            if any(s.is_even for s in ho) or len(ho & he) != 1:
                continue
            cross = next(iter(ho & he))
            odd, even = next(iter(ho - {cross})), next(iter(he - {cross}))
            if (p.is_playable(cross) or not even.is_even or odd.column == cross.column
                    or even.column != odd.column or abs(even.row - odd.row) != 1):
                continue
            higher = even.row > odd.row
            kind = ('even_above_odd' if higher else
                    'odd_above_even_playable' if p.is_playable(even) else
                    'odd_above_even_unplayable')
            claims = crossing_claims(cross)
            claims.append(SolutionClause('both-above', (above_range(cross), above_range(odd))))
            if higher:
                claims.append(SolutionClause('crossing-successor+odd',
                    (frozenset((Square(cross.row_index - 1, cross.column),)), frozenset((odd,)))))
                if p.is_playable(odd):
                    claims.append(SolutionClause('other-odd-playable-height-limit',
                        (frozenset(s for s in above_range(cross) if s.row > cross.row + 1),)))
            # Rules 5/6 of case 1, 3/4 of case 2a. Case 2b has ONLY two claims.
            if higher or not p.is_playable(even):
                first, other = landings[cross.column], landings[odd.column]
                base = not first.is_even and not p.is_playable(odd)
                if base:
                    claims.append(SolutionClause('reserved-baseinverse',
                                                (frozenset((first,)), frozenset((other,)))))
                start = other.row + int(base)
                for row in range(start, odd.row, 2):
                    claims.append(SolutionClause('reserved-vertical:' + str(row), (
                        frozenset((Square(6 - row, odd.column),)),
                        frozenset((Square(5 - row, odd.column),)))))
            add(kind, (go, ge), cross, odd, even, claims)
    return tuple(contexts)


@dataclass(frozen=True)
class WhiteCover:
    context: WhiteContext
    status: SearchStatus
    evidence: tuple[RuleEvidence, ...]
    nodes: int
    certification: str = 'conditional_source_coverage_only'


def search_white_covers(position: Position, *, node_budget=10_000,
                        context_budget=8, rules=ALL_RULES) -> tuple[WhiteCover, ...]:
    """Same enumerator and §7.4 search; restricted regions and Black targets.

    Context truncation is explicit: callers can compare with the complete
    white_evaluation_contexts inventory. An empty tuple never means no win.
    """
    if any(type(v) is not int or v < 0 for v in (node_budget, context_budget)):
        raise ValueError('budgets must be nonnegative integers')
    contexts = white_evaluation_contexts(position)[:context_budget]
    if not contexts:
        return ()
    report = analyze_nine_rules(position, defender=0, rules=rules)
    results = []
    for context in contexts:
        useful = tuple(RuleEvidence(e.candidate, e.clauses, solved)
                       for e in report.evidence
                       if footprint(e.candidate).squares <= context.permitted_squares
                       if (solved := tuple(g for g in context.target_groups
                                           if any(c.solves(g) for c in e.clauses))))
        index = {g: i for i, g in enumerate(context.target_groups)}
        masks = tuple(sum(1 << index[g] for g in e.conditional_solved_groups) for e in useful)
        conflicts, _ = conflict_masks(tuple(e.candidate for e in useful))
        found = backtrack_cover(len(index), masks, conflicts, node_budget, memoize=True)
        status = (SearchStatus.FOUND if found.selected is not None else
                  SearchStatus.BUDGET_EXHAUSTED if found.exhausted else SearchStatus.EXHAUSTIVE_NO_COVER)
        results.append(WhiteCover(context, status,
            () if found.selected is None else tuple(useful[i] for i in sorted(found.selected)), found.expanded))
    return tuple(results)
