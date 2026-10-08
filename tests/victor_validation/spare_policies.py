"""Phase 6B.4A spare-move policy explorer; test-only, NO certificate acceptance.

Clarifies the Phase 6B.3 wording gap: the strategy sigma_R permits ANY landing
square that is not an active Claimeven lower, while the old Lemma L4 text spoke
of even-row spares. This module explores three set-valued Black spare policies
against EVERY White move, with the forced pair reply when one is triggered:

- ``permissive``: sigma_R as stated (any landing square except an active CL lower).
- ``even_row``: the deterministic-friendly refinement (even-row landings only).
- ``unrestricted``: deliberately UNSOUND (any landing square, even an active CL
  lower), so the comparison is shown not to be vacuous.

It reuses the immutable ``Board`` and geometry of the standalone 6B.3 audit
model without modifying that model. Raw rules are ``(kind, a, b)`` tuples of
``(column, height)`` squares, CL/VE as ``(lower, upper)``.
"""
from victor_validation.independent_audit import Cutoff, bit, name

POLICIES = ('permissive', 'even_row', 'unrestricted')


def _active(board, rule):
    kind, a, b = rule
    if kind == 'CL':
        return not board.black & bit(*b)
    return not board.black & (bit(*a) | bit(*b))


def _spares(after, live, policy):
    landings = after.landings()
    if policy == 'unrestricted':
        return landings
    forbidden = {r[1] for r in live if r[0] == 'CL'}
    permitted = tuple(s for s in landings if s not in forbidden)
    if policy == 'even_row':
        return tuple(s for s in permitted if (s[1] + 1) % 2 == 0)
    return permitted


def explore(board, rules, policy, *, budget=300_000):
    """Return ``(status, detail, stats)``; status is 'held', 'violated' or 'unknown'.

    ``stats`` counts explored states, odd-row spares actually explored, and
    invariant breaks (an active rule partly occupied, or Black on a CL lower).
    Only a White four is a violation; invariant breaks are reported separately
    so the unsound policy's failure mechanism is visible.
    """
    if policy not in POLICIES:
        raise ValueError('unknown policy')
    cl_lowers = {r[1] for r in rules if r[0] == 'CL'}
    stats = {'states': 0, 'odd_row_spares': 0, 'invariant_breaks': 0}
    seen = set()

    def visit(b, path):
        if b.key() in seen:
            return None
        stats['states'] += 1
        if stats['states'] > budget:
            raise Cutoff
        seen.add(b.key())
        if b.empty_count == 0:
            return None
        live = [r for r in rules if _active(b, r)]
        if any(not (b.is_empty(r[1]) and b.is_empty(r[2])) for r in live) or (b.black & sum(
                bit(*s) for s in cl_lowers)):
            stats['invariant_breaks'] += 1
        for s in b.landings():
            if b.wins_at(0, s):
                return f'White wins at {name(s)}', path + (s[0],)
            after = b.play(s)
            touched = [r for r in live if s in (r[1], r[2])]
            if len(touched) == 1 and b.is_empty(touched[0][1]) and b.is_empty(touched[0][2]):
                kind, a, u = touched[0]
                replies = (u if s == a else a,) if not (kind in ('CL', 'VE') and s == u) else ()
                replies = tuple(t for t in replies if t in after.landings())
            else:
                replies = ()
            if not replies:
                replies = _spares(after, live, policy)
                stats['odd_row_spares'] += sum((t[1] + 1) % 2 == 1 for t in replies)
            if not replies:
                return 'no permitted Black move', path + (s[0],)
            for t in replies:
                if after.wins_at(1, t):
                    continue
                failed = visit(after.play(t), path + (s[0], t[0]))
                if failed:
                    return failed
        return None

    try:
        failed = visit(board, ())
    except Cutoff:
        return 'unknown', None, stats
    return ('violated', failed, stats) if failed else ('held', None, stats)
