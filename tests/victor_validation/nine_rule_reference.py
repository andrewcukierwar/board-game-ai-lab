"""Test-only independent reference for the nine Allis rules.

Standard library only; imports nothing from Victor, the engine or grounding.
Boards are top-first 6x7 matrices; cells are (column 0..6, row 1..6) with row 1
at the bottom. Rule identities are plain tuples so producer objects can be
compared without trusting their properties:

    (rule, group_cells | None, ((kind, lower, upper), ...), roles)

The predicates below are written directly from the thesis "Required"/
"Solutions" paragraphs (§§6.1–6.9), in a different style from both the
producer (``composite.py``) and the witness verifier.
"""
from itertools import permutations, product

# Legal replays of the thesis diagrams (filled = Black/O, open = White/X);
# reconstructions of the printed boards, not move lists attributed to Allis.
DIAGRAMS = {
    '6.1': [2, 3, 3, 3, 3, 3, 3, 4],
    '6.4': [2, 2, 3, 3, 2, 4, 3, 4, 4, 0],
    '6.5': [1, 0, 1, 0, 2, 1, 1, 4, 2, 4, 2, 2, 3, 3, 3, 3, 4, 4],
    '6.6': [3, 2],
    '6.7': [3, 3, 3, 3, 3, 0, 5, 6],
    '6.8': [3, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 6, 6, 6, 6, 6, 5],
    '6.9': [3, 2, 2, 3, 3, 2],
    '6.10': [2, 4, 3, 3, 2, 2, 2, 2],
}
DIAGRAM_BOARDS = {  # Visually transcribed, top row first.
    '6.4': ['       ', '       ', '       ', '  XXX  ', '  OOO  ', 'O XXO  '],
    '6.5': ['       ', '       ', ' XOOO  ', ' OXXX  ', 'OXXOO  ', 'OXXXO  '],
    '6.6': ['       ', '       ', '       ', '       ', '       ', '  OX   '],
    '6.7': ['       ', '   X   ', '   O   ', '   X   ', '   O   ', 'O  X XO'],
    '6.8': ['       ', '      X', '  XOO O', '  OXX X', '  XOO O', '  OXXOX'],
    '6.9': ['       ', '       ', '       ', '  OX   ', '  XO   ', '  OX   '],
    '6.10': ['       ', '  O    ', '  X    ', '  O    ', '  XO   ', '  XXO  '],
}


def cell(name):
    return 'abcdefg'.index(name[0]), int(name[1])


def at(board, c):
    return board[6 - c[1]][c[0]]


def empty(board, c):
    return at(board, c) == ' '


def playable(board, c):
    return empty(board, c) and (c[1] == 1 or not empty(board, (c[0], c[1] - 1)))


def lines():
    """All 69 windows by endpoint interpolation."""
    cells = [(c, r) for c in range(7) for r in range(1, 7)]
    result = set()
    for a in cells:
        for b in cells:
            d = (b[0] - a[0], b[1] - a[1])
            if d in ((3, 0), (0, 3), (3, 3), (3, -3)):
                result.add(frozenset((a[0] + i * d[0] // 3, a[1] + i * d[1] // 3) for i in range(4)))
    assert len(result) == 69
    return result


LINES = lines()


def white_groups(board):
    return {g for g in LINES if all(at(board, c) != 'O' for c in g)}


def _vpairs(board, upper_parity):
    return [((c, r), (c, r + 1)) for c in range(7) for r in range(1, 6)
            if (r + 1) % 2 == upper_parity and empty(board, (c, r)) and empty(board, (c, r + 1))]


def reference_identities(board):
    """Every rule application with Black controlling Zugzwang, as identity tuples."""
    out = set()
    play = [(c, r) for c in range(7) for r in range(1, 7) if playable(board, (c, r))]
    for lo, up in _vpairs(board, 0):
        out.add(('claimeven', None, (('claimeven', lo, up),), ()))
    for lo, up in _vpairs(board, 1):
        out.add(('vertical', None, (('vertical', lo, up),), ()))
    for a in play:
        for b in play:
            if a[0] < b[0]:
                out.add(('baseinverse', None, (), (a, b)))
    for g in LINES:
        if any(at(board, c) == 'X' for c in g):
            continue
        holes = sorted(c for c in g if empty(board, c))
        if not holes:
            continue
        # Aftereven: every hole an even Claimeven upper square.
        if all(h[1] % 2 == 0 and empty(board, (h[0], h[1] - 1)) for h in holes):
            comps = tuple(('claimeven', (h[0], h[1] - 1), h) for h in holes)
            if _parts_ok(comps, g, ()):
                out.add(('aftereven', g, comps, ()))
        if any(h[1] == 6 for h in holes):
            continue
        options = {h: _options(board, h) for h in holes}
        for choice in product(*(options[h] for h in holes)):
            if _parts_ok(choice, g, ()) and any(k == 'vertical' for k, _, _ in choice):
                out.add(('before', g, tuple(sorted(choice, key=lambda k: k[1])), ()))
        for p in holes:
            if not playable(board, p):
                continue
            rest = [h for h in holes if h != p]
            for x in play:
                if x in g or x[0] == p[0]:
                    continue
                for choice in product(*(options[h] for h in rest)):
                    if _parts_ok(choice, g, (p, x)):
                        out.add(('specialbefore', g, tuple(sorted(choice, key=lambda k: k[1])), (p, x)))
    for (a, b) in [(a, b) for a in _vpairs(board, 1) for b in _vpairs(board, 1) if a[0][0] < b[0][0]]:
        out.add(('lowinverse', None, (('vertical',) + a, ('vertical',) + b), ()))
    triples = [((c, r), (c, r + 1), (c, r + 2)) for c in range(7) for r in (2, 4)
               if all(empty(board, (c, r + i)) for i in range(3))]
    for a in triples:
        for b in triples:
            if a[0][0] < b[0][0]:
                out.add(('highinverse', None, (), a + b))
    for roles in permutations(play, 3):
        if roles[1][1] % 2 == 1:
            out.add(('baseclaim', None, (), roles))
    return out


def _options(board, h):
    result = [('vertical', h, (h[0], h[1] + 1))]
    if h[1] % 2 == 0 and empty(board, (h[0], h[1] - 1)):
        result.append(('claimeven', (h[0], h[1] - 1), h))
    return result


def _parts_ok(comps, group, special):
    squares = [s for _, lo, up in comps for s in (lo, up)] + list(special)
    if len(squares) != len(set(squares)):
        return False
    handled = {up if k == 'claimeven' else lo for k, lo, up in comps}
    return all(s in handled or s not in group for _, lo, up in comps for s in (lo, up))


def solved(identity, board, group):
    """Thesis 'Solutions' paragraphs, evaluated on one group (a frozenset of cells)."""
    rule, g, comps, roles = identity
    has = group.__contains__
    def comp_solves(k, lo, up):
        return has(up) if k == 'claimeven' else has(lo) and has(up)
    if rule in ('claimeven', 'vertical'):
        return comp_solves(*comps[0])
    if rule == 'baseinverse':
        return all(map(has, roles))
    if any(comp_solves(*k) for k in comps):
        return True
    if rule == 'aftereven':
        return all(any(s[0] == up[0] and s[1] > up[1] for s in group) for _, _, up in comps)
    if rule == 'lowinverse':
        return has(comps[0][2]) and has(comps[1][2])
    if rule == 'highinverse':
        (l1, m1, u1), (l2, m2, u2) = roles[:3], roles[3:]
        return ((has(u1) and has(u2)) or (has(m1) and has(m2)) or (has(m1) and has(u1))
                or (has(m2) and has(u2)) or (playable(board, l1) and has(l1) and has(u2))
                or (playable(board, l2) and has(l2) and has(u1)))
    if rule == 'baseclaim':
        first, second, third = roles
        return (has(first) and has((second[0], second[1] + 1))) or (has(second) and has(third))
    handled = [up if k == 'claimeven' else lo for k, lo, up in comps]
    if rule == 'before':
        return all(has((h[0], h[1] + 1)) for h in handled)
    p, x = roles
    succ = [(h[0], h[1] + 1) for h in handled + [p]]
    return (all(map(has, succ)) and has(x)) or (has(p) and has(x))


def identity_of(candidate):
    """Producer object -> identity tuple, from raw fields only."""
    def c(s):
        return s.column, 6 - s.row_index
    rule = candidate.rule.value
    if hasattr(candidate, 'squares'):  # Two-square RuleCandidate.
        a, b = (c(s) for s in candidate.squares)
        if rule == 'baseinverse':
            return rule, None, (), tuple(sorted((a, b)))
        return rule, None, ((rule, a, b),), ()
    group = None if candidate.group is None else frozenset(c(s) for s in candidate.group.squares)
    comps = tuple((k.rule.value, c(k.lower), c(k.upper)) for k in candidate.components)
    return rule, group, comps, tuple(c(s) for s in candidate.roles)


# §7.4 as a dictionary of sets of constraint codes.
_TABLE = ['CL 1', 'BI 1 1', 'VE 1 1 1', 'AE 1 1 1 3', 'LI 2 1 1 12 4', 'HI 2 1 1 12 4 4',
          'BC 1 1 1 1 12 12 1', 'BE 1 1 1 3 23 12 1 3', 'SB 1 1 1 3 23 12 1 3 3']
_NAMES = dict(CL='claimeven', BI='baseinverse', VE='vertical', AE='aftereven', LI='lowinverse',
              HI='highinverse', BC='baseclaim', BE='before', SB='specialbefore')
CODES = {}
for _i, _row in enumerate(_TABLE):
    _parts = _row.split()
    for _j, _code in enumerate(_parts[1:]):
        _other = _TABLE[_j].split()[0]
        CODES[frozenset((_NAMES[_parts[0]], _NAMES[_other]))] = {int(d) for d in _code}


def squares_of(identity):
    rule, g, comps, roles = identity
    out = {s for _, lo, up in comps for s in (lo, up)} | set(roles)
    if rule == 'baseclaim':
        out.add((roles[1][0], roles[1][1] + 1))
    return out


def claimevens_of(identity):
    rule, g, comps, roles = identity
    if rule == 'baseclaim':
        return [(roles[1], (roles[1][0], roles[1][1] + 1))]
    return [(lo, up) for k, lo, up in comps if k == 'claimeven']


def inverse_columns_of(identity):
    rule, g, comps, roles = identity
    if rule == 'lowinverse':
        return {lo[0]: {lo, up} for _, lo, up in comps}
    if rule == 'highinverse':
        return {roles[0][0]: set(roles[:3]), roles[3][0]: set(roles[3:])}
    return {}


def compatible(a, b):
    """§7.4 with the documented constraint-2 and N.B.(ii) interpretations."""
    if a == b:
        return False
    codes = CODES[frozenset((a[0], b[0]))]
    sa, sb = squares_of(a), squares_of(b)
    if 1 in codes and sa & sb:
        return False
    if 2 in codes:
        inv, other = (a, b) if inverse_columns_of(a) else (b, a)
        columns = inverse_columns_of(inv)
        for lo, _ in claimevens_of(other):
            if lo[0] in columns and lo[1] <= max(r for _, r in columns[lo[0]]):
                return False
    if 3 in codes:
        special = set(a[3]) if a[0] == 'specialbefore' else set()
        special |= set(b[3]) if b[0] == 'specialbefore' else set()
        for column in {s[0] for s in sa & sb}:
            pa = {s for s in sa if s[0] == column}
            pb = {s for s in sb if s[0] == column}
            if pa != pb or pa & special:
                return False
    if 4 in codes:
        ca, cb = set(inverse_columns_of(a)), set(inverse_columns_of(b))
        if sa & sb or not (ca == cb or not ca & cb):
            return False
    return True
