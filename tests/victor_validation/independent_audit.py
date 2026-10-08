"""Phase 6B.3 independent audit model; test-only, NO certificate acceptance.

Self-contained standard-library code. It imports no engine, grounding, Victor,
oracle or harness module, so it can disagree with all of them. Geometry is a
42-bit column-major board (bit = 6 * column + height) with 69 precomputed line
masks; winners are detected only on lines through the last move.

Rules are raw tuples ``(kind, a, b)`` of ``(column, height)`` squares, heights
0-based from the bottom (thesis row = height + 1). CL/VE use ``(lower, upper)``.
Mutation flags deliberately relax one theorem hypothesis at a time so that the
exact comparison can be shown to detect a false implication.
"""
from itertools import combinations
from random import Random

COLS, ROWS = 7, 6
FULL = (1 << 42) - 1


def bit(column, height):
    return 1 << (ROWS * column + height)


def _lines():
    lines = []
    for c in range(COLS):
        for h in range(ROWS):
            for dc, dh in ((1, 0), (0, 1), (1, 1), (1, -1)):
                cells = [(c + i * dc, h + i * dh) for i in range(4)]
                if all(0 <= cc < COLS and 0 <= hh < ROWS for cc, hh in cells):
                    lines.append(sum(bit(cc, hh) for cc, hh in cells))
    return tuple(sorted(set(lines)))


LINES = _lines()
LINES_THROUGH = {(c, h): tuple(m for m in LINES if m & bit(c, h))
                 for c in range(COLS) for h in range(ROWS)}


class Board:
    """Immutable bitboard snapshot. ``turn`` follows from counts (White first)."""

    __slots__ = ('white', 'black')

    def __init__(self, white=0, black=0):
        self.white, self.black = white, black

    @property
    def occupied(self):
        return self.white | self.black

    @property
    def turn(self):
        return 0 if self.white.bit_count() == self.black.bit_count() else 1

    @property
    def empty_count(self):
        return 42 - self.occupied.bit_count()

    def height(self, c):
        h = 0
        while h < ROWS and self.occupied & bit(c, h):
            h += 1
        return h

    def landing(self, c):
        h = self.height(c)
        return None if h == ROWS else (c, h)

    def landings(self):
        return tuple(s for c in range(COLS) if (s := self.landing(c)) is not None)

    def is_empty(self, square):
        return not self.occupied & bit(*square)

    def play(self, square):
        b = bit(*square)
        return Board(self.white | b, self.black) if self.turn == 0 else Board(self.white, self.black | b)

    def wins_at(self, player, square):
        stones = (self.white if player == 0 else self.black) | bit(*square)
        return any(stones & m == m for m in LINES_THROUGH[square])

    def has_four(self, player):
        stones = self.white if player == 0 else self.black
        return any(stones & m == m for m in LINES)

    def key(self):
        return self.white, self.black

    @classmethod
    def from_matrix(cls, rows):
        """Top-first 6x7 of ' ', 'X', 'O'. Checks gravity/counts only, not reachability."""
        white = black = 0
        for r, row in enumerate(rows):
            for c, cell in enumerate(row):
                if cell == 'X':
                    white |= bit(c, ROWS - 1 - r)
                elif cell == 'O':
                    black |= bit(c, ROWS - 1 - r)
                elif cell != ' ':
                    raise ValueError('bad cell')
        board = cls(white, black)
        for c in range(COLS):
            if any(board.occupied & bit(c, h + 1) and not board.occupied & bit(c, h)
                   for h in range(ROWS - 1)):
                raise ValueError('gravity')
        if white.bit_count() - black.bit_count() not in (0, 1):
            raise ValueError('counts')
        return board

    def matrix(self):
        return tuple(''.join('X' if self.white & bit(c, h) else 'O' if self.black & bit(c, h)
                             else ' ' for c in range(COLS)) for h in range(ROWS - 1, -1, -1))


def replay(columns):
    """Legal alternating play from empty; rejects moves after a win or full column."""
    board = Board()
    for c in columns:
        if board.has_four(0) or board.has_four(1):
            raise ValueError('move after terminal position')
        square = board.landing(c)
        if square is None:
            raise ValueError('full column')
        board = board.play(square)
    return board


def name(square):
    return f'{"abcdefg"[square[0]]}{square[1] + 1}'


# ---------------------------------------------------------------- rule model

def enumerate_rules(board, *, claimodd=False, floating_bi=False):
    """Allis §§6.1-6.3 from the definitions; flags add deliberately UNSOUND shapes."""
    rules = []
    for c in range(COLS):
        for lower in range(board.height(c), ROWS - 1):
            upper_row = lower + 2  # thesis numbering of the upper square
            kind = 'CL' if upper_row % 2 == 0 else 'VE'
            rules.append((kind, (c, lower), (c, lower + 1)))
            if claimodd and kind == 'VE':
                rules.append(('CLODD', (c, lower), (c, lower + 1)))
    for a, b in combinations(board.landings(), 2):
        rules.append(('BI', a, b))
    if floating_bi:
        for a in board.landings():
            for c in range(COLS):
                above = (c, board.height(c) + 1)
                if c != a[0] and above[1] < ROWS:
                    rules.append(('BIFLOAT', a, above))
    return tuple(rules)


def coverage_mask(rule):
    """Squares a target line must contain to be solved by ``rule``."""
    kind, a, b = rule
    if kind in ('CL', 'CLODD'):
        return bit(*b)
    return bit(*a) | bit(*b)


def targets(board):
    """Every line with no Black stone (White's potential groups)."""
    return tuple(m for m in LINES if not m & board.black)


def find_cover(board, *, allow_overlap=False, claimodd=False, floating_bi=False, budget=200_000):
    """First-uncovered-line DFS, independent of production's most-constrained search.

    Returns ``(status, rules)`` with status 'found', 'none' or 'unknown'.
    """
    rules = tuple(r for r in enumerate_rules(board, claimodd=claimodd, floating_bi=floating_bi)
                  if any(coverage_mask(r) & m == coverage_mask(r) for m in targets(board)))
    lines = targets(board)
    used_by = [bit(*r[1]) | bit(*r[2]) for r in rules]
    covers = [tuple(i for i, r in enumerate(rules) if coverage_mask(r) & m == coverage_mask(r))
              for m in lines]
    steps = 0

    def dfs(index, used, chosen):
        nonlocal steps
        steps += 1
        if steps > budget:
            raise TimeoutError
        while index < len(lines) and any(rules[i] in chosen for i in covers[index]):
            index += 1
        if index == len(lines):
            return chosen
        for i in covers[index]:
            if allow_overlap or not used & used_by[i]:
                found = dfs(index + 1, used | used_by[i], chosen + (rules[i],))
                if found is not None:
                    return found
        return None

    try:
        found = dfs(0, 0, ())
    except TimeoutError:
        return 'unknown', None
    return ('found', found) if found is not None else ('none', None)


def check_hypotheses(board, rules):
    """Theorem hypotheses H1-H5 for an UNTRUSTED raw rule set. Returns reasons."""
    reasons = []
    if board.turn != 0:
        reasons.append('H1: White must be to move')
    if board.has_four(0) or board.has_four(1):
        reasons.append('H1: position must be nonterminal')
    landings = set(board.landings())
    used = 0
    for rule in rules:
        kind, a, b = rule
        if not (board.is_empty(a) and board.is_empty(b)):
            reasons.append(f'H2: {rule} occupies a nonempty square')
        if kind in ('CL', 'VE'):
            even_upper = (b[1] + 1) % 2 == 0
            if a[0] != b[0] or b[1] != a[1] + 1 or even_upper != (kind == 'CL'):
                reasons.append(f'H2: {rule} has wrong shape/parity')
        elif kind == 'BI':
            if a not in landings or b not in landings or a == b:
                reasons.append(f'H2: {rule} needs two distinct landing squares')
        else:
            reasons.append(f'H2: {kind} is not a supported rule')
        cells = bit(*a) | bit(*b)
        if used & cells:
            reasons.append(f'H3: {rule} overlaps another rule')
        used |= cells
    for line in targets(board):
        if not any(coverage_mask(r) & line == coverage_mask(r) for r in rules):
            reasons.append(f'H4: line {line:#x} is uncovered')
    return tuple(reasons)


# ------------------------------------------------------------- exact search

class Cutoff(Exception):
    pass


def white_can_force_win(board, *, budget=2_000_000):
    """Exact boolean AND/OR search. ``None`` on budget cutoff (never a guess).

    Sound shortcuts only: immediate wins; two or more immediate opponent wins at
    landing squares (they are in different columns); a single one must be blocked.
    """
    memo = {}
    count = 0

    def threats(b, player):
        return [s for s in b.landings() if b.wins_at(player, s)]

    def visit(b):
        nonlocal count
        k = b.key()
        if k in memo:
            return memo[k]
        count += 1
        if count > budget:
            raise Cutoff
        if b.empty_count == 0:
            result = False
        elif b.turn == 0:
            if threats(b, 0):
                result = True
            else:
                black = threats(b, 1)
                if len(black) >= 2:
                    result = False
                elif black:
                    result = visit(b.play(black[0]))
                else:
                    result = any(visit(b.play(s)) for s in b.landings())
        else:
            if threats(b, 1):
                result = False
            else:
                white = threats(b, 0)
                if len(white) >= 2:
                    result = True
                elif white:
                    result = visit(b.play(white[0]))
                else:
                    result = all(visit(b.play(s)) for s in b.landings())
        memo[k] = result
        return result

    if board.has_four(0):
        return True, 0
    if board.has_four(1):
        return False, 0
    try:
        return visit(board), count
    except Cutoff:
        return None, count


def exact_value(board, *, budget=2_000_000):
    """White-perspective value in {-1, 0, 1} via two boolean searches, or None."""
    white_wins, n1 = white_can_force_win(board, budget=budget)
    if white_wins is None:
        return None, n1
    if white_wins:
        return 1, n1
    black_wins, n2 = _player_can_force_win(board, 1, budget=budget)
    if black_wins is None:
        return None, n1 + n2
    return (-1 if black_wins else 0), n1 + n2


def _player_can_force_win(board, player, *, budget):
    memo = {}
    count = 0

    def visit(b):
        nonlocal count
        k = b.key()
        if k in memo:
            return memo[k]
        count += 1
        if count > budget:
            raise Cutoff
        mover = b.turn
        if b.empty_count == 0:
            result = False
        elif any(b.wins_at(mover, s) for s in b.landings()):
            result = mover == player
        else:
            children = [visit(b.play(s)) for s in b.landings()]
            result = any(children) if mover == player else all(children)
        memo[k] = result
        return result

    try:
        return visit(board), count
    except Cutoff:
        return None, count


# ------------------------------------------------- constructive strategy check

def check_strategy(board, rules, *, budget=2_000_000):
    """Every White move x EVERY Black move the proof permits; White must never win.

    Returns ``(status, detail, states)``. Status is 'held', 'violated' or 'unknown'.
    Black's permitted replies: the unique obligation if White touched an active
    pair, otherwise ANY landing square that is not an active CL lower.
    """
    seen = set()
    count = 0

    def active(b, rule):
        kind, a, u = rule
        if kind == 'CL':
            return not b.black & bit(*u)
        return not b.black & (bit(*a) | bit(*u))

    def visit(b, path):
        nonlocal count
        if b.key() in seen:
            return None
        count += 1
        if count > budget:
            raise Cutoff
        seen.add(b.key())
        if b.empty_count == 0:
            return None
        live = [r for r in rules if active(b, r)]
        for r in live:
            if not (b.is_empty(r[1]) and b.is_empty(r[2])):
                return f'invariant broken: {r} partly occupied by White', path
        for s in b.landings():
            if b.wins_at(0, s):
                return f'White wins at {name(s)}', path + (s[0],)
            after = b.play(s)
            touched = [r for r in live if s in (r[1], r[2])]
            if len(touched) > 1:
                return 'two obligations from one move', path + (s[0],)
            if touched:
                kind, a, u = touched[0]
                if kind in ('CL', 'VE') and s == u:
                    return 'White reached an active upper square', path + (s[0],)
                reply = u if s == a else a
                if reply not in after.landings():
                    return 'obligatory reply not playable', path + (s[0],)
                replies = (reply,)
            else:
                forbidden = {r[1] for r in live if r[0] == 'CL'}
                replies = tuple(t for t in after.landings() if t not in forbidden)
                if not replies:
                    return 'no permitted spare move', path + (s[0],)
                if not any((t[1] + 1) % 2 == 0 for t in replies):
                    return 'parity lemma failed: no even-row spare', path + (s[0],)
            for t in replies:
                if after.wins_at(1, t):
                    continue  # Black wins: branch ends in Black's favour.
                failed = visit(after.play(t), path + (s[0], t[0]))
                if failed:
                    return failed
        return None

    try:
        failed = visit(board, ())
    except Cutoff:
        return 'unknown', None, count
    return ('violated', failed, count) if failed else ('held', None, count)


# ------------------------------------------------------------ generators

def sample_history(rng, plies, *, follow_up=0.0):
    """Legal non-terminating play; with prob ``follow_up`` Black answers in White's column.

    Follow-up bias produces Claimeven-friendly structure (Allis §4.1) that a
    uniform sampler rarely reaches with many empty cells. Not uniform.
    """
    board, moves = Board(), []
    for _ in range(plies):
        options = [s for s in board.landings()
                   if not board.wins_at(board.turn, s) and board.play(s).empty_count > 0]
        if not options:
            return None
        pick = None
        if board.turn == 1 and moves and rng.random() < follow_up:
            same = [s for s in options if s[0] == moves[-1]]
            pick = same[0] if same else None
        if pick is None:
            pick = rng.choice(options)
        board = board.play(pick)
        moves.append(pick[0])
    return tuple(moves)


def parity_lemma_exhaustive():
    """Over all 7^7 height vectors: odd empties => some landing on an even row.

    Returns the number of Black-to-move height vectors checked (odd empties).
    """
    checked = 0
    for code in range(7 ** 7):
        heights, x = [], code
        for _ in range(COLS):
            heights.append(x % 7)
            x //= 7
        empties = sum(ROWS - h for h in heights)
        if empties % 2 == 0:
            continue
        checked += 1
        # Landing height h has thesis row h + 1; even row <=> h odd.
        if not any(h < ROWS and h % 2 == 1 for h in heights):
            raise AssertionError(f'parity lemma fails for heights {heights}')
    return checked


# ------------------------------------------------------- reachability check

def reachable(board, *, budget=200_000):
    """Backward search: can top stones be removed alternately down to empty?

    Returns True/False, or None on budget cutoff. Prefix wins are impossible
    when the final board is nonterminal (stones are only ever added).
    """
    memo = {}
    count = 0

    def back(b):
        nonlocal count
        if b.occupied == 0:
            return True
        k = b.key()
        if k in memo:
            return memo[k]
        count += 1
        if count > budget:
            raise Cutoff
        last = 1 - b.turn  # the player who made the previous move
        stones = b.white if last == 0 else b.black
        ok = False
        for c in range(COLS):
            h = b.height(c)
            if h and stones & bit(c, h - 1):
                top = bit(c, h - 1)
                prev = Board(b.white & ~top, b.black) if last == 0 else Board(b.white, b.black & ~top)
                if back(prev):
                    ok = True
                    break
        memo[k] = ok
        return ok

    try:
        return back(board)
    except Cutoff:
        return None


def all_tops_white(rng, board):
    """Swap each Black column top with a random non-top White stone.

    Counts and gravity are kept. With White to move the previous mover was Black,
    so a board whose every column top is White is UNREACHABLE. Random single
    recolourings almost always stay reachable, hence this construction.
    Returns None when there are not enough non-top White stones.
    """
    white, black = board.white, board.black
    tops = [bit(c, h - 1) for c in range(COLS) if (h := board.height(c))]
    spare = [1 << i for i in range(42) if white >> i & 1 and (1 << i) not in tops]
    for top in tops:
        if black & top:
            if not spare:
                return None
            x = spare.pop(rng.randrange(len(spare)))
            white, black = white & ~x | top, black & ~top | x
    return Board(white, black)


# ---------------------------------------------------------------- campaign

def victor_rules(witness):
    """Production witness -> raw audit tuples (Square row_index is top-first)."""
    kinds = {'claimeven': 'CL', 'baseinverse': 'BI', 'vertical': 'VE'}
    return tuple((kinds[e.candidate.rule.value],)
                 + tuple((s.column, ROWS - 1 - s.row_index) for s in e.candidate.squares)
                 for e in witness.evidence)


MUTATIONS = {
    'overlap': {'allow_overlap': True},        # drop §7.4 disjointness (§5.3 conflict)
    'claimodd': {'claimodd': True},            # Claimeven with an ODD upper square
    'floating_bi': {'floating_bi': True},      # Baseinverse with a non-landing square
}


def run_campaign(*, seed=6203, per_setting=40, empties=(10, 12, 14, 16, 18, 20),
                 follow_ups=(0.0, 0.5, 0.9), search_budget=2_000_000, strategy_budget=300_000):
    """Bounded comparison. Production imports are local so the model stays standalone."""
    from games.connect4.victor import (Position, SearchStatus, VerificationStatus,
                                       search_covering_set, verify_coverage_witness)
    rng = Random(seed)
    seen = set()
    stats = {'positions': 0, 'cover_found': 0, 'cover_none': 0, 'cover_unknown': 0,
             'existence_disagreements': [], 'theorem_violations': [], 'unknown_values': 0,
             'strategy_held': 0, 'strategy_unknown': 0, 'strategy_violations': [],
             'hypothesis_rejections': [], 'by_empties': {},
             'mutation_examples': {k: [] for k in MUTATIONS}, 'mutation_safe': dict.fromkeys(MUTATIONS, 0),
             'mutation_false_implications': dict.fromkeys(MUTATIONS, 0),
             'witness_kinds': {}, 'witness_values': {}, 'max_witness_empties': 0,
             'unreachable_checked': 0, 'unreachable_covered': 0, 'unreachable_violations': []}
    for e in empties:
        for f in follow_ups:
            for _ in range(per_setting):
                h = sample_history(rng, 42 - e, follow_up=f)
                if h is None:
                    continue
                board = replay(h)
                if board.key() in seen or board.turn != 0:
                    continue
                seen.add(board.key())
                stats['positions'] += 1
                row = stats['by_empties'].setdefault(e, [0, 0, 0])
                row[0] += 1
                position = Position.from_board([list(r) for r in board.matrix()], 0)
                result = search_covering_set(position, node_budget=200_000)
                mine, rules = find_cover(board)
                prod = {SearchStatus.FOUND: 'found', SearchStatus.EXHAUSTIVE_NO_COVER: 'none'}.get(
                    result.status, 'unknown')
                if 'unknown' not in (prod, mine) and prod != mine:
                    stats['existence_disagreements'].append({'moves': h, 'production': prod, 'audit': mine})
                if prod == 'found':
                    stats['cover_found'] += 1
                    row[1] += 1
                    witness = result.witness
                    if verify_coverage_witness(position, witness).status != VerificationStatus.VERIFIED_UNCERTIFIED:
                        stats['hypothesis_rejections'].append({'moves': h, 'why': 'production verifier'})
                    raw = victor_rules(witness)
                    reasons = check_hypotheses(board, raw)
                    if reasons:
                        stats['hypothesis_rejections'].append({'moves': h, 'why': reasons[:3]})
                    win, _ = white_can_force_win(board, budget=search_budget)
                    if win is None:
                        stats['unknown_values'] += 1
                    elif win:
                        stats['theorem_violations'].append({'moves': h, 'rules': raw})
                    else:
                        row[2] += 1
                    kinds = '+'.join(sorted({r[0] for r in raw})) or 'empty'
                    stats['witness_kinds'][kinds] = stats['witness_kinds'].get(kinds, 0) + 1
                    stats['max_witness_empties'] = max(stats['max_witness_empties'], e)
                    value, _ = exact_value(board, budget=search_budget)
                    stats['witness_values'][str(value)] = stats['witness_values'].get(str(value), 0) + 1
                    status, detail, _ = check_strategy(board, raw, budget=strategy_budget)
                    if status == 'held':
                        stats['strategy_held'] += 1
                    elif status == 'unknown':
                        stats['strategy_unknown'] += 1
                    else:
                        stats['strategy_violations'].append({'moves': h, 'rules': raw, 'detail': detail})
                elif prod == 'none':
                    stats['cover_none'] += 1
                else:
                    stats['cover_unknown'] += 1
                if prod == 'none':
                    for label, flags in MUTATIONS.items():
                        status, mutated = find_cover(board, **flags)
                        if status != 'found':
                            continue
                        win, _ = white_can_force_win(board, budget=search_budget)
                        if win:
                            stats['mutation_false_implications'][label] += 1
                            if len(stats['mutation_examples'][label]) < 3:
                                stats['mutation_examples'][label].append(
                                    {'moves': h, 'rules': mutated, 'board': board.matrix()})
                        elif win is False:
                            stats['mutation_safe'][label] += 1
                # Basic-valid but UNREACHABLE variant of the same board.
                variant = all_tops_white(rng, board)
                if (variant is None or variant.has_four(0) or variant.has_four(1)
                        or reachable(variant) is not False):
                    continue
                stats['unreachable_checked'] += 1
                status, raw = find_cover(variant)
                if status == 'found':
                    stats['unreachable_covered'] += 1
                    win, _ = white_can_force_win(variant, budget=search_budget)
                    held, detail, _ = check_strategy(variant, raw, budget=strategy_budget)
                    if win or held == 'violated':
                        stats['unreachable_violations'].append({'board': variant.matrix(), 'rules': raw})
    return stats



def enumerate_covers(board, *, limit=200, budget=200_000, prefer_inverse=True):
    """Distinct valid (disjoint, complete) covers, NOT only the first one found.

    The theorem quantifies over every hypothesis-satisfying rule set, so each is
    checked separately. ``prefer_inverse`` branches on BI/VE before CL so that
    proactive-occupation cases are reached within the limit.
    """
    lines = targets(board)
    rules = [r for r in enumerate_rules(board)
             if any(coverage_mask(r) & m == coverage_mask(r) for m in lines)]
    if prefer_inverse:
        rules.sort(key=lambda r: r[0] == 'CL')
    found, steps = set(), 0

    def dfs(index, used, chosen):
        nonlocal steps
        steps += 1
        if steps > budget or len(found) >= limit:
            return
        while index < len(lines) and any(coverage_mask(r) & lines[index] == coverage_mask(r)
                                         for r in chosen):
            index += 1
        if index == len(lines):
            found.add(frozenset(chosen))
            return
        for r in rules:
            cells = bit(*r[1]) | bit(*r[2])
            if coverage_mask(r) & lines[index] == coverage_mask(r) and not used & cells:
                dfs(index + 1, used | cells, chosen + (r,))

    dfs(0, 0, ())
    return tuple(tuple(sorted(c)) for c in sorted(found, key=sorted))


def run_cover_diversity(*, seed=7403, per_setting=300, empties=(2, 4, 6, 8, 10, 12, 14, 16),
                        follow_ups=(0.5, 0.9), covers_per_position=40, strategy_budget=300_000):
    """Check EVERY enumerated valid cover (up to a limit) with the all-spares model."""
    rng = Random(seed)
    seen = set()
    stats = {'positions_with_cover': 0, 'covers_checked': 0, 'held': 0, 'unknown': 0,
             'violations': [], 'kinds': {}, 'covers_with_bi': 0, 'covers_with_ve': 0,
             'covers_mixed_all_three': 0, 'exact_white_wins_with_cover': 0}
    for e in empties:
        for f in follow_ups:
            for _ in range(per_setting):
                h = sample_history(rng, 42 - e, follow_up=f)
                if h is None:
                    continue
                board = replay(h)
                if board.key() in seen or board.turn != 0:
                    continue
                seen.add(board.key())
                covers = enumerate_covers(board, limit=covers_per_position)
                if not covers:
                    continue
                stats['positions_with_cover'] += 1
                if white_can_force_win(board)[0]:
                    stats['exact_white_wins_with_cover'] += 1
                for rules in covers:
                    kinds = {r[0] for r in rules}
                    key = '+'.join(sorted(kinds)) or 'empty'
                    stats['kinds'][key] = stats['kinds'].get(key, 0) + 1
                    stats['covers_with_bi'] += 'BI' in kinds
                    stats['covers_with_ve'] += 'VE' in kinds
                    stats['covers_mixed_all_three'] += kinds == {'CL', 'BI', 'VE'}
                    if check_hypotheses(board, rules):
                        raise AssertionError('enumerated cover fails hypotheses')
                    stats['covers_checked'] += 1
                    status, detail, _ = check_strategy(board, rules, budget=strategy_budget)
                    if status == 'held':
                        stats['held'] += 1
                    elif status == 'unknown':
                        stats['unknown'] += 1
                    else:
                        stats['violations'].append({'moves': h, 'rules': rules, 'detail': detail})
    return stats

if __name__ == '__main__':  # pragma: no cover - research CLI, prints to stdout only
    import json
    import sys
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 6203
    print(json.dumps(run_campaign(seed=seed), default=str, indent=1))
