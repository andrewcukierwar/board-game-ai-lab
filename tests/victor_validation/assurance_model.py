"""6B.5 reference model: column strings and geometric four-subsets.

No production or previous audit imports. This is bounded test tooling, not a
solver or a proof assistant. Coordinates are (column, bottom-based row 1..6).
"""
from itertools import combinations


def groups():
    """Classify all 111,930 four-subsets, rather than scan directional windows."""
    found = []
    for cells in combinations(((c, r) for c in range(7) for r in range(1, 7)), 4):
        xs, ys = [c for c, _ in cells], [r for _, r in cells]
        vertical = len(set(xs)) == 1 and max(ys) - min(ys) == 3
        across = xs == list(range(xs[0], xs[0] + 4))
        horizontal = across and len(set(ys)) == 1
        diagonal = across and (len({r - c for c, r in cells}) == 1
                               or len({r + c for c, r in cells}) == 1)
        if vertical or horizontal or diagonal:
            found.append(frozenset(cells))
    return tuple(found)


GROUPS = groups()
EMPTY = ('',) * 7


def matrix(board):
    return tuple(tuple(cell(board, (c, r)) for c in range(7)) for r in range(6, 0, -1))


def cell(board, square):
    c, r = square
    return board[c][r - 1] if len(board[c]) >= r else ' '


def landings(board):
    return tuple((c, len(column) + 1) for c, column in enumerate(board) if len(column) < 6)


def winner(board, stone):
    return any(all(cell(board, s) == stone for s in g) for g in GROUPS)


def terminal(board):
    return winner(board, 'X') or winner(board, 'O') or not landings(board)


def drop(board, column):
    assert type(column) is int and 0 <= column < 7
    assert not terminal(board) and len(board[column]) < 6
    stone = 'XO'[sum(map(len, board)) % 2]
    return board[:column] + (board[column] + stone,) + board[column + 1:]


def replay(history):
    board = EMPTY
    for column in history:
        board = drop(board, column)
    return board


def candidates(board):
    pairs = []
    for c, column in enumerate(board):
        for low in range(len(column) + 1, 6):
            pairs.append(('CL' if low % 2 else 'VE', (c, low), (c, low + 1)))
    pairs.extend(('BI', a, b) for a, b in combinations(landings(board), 2))
    return tuple(pairs)


def needs(rule):
    return frozenset(rule[2:] if rule[0] == 'CL' else rule[1:])


def hypotheses(board, rules):
    flat = ''.join(board)
    assert all(len(col) <= 6 and set(col) <= {'X', 'O'} for col in board)
    assert flat.count('X') == flat.count('O') and not terminal(board)
    assert all(r in candidates(board) for r in rules)
    squares = [s for r in rules for s in r[1:]]
    assert len(set(squares)) == len(squares)
    assert all(any(needs(r) <= g for r in rules) for g in GROUPS
               if all(cell(board, s) != 'O' for s in g))


def active(board, rule):
    return all(cell(board, s) != 'O' for s in needs(rule))


def invariant(board, rules):
    for rule in rules:
        if active(board, rule):
            assert all(cell(board, s) == ' ' for s in rule[1:]), (board, rule)
        if rule[0] == 'CL':
            assert cell(board, rule[1]) != 'O', (board, rule)


def replies(before, rules, white_column):
    """All permissive choices, calculated from the PRE-White rule state."""
    invariant(before, rules)
    white = (white_column, len(before[white_column]) + 1)
    live = [r for r in rules if active(before, r)]
    touched = [r for r in live if white in r[1:]]
    assert len(touched) <= 1
    after = drop(before, white_column)
    assert not terminal(after), (before, rules, white_column)
    if touched:
        k, a, b = touched[0]
        assert k == 'BI' or white == a
        reply = b if white == a else a
        assert reply in landings(after)
        return after, (reply,), True
    forbidden = {a for k, a, _ in live if k == 'CL'}
    allowed = tuple(s for s in landings(after) if s not in forbidden)
    assert any(r % 2 == 0 for _, r in allowed)
    return after, allowed, False


def explore(board, rules, *, node_budget=20_000, round_limit=None):
    """Universal White AND all-permitted-Black traversal, memoized by board.

    Every visited White state includes its two-ply successors. Depth boundaries
    and exhausted budgets are explicitly UNKNOWN, never a completed proof.
    Assertion contexts retain the initial board, rules and exact continuation.
    """
    hypotheses(board, rules)
    seen = set()
    stats = dict(states=0, transitions=0, black_wins=0, draws=0,
                 odd_spares=0, proactive_bi=0, proactive_ve=0, frontier=0)
    cutoff = False

    def visit(current, path):
        nonlocal cutoff
        if current in seen:
            return
        if len(seen) >= node_budget:
            cutoff = True
            return
        seen.add(current)
        stats['states'] += 1
        invariant(current, rules)
        if round_limit is not None and len(path) // 2 >= round_limit:
            stats['frontier'] += 1
            return
        for white, _ in landings(current):
            after, options, forced = replies(current, rules, white)
            for black, row in options:
                continuation = path + (white, black)
                context = (board, rules, continuation)
                completed = drop(after, black)
                try:
                    invariant(completed, rules)
                    assert not winner(completed, 'X')
                except AssertionError as exc:
                    raise AssertionError(context) from exc
                stats['transitions'] += 1
                if not forced:
                    stats['odd_spares'] += row % 2
                    for rule in rules:
                        if active(current, rule) and not active(completed, rule):
                            assert rule[0] in ('BI', 'VE'), context
                            stats['proactive_' + rule[0].lower()] += 1
                if winner(completed, 'O'):
                    stats['black_wins'] += 1
                elif not landings(completed):
                    stats['draws'] += 1
                else:
                    visit(completed, continuation)
    visit(board, ())
    status = 'unknown_node_budget' if cutoff else (
        'unknown_depth_frontier' if stats['frontier'] else 'complete')
    return status, stats
