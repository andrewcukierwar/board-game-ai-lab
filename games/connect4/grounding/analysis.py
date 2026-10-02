"""Pure, bounded tactical evidence. No agent evaluation or Allis rule solver.

Matrices are top-first (row_index 0..5); named squares use a1..g6,
with row 1 at the bottom. Players 0/X and 1/O correspond to White/Black.
"""
from copy import deepcopy

ROWS, COLS = 6, 7
PIECES = ('X', 'O')
GROUPS = tuple(
    tuple((r + i * dr, c + i * dc) for i in range(4))
    for r in range(ROWS) for c in range(COLS)
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1))
    if 0 <= r + 3 * dr < ROWS and 0 <= c + 3 * dc < COLS
)


def square(r, c):
    return {'row_index': r, 'column': c, 'row': ROWS - r,
            'name': f'{chr(97 + c)}{ROWS - r}'}


def _lines(board, player):
    return [g for g in GROUPS if all(board[r][c] == PIECES[player] for r, c in g)]


def outcome(board):
    for player in (0, 1):
        if _lines(board, player):
            return {'status': 'win', 'winner': player}
    return {'status': 'draw' if all(v != ' ' for row in board for v in row) else 'ongoing',
            'winner': None}


def validate_position(board, player):
    """Reject malformed/basic-illegal positions; not a full reachability proof."""
    if type(player) is not int or player not in (0, 1):
        raise ValueError('player_to_move must be 0 or 1')
    if (not isinstance(board, (list, tuple)) or len(board) != ROWS
            or any(not isinstance(row, (list, tuple)) or len(row) != COLS for row in board)
            or any(v not in (' ', 'X', 'O') for row in board for v in row)):
        raise ValueError('board must be a 6 by 7 matrix of spaces, X and O')
    for c in range(COLS):
        if any(board[r][c] != ' ' and board[r + 1][c] == ' ' for r in range(ROWS - 1)):
            raise ValueError('board violates gravity')
    x = sum(row.count('X') for row in board)
    o = sum(row.count('O') for row in board)
    if x - o not in (0, 1) or player != (x - o):
        raise ValueError('piece counts and side to move disagree')
    winners = [p for p in (0, 1) if _lines(board, p)]
    if len(winners) > 1 or (winners and winners[0] != 1 - player):
        raise ValueError('invalid terminal position')
    if winners:
        # A legal last move must be a topmost stone whose removal removes every win.
        for c in range(COLS):
            r = next((r for r in range(ROWS) if board[r][c] != ' '), None)
            if r is not None and board[r][c] == PIECES[winners[0]]:
                previous = [list(row) for row in board]
                previous[r][c] = ' '
                if not _lines(previous, winners[0]):
                    break
        else:
            raise ValueError('position contains a win predating the last move')


def _landing(board, c):
    return next((r for r in range(ROWS - 1, -1, -1) if board[r][c] == ' '), None)


def _drop(board, player, c):
    result = [list(row) for row in board]
    r = _landing(board, c)
    result[r][c] = PIECES[player]
    return result, square(r, c)


def _winning_squares(board, player):
    """Geometric completion squares, deduplicated; playability is separate."""
    found = {}
    for group in GROUPS:
        empty = [(r, c) for r, c in group if board[r][c] == ' ']
        if len(empty) != 1 or sum(board[r][c] == PIECES[player] for r, c in group) != 3:
            continue
        r, c = empty[0]
        entry = found.setdefault((r, c), {
            'square': square(r, c), 'playable': _landing(board, c) == r,
            'parity': 'even' if (ROWS - r) % 2 == 0 else 'odd', 'groups': []})
        entry['groups'].append([square(rr, cc) for rr, cc in group])
    return [found[key] for key in sorted(found)]


def _immediate(board, player):
    return [s for s in _winning_squares(board, player) if s['playable']]


def analyze_position(board, player_to_move):
    """JSON-ready facts, including exhaustive legal moves and one-ply replies.

Opponent winning squares are counterfactual opportunities on the unchanged
board. They are not claims that the opponent has the turn. A survival column
only avoids a loss on the next reply; it is not a long-term safe move.
"""
    validate_position(board, player_to_move)
    board = [list(row) for row in board]
    result = outcome(board)
    terminal = result['status'] != 'ongoing'
    threats = [{'player': p, 'squares': _winning_squares(board, p)} for p in (0, 1)]
    legal = [] if terminal else [c for c in range(COLS) if board[0][c] == ' ']
    alternatives = []
    for c in legal:
        after, landing = _drop(board, player_to_move, c)
        end = outcome(after)
        replies = _immediate(after, 1 - player_to_move) if end['status'] == 'ongoing' else []
        alternatives.append({'column': c, 'landing_square': landing, 'outcome': end,
                             'opponent_winning_replies': replies,
                             'avoids_immediate_loss': not replies})
    own_wins = [a['column'] for a in alternatives if a['outcome']['status'] == 'win']
    opponent_wins = [] if terminal else _immediate(board, 1 - player_to_move)
    survivors = [a['column'] for a in alternatives if a['avoids_immediate_loss']]
    if terminal:
        defense = 'not_applicable_terminal'
    elif own_wins:
        defense = 'immediate_win_available'
    elif not opponent_wins:
        defense = 'no_immediate_threat'
    elif not survivors:
        defense = 'unavoidable_loss_next_reply'
    elif len(survivors) == 1:
        defense = 'mandatory_block'
    else:
        defense = 'defensive_options'
    return {'player_to_move': player_to_move, 'outcome': result,
            'winning_lines': [{'player': p, 'squares': [square(r, c) for r, c in g]}
                              for p in (0, 1) for g in _lines(board, p)],
            'legal_columns': legal, 'winning_squares': threats,
            'immediate_winning_columns': own_wins,
            'opponent_immediate_winning_squares': opponent_wins,
            'alternatives': alternatives,
            'defense': {'status': defense,
                        'mandatory_column': survivors[0] if defense == 'mandatory_block' else None,
                        'survival_columns': survivors}}


def analyze_move(board_before, player, column):
    """Verify one legal transition and compare its geometric winning squares.

Removed squares may be occupied by the move (including a completed win).
Newly playable squares may be pre-existing patterns supported by this move.
These deltas do not prove forced continuations.
"""
    before = analyze_position(board_before, player)
    if type(column) is not int or column not in before['legal_columns']:
        raise ValueError('move must be legal in a nonterminal position')
    after_board, landing = _drop(board_before, player, column)
    after = analyze_position(after_board, 1 - player)
    changes = []
    for p in (0, 1):
        old = {s['square']['name']: s for s in before['winning_squares'][p]['squares']}
        new = {s['square']['name']: s for s in after['winning_squares'][p]['squares']}
        changes.append({'player': p,
                        'created': [new[k] for k in sorted(new.keys() - old.keys())],
                        'removed': [old[k] for k in sorted(old.keys() - new.keys())],
                        'newly_playable': [new[k] for k in sorted(new)
                                           if new[k]['playable'] and
                                           (k not in old or not old[k]['playable'])]})
    return {'player': player, 'column': column, 'landing_square': landing,
            'board_after': after_board, 'outcome': after['outcome'],
            'was_immediate_win': column in before['immediate_winning_columns'],
            'was_mandatory_block': column == before['defense']['mandatory_column'],
            'winning_square_changes': changes,
            'alternatives_before_move': deepcopy(before['alternatives'])}
