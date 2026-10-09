"""Slow matrix reference mechanics, distinct from the bitboard oracle and engine."""


def winner(board):
    # Count contiguous runs along entire rows, columns and diagonals. No windows.
    for r in range(6):
        for c in range(7):
            piece = board[r][c]
            if piece == ' ':
                continue
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                if (0 <= r - dr < 6 and 0 <= c - dc < 7
                        and board[r - dr][c - dc] == piece):
                    continue  # Only start at the beginning of a run.
                rr, cc, length = r, c, 0
                while 0 <= rr < 6 and 0 <= cc < 7 and board[rr][cc] == piece:
                    length += 1
                    rr, cc = rr + dr, cc + dc
                if length >= 4:
                    return 'XO'.index(piece)
    return None


def legal_moves(board):
    if winner(board) is not None:
        return ()
    return tuple(c for c in range(7) if board[0][c] == ' ')


def landing(board, column):
    if type(column) is not int or column not in legal_moves(board):
        raise ValueError('illegal reference move')
    return next((r, column) for r in range(5, -1, -1) if board[r][column] == ' ')


def drop(board, turn, column):
    r, c = landing(board, column)
    rows = [list(row) for row in board]
    rows[r][c] = 'XO'[turn]
    return tuple(tuple(row) for row in rows)


def exhaustive_value(board, turn):
    """Uncached full tree reference; callers MUST restrict to <=4 empty cells."""
    if sum(cell == ' ' for row in board for cell in row) > 4:
        raise ValueError('reference restricted to four remaining moves')
    won = winner(board)
    if won is not None:
        return 1 if won == turn else -1
    columns = legal_moves(board)
    if not columns:
        return 0
    return max(-exhaustive_value(drop(board, turn, c), 1 - turn) for c in columns)
