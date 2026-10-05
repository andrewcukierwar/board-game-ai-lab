from numbers import Integral

from .board import Board

# Every horizontal, vertical and diagonal four-cell window as (row, col) cells.
_WINDOWS = tuple(
    tuple((row + i * dr, col + i * dc) for i in range(4))
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1))
    for row in range(6) for col in range(7)
    if 0 <= row + 3 * dr < 6 and 0 <= col + 3 * dc < 7)


class Connect4:
    def __init__(self, board=None, current_player=None):
        self.board = Board(board) if board else Board()
        self.current_player = current_player if current_player else 0
        self.piece = 'X' if self.current_player == 0 else 'O'

    def is_valid_move(self, col):
        """Physical column availability: an integer 0..6 (never bool) that is not full.

        Termination is not considered here, preserving get_valid_moves on finished
        boards for historical callers; make_move additionally rejects terminal games.
        """
        return (isinstance(col, Integral) and not isinstance(col, bool)
                and 0 <= col < self.board.width and self.board[0][col] == " ")

    def get_valid_moves(self):
        cols = [3, 2, 4, 1, 5, 0, 6]
        valid_moves = [col for col in cols if self.is_valid_move(col)]
        return valid_moves

    def _has_terminated(self):
        """Exact, faster equivalent of is_game_over() for guarding every move."""
        board = self.board
        for (r0, c0), (r1, c1), (r2, c2), (r3, c3) in _WINDOWS:
            piece = board[r0][c0]
            if piece in ("X", "O") and piece == board[r1][c1] == board[r2][c2] == board[r3][c3]:
                return True
        return self.is_board_full()

    def make_move(self, col):
        """Drop the mover's piece; False (state unchanged) for invalid or post-terminal moves."""
        if not self.is_valid_move(col):
            print('Invalid column')
            return False
        if self._has_terminated():
            print('Game is over')
            return False
        for row in range(5, -1, -1):
            if self.board[row][col] == " ":
                self.board[row][col] = self.piece
                self.current_player = 1 - self.current_player
                self.piece = 'X' if self.current_player == 0 else 'O'
                return True
        raise RuntimeError('Validated column had no empty cell')
                
    def check_winner(self):
        # Check horizontal
        for row in range(6):
            for col in range(4):
                if all(self.board[row][col + i] == 'X' for i in range(4)):
                    return 0
                elif all(self.board[row][col + i] == 'O' for i in range(4)):
                    return 1

        # Check vertical
        for row in range(3):
            for col in range(7):
                if all(self.board[row + i][col] == 'X' for i in range(4)):
                    return 0
                elif all(self.board[row + i][col] == 'O' for i in range(4)):
                    return 1

        # Check diagonals (positive slope)
        for row in range(3):
            for col in range(4):
                if all(self.board[row + i][col + i] == 'X' for i in range(4)):
                    return 0
                elif all(self.board[row + i][col + i] == 'O' for i in range(4)):
                    return 1

        # Check diagonals (negative slope)
        for row in range(3):
            for col in range(3, 7):
                if all(self.board[row + i][col - i] == 'X' for i in range(4)):
                    return 0
                elif all(self.board[row + i][col - i] == 'O' for i in range(4)):
                    return 1

        return -1

    def is_board_full(self):
        return all(cell != " " for row in self.board for cell in row)

    def is_game_over(self):
        if self.is_board_full() or self.check_winner() != -1:
            return True
        return False

    def battle(self, agent1, agent2, output=True):
        while True:
            if output:
                print(self.board)
            
#             try:
            if self.current_player == 0:
                column = agent1.choose_move(self)
                if output:
                    print(f"{agent1} chooses column {column}.")
            else:
                column = agent2.choose_move(self)
                if output:
                    print(f"{agent2} chooses column {column}.")

            if self.is_valid_move(column):
                self.make_move(column)
                
                check_winner = self.check_winner()
                if check_winner != -1:
                    if output:
                        print(self.board)
                    winner = agent1 if check_winner == 0 else agent2
                    print(f"{winner} wins!")
                    break
                elif self.is_board_full():
                    if output:
                        print(self.board)
                    print("It's a draw!")
                    break

#                 self.current_player = 1 - self.current_player
#                 self.piece = 'X' if self.current_player == 0 else 'O'
            else:
                print("Invalid move. Try again.")

    def step(self, action):
        if not self.is_valid_move(action):
            raise ValueError("Invalid move")

        # Make the move
        if not self.make_move(action):
            raise ValueError("Invalid move")

        # Check if the game is over
        done = self.is_game_over()
        winner = self.check_winner()

        # Assign rewards
        if winner == 0:  # Player 1 wins
            reward = 1
        elif winner == 1:  # Player 2 wins
            reward = -1
        elif self.is_board_full():  # Draw
            reward = 0
        else:  # Game is not over
            reward = 0

        # Prepare the next state
        next_state = self.board

        return next_state, reward, done, {}