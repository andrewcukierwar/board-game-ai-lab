import numpy as np
from functools import lru_cache

class NegamaxAgent:
    def __init__(self, depth):
        self.depth = depth
        self.lookup = {}
        # Precompute all possible window starting positions
        self.window_positions = self.generate_window_positions()

    def __str__(self):
        return f"Negamax Agent {self.depth}"
    
    def __repr__(self):
        return f"Negamax Agent {self.depth}"
    
    def generate_window_positions(self):
        positions = []
        # Horizontal
        for row in range(6):
            for col in range(4):
                positions.append([(row, col + i) for i in range(4)])
        # Vertical
        for row in range(3):
            for col in range(7):
                positions.append([(row + i, col) for i in range(4)])
        # Positive Diagonal
        for row in range(3):
            for col in range(4):
                positions.append([(row + i, col + i) for i in range(4)])
        # Negative Diagonal
        for row in range(3):
            for col in range(3, 7):
                positions.append([(row + i, col - i) for i in range(4)])
        return positions

    def evaluate_board(self, board, color):
        # Convert board to NumPy array for faster access
        board_np = np.array(board)
        evaluation = 0
        weights = [1, 3, 9, 81]

        for window in self.window_positions:
            window_values = board_np[tuple(zip(*window))]
            for length, weight in enumerate(weights, start=1):
                for piece in ('X', 'O'):
                    if (window_values == piece).sum() == length and (window_values == ' ').sum() == (4 - length):
                        evaluation += weight * (1 if piece == 'X' else -1)
        return evaluation * color

    def negamax(self, board, depth, alpha, beta, color):
        # Create a hashable representation of the board
        board_key = tuple(tuple(row) for row in board)
        lookup_key = (board_key, depth, color)
        
        if lookup_key in self.lookup:
            return self.lookup[lookup_key]
        
        # Check for terminal node
        valid_moves = self.get_valid_moves(board)
        winner = self.check_winner(board)
        if depth == 0 or not valid_moves or winner != -1:
            eval = self.evaluate_board(board, color)
            self.lookup[lookup_key] = (-1, eval)
            return -1, eval
        
        max_eval = float('-inf')
        best_move = valid_moves[0]
        
        # Move ordering: prioritize center columns
        center = 3
        sorted_moves = sorted(valid_moves, key=lambda x: abs(center - x))
        
        for move in sorted_moves:
            child_board = self.make_move(board, move, 'X' if color == 1 else 'O')
            _, eval = self.negamax(child_board, depth - 1, -beta, -alpha, -color)
            eval = -eval
            if eval > max_eval:
                max_eval = eval
                best_move = move
            alpha = max(alpha, eval)
            if alpha >= beta:
                break  # Beta cut-off

            if depth == self.depth:
                print(move, eval)
        
        self.lookup[lookup_key] = (best_move, max_eval)
        return best_move, max_eval

    def choose_move(self, node):
        color = 1 if node.current_player == 0 else -1  # Player 0 -> 1, Player 1 -> -1
        board = node.board
        best_move, _ = self.negamax(board, self.depth, float('-inf'), float('inf'), color)
        return best_move

    def get_valid_moves(self, board):
        # Return list of columns that are not full
        return [c for c in range(7) if board[0][c] == ' ']

    def make_move(self, board, move, piece):
        # Create a new board with the move applied
        new_board = [list(row) for row in board]
        for row in reversed(range(6)):
            if new_board[row][move] == ' ':
                new_board[row][move] = piece
                break
        return tuple(tuple(row) for row in new_board)

    def check_winner(self, board):
        # Check all window positions for a win
        board_np = np.array(board)
        for window in self.window_positions:
            window_values = board_np[tuple(zip(*window))]
            if np.all(window_values == 'X'):
                return 1
            elif np.all(window_values == 'O'):
                return 0
        return -1