from ..board import Board
from ..connect4 import Connect4

class NegamaxAgent:
    def __init__(self, depth):
        self.depth = depth
        self.lookup = {}
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

    # A simple evaluation function that computes the difference in scores
    def evaluate_board(self, board, color):
        weights = {1:1, 2:3, 3:9, 4:81}
        evaluation = 0
        for window in self.window_positions:
            window_values = [board[row][col] for row, col in window]
            x_count = window_values.count('X')
            o_count = window_values.count('O')
            empty_count = window_values.count(' ')
            for length, weight in weights.items():
                if x_count == length and empty_count == (4 - length):
                    evaluation += weight
                if o_count == length and empty_count == (4 - length):
                    evaluation -= weight
        evaluation *= color
        # print(f'{"".join(["-" for _ in range(self.depth + 1)]) + ">"} Evaluation {evaluation}')
        return evaluation

    def negamax(self, node, depth, alpha, beta, color):
        if depth == 0 or not node.get_valid_moves() or node.check_winner() > -1:
            return -1, self.evaluate_board(node.board, color)
        move_evals = {}
        for move in node.get_valid_moves():
            child_node = Connect4(node.board, node.current_player)
            child_node.make_move(move)
            # print(f'{"".join(["-" for _ in range(self.depth - depth + 1)]) + ">"} Depth {depth}, Player {node.current_player+1}, Move {move}')
            # display_board(child_node.board)
            eval = float('-inf')
            lookup_key = (Board(node.board), depth, color, move)
            if lookup_key in self.lookup:
                eval = self.lookup[lookup_key]
            else:
                eval = -self.negamax(child_node, depth-1, -beta, -alpha, -color)[1]
                self.lookup[lookup_key] = eval
            move_evals[move] = eval
            alpha = max(alpha, eval)
            if beta <= alpha: # beta cut-off
                break
            if depth == self.depth: # print eval for each move
                print(move, eval)
        max_eval = max(move_evals.values())
        max_moves = [move for move, eval in move_evals.items() if eval == max_eval]
        best_move = max_moves[0] # random.choice(max_moves)
        return best_move, max_eval

    def choose_move(self, node):
        color = 1 - 2 * node.current_player # Player 0 -> 1, Player 1 -> -1
        best_move, max_eval = self.negamax(node, self.depth, float('-inf'), float('inf'), color)
        return best_move