from board import Board
from connect4 import Connect4

class NegamaxAgent:
    def __init__(self, depth):
        self.depth = depth
        self.lookup = {}
        
    def __str__(self):
        return f"Negamax Agent {self.depth}"
    
    def __repr__(self):
        return f"Negamax Agent {self.depth}"

    # A simple evaluation function that computes the difference in scores
    def evaluate_board(self, board, color):
        windows = []
                
        # Check horizontal
        for row in range(6):
            for col in range(4):
                window = [board[row][col + i] for i in range(4)]
                windows.append(window)

        # Check vertical
        for row in range(3):
            for col in range(7):
                window = [board[row + i][col] for i in range(4)]
                windows.append(window)

        # Check diagonals (positive slope)
        for row in range(3):
            for col in range(4):
                window = [board[row + i][col + i] for i in range(4)]
                windows.append(window)

        # Check diagonals (negative slope)
        for row in range(3):
            for col in range(3, 7):
                window = [board[row + i][col - i] for i in range(4)]
                windows.append(window)        
    
        weights = {1: 1, 2: 3, 3: 9, 4: 81}
        
        evaluation = 0
        for length, weight in weights.items():
            count = {}
            for piece in ('X','O'):
                count[piece] = 0
                for window in windows:
                    if window.count(piece) == length and window.count(" ") == (4 - length):
                        count[piece] += 1
            evaluation += weight * (count['X'] - count['O'])
        evaluation *= color
        
#         print(f'{"".join(["-" for _ in range(self.depth + 1)]) + ">"} Evaluation {evaluation}')
                                
        return evaluation

    def negamax(self, node, depth, alpha, beta, color):
        if depth == 0 or not node.get_valid_moves() or node.check_winner() > -1:
            return -1, self.evaluate_board(node.board, color)

        move_evals = {}
        for move in node.get_valid_moves():
            child_node = Connect4(node.board, node.current_player)
            child_node.make_move(move)
#             print(f'{"".join(["-" for _ in range(self.depth - depth + 1)]) + ">"} Depth {depth}, Player {node.current_player+1}, Move {move}')
#             display_board(child_node.board)     
            
            eval = float('-inf')
            lookup_key = (Board(node.board), depth, color, move)
            if lookup_key in self.lookup:
                eval = self.lookup[lookup_key]
            else:
                eval = -self.negamax(child_node, depth-1, -beta, -alpha, -color)[1]
                self.lookup[lookup_key] = eval
            move_evals[move] = eval
            alpha = max(alpha, eval)
            if beta <= alpha: # Beta cut-off
                break
                
            if depth == self.depth:
                print(move, eval)
                
        max_eval = max(move_evals.values())
        max_moves = [move for move, eval in move_evals.items() if eval == max_eval]
        best_move = max_moves[0] # random.choice(max_moves) # max_moves[0]
        
        return best_move, max_eval

    def choose_move(self, node):
        color = 1 - 2 * node.current_player # Player 0 -> 1, Player 1 -> -1
        best_move, max_eval = self.negamax(node, self.depth, float('-inf'), float('inf'), color)
        return best_move