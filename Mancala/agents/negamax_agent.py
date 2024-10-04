from mancala import Mancala

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
        player_store = 6 if color == 1 else 13 # player_store
        opponent_store = 13 if color == 1 else 6 # opponent_store
        return board[player_store] - board[opponent_store]

    def negamax(self, node, depth, alpha, beta, color):
        if depth == 0 or not node.get_valid_moves():
            return -1, self.evaluate_board(node.board, color)

        move_evals = {}
        for move in node.get_valid_moves():
            child_node = Mancala(node.board, node.current_player)
            child_node.make_move(move)
            eval = float('-inf')
            lookup_key = (tuple(node.board), depth, color, move)
            if lookup_key in self.lookup:
                eval = self.lookup[lookup_key]
            elif child_node.current_player == node.current_player:
                eval = self.negamax(child_node, depth, alpha, beta, color)[1]
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
        color = 1 - 2 * node.current_player
        best_move, max_eval = self.negamax(node, self.depth, float('-inf'), float('inf'), color)
        return best_move