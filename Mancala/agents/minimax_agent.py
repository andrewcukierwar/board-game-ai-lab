from mancala import Mancala

class MiniMaxAgent:
    def __init__(self, depth):
        self.depth = depth
        self.lookup = {}
        
    def __str__(self):
        return f"MiniMax Agent {self.depth}"
    
    def __repr__(self):
        return f"MiniMax Agent {self.depth}"

    # A simple evaluation function that computes the difference in scores
    def evaluate_board(self, board, player):
        player_store = 6 if player == 0 else 13 # player_store
        opponent_store = 13 if player == 0 else 6 # opponent_store
        # print('-- Player Store:', board[player_store], ', Opponent Store:', board[opponent_store])
        return board[player_store] - board[opponent_store]

    def minimax(self, node, depth, alpha, beta, maximizing_player):
        if depth == 0 or not node.get_valid_moves():
            player = node.current_player if maximizing_player else 1 - node.current_player
            # player = 1 - node.current_player if not maximizing_player else node.current_player
            return self.evaluate_board(node.board, player)

        if maximizing_player:
            max_eval = float('-inf')
            for move in node.get_valid_moves():
#                 child_node = Mancala()
#                 child_node.board = node.board.copy()
#                 child_node.current_player = node.current_player
                child_node = Mancala(node.board, node.current_player)
                child_node.make_move(move)
                # print('- Max Node', move, node.current_player+1, depth, child_node.board)
                eval = float('-inf')
                if child_node.current_player == node.current_player:
                    # print("---GOING AGAIN---")
                    eval = self.minimax(child_node, depth, alpha, beta, True)
                else:
                    eval = self.minimax(child_node, depth-1, alpha, beta, False)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha: # Beta cut-off
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in node.get_valid_moves():
#                 child_node = Mancala()
#                 child_node.board = node.board.copy()
#                 child_node.current_player = node.current_player
                child_node = Mancala(node.board, node.current_player)
                child_node.make_move(move)
                # print('- Min Node', move, node.current_player+1, depth, child_node.board)
                eval = float('inf')
                if child_node.current_player == node.current_player:
                    # print("---GOING AGAIN---")
                    eval = self.minimax(child_node, depth, alpha, beta, False)
                else:
                    eval = self.minimax(child_node, depth-1, alpha, beta, True)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha: # Alpha cut-off
                    break
            return min_eval

    def choose_move(self, node):
#         print(self.depth)
        valid_moves = node.get_valid_moves()
        move_evals = {}
        for move in valid_moves:
#                 child_node = Mancala()
#                 child_node.board = node.board.copy()
#                 child_node.current_player = node.current_player
            child_node = Mancala(node.board, node.current_player)
            child_node.make_move(move)
            # print('Root', move, node.current_player+1, self.depth, child_node.board)
            eval = float('-inf')
            if child_node.current_player == node.current_player:
                # print("---GOING AGAIN---")
                eval = self.minimax(child_node, self.depth, float('-inf'), float('inf'), True)
            else:
                eval = self.minimax(child_node, self.depth-1, float('-inf'), float('inf'), False)
            move_evals[move] = eval
            print(move, eval)
        max_eval = max(move_evals.values())
        max_moves = [move for move, eval in move_evals.items() if eval == max_eval]
        best_move = max_moves[0] # random.choice(max_moves) # max_moves[0]
#         self.depth += 1
        return best_move