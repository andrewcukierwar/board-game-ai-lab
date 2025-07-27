import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import os
from copy import deepcopy
from dataclasses import dataclass
from typing import List #, Tuple

@dataclass
class TrainingExample:
    board: np.ndarray
    policy: np.ndarray  # MCTS improved policy
    value: float        # Final game outcome

class Connect4Net(nn.Module):
    def __init__(self):
        super(Connect4Net, self).__init__()
        # Shared layers
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, padding=1)
        
        # Policy head
        self.policy_conv = nn.Conv2d(128, 32, 1)
        self.policy_fc = nn.Linear(32 * 6 * 7, 7)  # 7 possible moves
        
        # Value head - now outputs 3 classes [win, draw, loss]
        self.value_conv = nn.Conv2d(128, 32, 1)
        self.value_fc1 = nn.Linear(32 * 6 * 7, 64)
        self.value_fc2 = nn.Linear(64, 3)  # Changed to output 3 classes

    def forward(self, x):
        # Shared layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Policy head
        policy = F.relu(self.policy_conv(x))
        policy = policy.view(-1, 32 * 6 * 7)
        policy = self.policy_fc(policy)
        policy = F.softmax(policy, dim=1)
        
        # Value head - now outputs probabilities for [win, draw, loss]
        value = F.relu(self.value_conv(x))
        value = value.view(-1, 32 * 6 * 7)
        value = F.relu(self.value_fc1(value))
        value = F.softmax(self.value_fc2(value), dim=1)  # Changed to softmax
        
        return policy, value
    
class Node:
    def __init__(self, game_state, parent=None, move=None, prior_p=0):
        self.game_state = game_state
        self.parent = parent
        self.move = move
        self.children = {}
        self.wins = 0
        self.visits = 0
        self.prior_p = prior_p
        self.untried_moves = game_state.get_valid_moves()
        self.q_value = 0.0  # Track running average of values
        # Add board hash for transposition table
        self.board_hash = self._compute_board_hash()
        
    def _compute_board_hash(self):
        # Convert board to tuple of tuples for hashing
        # Convert X/O/space to numbers first
        board_nums = [[1 if cell == 'X' else -1 if cell == 'O' else 0 
                      for cell in row] for row in self.game_state.board]
        return tuple(tuple(row) for row in board_nums)

    def ucb1(self, c=1.41):
        if self.visits == 0:
            return float('inf')
        # Use Q-value directly instead of wins/visits
        u_value = c * self.prior_p * math.sqrt(self.parent.visits) / (1 + self.visits)
        return self.q_value + u_value
    
    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def is_terminal(self):
        return self.game_state.is_game_over()

class MCTSNNAgent:
    def __init__(self, model, simulation_limit=1000, temperature=1.0):
        self.model = model
        self.simulation_limit = simulation_limit
        self.temperature = temperature
        self.transposition_table = {}  # Hash -> Node mapping
        self.training_examples = []  # Store (state, policy, value) triplets

    # def _get_state_tensor(self, game_state):
    #     board = game_state.board.reshape(6, 7)
    #     # Convert to network input format (1, 1, 6, 7)
    #     tensor = torch.FloatTensor(board).unsqueeze(0).unsqueeze(0)
    #     return tensor
    
    def _reshape_board(self, board):
        # Convert board to numpy array first
        board_num = [[1 if cell == 'X' else -1 if cell == 'O' else 0 for cell in row] for row in board]
        board_arr = np.array(board_num, dtype=np.float32)
        # Now we can reshape
        board_reshaped = board_arr.reshape(1, 6, 7)  # Add batch dimension
        return board_reshaped

    def _get_state_tensor(self, game_state):
        """Convert game state to tensor for neural network input"""
        # # Convert board to numpy array first
        # board_num = [[1 if cell == 'X' else -1 if cell == 'O' else 0 for cell in row] for row in game_state.board]
        # board = np.array(board_num, dtype=np.float32)
        # # Now we can reshape
        # board = board.reshape(1, 6, 7)  # Add batch dimension
        board = self._reshape_board(game_state.board)
        return torch.FloatTensor(board)

    def _predict(self, game_state):
        self.model.eval()
        with torch.no_grad():
            tensor = self._get_state_tensor(game_state)
            policy, value = self.model(tensor)
            # Convert 3-class probability to scalar value
            value_probs = value.squeeze()
            # Calculate expected value: 1 * p(win) + 0 * p(draw) + (-1) * p(loss)
            scalar_value = value_probs[0] - value_probs[2]
            return policy.squeeze(), scalar_value

    def _check_forced_move(self, game):
        """Check if there's a winning move or a move to block opponent's win"""
        # Check for immediate winning moves
        for move in game.get_valid_moves():
            game_copy = deepcopy(game)
            game_copy.make_move(move)
            if game_copy.check_winner() == game.current_player:
                return move

        # Check for moves that block opponent's win
        opponent = 1 - game.current_player  # Switch between -1 and 1
        game_copy = deepcopy(game)
        game_copy.current_player = opponent
        for move in game.get_valid_moves():
            test_game = deepcopy(game_copy)
            test_game.make_move(move)
            if test_game.check_winner() == opponent:
                return move

        return None

    def choose_move(self, game):
        # First check for forced moves
        forced_move = self._check_forced_move(game)
        if forced_move is not None:
            # Create one-hot policy for forced moves
            policy = np.zeros(7)
            policy[forced_move] = 1.0
            self._store_training_example(game.board, policy)
            return forced_move

        root = Node(deepcopy(game))
        self.transposition_table = {}
        self.transposition_table[root.board_hash] = root
        
        for n in range(self.simulation_limit):
            node = root # deepcopy(root)
            game_copy = deepcopy(game)

            # print('Simulation', n, '- Untried Moves:', root.untried_moves)
            
            # Selection
            # while node.is_fully_expanded() and not node.is_terminal():
            #     child = self._select_child(node)
            #     if not game_copy.make_move(child.move):  # Check if move was successful
            #         continue
            #     node = child
            #     board_hash = node.board_hash
            #     if board_hash in self.transposition_table:
            #         node = self.transposition_table[board_hash]

            # print(game_copy.board)
            # print(game_copy.get_valid_moves())

            while node.is_fully_expanded() and not node.is_terminal():
                # print(game_copy.board)
                # Recalculate valid moves from the current game_copy state
                current_valid_moves = game_copy.get_valid_moves()
                
                if not current_valid_moves:
                    break  # No valid moves remain

                # Select a child; ensure its move is still valid
                child_move, child = self._select_child(node)
                if child_move not in current_valid_moves: # child.move
                    print(game_copy.board)
                    print('Current Valid Moves:', current_valid_moves)
                    print('Current Child Nodes:', list(node.children.keys()))
                    print('Selected Child Move:', child_move)
                    # Skip this child if its move is no longer valid
                    # Optionally, remove it from the node's children to avoid reconsideration.
                    node.children.pop(child_move, None)
                    continue

                # Attempt to make the move on the game_copy
                if not game_copy.make_move(child_move):
                    print('Failed Attempting Child Move:', child_move)
                    # If make_move fails, remove the move from the valid set and try again
                    if child_move in current_valid_moves:
                        current_valid_moves.remove(child_move)
                    continue

                node = child
                board_hash = node.board_hash
                if board_hash in self.transposition_table:
                    node = self.transposition_table[board_hash]

            # Rest of the MCTS logic remains the same...
            if not node.is_terminal():
                policy, value = self._predict(game_copy)
                
                valid_moves = game_copy.get_valid_moves()
                policy = policy.numpy()
                valid_policy = np.zeros(7)
                valid_policy[valid_moves] = policy[valid_moves]
                # Add safety check for zero sum
                policy_sum = valid_policy.sum()
                if policy_sum > 0:
                    valid_policy /= policy_sum
                else:
                    # If all probabilities are zero, use uniform distribution over valid moves
                    valid_policy[valid_moves] = 1.0 / len(valid_moves)
                
                for move in valid_moves:
                    game_next = deepcopy(game_copy)
                    game_next.make_move(move)
                    # Convert board state to numbers before hashing
                    board_hash = tuple(tuple(1 if cell == 'X' else -1 if cell == 'O' else 0 for cell in row) for row in game_next.board)
                    
                    # Reuse existing node or create new one
                    if board_hash in self.transposition_table:
                        node.children[move] = self.transposition_table[board_hash]
                    else:
                        new_node = Node(
                            game_next,
                            parent=node,
                            move=move,
                            prior_p=valid_policy[move]
                        )
                        node.children[move] = new_node
                        self.transposition_table[board_hash] = new_node
                
                node.untried_moves = []
                winner_value = float(value)
            else:
                # Explicitly handle terminal states including draws
                winner = game_copy.check_winner()
                if winner == -1:  # Draw
                    winner_value = 0.0
                else:
                    winner_value = 1.0 if winner == game_copy.current_player else -1.0
            
            # Backpropagation with Q-value updates
            while node:
                node.visits += 1
                # Update Q-value using incremental mean formula
                node.q_value += (winner_value - node.q_value) / node.visits
                winner_value = -winner_value  # Flip value for opponent
                node = node.parent
        
        # Store MCTS policy (visit distribution)
        visits = np.array([child.visits for child in root.children.values()])
        moves = list(root.children.keys())
        policy = np.zeros(7)
        total_visits = visits.sum()
        for move, visit_count in zip(moves, visits):
            policy[move] = visit_count / total_visits
        
        self._store_training_example(game.board, policy)
        
        # Choose move based on visit counts and temperature
        if self.temperature == 0:
            move = moves[np.argmax(visits)]
        else:
            visits = visits ** (1/self.temperature)
            visits = visits / visits.sum()
            move = np.random.choice(moves, p=visits)
        
        return move

    # def _select_child(self, node):
    #     """Select the child with the highest UCB1 value"""
    #     child_keys = list(node.children.keys())
    #     ucb1_values = [child.ucb1() for child in node.children.values()]
    #     print('Selecting child from', child_keys, 'with UCB1 values:', ucb1_values)
    #     if not node.children:
    #         raise ValueError("Node has no children")
    #     print('Child Values:', node.children.values())
    #     max_child = max(node.children.values(), key=lambda x: x.ucb1())
    #     print('Max child', max_child)
    #     return max_child

    def _select_child(self, node):
        """Select the child with the highest UCB1 value"""
        if not node.children:
            raise ValueError("Node has no children")
        best_move, best_child = max(node.children.items(), key=lambda x: x[1].ucb1())
        # assert best_child.move == best_move, f"Move mismatch: {best_child.move} != {best_move}"
        return best_move, best_child

    def _store_training_example(self, board: np.ndarray, policy: np.ndarray):
        """Store a training example without the final value (updated later)"""
        self.training_examples.append(TrainingExample(
            board=board.copy(),
            policy=policy.copy(),
            value=None  # Will be updated when game ends
        ))

    def update_game_result(self, winner: int):
        """Update training examples with final game outcome"""
        if not self.training_examples:
            return
        
        # Convert winner to values for each position
        for example in self.training_examples:
            if winner == -1:  # Draw
                example.value = 0.0
            else:
                # Determine if position was player's turn or opponent's
                player = 1 if np.sum(example.board == 1) == np.sum(example.board == 2) else 2
                example.value = 1.0 if player == winner else -1.0

    def get_training_data(self) -> List[TrainingExample]:
        """Return collected training examples and clear buffer"""
        examples = self.training_examples
        self.training_examples = []
        return examples

def load_pretrained_mcts_nn_agent(model_path=None, simulation_limit=1000, temperature=0.1):
    """
    Loads a pre-trained Connect4Net model and returns an MCTSNNAgent.
    If model_path is None, tries to load from 'training/connect4_model_iter_100.pt' relative to this file.
    """
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    if model_path is None:
        # model_path = os.path.join(script_dir, '..', 'training', 'connect4_model_iter_100.pt')
        model_path = 'models/connect4/connect4_model_iter_100.pt'  # Adjust path as needed
    model = Connect4Net()
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return MCTSNNAgent(model, simulation_limit=simulation_limit, temperature=temperature)

def test_mcts_nn_agent(game_class, model_path=None):
    """
    Example test function to play one move using the MCTSNNAgent and a pre-trained model.
    game_class: a class implementing the Connect4 game interface (must have .board, .current_player, .get_valid_moves(), .make_move(), etc.)
    """
    agent = load_pretrained_mcts_nn_agent(model_path)
    game = game_class()  # Assumes game_class() creates a new game
    move = agent.choose_move(game)
    print(f"MCTSNNAgent selects move: {move}")
    return move