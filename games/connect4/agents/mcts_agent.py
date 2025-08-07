import math
import random
import numpy as np
from copy import deepcopy
from ..connect4 import Connect4

class Node:
    def __init__(self, game_state, parent=None, move=None):
        self.game_state = game_state
        self.parent = parent
        self.move = move  # Move that led to this state
        self.children = {}  # Dict of move -> Node
        self.wins = 0
        self.visits = 0
        self.untried_moves = game_state.get_valid_moves()
        
    def ucb1(self, c=1.41):
        if self.visits == 0:
            return float('inf')
        return (self.wins / self.visits) + c * math.sqrt(math.log(self.parent.visits) / self.visits)

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def is_terminal(self):
        return self.game_state.is_game_over()

class MCTSAgent:
    def __init__(self, simulation_limit=1000):
        self.simulation_limit = simulation_limit

    def __str__(self):
        return f"MCTS Agent ({self.simulation_limit} sims)"
    
    def __repr__(self):
        return self.__str__()

    def choose_move(self, game):
        root = Node(deepcopy(game))
        
        # Run MCTS for simulation_limit iterations
        for _ in range(self.simulation_limit):
            node = root
            game_copy = deepcopy(game)
            
            # Selection
            while node.is_fully_expanded() and not node.is_terminal():
                node = self._select_child(node)
                game_copy.make_move(node.move)
            
            # Expansion
            if not node.is_terminal():
                move = random.choice(node.untried_moves)
                game_copy.make_move(move)
                node.untried_moves.remove(move)
                new_node = Node(deepcopy(game_copy), parent=node, move=move)
                node.children[move] = new_node
                node = new_node
            
            # Simulation
            winner = self._simulate(game_copy)
            
            # Backpropagation
            while node:
                node.visits += 1
                if winner == -1:  # Draw
                    node.wins += 0.5
                else:
                    ## # Update wins from the perspective of the player who made the move
                    ## node.wins += 1 if winner != game_copy.current_player else 0
                    # Award +1 if this node's current_player is indeed the winner
                    node.wins += 1 if winner == node.game_state.current_player else 0

                node = node.parent
        
        # Choose move with highest visit count
        return max(root.children.items(), key=lambda x: (x[1].visits, random.random()))[0]

    def _select_child(self, node):
        return max(node.children.values(), key=lambda x: x.ucb1())

    def _simulate(self, game):
        game_copy = deepcopy(game)
        
        while not game_copy.is_game_over():
            valid_moves = game_copy.get_valid_moves()
            move = random.choice(valid_moves)
            game_copy.make_move(move)
        
        return game_copy.check_winner()