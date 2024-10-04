import random

class RandomAgent:
    def __str__(self):
        return "Random Agent"
    
    def __repr__(self):
        return "Random Agent"
    
    def choose_move(self, mancala_game):
        valid_moves = mancala_game.get_valid_moves()
        return random.choice(valid_moves)