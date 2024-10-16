import random

class RandomAgent:
    def __str__(self):
        return "Random Agent"
    
    def __repr__(self):
        return "Random Agent"
    
    def choose_move(self, game):
        valid_moves = game.get_valid_moves()
        col = random.choice(valid_moves)
        return col