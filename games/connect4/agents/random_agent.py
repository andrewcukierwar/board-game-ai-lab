import random

class RandomAgent:
    def __init__(self, *, rng=None):
        self.rng = random if rng is None else rng

    def __str__(self):
        return "Random Agent"
    
    def __repr__(self):
        return "Random Agent"
    
    def choose_move(self, game):
        valid_moves = game.get_valid_moves()
        col = self.rng.choice(valid_moves)
        return col
