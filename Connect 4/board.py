import numpy as np

class Board(list):
    def __init__(self, board=None):
        arr = [[cell for cell in row] for row in board] if board else [[" " for _ in range(7)] for _ in range(6)]
        super().__init__(arr)
    
    def to_string(self):
        line_break = '–––––––––––––––––––––––––––––'
        rows = ['| ' + ' | '.join([cell for cell in row]) + ' |' for row in self]
        output = line_break + '\n' + f'\n{line_break}\n'.join(rows) + '\n' + line_break
        return output
    
    def to_tuple(self):
        tup = tuple([tuple([cell for cell in row]) for row in self])
        return tup

    def flatten(self): # NEW METHOD FOR Q-LEARNING
        return np.array(self).flatten()
    
#     def from_tuple(self, tup):
#         arr = [[cell for cell in row] for row in self]
#         board = Board(arr)
            
    def __repr__(self):
        output = self.to_string()
        return output
    
    def __str__(self):
        output = self.to_string()
        return output
    
    def __hash__(self):
        hashed_tup = hash(self.to_tuple())
        return hashed_tup