class Player():
    def __init__(self, player=None):
        self.player = player if player else 0
        self.piece = 'X' if self.player == 0 else 'O'