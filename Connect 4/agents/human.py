class Human:
    def __str__(self):
        return "Human"
    
    def __repr__(self):
        return "Human"
    
    def choose_move(self, game):
        message = f"Player {game.current_player+1}, choose a column (0-6): "
        column = int(input(message))
        return column