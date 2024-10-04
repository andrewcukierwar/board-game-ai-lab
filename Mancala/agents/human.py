class Human:
    def __str__(self):
        return "Human"
    
    def __repr__(self):
        return "Human"
    
    def choose_move(self, mancala_game):
        message = "Player 1, choose a pit (0-5): "
        if mancala_game.current_player == 1:
            message = "Player 2, choose a pit (7-12): "
        pit_index = int(input(message))
        return pit_index