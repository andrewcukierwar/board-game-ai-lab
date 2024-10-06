class Human:
    def __str__(self):
        return "Human"

    def __repr__(self):
        return "Human"

    def choose_move(self, game):
        while True:
            try:
                column = int(input(f"Player {game.current_player + 1}, choose a column (0-6): "))
                if 0 <= column < game.board.width and game.is_valid_move(column):
                    return column
                else:
                    print("Invalid column. Please choose again.")
            except ValueError:
                print("Please enter a valid number between 0 and 6.")
