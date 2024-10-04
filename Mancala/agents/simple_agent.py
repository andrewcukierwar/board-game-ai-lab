class SimpleAgent:
    def __str__(self):
        return "Simple Agent"
    
    def __repr__(self):
        return "Simple Agent"
        
    def choose_move(self, mancala_game):
        board = mancala_game.board
        current_player = mancala_game.current_player
        store = 6 if current_player == 0 else 13
        
        possible_moves = [i for i in range(6)] if current_player == 0 else [i + 7 for i in range(6)]
        valid_moves = mancala_game.get_valid_moves()
        
        # check in reverse order if player 2
        if current_player == 1:
            valid_moves = valid_moves[::-1]
            
        for index in valid_moves:
            landing_space = (index + mancala_game.board[index]) % 14
            if landing_space == store:
                # print("I'm getting another turn hehe")
                return index
        
        # Check for capture, capture max amount if available
        max_index, max_val = -1, -1
        for index in valid_moves:
            landing_space = (index + mancala_game.board[index]) % 14
            if landing_space in possible_moves and board[landing_space] == 0 and board[12 - landing_space] > 0:
                if max_index == -1:
                    max_index, max_val = index, board[12 - landing_space]
                elif board[12 - landing_space] > max_val:
                    max_index, max_val = index, board[12 - landing_space]
        if max_index > -1:
            # print("I'm stealing your stones hehe")
            return max_index
            
        # print("Performing default move")
        default_move = max(valid_moves)
        return default_move