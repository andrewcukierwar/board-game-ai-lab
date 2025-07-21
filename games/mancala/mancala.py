class Mancala:
    def __init__(self, board=None, current_player=None):
        # Initialize the game board
        # 6 pits for each player and 1 store for each player if no board provided
        self.board = board.copy() if board else [4] * 6 + [0] + [4] * 6 + [0]
        self.current_player = current_player if current_player else 0  # 0 for player 1, 1 for player 2

    def display_board(self):
        print(f"Current Player: {self.current_player + 1}")
        
        pot1 = self.board[6]
        pot2 = self.board[13]
        upper_pockets = self.board[0:6]
        lower_pockets = self.board[12:6:-1]
        
        print(" ________________________________________________________")
        print("|         ___    ___    ___    ___    ___    ___         |")
        print("|  ___   [_{}_]  [_{}_]  [_{}_]  [_{}_]  [_{}_]  [_{}_]   ___  |".format(*upper_pockets))
        print("| |   |                                            |   | |")
        print("| | {} |                                            | {} | |".format(pot2, pot1))
        print("| |___|   ___    ___    ___    ___    ___    ___   |___| |")
        print("|        [_{}_]  [_{}_]  [_{}_]  [_{}_]  [_{}_]  [_{}_]        |".format(*lower_pockets))
        print("|________________________________________________________|")
        
#         print(" _________________________________________________________")
#         print("|  ___    ___    ___    ___    ___    ___    ___          |")
#         print("| |   |  [_{}_]  [_{}_]  [_{}_]  [_{}_]  [_{}_]   [_{}_]   ___  |".format(*upper_pockets))
#         print("| | {} |                                             |   | |".format(pot2))
#         print("| |___|   ___    ___    ___    ___    ___    ___    | {} | |".format(pot1))
#         print("|        [_{}_]  [_{}_]  [_{}_]  [_{}_]  [_{}_]   [_{}_]  |___| |".format(*lower_pockets))
#         print("|_________________________________________________________|")

    def get_valid_moves(self):
        possible_moves = [i for i in range(6)] if self.current_player == 0 else [i + 7 for i in range(6)]
        valid_moves = [i for i in possible_moves if self.board[i] > 0]
        return valid_moves

    def make_move(self, pit_index):
        store_index = 6 if self.current_player == 0 else 13
        # Check if the selected pit is valid for the current player
        if self.current_player == 0 and (pit_index < 0 or pit_index >= 6 or self.board[pit_index] == 0):
            print(self.current_player)
            print("Invalid pit index for Player 1.")
            return False
        elif self.current_player == 1 and (pit_index < 7 or pit_index >= 13 or self.board[pit_index] == 0):
            print(self.current_player)
            print("Invalid pit index for Player 2.")
            return False

        stones = self.board[pit_index]
        self.board[pit_index] = 0

        # Distribute the stones to the pits
        while stones > 0:
            pit_index = (pit_index + 1) % 14
            # Skip the opponent's store
            if self.current_player == 0 and pit_index == 13:
                pit_index = (pit_index + 1) % 14
            elif self.current_player == 1 and pit_index == 6:
                pit_index = (pit_index + 1) % 14
            self.board[pit_index] += 1
            stones -= 1

        # Switch player if the last stone doesn't end in the player's store
        if pit_index != 6 and pit_index != 13:
            
            possible_moves = [i for i in range(6)] if self.current_player == 0 else [i + 7 for i in range(6)]
             # Check for capture and extra move
            if self.board[pit_index] == 1 and self.board[12 - pit_index] > 0 and pit_index in possible_moves:
                self.board[store_index] += self.board[pit_index] + self.board[12 - pit_index]
                self.board[pit_index] = 0
                self.board[12 - pit_index] = 0
                
            self.current_player = 1 - self.current_player
            
        if sum(self.board[0:6]) == 0 or sum(self.board[7:13]) == 0:
            self.board[6] += sum(self.board[0:6])
            self.board[0:6] = [0,0,0,0,0,0]

            self.board[13] += sum(self.board[7:13])
            self.board[7:13] = [0,0,0,0,0,0]

        return True
    
    def is_game_over(self):
        return sum(self.board[0:6]) == 0 or sum(self.board[7:13]) == 0

    def battle(self, agent1, agent2, output=True):
        while True:
            if output:
                self.display_board()

            try:
                if self.current_player == 0:
                    pit_index = agent1.choose_move(self)
                    if output:
                        print(f"{agent1} chooses pit {pit_index}.")
                        print()
                else:
                    pit_index = agent2.choose_move(self)
                    if output:
                        print(f"{agent2} chooses pit {pit_index}.")
                        print()
            except ValueError:
                print("Invalid input. Please enter a number.")
                continue

            self.make_move(pit_index)

            # Check for game over
            if self.is_game_over():
                if output:
                    self.display_board()
                    print("Game over!")

                player_store1 = self.board[6]
                player_store2 = self.board[13]

                if player_store1 > player_store2:
                    print(f'{agent1} wins {player_store1}-{player_store2}')
                elif player_store1 < player_store2:
                    print(f'{agent2} wins {player_store2}-{player_store1}')
                else:
                    print('Tie!')
                
                return player_store1, player_store2