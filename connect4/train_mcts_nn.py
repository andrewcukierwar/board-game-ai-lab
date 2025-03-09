import torch
import torch.optim as optim
from connect4 import Connect4
from agents.mcts_nn_agent import MCTSNNAgent, Connect4Net
import numpy as np
from tqdm import tqdm

def self_play_episode(agent1, agent2):
    """Play one full game between two agents and collect training data"""
    game = Connect4()
    
    while not game.is_game_over():
        # Get move from current player
        current_agent = agent1 if game.current_player == 0 else agent2
        # print('Valid Moves', game.get_valid_moves())
        # print('Untried Moves', current_agent.untried_moves)
        move = current_agent.choose_move(game)
        game.make_move(move)
        # print(game.board)
    
    # Get game result
    winner = game.check_winner()
    print('Winner is', winner)
    
    # Update training examples with final result
    agent1.update_game_result(winner)
    agent2.update_game_result(winner)
    
    return agent1.get_training_data() + agent2.get_training_data()

def reshape_board(board):
        # Convert board to numpy array first
        board_num = [[1 if cell == 'X' else -1 if cell == 'O' else 0 for cell in row] for row in board]
        board_arr = np.array(board_num, dtype=np.float32)
        # Now we can reshape
        board_reshaped = board_arr.reshape(1, 6, 7)  # Add batch dimension
        return board_reshaped

def train_network(model, examples, batch_size=32, epochs=10, lr=0.001):
    """Train the neural network on collected examples"""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    policy_criterion = torch.nn.CrossEntropyLoss()
    value_criterion = torch.nn.CrossEntropyLoss()
    
    # Prepare data
    # states = torch.FloatTensor([ex.board.reshape(1, 6, 7) for ex in examples])
    states = torch.FloatTensor(np.array([reshape_board(ex.board) for ex in examples]))
    policies = torch.FloatTensor(np.array([ex.policy for ex in examples]))
    values = torch.FloatTensor(np.array([ex.value for ex in examples]))
    
    # Convert values to 3-class format
    value_targets = torch.zeros((len(examples), 3))
    value_targets[values == 1.0, 0] = 1.0  # win
    value_targets[values == 0.0, 1] = 1.0  # draw
    value_targets[values == -1.0, 2] = 1.0  # loss
    
    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        for i in range(0, len(examples), batch_size):
            batch_states = states[i:i+batch_size]
            batch_policies = policies[i:i+batch_size]
            batch_values = value_targets[i:i+batch_size]
            
            # Forward pass
            pred_policies, pred_values = model(batch_states)
            
            # Calculate losses
            policy_loss = policy_criterion(pred_policies, batch_policies)
            value_loss = value_criterion(pred_values, batch_values)
            loss = policy_loss + value_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(examples):.4f}")

def main():
    # Initialize model and agents
    model = Connect4Net()
    agent1 = MCTSNNAgent(model, simulation_limit=100, temperature=1.0)
    agent2 = MCTSNNAgent(model, simulation_limit=100, temperature=1.0)
    
    num_iterations = 10  # Number of training iterations
    games_per_iteration = 10  # Number of self-play games per iteration
    
    for iteration in tqdm(range(num_iterations), desc="Training Progress"):
        print('Iteration', iteration)
        examples = []
        
        print('Playing games')
        # Self-play phase
        for n in range(games_per_iteration):
            print('Playing game', n, 'of iteration', iteration)
            game_examples = self_play_episode(agent1, agent2)
            examples.extend(game_examples)
        
        print('Training')
        # Training phase
        train_network(model, examples)
        
        # Optionally save the model periodically
        if (iteration + 1) % 10 == 0:
            print('Saving Model')
            torch.save({
                'model_state_dict': model.state_dict(),
                'iteration': iteration
            }, f'connect4_model_iter_{iteration+1}.pt')

if __name__ == "__main__":
    main()