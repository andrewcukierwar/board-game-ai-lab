from flask import Flask, request, jsonify, send_from_directory
from connect4 import Connect4
from agents.human import Human
from agents.negamax_agent import NegamaxAgent

app = Flask(__name__, static_folder=".")

game = None
human = Human()
bot_agent = NegamaxAgent(depth=3)  # Default depth

agent1 = human
agent2 = bot_agent

@app.route('/start_game', methods=['POST'])
def start_game():
    global game, bot_agent, agent2
    data = request.get_json()
    
    # Extract depth from request, default to 3 if not provided
    depth = data.get('depth', 3)
    
    # Validate depth
    if not isinstance(depth, int) or depth < 1 or depth > 6:
        return jsonify({'error': 'Invalid depth. Please choose a depth between 1 and 6.'}), 400
    
    # Initialize a new NegamaxAgent with the specified depth
    bot_agent = NegamaxAgent(depth=depth)
    agent2 = bot_agent  # Update agent2 to the new bot_agent
    
    # Initialize the game
    game = Connect4()
    
    return jsonify({'message': 'Game started!', 'board': game.board})

@app.route('/make_move', methods=['POST'])
def make_move():
    if not game:
        return jsonify({'error': 'Game has not started.'}), 400

    column = -1
    if game.current_player == 0:
        column = request.json.get('column')
        if column is None:
            return jsonify({'error': 'No column provided.'}), 400
        if not game.is_valid_move(column):
            return jsonify({'error': 'Invalid move.'}), 400
    else:
        column = bot_agent.choose_move(game)

    game.make_move(column)

    check_winner = game.check_winner()
    if check_winner != -1:
        winner = 'Human' if check_winner == 0 else 'AI'
        return jsonify({'board': game.board, 'winner': winner})
    elif game.is_board_full():
        return jsonify({'board': game.board, 'winner': 'Draw'})

    return jsonify({
        'board': game.board,
        'currentPlayer': game.current_player,
        'checkWinner': check_winner
    })

# Serve the HTML frontend
@app.route('/')
def serve_frontend():
    return send_from_directory('.', 'connect4.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
