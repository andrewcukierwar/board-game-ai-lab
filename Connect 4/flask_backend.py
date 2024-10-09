from flask import Flask, request, jsonify, send_from_directory
from connect4 import Connect4
from agents.human import Human
from agents.negamax_agent import NegamaxAgent

app = Flask(__name__, static_folder=".")

game = None
human = Human()
bot_agent = NegamaxAgent(depth=8)
agent1 = human
agent2 = bot_agent

@app.route('/start_game', methods=['POST'])
def start_game():
    global game
    game = Connect4()
    # game.battle(human, bot_agent)
    return jsonify({'message': 'Game started!', 'board': game.board})

# @app.route('/make_move', methods=['POST'])
# def battle():
#     if not game:
#         return jsonify({'error': 'Game has not started.'}), 400



#     move = request.json.get('column')
#     if not game.is_valid_move(move):
#         return jsonify({'error': 'Invalid move.'}), 400

#     game.make_move(move)
#     return jsonify({'board': game.board, 'current_player': game.current_player})

@app.route('/make_move', methods=['POST'])
def make_move():
    if not game:
        return jsonify({'error': 'Game has not started.'}), 400

    column = -1
    if game.current_player == 0:
        column = request.json.get('column')
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

    return jsonify({'board': game.board, 'currentPlayer': game.current_player, 'checkWinner': check_winner})

# Serve the HTML frontend
@app.route('/')
def serve_frontend():
    return send_from_directory('.', 'connect4.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
