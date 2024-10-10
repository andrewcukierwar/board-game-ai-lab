from flask import Flask, request, jsonify, send_from_directory
from connect4 import Connect4
from agents.agent_factory import create_agent
from agents.human import Human
import logging

app = Flask(__name__, static_folder=".")

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize global game and agents
game = None
player1_agent = None
player2_agent = None

@app.route('/start_game', methods=['POST'])
def start_game():
    global game, player1_agent, player2_agent
    data = request.get_json()

    # Extract player configurations
    player1_config = data.get('player1', {'type': 'human'})
    player2_config = data.get('player2', {'type': 'negamax', 'depth': 2})

    try:
        player1_agent = create_agent(player1_config)
        player2_agent = create_agent(player2_config)
    except ValueError as ve:
        logger.error(f"Agent creation error: {ve}")
        return jsonify({'error': str(ve)}), 400

    # Initialize the game
    game = Connect4()

    logger.info("Game started.")
    logger.debug(f"Initial Board: {game.board}")
    logger.debug(f"Current Player: {game.current_player}")

    return jsonify({
        'message': 'Game started!',
        'board': game.board,
        'currentPlayer': game.current_player  # Include currentPlayer in the response
    })

@app.route('/make_move', methods=['POST'])
def make_move():
    if not game:
        logger.error("Game has not started.")
        return jsonify({'error': 'Game has not started.'}), 400

    column = -1
    try:
        if game.current_player == 0:
            # Player 1's turn
            if isinstance(player1_agent, Human):
                move = request.json.get('column')
                logger.debug(f"Player 1 (Human) Move: {move}")
                if move is None:
                    logger.error("No column provided by Player 1 (Human).")
                    return jsonify({'error': 'No column provided for Player 1 (Human).'}), 400
                if not game.is_valid_move(move):
                    logger.error(f"Invalid move by Player 1 (Human): Column {move} is full.")
                    return jsonify({'error': 'Invalid move by Player 1 (Human).'}), 400
                column = move
            else:
                # Player 1 is an agent
                logger.debug("Player 1 (Agent) is making a move.")
                column = player1_agent.choose_move(game)
                logger.debug(f"Player 1 (Agent) Move: {column}")
                if column == -1:
                    logger.error("Player 1 (Agent) has no valid moves.")
                    return jsonify({'error': 'No valid moves available for Player 1 (Agent).'}), 400

        else:
            # Player 2's turn
            if isinstance(player2_agent, Human):
                move = request.json.get('column')
                logger.debug(f"Player 2 (Human) Move: {move}")
                if move is None:
                    logger.error("No column provided by Player 2 (Human).")
                    return jsonify({'error': 'No column provided for Player 2 (Human).'}), 400
                if not game.is_valid_move(move):
                    logger.error(f"Invalid move by Player 2 (Human): Column {move} is full.")
                    return jsonify({'error': 'Invalid move by Player 2 (Human).'}), 400
                column = move
            else:
                # Player 2 is an agent
                logger.debug("Player 2 (Agent) is making a move.")
                column = player2_agent.choose_move(game)
                logger.debug(f"Player 2 (Agent) Move: {column}")
                if column == -1:
                    logger.error("Player 2 (Agent) has no valid moves.")
                    return jsonify({'error': 'No valid moves available for Player 2 (Agent).'}), 400

        # Make the move
        game.make_move(column)
        logger.debug(f"Move made at column {column}. Current Board: {game.board}")

        # Check for winner or draw
        check_winner = game.check_winner()
        logger.debug(f"Check Winner Result: {check_winner}")
        if check_winner != -1:
            winner = 'Player 1' if check_winner == 0 else 'Player 2'
            # winning_sequence = game.get_winning_sequence()  # Ensure this method exists
            logger.info(f"Game Over: {winner} wins.")
            # logger.debug(f"Winning Sequence: {winning_sequence}")
            return jsonify({
                'board': game.board,
                'winner': winner,
                # 'winningSequence': winning_sequence
            })
        elif game.is_board_full():
            logger.info("Game Over: Draw.")
            return jsonify({
                'board': game.board,
                'winner': 'Draw'
            })

        return jsonify({
            'board': game.board,
            'currentPlayer': game.current_player,
            'checkWinner': check_winner
        })
    except Exception as e:
        logger.exception("An error occurred during make_move.")
        return jsonify({'error': str(e)}), 500

# Serve the HTML frontend
@app.route('/')
def serve_frontend():
    return send_from_directory('.', 'connect4.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
