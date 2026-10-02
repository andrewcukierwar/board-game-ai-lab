"""Connect 4 API: each mutation is locked and guarded by a board revision."""
from copy import deepcopy

from flask import Blueprint, current_app, jsonify, request

from games.connect4.agents.negamax_agent import NegamaxAgent
from games.connect4.agents.random_agent import RandomAgent
from games.connect4.connect4 import Connect4
from games.connect4.grounding.history import record_move
from .state import GameError

bp = Blueprint('connect4', __name__)
MAX_DEPTH = 4


def store():
    return current_app.extensions['connect4_games']


def json_object():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        raise GameError('invalid_request', 'Send a JSON object with Content-Type: application/json.')
    return data


def game_id(value):
    if not isinstance(value, str) or not value or len(value) > 64:
        raise GameError('invalid_game_id', 'Provide the game_id returned when starting a game.')
    return value


def player_config(value):
    if not isinstance(value, dict):
        raise GameError('invalid_agent', 'Each player configuration must be an object.')
    kind = value.get('type')
    if kind not in ('human', 'random', 'negamax'):
        raise GameError('invalid_agent', 'Supported player types are human, random, and negamax.')
    allowed = {'type', 'depth'} if kind == 'negamax' else {'type'}
    if set(value) - allowed:
        raise GameError('invalid_agent', 'Only Negamax accepts a depth; no other agent settings are supported.')
    config = {'type': kind}
    if kind == 'negamax':
        depth = value.get('depth', 2)
        if type(depth) is not int or not 1 <= depth <= MAX_DEPTH:
            raise GameError('invalid_agent', f'Negamax depth must be an integer from 1 to {MAX_DEPTH}.')
        config['depth'] = depth
    return config


def snapshot(gid, session):
    game = session.game
    winner = game.check_winner()
    terminal = game.is_game_over()
    return dict(game_id=gid, revision=session.revision, board=deepcopy(game.board),
                currentPlayer=game.current_player, players=deepcopy(session.players),
                gameOver=terminal, legalMoves=[] if terminal else game.get_valid_moves(),
                winner=(f'Player {winner + 1}' if winner != -1 else 'Draw' if terminal else None))


@bp.after_request
def no_cache(response):
    response.headers['Cache-Control'] = 'no-store'
    return response


@bp.errorhandler(GameError)
def client_error(error):
    return jsonify(error=error.message, code=error.code), error.status


@bp.route('/start_game', methods=['POST'])
def start_game():
    data = json_object()
    if set(data) - {'player1', 'player2', 'replace_game_id'}:
        raise GameError('invalid_request', 'Unknown start-game field.')
    players = [player_config(data.get('player1', {'type': 'human'})),
               player_config(data.get('player2', {'type': 'negamax', 'depth': 2}))]
    replace_id = game_id(data['replace_game_id']) if 'replace_game_id' in data else None
    gid, session = store().create(players, replace_id)
    return jsonify(snapshot(gid, session)), 201


@bp.route('/games/<gid>', methods=['GET'])
def get_game(gid):
    with store().access(game_id(gid)) as session:
        return jsonify(snapshot(gid, session))


@bp.route('/make_move', methods=['POST'])
def make_move():
    data = json_object()
    if set(data) - {'game_id', 'revision', 'column'}:
        raise GameError('invalid_request', 'Unknown move field.')
    gid = game_id(data.get('game_id'))
    revision = data.get('revision')
    if type(revision) is not int or revision < 0:
        raise GameError('invalid_revision', 'Provide the non-negative integer revision from the latest game response.')
    if 'column' in data and (type(data['column']) is not int or not 0 <= data['column'] <= 6):
        raise GameError('invalid_move', 'Column must be an integer from 0 to 6.')
    with store().access(gid) as session:
        if revision != session.revision:
            raise GameError('stale_revision', 'The board changed. Refresh the game before moving again.', 409)
        if session.game.is_game_over():
            raise GameError('game_over', 'This game is finished. Start a new game.', 409)
        config = session.players[session.game.current_player]
        candidate = Connect4(session.game.board, session.game.current_player)
        if config['type'] == 'human':
            if 'column' not in data:
                raise GameError('invalid_move', 'Choose a column from 0 to 6 for the human player.')
            column = data['column']
            if not candidate.is_valid_move(column):
                raise GameError('invalid_move', 'That column is full. Choose another column.')
        else:
            if 'column' in data:
                raise GameError('invalid_move', 'It is the AI turn. Request its move without a column.')
            # A fresh bounded search avoids retaining caches across requests.
            agent = RandomAgent() if config['type'] == 'random' else NegamaxAgent(config['depth'])
            try:
                # Never expose the live game to agent code.
                column = agent.choose_move(Connect4(candidate.board, candidate.current_player))
                if type(column) is not int or not candidate.is_valid_move(column):
                    raise ValueError('Agent returned an illegal move')
            except Exception:
                current_app.logger.exception('Connect 4 agent failed')
                raise GameError('agent_failed', 'The AI could not make a legal move. Retry or start a new game.', 503)
        if not candidate.make_move(column):
            raise GameError('invalid_move', 'That move could not be played. Refresh the game.', 409)
        # Prepare the immutable evidence before committing; failed requests add nothing.
        record = record_move(session.game.board, candidate.board,
                             session.game.current_player, config, column, session.revision)
        history = session.history + (record,)
        session.game, session.revision, session.history = candidate, record.revision, history
        return jsonify(snapshot(gid, session))


@bp.route('/health')
def health():
    return 'OK', 200
