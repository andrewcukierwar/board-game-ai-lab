"""Connect 4 API: each mutation is locked and guarded by a board revision."""
from copy import deepcopy
import random
from threading import BoundedSemaphore

from flask import Blueprint, current_app, jsonify, request

from games.connect4.agents.negamax_agent import NegamaxAgent
from games.connect4.agents.random_agent import RandomAgent
from games.connect4.agents.mcts_agent import MCTSAgent
from games.connect4.connect4 import Connect4
from games.connect4.grounding.history import record_move
from .state import GameError
from .provenance import public_provenance

bp = Blueprint('connect4', __name__)
MAX_DEPTH = 8
MCTS_SIMULATION_LIMITS = (50, 100, 250, 400, 800)
# Shared by all app instances in this process; never wait/queue for a search.
_mcts_reservation = BoundedSemaphore(1)
# The opt-in research agent gets its own reservation, so it never blocks MCTS.
_victor_reservation = BoundedSemaphore(1)
VICTOR_RESEARCH = 'victor_research'


def ply_seed(seed, revision, player):
    """32-bit avalanche of seed XOR ply/player salts (no runtime hash/global RNG)."""
    value = (seed ^ ((revision + 1) * 0x9E3779B9) ^ ((player + 1) * 0x85EBCA6B)) & 0xFFFFFFFF
    value = ((value ^ (value >> 16)) * 0x7FEB352D) & 0xFFFFFFFF
    value = ((value ^ (value >> 15)) * 0x846CA68B) & 0xFFFFFFFF
    return (value ^ (value >> 16)) & 0xFFFFFFFF


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
    research = kind == VICTOR_RESEARCH and current_app.config.get('VICTOR_RESEARCH_ENABLED') is True
    if kind not in ('human', 'random', 'negamax', 'mcts') and not research:
        raise GameError('invalid_agent', 'Supported player types are human, random, negamax, and mcts.')
    allowed = {'type'}
    if kind == 'negamax':
        allowed.add('depth')
    elif kind == 'mcts':
        allowed.add('simulation_limit')
    if set(value) - allowed:
        raise GameError('invalid_agent', 'Only Negamax accepts depth; only MCTS accepts simulation_limit. No other agent settings are supported.')
    config = {'type': kind}
    if kind == 'negamax':
        depth = value.get('depth', 2)
        if type(depth) is not int or not 1 <= depth <= MAX_DEPTH:
            raise GameError('invalid_agent', f'Negamax depth must be an integer from 1 to {MAX_DEPTH}.')
        config['depth'] = depth
    elif kind == 'mcts':
        limit = value.get('simulation_limit', 100)
        if type(limit) is not int or limit not in MCTS_SIMULATION_LIMITS:
            raise GameError('invalid_agent', f'MCTS simulation_limit must be an integer in {MCTS_SIMULATION_LIMITS}.')
        config['simulation_limit'] = limit
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
    if set(data) - {'player1', 'player2', 'replace_game_id', 'rng_seed'}:
        raise GameError('invalid_request', 'Unknown start-game field.')
    seed = data.get('rng_seed')
    if 'rng_seed' in data and (type(seed) is not int or not 0 <= seed <= 0xFFFFFFFF):
        raise GameError('invalid_rng_seed', 'rng_seed must be an integer from 0 to 4294967295.')
    players = [player_config(data.get('player1', {'type': 'human'})),
               player_config(data.get('player2', {'type': 'negamax', 'depth': 2}))]
    replace_id = game_id(data['replace_game_id']) if 'replace_game_id' in data else None
    gid, session = store().create(players, replace_id, seed)
    return jsonify(snapshot(gid, session)), 201


@bp.route('/games/<gid>', methods=['GET'])
def get_game(gid):
    with store().access(game_id(gid)) as session:
        return jsonify(snapshot(gid, session))


@bp.route('/games/<gid>/history', methods=['GET'])
def get_history(gid):
    # Capture game and immutable records under the same existing session lock.
    # Only safe game evidence is serialized, never an agent/search object.
    with store().access(game_id(gid)) as session:
        state = snapshot(gid, session)
        return jsonify(game_id=gid, revision=session.revision,
                       provenance=public_provenance(current_app.config.get('SOURCE_COMMIT')),
                       **({'rng_seed': session.rng_seed} if session.rng_seed is not None else {}),
                       players=state['players'], state=state,
                       moves=[record.to_dict() for record in session.history])


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
            if config['type'] == VICTOR_RESEARCH and current_app.config.get('VICTOR_RESEARCH_ENABLED') is not True:
                raise GameError('invalid_agent', 'The Victor research agent is not enabled on this server.', 409)
            reservation = {'mcts': _mcts_reservation, VICTOR_RESEARCH: _victor_reservation}.get(config['type'])
            reserved = reservation is not None
            if reserved and not reservation.acquire(blocking=False):
                name = 'MCTS' if config['type'] == 'mcts' else 'Victor research'
                raise GameError('agent_busy', f'Another {name} search is running. Retry the AI move shortly.', 503)
            try:
                rng = (random.Random(ply_seed(session.rng_seed, session.revision, candidate.current_player))
                       if session.rng_seed is not None else None)
                # Explicit production-safe imports; fresh search state per request.
                if config['type'] == 'mcts':
                    agent = (MCTSAgent(config['simulation_limit'], rng=rng) if rng is not None
                             else MCTSAgent(config['simulation_limit']))
                elif config['type'] == 'random':
                    agent = RandomAgent(rng=rng) if rng is not None else RandomAgent()
                elif config['type'] == VICTOR_RESEARCH:
                    # Imported only when enabled and used. Stateless, deterministic
                    # under node budgets, deadline-bounded; always a legal move.
                    from games.connect4.agents.victor_research_agent import VictorResearchAgent
                    agent = VictorResearchAgent()
                else:
                    agent = NegamaxAgent(config['depth'])
                # Never expose the live game to agent code.
                column = agent.choose_move(Connect4(candidate.board, candidate.current_player))
                if type(column) is not int or not candidate.is_valid_move(column):
                    raise ValueError('Agent returned an illegal move')
            except Exception:
                current_app.logger.exception('Connect 4 agent failed')
                raise GameError('agent_failed', 'The AI could not make a legal move. Retry or start a new game.', 503)
            finally:
                if reserved:
                    reservation.release()
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


@bp.route('/provenance', methods=['GET'])
def provenance():
    return jsonify(public_provenance(current_app.config.get('SOURCE_COMMIT')))


@bp.route('/explain', methods=['POST'])
def explain():
    from .explanations import MODES
    data = json_object()
    if set(data) - {'game_id', 'revision', 'mode', 'column', 'question'}:
        raise GameError('invalid_request', 'Unknown explanation field.')
    gid = game_id(data.get('game_id'))
    revision = data.get('revision')
    if type(revision) is not int or revision < 0:
        raise GameError('invalid_revision', 'Provide the non-negative integer revision from the latest game response.')
    mode = data.get('mode')
    if mode not in MODES:
        raise GameError('invalid_mode', 'Choose last_move, position, or what_if.')
    column = data.get('column')
    if mode == 'what_if':
        if type(column) is not int or not 0 <= column <= 6:
            raise GameError('invalid_move', 'Hypothetical column must be an integer from 0 to 6.')
    elif 'column' in data:
        raise GameError('invalid_request', 'Only what_if accepts a column.')
    question = data.get('question', '')
    if not isinstance(question, str) or len(question) > current_app.config['EXPLANATION_MAX_QUESTION_LENGTH']:
        raise GameError('invalid_question', 'The question is invalid or too long.')
    result = current_app.extensions['connect4_explanations'].explain(
        store(), gid, revision, mode, column, question.strip(), request.remote_addr or 'unknown')
    return jsonify(result)
