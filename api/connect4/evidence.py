"""Internal service boundary for future explanation requests (no HTTP route)."""
from copy import deepcopy

from games.connect4.grounding import build_explanation_context
from .state import GameError


def get_explanation_context(store, game_id, expected_revision, *, concept_ids=()):
    """Capture under the session lock, analyze a detached revision afterward.

Uses the same busy/expiry behavior as gameplay. The returned revision remains
its identity even if a new move occurs while analysis runs.
"""
    if type(expected_revision) is not int or expected_revision < 0:
        raise GameError('invalid_revision', 'Provide a non-negative integer revision.')
    with store.access(game_id) as session:
        if session.revision != expected_revision:
            raise GameError('stale_revision', 'The board changed. Refresh before requesting evidence.', 409)
        captured = {'board': [list(row) for row in session.game.board],
                    'player_to_move': session.game.current_player,
                    'revision': session.revision, 'move_history': session.history,
                    'players': deepcopy(session.players), 'game_id': game_id}
    return build_explanation_context(**captured, concept_ids=concept_ids)
