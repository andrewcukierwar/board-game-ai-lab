"""Conventional one-ply filtering; original PUCT is never changed."""
from numbers import Integral
from time import perf_counter_ns
from games.connect4.alphazero_v2.bitboard import BitboardState

CONTRACT = 'connect4-tactical-root-guard-v1'


def guard(root, visits, original):
    """Override only for immediate wins or an unsafe original with a safe alternative.

    Override ties: greatest original visit count, then lowest physical column.
    The original seeded maximum-visit choice is preserved exactly when safe.
    A terminal successor has no opponent replies (including a final-cell draw).
    """
    start = perf_counter_ns()
    if not isinstance(root, BitboardState):
        raise TypeError('Guard requires the actual immutable bitboard root')
    if root.is_game_over():
        raise ValueError('Cannot guard a terminal position')
    legal = sorted(root.get_valid_moves())
    if len(visits) != 7 or any(isinstance(n, bool) or not isinstance(n, Integral) or n < 0 for n in visits):
        raise ValueError('Expected seven original nonnegative integer visits')
    if any(visits[a] for a in range(7) if a not in legal):
        raise ValueError('Illegal action has visits')
    if isinstance(original, bool) or not isinstance(original, Integral) or original not in legal:
        raise ValueError('Original action must be legal')
    winning, safe, unsafe, replies = [], [], [], {}
    inspected = 0
    classification_start = perf_counter_ns()
    for action in legal:
        after = root.drop(action)
        if after.winner == root.current_player:
            winning.append(action)
        losses = []
        if not after.is_game_over():
            for reply in sorted(after.get_valid_moves()):
                inspected += 1
                if after.drop(reply).winner == after.current_player:
                    losses.append(reply)
        if losses:
            unsafe.append(action)
            replies[str(action)] = losses
        else:
            safe.append(action)
    classification_seconds = (perf_counter_ns()-classification_start)/1e9
    choose = lambda candidates: min(candidates, key=lambda a: (-visits[a], a))
    if winning:
        final, reason = choose(winning), 'IMMEDIATE_WIN'
    elif not safe:
        final, reason = int(original), 'NO_SAFE_MOVE'
    elif original not in safe:
        final, reason = choose(safe), 'AVOID_IMMEDIATE_LOSS'
    else:
        final, reason = int(original), 'UNCHANGED'
    return dict(contract=CONTRACT, original_action=int(original), final_action=final,
                changed=final != original, legal_actions=legal, immediate_wins=winning,
                safe_actions=safe, unsafe_actions=unsafe, opponent_winning_replies=replies,
                original_visits=[int(n) for n in visits], selected_visits=int(visits[final]),
                selected_zero_visits=visits[final] == 0, reason=reason, no_safe_action=not safe,
                candidate_count=len(legal), opponent_replies_inspected=inspected,
                occupancy=root.occupied.bit_count(), classification_seconds=classification_seconds, guard_seconds=(perf_counter_ns()-start)/1e9)
