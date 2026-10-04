"""Fixed, legally replayed probes and bounded greedy-vs-uniform-Random games."""

import random
import time

import torch

from ..connect4 import Connect4
from ..agents.dqn_agent import DQNAgent
from .encoding import encode_game, legal_mask
from .network import checked_q

# Frozen sequences, never generated from a model or consumed by training.
import hashlib
import json
import math
from collections import Counter
from copy import deepcopy
from pathlib import Path

FIXTURE_PATH = Path(__file__).with_name('diagnostic_positions.json')
FIXTURES = tuple(json.loads(FIXTURE_PATH.read_text())['positions'])
POSITIONS = tuple((f['name'], tuple(f['moves']), f['expected_action']) for f in FIXTURES)


def suite_identity():
    return dict(version=1, sha256=hashlib.sha256(FIXTURE_PATH.read_bytes()).hexdigest(),
                positions=len(FIXTURES))


def winning_moves(game):
    wins = []
    for move in game.get_valid_moves():
        child = deepcopy(game)
        child.make_move(move)
        if child.check_winner() == game.current_player:
            wins.append(move)
    return wins


def winning_directions(game, move):
    child = deepcopy(game)
    piece = child.piece
    child.make_move(move)
    found = set()
    for row in range(6):
        for col in range(7):
            for dr, dc, name in ((0, 1, 'horizontal'), (1, 0, 'vertical'),
                                 (1, 1, 'diagonal'), (1, -1, 'diagonal')):
                if (0 <= row + 3 * dr < 6 and 0 <= col + 3 * dc < 7
                        and all(child.board[row + i * dr][col + i * dc] == piece for i in range(4))):
                    found.add(name)
    return found


def validate_fixtures():
    """Engine-derived labels: block means safe against the next reply only."""
    by_name = {f['name']: f for f in FIXTURES}
    if len(by_name) != len(FIXTURES):
        raise ValueError('Duplicate fixture names')
    for f in FIXTURES:
        game = position(f['moves'])
        expected = f['expected_action']
        if game.current_player != f['player']:
            raise ValueError('Fixture perspective mismatch')
        threat = game
        if f['category'] == 'win':
            if winning_moves(game) != [expected]:
                raise ValueError('Win fixture must have exactly one winning move')
        elif f['category'] == 'block':
            if winning_moves(game):
                raise ValueError('Block fixture has an own immediate win')
            safe = []
            for move in game.get_valid_moves():
                child = deepcopy(game)
                child.make_move(move)
                if child.is_game_over() or not winning_moves(child):
                    safe.append(move)
            if safe != [expected]:
                raise ValueError('Block fixture must have exactly one safe move')
            threat = deepcopy(game)
            threat.current_player = 1 - game.current_player
            threat.piece = 'XO'[threat.current_player]
            if winning_moves(threat) != [expected]:
                raise ValueError('Block fixture must have an immediate opposing threat')
        if expected is not None and winning_directions(threat, expected) != {f['direction']}:
            raise ValueError('Threat orientation mismatch')
        if f['mirror_of'] is not None:
            original = by_name[f['mirror_of']]
            mirrored_expected = original['expected_action']
            if (f['moves'] != [6 - c for c in original['moves']]
                    or expected != (None if mirrored_expected is None else 6 - mirrored_expected)):
                raise ValueError('Incorrect physical-column mirror')
    return suite_identity()


def position(moves):
    game = Connect4()
    for move in moves:
        if game.is_game_over() or not game.is_valid_move(move):
            raise ValueError('Illegal diagnostic sequence')
        game.make_move(move)
    if game.is_game_over():
        raise ValueError('Diagnostic position must be nonterminal')
    return game


def predictions(network):
    network.eval()
    rows = []
    with torch.inference_mode():
        for fixture in FIXTURES:
            name, moves, expected = (fixture[k] for k in ('name', 'moves', 'expected_action'))
            game = position(moves)
            state = encode_game(game)
            mask = legal_mask(state)
            q = checked_q(network, torch.tensor([state], dtype=torch.float32))[0]
            action = int(q.masked_fill(~torch.tensor(mask), -torch.inf).argmax().item())
            legal_q = [float(q[c]) for c in range(7) if mask[c]]
            margin = (None if expected is None else float(q[expected]) -
                      max(float(q[c]) for c in range(7) if mask[c] and c != expected))
            rows.append(dict(category=fixture['category'], direction=fixture['direction'],
                             mirror_of=fixture['mirror_of'], legal_q_min=min(legal_q),
                             legal_q_max=max(legal_q), tactical_margin=margin, name=name, moves=list(moves), player=game.current_player,
                             legal=list(mask), q=q.tolist(), action=action,
                             expected_action=expected,
                             tactical_correct=None if expected is None else action == expected))
    return rows


def action_distribution(actions):
    counts = Counter(actions)
    total = sum(counts.values())
    return dict(counts=[counts[c] for c in range(7)], total=total,
                dominant_column=max(range(7), key=lambda c: counts[c]) if total else None,
                dominant_fraction=max(counts.values()) / total if total else None,
                entropy_bits=-sum((n / total) * math.log2(n / total) for n in counts.values()))


def summarize_predictions(rows):
    tactical = [r for r in rows if r['expected_action'] is not None]
    def accuracy(items):
        margins = [r['tactical_margin'] for r in items]
        return dict(correct=sum(r['tactical_correct'] for r in items), total=len(items),
                    margin_mean=sum(margins) / len(margins) if margins else None,
                    margin_min=min(margins) if margins else None,
                    margin_max=max(margins) if margins else None)
    groups = {}
    for field in ('category', 'direction', 'player', 'expected_action'):
        groups[field] = {str(key): accuracy([r for r in tactical if r[field] == key])
                         for key in sorted({r[field] for r in tactical})}
    by_name = {r['name']: r for r in rows}
    mirrors = []
    for row in rows:
        if row['mirror_of']:
            original = by_name[row['mirror_of']]
            differences = [abs(original['q'][c] - row['q'][6-c])
                           for c in range(7) if original['legal'][c]]
            mirrors.append(dict(original=original['name'], mirror=row['name'],
                                action_equivariant=row['action'] == 6-original['action'],
                                legal_q_mae=sum(differences)/len(differences),
                                legal_q_max_difference=max(differences)))
    return dict(tactical=accuracy(tactical), by=groups,
                greedy_actions=action_distribution(r['action'] for r in rows),
                legal_q_min=min(r['legal_q_min'] for r in rows),
                legal_q_max=max(r['legal_q_max'] for r in rows), mirrors=mirrors,
                mirror_action_matches=sum(m['action_equivariant'] for m in mirrors),
                mirror_pairs=len(mirrors))


def random_protocol(games, seed):
    if type(games) is not int or not 0 <= games <= 100 or games % 2:
        raise ValueError('Each evaluation must request an even count in 0..100')
    if type(seed) is not int or seed < 0:
        raise ValueError('Evaluation seed must be a nonnegative integer')
    return [dict(index=i, seed=seed+i, side=i % 2) for i in range(games)]


def evaluate_random(network, *, games, seed, seconds, clock=time.monotonic,
                    should_stop=lambda: False):
    """No replay/learning. Alternate DQN sides, restart the same seeded suite per model.

    Random uses the existing RandomAgent's uniform legal policy with a private
    RNG, avoiding any mutation of training/global RNG state. Partial games are
    explicitly excluded from wins/losses/draws and side counts.
    """
    protocol = random_protocol(games, seed)
    if not math.isfinite(seconds) or not 0 < seconds <= 60:
        raise ValueError('Each evaluation is limited to 60 seconds')
    start = clock()
    agent = DQNAgent(network)
    results = []
    partial = None
    actions = []
    side_actions = [[], []]
    for index in range(games):
        rng = random.Random(seed + index)
        side = index % 2
        game = Connect4()
        moves = []
        while not game.is_game_over():
            if should_stop() or clock() - start >= seconds:
                partial = dict(index=index, side=side, moves=moves)
                break
            move = (agent.choose_move(game) if game.current_player == side
                    else rng.choice(game.get_valid_moves()))
            if should_stop() or clock() - start >= seconds:
                partial = dict(index=index, side=side, moves=moves)
                break
            if game.current_player == side:
                actions.append(move)
                side_actions[side].append(move)
            game.make_move(move)
            moves.append(move)
        if partial is not None:
            break
        winner = game.check_winner()
        results.append(dict(index=index, seed=seed + index, side=side, moves=moves,
                            result='draw' if winner == -1 else 'win' if winner == side else 'loss'))
    by_side = {str(side): dict(
        **{{'win': 'wins', 'loss': 'losses', 'draw': 'draws'}[result]: sum(r['side'] == side and r['result'] == result for r in results)
           for result in ('win', 'loss', 'draw')},
        greedy_actions=action_distribution(side_actions[side])) for side in (0, 1)}
    # Action counts include any explicitly recorded partial game; outcomes do not.
    return dict(protocol=protocol, by_side=by_side, greedy_actions=action_distribution(actions),
                requested_games=games, completed_games=len(results), games=results,
                partial=partial, elapsed_seconds=clock() - start,
                wins=sum(r['result'] == 'win' for r in results),
                draws=sum(r['result'] == 'draw' for r in results),
                losses=sum(r['result'] == 'loss' for r in results),
                starting_sides=[sum(r['side'] == side for r in results) for side in (0, 1)])
