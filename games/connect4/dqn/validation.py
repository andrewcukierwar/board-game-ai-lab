"""Inference-only, predeclared paired-opening checkpoint validation. No trainer imports."""
import argparse
import contextlib
import io
import json
import platform
import random
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch

from ..agents.dqn_agent import DQNAgent
from ..agents.negamax_agent import NegamaxAgent
from ..agents.random_agent import RandomAgent
from .checkpoint import load_checkpoint
from .diagnostics import summarize_predictions
from .encoding import encode_game, legal_mask
from .network import checked_q
from .validation_holdout import PATH, OLD_PATH, digest, key, replay, validate

OUTPUT = Path('experiment-output/phase4c2d-validation')
CHECKPOINTS = {
    'baseline': ('experiment-output/phase4c2b-seed42-20261003-2059/candidate.pt',
                 '29c1b6c3288811b449b4d90744607e572fca17bc631d5cff4334fc2b4efaf56e'),
    'augmented': ('experiment-output/phase4c2c-seed42-20261003/candidate.pt',
                  '3327863c8d6ed45b70871cd7b336d35950f2c9962a204ebcc67baa86f6c86f08')}
OLD_SHA = 'ec99e0ee080370ab8cfc678c00a49ff552466de73ffe0955778ac87ee340543a'


def configuration():
    rng = random.Random(420205)
    originals, seen = [], set()
    # Ten independently drawn prefixes, each followed by its reflection.
    while len(originals) < 10:
        moves = [rng.randrange(7) for _ in range((2, 3, 4, 5, 6)[len(originals) % 5])]
        try:
            game = replay(moves)
        except ValueError:
            continue
        if key(game) in seen or moves == [6-c for c in moves]:
            continue
        seen.add(key(game))
        originals.append(moves)
    openings = [p for moves in originals for p in (moves, [6-c for c in moves])]
    return dict(version=1, opening_seed=420205, random_seed=420206,
                openings=openings, sides=[0, 1], threads=1, interop_threads=1,
                deterministic_algorithms=True, epsilon=0, learning=False,
                negamax_depths=[1, 2], games_per_model_per_depth=40,
                random_games_per_model=20, head_to_head_games=20,
                aggregate_seconds=600, random_seconds=30, head_to_head_seconds=30,
                schedule='depth, opening, side, model; then Random; then head-to-head',
                negamax_cache='fresh existing agent each game; existing cache retained within game',
                tie_break='DQN lowest legal column; Negamax existing center-first order',
                holdout_sha256=digest(PATH), original_suite_sha256=OLD_SHA,
                checkpoints={k: dict(path=p, sha256=h) for k, (p, h) in CHECKPOINTS.items()})


def predictions(network, fixtures):
    agent = DQNAgent(network)
    rows = []
    with torch.inference_mode():
        for fixture in fixtures:
            game = replay(fixture['moves'])
            state = encode_game(game)
            mask = legal_mask(state)
            q = checked_q(network, torch.tensor([state], dtype=torch.float32))[0].tolist()
            expected = fixture['expected_action']
            legal = [c for c in range(7) if mask[c]]
            action = agent.choose_move(game)
            rows.append(dict(fixture, q=q, legal=list(mask), action=action,
                             legal_q_min=min(q[c] for c in legal), legal_q_max=max(q[c] for c in legal),
                             tactical_margin=None if expected is None else q[expected]-max(q[c] for c in legal if c != expected),
                             tactical_correct=None if expected is None else action == expected))
    return dict(rows=rows, summary=summarize_predictions(rows))


def outcome(winner, side):
    return 'draw' if winner == -1 else 'win' if winner == side else 'loss'


def play(network, opponent, opening, side, seed, deadline, *, clock=time.monotonic):
    start = clock()
    game = replay(opening)
    agent = DQNAgent(network)
    moves = list(opening)
    decisions = []
    # Existing RandomAgent uses module random. Save/restore it for private deterministic games.
    rng_state = random.getstate()
    random.seed(seed)
    try:
        while not game.is_game_over():
            if clock() >= deadline:
                break
            actor = game.current_player
            with contextlib.redirect_stdout(io.StringIO()):
                col = (agent if actor == side else opponent).choose_move(game)
            if clock() >= deadline:
                break
            if not game.is_valid_move(col):
                raise RuntimeError('Agent selected illegal move')
            decisions.append(dict(ply=len(moves), player=actor, action=col,
                                  agent='model' if actor == side else 'opponent'))
            game.make_move(col)
            moves.append(col)
    finally:
        random.setstate(rng_state)
    complete = game.is_game_over()
    winner = game.check_winner() if complete else None
    return dict(opening=list(opening), side=side, seed=seed, moves=moves, decisions=decisions,
                complete=complete, winner=winner, result=outcome(winner, side) if complete else None,
                elapsed_seconds=clock()-start)


def summarize_games(games):
    completed = [g for g in games if g['complete']]
    counts = Counter(g['result'] for g in completed)
    return dict(completed=len(completed), partial=len(games)-len(completed),
                wins=counts['win'], losses=counts['loss'], draws=counts['draw'],
                by_side={str(s): dict(Counter(g['result'] for g in completed if g['side'] == s)) for s in (0, 1)},
                starting_sides=[sum(g['side'] == s for g in completed) for s in (0, 1)],
                elapsed_seconds=sum(g['elapsed_seconds'] for g in games))


def compare(before, after):
    groups = {k: [] for k in ('previously_correct', 'retained_correct', 'improved', 'regressed', 'neither_correct')}
    for a, b in zip(before['rows'], after['rows']):
        if a['name'] != b['name']:
            raise ValueError('Position order mismatch')
        if a['expected_action'] is None:
            continue
        row = dict(name=a['name'], category=a['category'], direction=a['direction'],
                   player=a['player'], expected_action=a['expected_action'],
                   baseline_action=a['action'], augmented_action=b['action'])
        if a['tactical_correct']:
            groups['previously_correct'].append(row)
        group = ('retained_correct' if b['tactical_correct'] else 'regressed') if a['tactical_correct'] else ('improved' if b['tactical_correct'] else 'neither_correct')
        groups[group].append(row)
    return groups


def freeze(output):
    validate(json.loads(PATH.read_text())['positions'])
    if digest(OLD_PATH) != OLD_SHA:
        raise ValueError('Original suite changed')
    for path, expected in CHECKPOINTS.values():
        if digest(path) != expected:
            raise ValueError('Checkpoint changed')
    output.mkdir(parents=True, exist_ok=False)
    config = configuration()
    (output/'configuration.json').write_text(json.dumps(config, indent=2)+'\n')
    (output/'freeze.json').write_text(json.dumps(dict(
        frozen_utc=datetime.now(timezone.utc).isoformat(), holdout_sha256=digest(PATH),
        configuration_sha256=digest(output/'configuration.json'),
        predictions_inspected=False), indent=2)+'\n')
    return config


def run(output):
    config = json.loads((output/'configuration.json').read_text())
    frozen = json.loads((output/'freeze.json').read_text())
    if config != configuration() or frozen['configuration_sha256'] != digest(output/'configuration.json'):
        raise ValueError('Frozen configuration changed')
    if (output/'report.json').exists():
        raise ValueError('Refuse repeated evaluation output')
    validate(json.loads(PATH.read_text())['positions'])
    for path, expected in CHECKPOINTS.values():
        if digest(path) != expected:
            raise ValueError('Checkpoint changed')
    if digest(OLD_PATH) != OLD_SHA:
        raise ValueError('Original suite changed')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    start = time.monotonic()
    deadline = start + config['aggregate_seconds']
    nets = {name: load_checkpoint(path)[0] for name, (path, _) in CHECKPOINTS.items()}
    weights = {name: {k: t.clone() for k, t in net.state_dict().items()} for name, net in nets.items()}
    report = dict(configuration=config, started_utc=datetime.now(timezone.utc).isoformat(),
                  runtime=dict(python=platform.python_version(), torch=torch.__version__, platform=platform.platform()),
                  diagnostics={}, comparisons={}, games=[], matches={})
    for suite, path in [('existing', OLD_PATH), ('holdout', PATH)]:
        fixtures = json.loads(path.read_text())['positions']
        report['diagnostics'][suite] = {n: predictions(net, fixtures) for n, net in nets.items()}
        report['comparisons'][suite] = compare(**dict(zip(('before', 'after'), report['diagnostics'][suite].values())))
    with (output/'games.jsonl').open('x') as log:
        def record(model, opponent_name, opening_index, side, opponent, limit):
            game = play(nets[model], opponent, config['openings'][opening_index], side,
                        config['random_seed']+opening_index*2+side, limit)
            game.update(model=model, opponent=opponent_name, opening_index=opening_index)
            report['games'].append(game)
            log.write(json.dumps(game)+'\n')
            log.flush()
        for depth in config['negamax_depths']:
            for i in range(20):
                for side in (0, 1):
                    for model in nets:
                        if time.monotonic() < deadline:
                            record(model, f'negamax_{depth}', i, side, NegamaxAgent(depth), deadline)
        random_deadline = min(deadline, time.monotonic()+config['random_seconds'])
        for i in range(10):
            for side in (0, 1):
                for model in nets:
                    if time.monotonic() < random_deadline:
                        record(model, 'random', i, side, RandomAgent(), random_deadline)
        h2h_deadline = min(deadline, time.monotonic()+config['head_to_head_seconds'])
        for i in range(10):
            for side in (0, 1):
                if time.monotonic() < h2h_deadline:
                    record('baseline', 'augmented', i, side, DQNAgent(nets['augmented']), h2h_deadline)
    for model, opponent in sorted({(g['model'], g['opponent']) for g in report['games']}):
        report['matches'][model+'/'+opponent] = summarize_games([g for g in report['games'] if (g['model'], g['opponent']) == (model, opponent)])
    report['weights_unchanged'] = all(torch.equal(t, weights[n][k]) for n, net in nets.items() for k, t in net.state_dict().items())
    report['files_unchanged'] = all(digest(p) == h for p, h in CHECKPOINTS.values()) and digest(PATH) == config['holdout_sha256'] and digest(OLD_PATH) == OLD_SHA
    if not report['weights_unchanged'] or not report['files_unchanged']:
        raise RuntimeError('Evaluation mutated preserved data')
    report['elapsed_seconds'] = time.monotonic()-start
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(matches=report['matches'], elapsed_seconds=report['elapsed_seconds']), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['freeze', 'evaluate'])
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    (freeze if args.operation == 'freeze' else run)(args.output)
