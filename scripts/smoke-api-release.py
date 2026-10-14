#!/usr/bin/env python3
"""Bounded, provider-free release probe. Invoked by staging/approved deployment."""
import argparse
import json
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


def smoke(base, mode, commit):
    if urlparse(base).hostname not in ('localhost', '127.0.0.1', '::1'):
        raise ValueError('Release smoke must use localhost')

    def call(path, body=None):
        request = Request(base + '/v1/connect4/' + path,
                          data=None if body is None else json.dumps(body).encode(),
                          headers={'Content-Type': 'application/json'})
        with urlopen(request, timeout=30) as response:
            payload = response.read()
            return payload.decode().strip() if path == 'health' else json.loads(payload)

    assert call('health') == 'OK'
    p = call('provenance')
    assert p == dict(source_commit=commit, api_provenance_version=1,
                     api_contract_version=3, connect4_engine_version=1, stochastic_seed_version=1,
                     agents={'random': 2, 'negamax': 2, 'mcts': 2, 'mcts_tactical': 1})
    status = call('agents/alphazero')
    assert status['enabled'] is (mode == 'alphazero') and status['available'] is (mode == 'alphazero')
    configs = [{'type': 'random'}, {'type': 'negamax', 'depth': 2},
               {'type': 'mcts', 'simulation_limit': 100}, {'type': 'mcts_tactical', 'simulation_limit': 100},
               {'type': 'victor_research'}]
    if mode == 'alphazero':
        evidence = status['provenance']
        assert evidence['version'] == 1 and evidence['model']['id'] == 'az2-hc-g6-cnn-s42-v1'
        assert evidence['search']['simulations'] == 256 and evidence['rng_seed'] == 'ignored'
        assert evidence['model']['artifact_sha256'] == '4c3cf26aa58495f43094436f231486969658d70b94b246a912a5bb54756dc528'
        configs += [{'type': 'alphazero', 'tactical_guard': guard} for guard in (False, True)]
    retained = None
    for config in configs:
        body = dict(player1=config, player2={'type': 'human'}, **({'replace_game_id': retained} if retained else {}))
        try:
            game = call('start_game', body)
        except HTTPError as error:
            if config['type'] == 'victor_research' and error.code == 400 and json.load(error)['code'] == 'invalid_agent':
                continue  # Optional Victor remains operator-controlled.
            raise
        retained = game['game_id']
        assert game['revision'] == 0 and game['players'][0] == config
        moved = call('make_move', dict(game_id=retained, revision=0))
        assert moved['revision'] == 1 and moved['players'] == game['players']
        history = call(f'games/{retained}/history')
        assert history['revision'] == 1 and len(history['moves']) == 1
        if config['type'] == 'alphazero':
            assert history['alphazero_provenance'] == evidence
            move = history['moves'][0]['alphazero']
            assert move['model_id'] == evidence['model']['id'] and move['tactical_guard'] is config['tactical_guard']
            assert sum(move['root_visits']) == 256
    print(f'PASS: {mode} health, provenance, availability, bounded existing-agent moves' + (' and AlphaZero moves' if mode == 'alphazero' else ''))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True)
    parser.add_argument('--mode', choices=('lite', 'alphazero'), required=True)
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    smoke(args.base.rstrip('/'), args.mode, args.commit)
