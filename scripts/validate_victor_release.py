"""Small, local-only release check; consumes frozen labels, never runs an oracle.

Run in the Linux runtime image with this script and a selected JSON fixture
mounted read-only. No compiler, pytest, research package or provider is needed.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import math
import platform
from random import Random
import resource
import shutil
import statistics
import subprocess
import sys
from threading import Barrier, Event
from time import perf_counter
from unittest.mock import patch
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from urllib.error import HTTPError

from api.app import create_app
from games.connect4.connect4 import Connect4
from games.connect4.agents.victor_research_agent import VictorResearchAgent, PUBLIC_BUDGET
from games.connect4.victor import Position
from games.connect4.victor import native
from games.connect4.victor.opening_book import default_book

BASE = '/v1/connect4'


def game(history):
    result = Connect4()
    for column in history:
        assert not result.is_game_over() and result.make_move(column)
    return result


def summary(values):
    ordered = sorted(values)
    return dict(n=len(values), median_ms=1000 * statistics.median(values),
                p95_ms=1000 * ordered[math.ceil(len(values) * .95) - 1],
                maximum_ms=1000 * max(values))


def controlled_failures():
    """Deterministic overlap/failure checks against the packaged Flask app."""
    app = create_app({'TESTING': True, 'EXPLANATIONS_ENABLED': False,
                      'VICTOR_RESEARCH_ENABLED': True})
    def start(agent='victor_research'):
        config = {'type': agent}
        if agent == 'mcts':
            config['simulation_limit'] = 50
        response = app.test_client().post(BASE + '/start_game', json={
            'player1': config, 'player2': {'type': 'human'}})
        assert response.status_code == 201
        return response.json
    def move(state):
        return app.test_client().post(BASE + '/make_move', json={
            'game_id': state['game_id'], 'revision': state['revision']})
    def snapshot(state):
        return app.test_client().get(BASE + '/games/' + state['game_id']).json
    first, second, mcts = start(), start(), start('mcts')
    entered, release = Event(), Event()
    original = native.prove_moves
    # Block inside the native step, while keeping the API reservation occupied.
    from games.connect4.victor import solver
    def slow(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)
    budget = replace(PUBLIC_BUDGET, opening_book=False)
    original_init = VictorResearchAgent.__init__
    try:
        with patch.object(VictorResearchAgent, '__init__',
                          lambda self: original_init(self, budget)), \
             patch.object(solver, 'prove_moves', slow), ThreadPoolExecutor(2) as pool:
            future = pool.submit(move, first)
            assert entered.wait(5)
            started = perf_counter()
            busy = move(second)
            assert busy.status_code == 503 and busy.json['code'] == 'agent_busy'
            assert snapshot(second) == second
            assert app.test_client().get(BASE + '/health').status_code == 200
            assert move(mcts).status_code == 200
            overlap_ms = (perf_counter() - started) * 1000
            release.set()
            assert future.result(5).status_code == 200
    finally:
        release.set()
    assert move(second).status_code == 200  # permit released, explicit retry works
    # A lost success response must be reconciled, never replayed as a new ply.
    assert snapshot(first)['revision'] == 1
    stale = move(first)
    assert stale.status_code == 409 and stale.json['code'] == 'stale_revision'
    assert snapshot(first)['revision'] == 1
    failed = start()
    with patch.object(VictorResearchAgent, 'choose_move', side_effect=RuntimeError('injected')):
        response = move(failed)
    assert response.status_code == 503 and response.json['code'] == 'agent_failed'
    assert snapshot(failed) == failed
    assert move(failed).status_code == 200
    disabled = start()
    app.config['VICTOR_RESEARCH_ENABLED'] = False
    assert move(disabled).status_code == 409
    assert snapshot(disabled) == disabled
    return dict(busy_retry=True, separate_mcts=True, health=True,
                failed_state_unchanged=True, reservation_released=True,
                lost_response_reconciled=True, disabled_session_safe=True,
                health_busy_and_mcts_total_ms=overlap_ms)


def http_games(origin):
    assert urlparse(origin).hostname in ('localhost', '127.0.0.1', '::1')
    def request(path, body=None):
        req = Request(origin + BASE + path,
                      data=None if body is None else json.dumps(body).encode(),
                      headers={'Content-Type': 'application/json'})
        with urlopen(req, timeout=15) as response:
            data = response.read()
            return json.loads(data) if 'json' in response.headers.get('Content-Type', '') else data.decode()
    request('/health')
    times = []
    lengths = []
    for research_player in (0, 1):
        players = [{'type': 'human'}, {'type': 'human'}]
        players[research_player] = {'type': 'victor_research'}
        state = request('/start_game', dict(zip(('player1', 'player2'), players)))
        rng = Random(9 + research_player)
        replay = Connect4()
        while not state['gameOver']:
            body = dict(game_id=state['game_id'], revision=state['revision'])
            ai = state['currentPlayer'] == research_player
            if not ai:
                body['column'] = rng.choice(state['legalMoves'])
            started = perf_counter()
            after = request('/make_move', body)
            if ai:
                times.append(perf_counter() - started)
            assert after['revision'] == state['revision'] + 1
            state = after
        history = request('/games/' + state['game_id'] + '/history')
        for revision, record in enumerate(history['moves'], 1):
            assert record['revision'] == revision
            assert [list(r) for r in replay.board] == record['board_before']
            assert replay.make_move(record['column'])
            assert [list(r) for r in replay.board] == record['board_after']
        assert [list(r) for r in replay.board] == state['board']
        forbidden = {'certificate', 'bound', 'exact_value', 'move_kind', 'research'}
        def walk(value):
            if isinstance(value, dict):
                assert not forbidden.intersection(value)
                for child in value.values():
                    walk(child)
            elif isinstance(value, list):
                for child in value:
                    walk(child)
        walk(state)
        walk(history)
        lengths.append(state['revision'])
    return dict(plies=lengths, ai_http_latency=summary(times))


def http_overlap(origin):
    """Three modest overlap trials against real one-worker/four-thread Gunicorn."""
    assert urlparse(origin).hostname in ('localhost', '127.0.0.1', '::1')
    def request(path, body=None):
        req = Request(origin + BASE + path,
                      data=None if body is None else json.dumps(body).encode(),
                      headers={'Content-Type': 'application/json'})
        try:
            response = urlopen(req, timeout=15)
        except HTTPError as error:
            response = error
        with response:
            raw = response.read()
            data = json.loads(raw) if 'json' in response.headers.get('Content-Type', '') else raw.decode()
            return response.status, data
    health_times, busy_count = [], 0
    for _ in range(3):
        states = []
        for agent in ('victor_research', 'victor_research', 'mcts'):
            config = {'type': agent}
            if agent == 'mcts':
                config['simulation_limit'] = 50
            status, state = request('/start_game', {
                'player1': config, 'player2': {'type': 'human'}})
            assert status == 201
            states.append(state)
        barrier = Barrier(4)
        def move(state):
            barrier.wait(5)
            return request('/make_move', dict(game_id=state['game_id'], revision=0))
        def health():
            barrier.wait(5)
            started = perf_counter()
            assert request('/health')[0] == 200
            return perf_counter() - started
        with ThreadPoolExecutor(4) as pool:
            futures = [pool.submit(move, state) for state in states]
            health_future = pool.submit(health)
            results = [future.result(15) for future in futures]
            health_times.append(health_future.result(15))
        assert results[2][0] == 200
        for state, (status, data) in zip(states[:2], results[:2]):
            if status == 503:
                assert data['code'] == 'agent_busy'
                busy_count += 1
                assert request('/games/' + state['game_id'])[1] == state
                status, data = request('/make_move', dict(game_id=state['game_id'], revision=0))
            assert status == 200 and data['revision'] == 1
    return dict(trials=3, busy_rejections=busy_count, retries_succeeded=True,
                mcts_succeeded=True, concurrent_health_latency=summary(health_times))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixtures', required=True)
    parser.add_argument('--api')
    args = parser.parse_args()
    rows = json.loads(open(args.fixtures).read())
    assert sys.platform == 'linux', 'This check requires Linux, not macOS evidence'
    cold_started = perf_counter()
    assert native.available()
    native_load_ms = (perf_counter() - cold_started) * 1000
    assert shutil.which('cc') is None and shutil.which('gcc') is None
    assert not __import__('importlib.util').util.find_spec('victor_validation')
    cold_started = perf_counter()
    book = default_book()
    book_load_ms = (perf_counter() - cold_started) * 1000
    assert book is not None and len(book) == 1722
    for _, _, history, _ in book.entries.values():
        g = game([int(c) - 1 for c in history])
        assert book.lookup(Position.from_board(g.board, g.current_player)).status == 'exact'
    exact_values = set()
    for row in rows['exact']:
        g = game(row['history'])
        p = Position.from_board(g.board, g.current_player)
        truth = {int(c): v for c, v in row['move_values'].items()}
        exact_values.add(max(truth.values()))
        for nodes, entries in ((0, 1), (128, 17), (1_000_000, 8192)):
            proof = native.prove_moves(p, native.NativeBudget(
                nodes=nodes, seconds=None, table_entries=entries, serial=True))
            assert {c for c, _, _ in proof.intervals} == set(truth)
            assert all(lo <= truth[c] <= hi for c, lo, hi in proof.intervals)
            assert all(truth[c] == max(truth.values()) for c in proof.optimal_moves)
            assert proof.nodes <= nodes
            if nodes == 1_000_000:
                assert proof.status == 'all_moves'
                assert all(lo == hi == truth[c] for c, lo, hi in proof.intervals)
        assert native.prove_moves(p, native.NativeBudget(seconds=0)).nodes == 0
        exhausted = native.prove_moves(p, native.NativeBudget(table_entries=0))
        assert exhausted.status == 'unavailable'
    assert exact_values == {-1, 0, 1}
    times, deadline_hits, optimal = [], 0, 0
    g = game(rows['timed'][0]['history'])
    p = Position.from_board(g.board, g.current_player)
    started = perf_counter()
    slow = native.prove_moves(p, native.NativeBudget(seconds=.001))
    slow_ms = (perf_counter() - started) * 1000
    truth = {int(c): v for c, v in rows['timed'][0]['move_values'].items()}
    assert all(lo <= truth[c] <= hi for c, lo, hi in slow.intervals)
    assert slow_ms < 1500
    exhausted_agent = VictorResearchAgent(replace(
        PUBLIC_BUDGET, opening_book=False, native=native.NativeBudget(table_entries=0)))
    assert exhausted_agent.choose_move(g) in g.get_valid_moves()
    for row in rows['timed']:
        g = game(row['history'])
        agent = VictorResearchAgent()
        started = perf_counter()
        move = agent.choose_move(g)
        times.append(perf_counter() - started)
        assert move in g.get_valid_moves()
        deadline_hits += agent.last_decision['deadline_reached']
        truth = {int(c): v for c, v in row['move_values'].items()}
        optimal += truth[move] == max(truth.values())
    # Hide the real binary, clear the loader cache, exercise real missing-file fallback.
    path = native.library_path()
    hidden = path.with_suffix('.hidden')
    path.rename(hidden)
    native._library.cache_clear()
    try:
        # dlopen can reuse an already loaded object after its file is removed.
        # A fresh interpreter models an image shipped without the accelerator.
        subprocess.run([sys.executable, '-c', '''
from dataclasses import replace
from games.connect4.victor import native
from games.connect4.connect4 import Connect4
from games.connect4.agents.victor_research_agent import VictorResearchAgent, PUBLIC_BUDGET
assert not native.available()
g = Connect4()
agent = VictorResearchAgent(replace(PUBLIC_BUDGET, opening_book=False))
assert agent.choose_move(g) in g.get_valid_moves()
'''], check=True)
    finally:
        hidden.rename(path)
        native._library.cache_clear()
    with patch('games.connect4.victor.solver.prove_moves', side_effect=RuntimeError('injected')):
        agent = VictorResearchAgent(replace(PUBLIC_BUDGET, opening_book=False))
        assert agent.choose_move(game(())) in range(7)
        assert agent.last_decision['kind'] == 'fallback'
    result = dict(platform=platform.platform(), architecture=platform.machine(),
                  native=True, opening_entries=len(book), compiler_required=False,
                  oracle_required=False, exact_rows=len(rows['exact']),
                  interrupted_and_exhausted_bounds=True, missing_native_fallback=True,
                  native_failure_fallback=True, fixed_move_latency=summary(times),
                  cold_native_load_ms=native_load_ms, cold_book_load_ms=book_load_ms,
                  tiny_native_budget_ms=slow_ms, tiny_native_budget_status=slow.status,
                  exhausted_allocation_fallback=True,
                  deadline_hits=deadline_hits, sampled_optimal=f'{optimal}/{len(times)}',
                  concurrency=controlled_failures())
    if args.api:
        result['http_overlap'] = http_overlap(args.api)
        result['http_games'] = http_games(args.api)
    result['peak_rss_mib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
