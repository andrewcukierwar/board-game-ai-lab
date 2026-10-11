"""Standalone, bounded audit of immutable v2 inference artifacts.

Run from the research checkout: python -m research.alphazero_baseline.audit
--campaign PATH --output NEW_DIRECTORY --mode {archive,profile,evaluate}.
Only development packages are used. Each invocation needs a fresh directory.
Timing instrumentation calls the original functions and uses a nested stack
to report both inclusive times and independently additive exclusive times.
"""
import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import cProfile
import functools
import gc
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import pstats
import random
import resource
import statistics
import subprocess
import sys
import time
import tracemalloc
from types import ModuleType
from unittest.mock import patch

import numpy as np
import torch

from games.connect4.connect4 import Connect4
from games.connect4.alphazero_v2 import network, search
from games.connect4.alphazero_v2.arena import StopEvaluation, run_paired_arena
from games.connect4.alphazero_v2.data import seeded_rng
from games.connect4.alphazero_v2.diagnostics import entropy, peak_rss_mib
from games.connect4.alphazero_v2.evaluation import V2SearchArenaAgent
from games.connect4.alphazero_v2.oracle import (engine_position, safe_moves_scan,
                                               winning_moves_scan)

DECLARATION = 'c8ab2227d4b2c707b3185da8695a991f0914faede88ecd5f1c7fc84fb017b716'
SEEDS = (42, 314159)
GENERATIONS = (0, 10, 15, 20)
DOMAIN = 'phase4d4a-exploratory-v1'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    # Never clobber evidence from another invocation.
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def command(*args):
    return subprocess.check_output(args, text=True).strip()


def runtime():
    return dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                machine=platform.machine(), processor=platform.processor(),
                cpu_count=os.cpu_count(), torch=torch.__version__, numpy=np.__version__,
                intraop=torch.get_num_threads(), interop=torch.get_num_interop_threads(),
                device='cpu', dtype='float32', torch_config=torch.__config__.show(),
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                matmul_precision=torch.get_float32_matmul_precision(),
                env={k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                         'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'PYTHONHASHSEED')},
                commit=command('git', 'rev-parse', 'HEAD'),
                tracked_status=command('git', 'status', '--short'),
                tool_sha256=sha(__file__))


def fresh_output(campaign, output):
    campaign, output = Path(campaign).resolve(), Path(output).resolve()
    if output == campaign or campaign in output.parents or output in campaign.parents:
        raise ValueError('Output must be disjoint from the immutable campaign')
    output.mkdir(parents=True, exist_ok=False)
    return output


def campaign_fingerprint(campaign):
    # Includes sealed file bytes for preservation verification, never evaluation.
    return {str(p.relative_to(campaign)): dict(bytes=p.stat().st_size, sha256=sha(p))
            for p in sorted(campaign.rglob('*')) if p.is_file()}


def checkpoint(campaign, seed, generation):
    return campaign / f'runs/seed-{seed}/generations/generation-{generation:04d}/generation-{generation:04d}.inference.pt'


def package(campaign, kind):
    p = campaign / f'packages/{kind}-development.json'
    data = read(p)
    if data['split'] != 'development':
        raise ValueError('Audit only accepts development packages')
    return data['rows']


class Deadline:
    def __init__(self, seconds):
        self.started = time.perf_counter()
        self.limit = seconds

    def __call__(self):
        if time.perf_counter() - self.started >= self.limit:
            raise StopEvaluation('Audit cooperative wall-time ceiling reached')


class Timings:
    """Single-thread instrumentation; exclusive categories add, inclusive overlap."""
    def __init__(self):
        self.stack = []
        self.inclusive = defaultdict(float)
        self.exclusive = defaultdict(float)
        self.calls = Counter()

    def wrap(self, category, function):
        @functools.wraps(function)
        def measured(*args, **kwargs):
            frame = [time.perf_counter(), 0.0]
            self.stack.append(frame)
            try:
                return function(*args, **kwargs)
            finally:
                duration = time.perf_counter() - frame[0]
                self.stack.pop()
                self.calls[category] += 1
                self.inclusive[category] += duration
                self.exclusive[category] += duration - frame[1]
                if self.stack:
                    self.stack[-1][1] += duration
        return measured

    @contextmanager
    def instrument(self, inference):
        from contextlib import ExitStack
        targets = [(search, 'deepcopy', 'state_copy'),
                   (search, 'backup', 'backup'),
                   (network, 'encode_current_player', 'encoding_tensor'),
                   (Connect4, 'make_move', 'state_transition'),
                   (Connect4, 'is_game_over', 'terminal'),
                   (Connect4, 'check_winner', 'terminal'),
                   (Connect4, '_has_terminated', 'terminal'),
                   (Connect4, 'get_valid_moves', 'legal_moves'),
                   (search.PUCTSearch, '_expand', 'expansion'),
                   (network.V2Inference, 'predict', 'inference_pipeline'),
                   (inference.model, 'forward', 'neural_forward')]
        with ExitStack() as patches:
            for owner, name, category in targets:
                patches.enter_context(patch.object(owner, name, self.wrap(category, getattr(owner, name))))
            # Retain the staticmethod descriptor when installing the wrapper.
            patches.enter_context(patch.object(search.PUCTSearch, '_select_child',
                staticmethod(self.wrap('selection', search.PUCTSearch._select_child))))
            yield self

    def result(self, elapsed):
        return dict(exclusive_seconds=dict(self.exclusive), inclusive_seconds=dict(self.inclusive),
                    calls=dict(self.calls), instrumented_wall_seconds=elapsed,
                    other_seconds=elapsed - sum(self.exclusive.values()),
                    note='Exclusive categories plus other are additive. Inclusive categories overlap; '
                         'neural_forward includes contract validation. Instrumented times include wrapper overhead.')


def search_signature(result):
    """Check more than selected action: every node's state, visits, Q sum and priors."""
    def node_signature(node):
        return (node.game_state.current_player, tuple(tuple(r) for r in node.game_state.board),
                node.visits, node.value_sum, node.prior_p,
                tuple((a, node_signature(child)) for a, child in node.children.items()))
    return (result.visits, result.root_logits, result.root_value, result.search_prior,
            result.max_depth, result.leaf_depth_sum, result.terminal_leaves, node_signature(result.root))


def descriptive(values):
    a = np.asarray(values, dtype=float)
    return dict(n=len(values), mean=float(a.mean()), median=float(np.median(a)),
                min=float(a.min()), max=float(a.max()),
                p95=float(np.quantile(a, .95)),
                stdev=float(a.std(ddof=1)) if len(values) > 1 else None)


def representative_positions(campaign):
    tactics = package(campaign, 'tactical')
    solved = package(campaign, 'solved')
    openings = package(campaign, 'openings')
    quiet = next(r for r in solved if 16 <= r['ply'] <= 24 and not winning_moves_scan(r['moves'])
                 and len(safe_moves_scan(r['moves'])) == len(engine_position(r['moves']).get_valid_moves()))
    tactical = next(r for r in tactics if r['ply'] >= 14 and r['category'] == 'unique_safe')
    late = next(r for r in solved if r['ply'] >= 32)
    return [dict(label='empty', id='audit-empty', moves=[]),
            dict(next(r for r in openings if r['moves']), label='early_opening'),
            dict(quiet, label='quiet_midgame'), dict(tactical, label='tactical_midgame'),
            dict(late, label='late_game'),
            dict(next(r for r in tactics if r['category'] == 'unique_win'), label='immediate_win'),
            dict(next(r for r in tactics if r['category'] == 'unique_safe'), label='mandatory_block')]


def profile(campaign, output, deadline, repeats):
    rows = representative_positions(campaign)
    write(output / 'positions.json', rows)
    records = []
    for seed in SEEDS:
        for generation in GENERATIONS:
            inference = network.load_inference_checkpoint(checkpoint(campaign, seed, generation))
            # Warm framework and convolutions once per model, outside timed trials.
            inference.predict(Connect4())
            for row in rows:
                game = engine_position(row['moves'])
                for budget in (256, 512):
                    engine = search.PUCTSearch(inference, budget)
                    trials = []
                    for repeat in range(repeats):
                        deadline()
                        gc.collect()
                        rng = seeded_rng(f'{DOMAIN}:profile:{row["id"]}:{repeat}', 4304001)
                        before_cpu, before = time.process_time(), time.perf_counter()
                        result = engine.run(game, rng=rng)
                        elapsed = time.perf_counter() - before
                        cpu = time.process_time() - before_cpu
                        trials.append(dict(wall_seconds=elapsed, cpu_seconds=cpu,
                                           simulations_per_second=budget / elapsed,
                                           inference_calls=1 + budget - result.terminal_leaves,
                                           terminal_leaves=result.terminal_leaves, visits=result.visits,
                                           max_depth=result.max_depth, leaf_depth_sum=result.leaf_depth_sum))
                        if repeat == 0:
                            signature, rng_after = search_signature(result), rng.getstate()
                        del result
                    meter = Timings()
                    rng = seeded_rng(f'{DOMAIN}:profile:{row["id"]}:0', 4304001)
                    with meter.instrument(inference):
                        before = time.perf_counter()
                        measured = engine.run(game, rng=rng)
                        elapsed = time.perf_counter() - before
                    if search_signature(measured) != signature or rng.getstate() != rng_after:
                        raise RuntimeError('Instrumentation changed the search or RNG')
                    records.append(dict(seed=seed, generation=generation, position=row['label'],
                                        position_id=row['id'], simulations=budget, trials=trials,
                                        wall=descriptive([t['wall_seconds'] for t in trials]),
                                        timing=meter.result(elapsed), instrumentation_semantics_verified=True,
                                        process_peak_rss_mib=peak_rss_mib()))
                    with (output / 'profile.jsonl').open('a') as stream:
                        stream.write(json.dumps(records[-1], allow_nan=False)+'\n')
                    del measured
            print(f'profile seed={seed} generation={generation} complete', flush=True)
    write(output / 'profile.json', records)
    # Separate allocation and cProfile probes; never include these in latency data.
    inference = network.load_inference_checkpoint(checkpoint(campaign, 42, 10))
    game = engine_position(rows[2]['moves'])
    engine = search.PUCTSearch(inference, 512)
    gc.collect()
    tracemalloc.start()
    result = engine.run(game, rng=random.Random(4304001))
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    write(output / 'memory.json', dict(seed=42, generation=10, position=rows[2]['id'], simulations=512,
                                      python_current_bytes=current, python_peak_bytes=peak,
                                      process_peak_rss_mib=peak_rss_mib(),
                                      note='tracemalloc excludes most native tensor storage; RSS is process lifetime peak'))
    del result
    profiler = cProfile.Profile()
    profiler.runcall(engine.run, game, rng=random.Random(4304001))
    profiler.dump_stats(str(output / 'search.pstats'))
    stats = pstats.Stats(profiler)
    functions = [dict(file=f, line=line, function=name, primitive_calls=v[0], calls=v[1],
                      self_seconds=v[2], cumulative_seconds=v[3])
                 for (f, line, name), v in stats.stats.items()]
    write(output / 'cprofile.json', sorted(functions, key=lambda r: r['self_seconds'], reverse=True))
    # Batching feasibility microbenchmark only; no search or model modification.
    tensor = network.encode_current_player(game)
    batches = []
    with torch.inference_mode():
        for batch in (1, 8, 32, 64):
            states = tensor.repeat(batch, 1, 1, 1)
            inference.model(states)
            samples = []
            for _ in range(20):
                deadline()
                before = time.perf_counter()
                inference.model(states)
                samples.append(time.perf_counter() - before)
            batches.append(dict(batch=batch, seconds=descriptive(samples),
                                positions_per_second=batch / statistics.median(samples)))
    write(output / 'forward_batches.json', batches)


def archives(campaign, output, deadline):
    generations, checks = [], []
    for seed in SEEDS:
        for generation in range(1, 21):
            deadline()
            directory = checkpoint(campaign, seed, generation).parent
            summary = read(directory / 'summary.json')
            games = [json.loads(line) for line in (directory / f'generation-{generation:04d}.games.jsonl').read_text().splitlines()]
            lengths = [len(g['moves']) for g in games]
            trajectories = Counter(tuple(g['moves']) for g in games)
            openings = Counter(tuple(g['moves'][:8]) for g in games)
            positions, families, after8 = Counter(), Counter(), Counter()
            tactical = Counter()
            for g in games:
                game = Connect4()
                for ply, move in enumerate(g['moves']):
                    board = tuple(tuple(row) for row in game.board)
                    mirror = tuple(row[::-1] for row in board)
                    positions[(game.current_player, board)] += 1
                    families[(game.current_player, min(board, mirror))] += 1
                    if ply >= 8:
                        after8[(game.current_player, board)] += 1
                    # All archived trajectories: independent one-reply oracle, no NN.
                    prefix = g['moves'][:ply]
                    wins, safe = winning_moves_scan(prefix), safe_moves_scan(prefix)
                    if wins:
                        tactical['immediate_win_states'] += 1
                        tactical['missed_immediate_win_actions'] += move not in wins
                    elif safe and len(safe) < len(game.get_valid_moves()):
                        tactical['mandatory_response_states'] += 1
                        tactical['unsafe_actions_with_safe_alternative'] += move not in safe
                    if not game.make_move(move):
                        raise ValueError('Invalid archived trajectory')
                if game.check_winner() != g['winner']:
                    raise ValueError('Archive winner mismatch')
            total = sum(lengths)
            generations.append(dict(seed=seed, generation=generation, length=descriptive(lengths),
                unique_trajectories=len(trajectories), max_trajectory_multiplicity=max(trajectories.values()),
                unique_opening8=len(openings), top_opening8_count=max(openings.values()),
                unique_positions=len(positions), unique_mirror_families=len(families), positions=total,
                duplicate_position_fraction=1-len(positions)/total,
                after8_duplicate_fraction=1-len(after8)/sum(after8.values()) if after8 else None,
                archived_tactics=dict(tactical), resources=summary['resources'], summary=summary['summary'],
                development_raw_value=summary.get('development_raw_value')))
        for generation in (5, 10, 15, 20):
            check = read(campaign / f'runs/seed-{seed}/champion-check-{generation:04d}.json')
            checks.append(dict(seed=seed, **{k:v for k,v in check.items() if k != 'records'}))
        print(f'archive seed={seed} complete', flush=True)
    write(output / 'archives.json', generations)
    write(output / 'official_development.json', checks)
    replay_records = []
    for seed in SEEDS:
        deadline()
        path = checkpoint(campaign, seed, 20).with_name('generation-0020.resume.pt')
        # Read storage only: never instantiate a runner or call load_boundary/resume.
        payload = torch.load(path, map_location='cpu', weights_only=True)
        for gen in payload['replay']['generations']:
            by_stage = defaultdict(list)
            for game in gen['games']:
                for ply, (move, visits, temperature) in enumerate(zip(
                        game['moves'], game['visits'], game['action_temperatures'])):
                    deadline()
                    if sum(visits) != 256 or temperature != (1. if ply < 8 else 0.):
                        raise ValueError('Archived replay visit/temperature mismatch')
                    if ply >= 8 and visits[move] != max(visits):
                        raise ValueError('Archived greedy action is not maximum-visit')
                    probabilities = [v / 256 for v in visits]
                    by_stage['before8' if ply < 8 else 'after8'].append(
                        (entropy(probabilities), max(probabilities), sum(v>0 for v in visits)))
            replay_records.append(dict(seed=seed, generation=gen['generation'],
                stages={stage:dict(positions=len(values), mean_target_entropy=statistics.mean(x[0] for x in values),
                                   mean_max_visit_mass=statistics.mean(x[1] for x in values),
                                   mean_visited_actions=statistics.mean(x[2] for x in values))
                        for stage,values in by_stage.items()}))
    write(output/'replay_exploration.json', replay_records)


def clustered_interval(records, resamples=4000):
    """Percentile bootstrap by reflection/opening family, preserving color pairs.

    Descriptive only for tiny samples; degenerate bounds are not certainty.
    """
    families = defaultdict(list)
    completed = [r for r in records if not r['abandoned']]
    for r in completed:
        families[r['family']].append({'win':1., 'draw':.5, 'loss':0.}[r['result']])
    if not families:
        return dict(games=0, score=None, interval95=None, clusters=0)
    rng = random.Random(4304002)
    groups = list(families.values())
    estimates = []
    for _ in range(resamples):
        sample = [x for group in rng.choices(groups, k=len(groups)) for x in group]
        estimates.append(statistics.mean(sample))
    by_color = {}
    for color in (0, 1):
        rows = [r for r in completed if r['agent_color'] == color]
        counts = Counter(r['result'] for r in rows)
        by_color['XO'[color]] = dict(games=len(rows), wins=counts['win'], draws=counts['draw'], losses=counts['loss'],
                                     score=(counts['win']+.5*counts['draw'])/len(rows) if rows else None)
    return dict(games=len(completed), wins=sum(r['result']=='win' for r in completed),
                draws=sum(r['result']=='draw' for r in completed), losses=sum(r['result']=='loss' for r in completed),
                score=statistics.mean([x for g in groups for x in g]), clusters=len(groups), by_color=by_color,
                interval95=[float(x) for x in np.quantile(estimates, [.025,.975])],
                caveat='Small exploratory sample; family bootstrap is unstable and may degenerate. No significance claim.')


def load_main_reference(path):
    """Load exact main opponents under an isolated namespace, without merging.

    Negamax has one absolute import of its TT helper; install that same helper
    under the canonical name, which is absent from the historical checkout.
    The neural search and historical MCTS modules are never replaced.
    """
    path = Path(path).resolve()
    name = '_phase4d4a_main_agents'
    package_module = ModuleType(name)
    package_module.__path__ = [str(path / 'games/connect4/agents')]
    sys.modules[name] = package_module
    tt = importlib.import_module(name + '.negamax_tt')
    canonical = 'games.connect4.agents.negamax_tt'
    if canonical in sys.modules and sys.modules[canonical] is not tt:
        raise RuntimeError('Conflicting TT module; use a clean audit process')
    sys.modules[canonical] = tt
    return {key: importlib.import_module(name + '.' + key) for key in
            ('mcts_agent', 'tactical_mcts_agent', 'negamax_agent', 'mcts_bitboard')}


class MainOpponent:
    def __init__(self, modules, kind, budget):
        self.modules, self.kind, self.budget = modules, kind, budget
        self.name = f'main-{kind}-{budget}'

    def describe(self):
        return dict(name=self.name, kind=self.kind, budget=self.budget,
                    max_seconds=None, root_guard=self.kind == 'classic',
                    exploration=.5 if self.kind == 'tactical' else None)

    def choose(self, game, rng):
        if self.kind == 'negamax':
            agent = self.modules['negamax_agent'].NegamaxAgent(self.budget)
        elif self.kind == 'classic':
            agent = self.modules['mcts_agent'].MCTSAgent(self.budget, rng=rng)
        else:
            agent = self.modules['tactical_mcts_agent'].TacticalMCTSAgent(self.budget, rng=rng)
        return int(agent.choose_move(game)), {}


def state_microbenchmark(campaign, output, deadline, main_reference):
    """Representation-only reference measurements, never a modified neural search."""
    from copy import deepcopy
    modules = load_main_reference(main_reference)
    bitboard = modules['mcts_bitboard'].BitboardState
    rows = representative_positions(campaign)
    checked = 0
    for row in package(campaign,'tactical') + package(campaign,'solved'):
        deadline()
        game = engine_position(row['moves'])
        state = bitboard.from_game(game)
        for action in game.get_valid_moves():
            after = deepcopy(game)
            assert after.make_move(action)
            dropped = state.drop(action)
            assert tuple(tuple(r) for r in after.board) == dropped.board
            assert after.current_player == dropped.current_player
            assert after.check_winner() == dropped.check_winner()
            assert after.is_game_over() == dropped.is_game_over()
            assert after.get_valid_moves() == dropped.get_valid_moves()
            checked += 1
    measurements = []
    for row in rows:
        deadline()
        game = engine_position(row['moves'])
        state = bitboard.from_game(game)
        actions = game.get_valid_moves()
        def engine_drop():
            for move in actions:
                after = deepcopy(game)
                after.make_move(move)
        def bitboard_drop():
            for move in actions:
                state.drop(move)
        for operation, function, count in (
                ('engine_copy_all_successors',engine_drop,len(actions)),
                ('bitboard_all_successors',bitboard_drop,len(actions)),
                ('engine_terminal',game.is_game_over,1),
                ('bitboard_terminal',state.is_game_over,1),
                ('engine_to_bitboard',lambda:bitboard.from_game(game),1)):
            trials = []
            for _ in range(5):
                deadline()
                before = time.perf_counter()
                for _ in range(500):
                    function()
                trials.append((time.perf_counter()-before)/(500*count))
            measurements.append(dict(position=row['label'],operation=operation,
                                     seconds_per_successor_or_call=descriptive(trials)))
    write(output/'state_microbenchmark.json',dict(transitions_checked=checked, measurements=measurements,
        main_commit=command('git','-C',str(main_reference),'rev-parse','HEAD'),
        bitboard_source_sha256=sha(Path(main_reference)/'games/connect4/agents/mcts_bitboard.py'),
        caveat='Microbenchmarks are not neural-search speedups. Direct bitboard encoding is not implemented.'))


def position_assessment(inference, rows, budget, deadline):
    records = []
    for row in rows:
        deadline()
        game = engine_position(row['moves'])
        prediction = inference.predict(game)
        result, action = search.EvaluationAgent(inference, budget).search(
            game, rng=seeded_rng(f'{DOMAIN}:tactics:{row["id"]}', 4304003))
        wins, safe = winning_moves_scan(row['moves']), safe_moves_scan(row['moves'])
        record = dict(id=row['id'], family=row['family'], actor=row['actor'], category=row['category'],
                      move=action.move, correct=action.move==row['expected_action'], visits=result.visits,
                      policy=prediction.policy, policy_entropy=entropy(prediction.policy), value=prediction.value,
                      nn_move=int(np.argmax(prediction.policy)), nn_correct=int(np.argmax(prediction.policy))==row['expected_action'],
                      avoidable_error=bool((wins and action.move not in wins) or (not wins and safe and action.move not in safe)))
        records.append(record)
    return records


def evaluate(campaign, output, deadline, main_reference, openings_count, budget, opening_rows=None, ladder_only=False):
    modules = load_main_reference(main_reference)
    reference = Path(main_reference)
    write(output / 'main_reference.json', dict(commit=command('git', '-C', str(reference), 'rev-parse', 'HEAD'),
        files={str(p.relative_to(reference)): sha(p) for p in (reference/'games/connect4/agents').glob('*.py')},
        opponents=[MainOpponent(modules, kind, b).describe() for kind,b in
                   [('negamax',2),('negamax',4),('negamax',6),('negamax',8),('classic',1000),('tactical',1000)]]))
    tactics, solved = package(campaign, 'tactical'), package(campaign, 'solved')
    # One empty row plus complete base/mirror prefix families, fixed before weights.
    rows = package(campaign, 'openings')
    openings = [next(r for r in rows if not r['moves'])] + [r for r in rows if r['moves']][:openings_count*2]
    if opening_rows is not None:
        openings = openings[:opening_rows]
    write(output / 'evaluation_design.json', dict(openings=openings, simulations=budget, tactics_rows=len(tactics),
        solved_rows=len(solved), search_tie_seed=4304003, arena_seed=4304004, domain=DOMAIN,
        ladder_only=ladder_only,
        note='Development reuse, exploratory. Identical RNG domain across checkpoints and opponents. '
             'All tactical rows once; all solved rows raw NN unless ladder-only. Paired X/O. '
             'Opening list above is authoritative; --opening-rows may omit mirrors.'))
    position_results, ladder = [], []
    # Assess all networks before the ladder so a time limit cannot omit a seed.
    for seed in (() if ladder_only else SEEDS):
        for generation in GENERATIONS:
            inference = network.load_inference_checkpoint(checkpoint(campaign, seed, generation))
            tactical_records = position_assessment(inference, tactics, budget, deadline)
            values = []
            for row in solved:
                deadline()
                prediction = inference.predict(engine_position(row['moves']))
                values.append(dict(id=row['id'], family=row['family'], actor=row['actor'],
                                   target=row['value'], value=prediction.value, policy=prediction.policy,
                                   nn_move=int(np.argmax(prediction.policy)),
                                   nn_optimal=int(np.argmax(prediction.policy)) in row['optimal_actions']))
            categories = {category: dict(rows=len(rs), accuracy=statistics.mean(r['correct'] for r in rs),
                                         nn_accuracy=statistics.mean(r['nn_correct'] for r in rs),
                                         by_actor={str(a):statistics.mean(r['correct'] for r in rs if r['actor']==a) for a in (0,1)})
                          for category in ('unique_win','unique_safe')
                          for rs in [[r for r in tactical_records if r['category']==category]]}
            value_metrics = {str(target): dict(rows=len(rs), mse=statistics.mean((r['value']-target)**2 for r in rs),
                                               mae=statistics.mean(abs(r['value']-target) for r in rs),
                                               correct_sign=statistics.mean(r['value']*target>0 for r in rs) if target else None,
                                               nn_optimal=statistics.mean(r['nn_optimal'] for r in rs))
                             for target in (-1,0,1) for rs in [[r for r in values if r['target']==target]]}
            position_results.append(dict(seed=seed, generation=generation, tactics=categories, values=value_metrics,
                                          tactical_records=tactical_records, solved_records=values))
            with (output / 'positions_assessment.jsonl').open('a') as stream:
                stream.write(json.dumps(position_results[-1], allow_nan=False)+'\n')
            print(f'positions seed={seed} generation={generation} complete', flush=True)
    write(output / 'positions_assessment.json', position_results)
    # Round robin opponent-first keeps checkpoint coverage matched if bounded.
    opponents = [('generation0',0),('negamax',2),('negamax',4),('negamax',6),('negamax',8),('classic',1000),('tactical',1000)]
    for kind, opponent_budget in opponents:
        for seed in SEEDS:
            for generation in GENERATIONS:
                if generation == 0 and kind == 'generation0':
                    continue
                deadline()
                inference = network.load_inference_checkpoint(checkpoint(campaign, seed, generation))
                agent = V2SearchArenaAgent(inference, budget, name=f'seed{seed}-g{generation}')
                opponent = (V2SearchArenaAgent(network.load_inference_checkpoint(checkpoint(campaign, seed, 0)),
                                               budget, name=f'seed{seed}-g0') if kind == 'generation0'
                            else MainOpponent(modules, kind, opponent_budget))
                records, status = run_paired_arena(agent, opponent, openings,
                    namespace=DOMAIN+':ladder', seed=4304004, check=deadline, keep_decisions=True)
                item = dict(seed=seed, generation=generation, opponent=opponent.name,
                            opponent_config=opponent.describe(), status=status, summary=clustered_interval(records), records=records)
                ladder.append(item)
                with (output / 'ladder.jsonl').open('a') as stream:
                    stream.write(json.dumps(item, allow_nan=False)+'\n')
                print(f'ladder seed={seed} generation={generation} opponent={opponent.name} '
                      f'score={item["summary"]["score"]} games={item["summary"]["games"]}', flush=True)
                if not status['complete']:
                    raise StopEvaluation(status['stop_reason'])
    write(output / 'ladder.json', ladder)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('archive','profile','evaluate','state'), required=True)
    parser.add_argument('--main-reference', type=Path)
    parser.add_argument('--seconds', type=float, default=1500)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--interop', type=int, default=1)
    parser.add_argument('--deterministic', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--opening-families', type=int, default=2)
    parser.add_argument('--opening-rows', type=int)
    parser.add_argument('--ladder-only', action='store_true')
    parser.add_argument('--simulations', type=int, default=512)
    args = parser.parse_args()
    if min(args.seconds,args.threads,args.interop,args.repeats,args.opening_families,args.simulations)<=0:
        parser.error('Budgets and trial counts must be positive')
    if args.opening_rows is not None and args.opening_rows < 1:
        parser.error('--opening-rows must be positive')
    if args.mode in ('evaluate','state') and args.main_reference is None:
        parser.error('--main-reference is required for evaluate/state')
    campaign = args.campaign.resolve()
    if sha(campaign/'declaration.json') != DECLARATION:
        raise ValueError('Official declaration hash mismatch')
    output = fresh_output(campaign, args.output)
    (output/'audit-used.py').write_bytes(Path(__file__).read_bytes())
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.interop)
    torch.use_deterministic_algorithms(args.deterministic)
    before = campaign_fingerprint(campaign)
    write(output/'campaign_before.json', before)
    source = read(campaign/'source/source.json')
    root = Path(__file__).resolve().parents[2]
    mismatch = [name for name,digest in source['execution']['files'].items() if sha(root/name)!=digest]
    if mismatch:
        raise ValueError(f'Frozen execution source mismatch: {mismatch}')
    models = []
    for seed in SEEDS:
        for generation in GENERATIONS:
            path = checkpoint(campaign,seed,generation)
            digest = sha(path)
            if generation:
                expected = read(path.parent/'summary.json')['files']['inference']['sha256']
            else:
                expected = read(campaign/f'runs/seed-{seed}/selection.json')['inference']['sha256']
            if digest != expected:
                raise ValueError('Checkpoint hash differs from campaign record')
            models.append(dict(seed=seed,generation=generation,path=str(path),sha256=digest))
    write(output/'provenance.json', dict(runtime=runtime(), parameters={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                                       checkpoints=models, execution_source=source, declaration_sha256=DECLARATION,
                                       started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    deadline = Deadline(args.seconds)
    status = dict(complete=False)
    try:
        if args.mode == 'archive':
            archives(campaign,output,deadline)
        elif args.mode == 'profile':
            profile(campaign,output,deadline,args.repeats)
        elif args.mode == 'state':
            state_microbenchmark(campaign,output,deadline,args.main_reference)
        else:
            evaluate(campaign,output,deadline,args.main_reference,args.opening_families,args.simulations,
                     args.opening_rows,args.ladder_only)
        deadline()
        status['complete'] = True
    except StopEvaluation as stop:
        status['stop_reason'] = str(stop)
    except BaseException as error:
        status['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        after = campaign_fingerprint(campaign)
        unchanged = before == after
        write(output/'preservation.json', dict(unchanged=unchanged, files=len(before),
            changed=[p for p in set(before)|set(after) if before.get(p)!=after.get(p)]))
        status.update(elapsed_seconds=time.perf_counter()-deadline.started, process_peak_rss_mib=peak_rss_mib())
        write(output/'status.json',status)
        if not unchanged:
            raise RuntimeError('Immutable campaign changed during audit')
    print(json.dumps(status),flush=True)
    return 0 if status['complete'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
