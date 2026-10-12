"""Experimental forks and strict, atomic generation-boundary continuation.

No official campaign resume; no mid-generation checkpoints. Every attempted
failed generation poisons its runner, including persistence failures. Reload
last published boundary before retrying. Inert on import.
"""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import random
import time

import numpy as np
import torch

from ..connect4 import Connect4
from .artifacts import atomic_torch_save, file_sha256
from .batching import PROTOCOL, TemperatureSchedule, CollectionStats, collect_batched
from .config import ARCHITECTURE, V2Config
from .data import (GenerationReplay, ReflectionAugmenter, V2Example, encode_board,
                   finalize_game, seeded_rng)
from .generation import (_update_digest, _python_rng_state, _python_rng_tuple, SAMPLING_DOMAIN,
                         check_process_sources)
from .experimental_models import model_factory
from .provenance import (execution_source_identity, runtime_identity, source_identity,
                         unavailable_runtime_fields)
from . import diagnostics

CONTRACT = dict(format='connect4-alphazero-experimental-resume', version=1,
                replay_schema='complete-games-generation-schedule-budget-protocol-v1',
                boundary='collection+validation+replay+declared-updates+atomic-persistence',
                exact='identical execution content, runtime and configuration')
LEGACY_PROTOCOL = 'connect4-az-historical-sequential-v2'
ORIGINAL = TemperatureSchedule('original', ((0, 1), (8, 0)))
EXTENDED = TemperatureSchedule('extended', ((0, 1), (16, 0)))
PROCESS_SOURCE = execution_source_identity()


def check_output_path(path):
    """Never publish beneath a campaign declaration, including through symlinks."""
    from pathlib import Path
    target = Path(path).resolve()
    for directory in (target, *target.parents):
        if (directory / 'declaration.json').is_file():
            raise ValueError('Experimental output cannot be inside a campaign directory')


def digest(value):
    result = hashlib.sha256()
    _update_digest(result, value)
    return result.hexdigest()


@dataclass(frozen=True)
class ExperimentalConfig:
    training: V2Config
    optimizer_updates: int
    architecture: str = ARCHITECTURE
    pool_size: int = 8
    rng_protocol: str = PROTOCOL

    def __post_init__(self):
        if not isinstance(self.training, V2Config):
            raise ValueError('Expected V2Config training contract')
        if type(self.optimizer_updates) is not int or self.optimizer_updates < 1:
            raise ValueError('Declare a positive optimizer-update budget')
        if self.pool_size != 8 or type(self.pool_size) is not int or self.rng_protocol != PROTOCOL:
            raise ValueError('Only pool-eight independent-game protocol is supported')
        model_factory(self.architecture)

    def to_dict(self):
        return dict(training=self.training.to_dict(), optimizer_updates=self.optimizer_updates,
                    architecture=self.architecture, pool_size=self.pool_size, rng_protocol=self.rng_protocol)

    @classmethod
    def from_dict(cls, data):
        if set(data) != {'training', 'optimizer_updates', 'architecture', 'pool_size', 'rng_protocol'}:
            raise ValueError('Experimental config fields differ')
        return cls(**dict(data, training=V2Config.from_dict(data['training'])))


def rng_payload():
    n = np.random.get_state()
    return dict(python=_python_rng_state(random.getstate()), torch=torch.get_rng_state().clone(),
                numpy=dict(algorithm=n[0], keys=[int(k) for k in n[1]], position=int(n[2]),
                           has_gauss=int(n[3]), cached_gaussian=float(n[4])))


def restore_rng(state):
    # Validate using disposable generators before changing any global state.
    random.Random().setstate(_python_rng_tuple(state['python']))
    n = state['numpy']
    numpy_state = (n['algorithm'], np.asarray(n['keys'], dtype=np.uint32), n['position'],
                   n['has_gauss'], n['cached_gaussian'])
    np.random.RandomState().set_state(numpy_state)
    torch.Generator().set_state(state['torch'])
    random.setstate(_python_rng_tuple(state['python']))
    np.random.set_state(numpy_state)
    torch.set_rng_state(state['torch'])


def game_record(game):
    return dict(index=game.index, moves=list(game.moves), winner=game.winner,
                visits=[list(e.visits) for e in game.examples],
                action_temperatures=[e.action_temperature for e in game.examples])


def rebuild_scheduled_game(generation, record, schedule, simulations):
    if set(record) != {'index', 'moves', 'winner', 'visits', 'action_temperatures'}:
        raise ValueError('Stored game fields differ')
    moves, visits, temperatures = (record[k] for k in ('moves', 'visits', 'action_temperatures'))
    if not len(moves) == len(visits) == len(temperatures):
        raise ValueError('Stored game arrays differ')
    board, pending = Connect4(), []
    for ply, (move, counts, temperature) in enumerate(zip(moves, visits, temperatures)):
        if board.is_game_over():
            raise ValueError('History continues after termination')
        if type(move) is not int or not 0 <= move < 7:
            raise ValueError('Invalid move')
        if temperature != schedule.temperature(ply):
            raise ValueError('Stored action temperature disagrees with generation schedule')
        example = V2Example(encode_board(board), board.current_player, ply, tuple(counts), move, temperature)
        if sum(counts) != simulations:
            raise ValueError('Visit budget differs')
        if temperature == 0 and counts[move] != max(counts):
            raise ValueError('Greedy action is not maximum-visit')
        pending.append(example)
        if not board.make_move(move):
            raise ValueError('Illegal move')
    result = finalize_game(generation, record['index'], moves, pending)
    if type(record['winner']) is not int or result.winner != record['winner']:
        raise ValueError('Stored outcome differs')
    return result


def validate_optimizer(trainer, stored, steps):
    """Exact parameter ordering, all AdamW options, complete finite moment state."""
    expected = trainer.optimizer.state_dict()
    if set(stored) != set(expected) or len(stored['param_groups']) != len(expected['param_groups']):
        raise ValueError('Optimizer structure differs')
    for actual, reference in zip(stored['param_groups'], expected['param_groups']):
        if actual != reference:
            raise ValueError('Optimizer parameter groups/configuration differ')
    ids = [p for g in expected['param_groups'] for p in g['params']]
    parameters = [p for g in trainer.optimizer.param_groups for p in g['params']]
    if set(stored['state']) != (set(ids) if steps else set()):
        raise ValueError('Optimizer state is incomplete')
    for identity, parameter in zip(ids, parameters):
        if not steps:
            continue
        state = stored['state'][identity]
        if set(state) != {'step', 'exp_avg', 'exp_avg_sq'}:
            raise ValueError('Optimizer state fields differ')
        step = state['step']
        if not isinstance(step, torch.Tensor) or step.numel() != 1 or float(step) != steps:
            raise ValueError('Optimizer step counter differs')
        for name in ('exp_avg', 'exp_avg_sq'):
            value = state[name]
            if (not isinstance(value, torch.Tensor) or value.shape != parameter.shape
                    or value.dtype != parameter.dtype or value.device.type != 'cpu'
                    or not torch.isfinite(value).all() or (name == 'exp_avg_sq' and (value < 0).any())):
                raise ValueError('Optimizer moment is incompatible')
    trainer.optimizer.load_state_dict(deepcopy(stored))
    trainer.steps = steps


class ExperimentalRunner:
    def __init__(self, config, *, schedule=ORIGINAL, lineage='scratch', arm='scratch'):
        self.config = config
        self._install(model_factory(config.architecture).construct(config.training.seed), schedule, lineage, arm)

    def _install(self, model, schedule, lineage, arm):
        if not isinstance(schedule, TemperatureSchedule) or not lineage or not arm:
            raise ValueError('Explicit schedule, lineage and arm required')
        check_process_sources()
        self.factory = model_factory(self.config.architecture)
        self.model = model
        self.trainer = self.factory.trainer(model, self.config.training)
        c = self.config.training
        self.replay = GenerationReplay(c.replay_generations, c.replay_max_games)
        self.sampling_rng = seeded_rng(SAMPLING_DOMAIN, c.seed)
        self.augmenter = ReflectionAugmenter(c.reflection_probability, seed=c.seed)
        self.schedule, self.lineage, self.arm = schedule, lineage, arm
        self.metadata, self.history, self.resources = {}, [], []
        self.parent = None
        self.completed_generations = self.total_games = self.total_positions = 0
        self.runtime = runtime_identity()
        if (not self.runtime['deterministic_algorithms'] or self.runtime['intra_op_threads'] != 1
                or self.runtime['inter_op_threads'] != 1
                or any(v != '1' for v in self.runtime['thread_environment'].values())):
            raise ValueError('Experimental runner requires deterministic one-thread CPU settings')
        if unavailable_runtime_fields(self.runtime):
            raise ValueError('Runtime identity has unavailable fields')
        self.failed, self.phase = False, 'idle'
        self.global_rng = rng_payload()

    def check_identity(self):
        check_process_sources()
        if execution_source_identity() != PROCESS_SOURCE:
            raise RuntimeError('Experimental source content differs')
        if runtime_identity() != self.runtime:
            raise RuntimeError('Experimental runtime identity differs')
        if self.failed:
            raise RuntimeError('Failed runner; reload last valid boundary')

    def counters(self):
        return dict(completed_generations=self.completed_generations, total_games=self.total_games,
                    total_positions=self.total_positions, trainer_steps=self.trainer.steps)

    def replay_payload(self):
        return dict(generations=[dict(generation=g, **deepcopy(self.metadata[g]),
                                      games=[game_record(x) for x in self.replay.iter_games() if x.generation == g])
                                 for g in self.replay.generations], digest=self.replay.digest())

    def _content(self):
        return dict(contract=CONTRACT, config=self.config.to_dict(), schedule=self.schedule.to_dict(),
                    lineage=self.lineage, arm=self.arm, parent=deepcopy(self.parent),
                    counters=self.counters(), model=self.factory.state(self.model),
                    model_hash=self.factory.hash(self.model), optimizer=deepcopy(self.trainer.optimizer.state_dict()),
                    replay=self.replay_payload(), sampling=_python_rng_state(self.sampling_rng.getstate()),
                    augmentation=dict(rng=_python_rng_state(self.augmenter.rng.getstate()),
                                      reflected=self.augmenter.reflected, unreflected=self.augmenter.unreflected),
                    global_rng=deepcopy(self.global_rng), history=deepcopy(self.history), resources=deepcopy(self.resources),
                    runtime=deepcopy(self.runtime), source=source_identity(PROCESS_SOURCE))

    def initial_state_sha256(self):
        """Matched computational state; excludes arm name and prospective action schedule."""
        p = self._content()
        for key in ('schedule', 'arm', 'source', 'resources'):
            p.pop(key)
        return digest(p)

    def boundary_payload(self):
        self.check_identity()
        if self.phase not in ('idle', 'persisting'):
            raise RuntimeError('Only complete generation boundaries may be saved')
        p = self._content()
        return dict(p, integrity=digest(p))

    def save(self, path, *, check=None):
        check_output_path(path)
        p = self.boundary_payload()
        def validate(temporary):
            if check:
                check()
            load_experimental(temporary, restore_global_rng=False)
            if check:
                check()
        return atomic_torch_save(path, p, validate)

    def run_generation(self, path, *, check=None, progress=None):
        """Mandatory persistence; progress/check callbacks must never consume runner RNGs."""
        self.check_identity()
        check_output_path(path)
        c = self.config.training
        if self.completed_generations >= c.max_generations:
            raise RuntimeError('Declared generation limit reached')
        def point(event):
            if check:
                check()
            if progress:
                progress(event)
        try:
            # Each arm owns its process-global stochastic state, too.
            restore_rng(self.global_rng)
            self.phase = 'collecting'
            point('collection')
            start = time.perf_counter()
            snapshot = self.factory.freeze(self.model)
            before = self.factory.hash(self.model)
            stats, observer = CollectionStats(), diagnostics.CollectionObserver()
            generation = self.completed_generations + 1
            games = collect_batched(self.factory.inference(snapshot), c, generation,
                                    range(c.games_per_generation), batch_size=self.config.pool_size,
                                    check=check, stats=stats, observer=observer,
                                    on_game=lambda g: point('game_finalized'), temperature_schedule=self.schedule)
            collected = time.perf_counter()
            point('validation')
            rebuilt = tuple(rebuild_scheduled_game(generation, game_record(g), self.schedule,
                                                   c.self_play_simulations) for g in games)
            if rebuilt != games or len(games) != c.games_per_generation:
                raise ValueError('Collected examples differ from validated games')
            if self.factory.hash(self.model) != before or self.factory.hash(snapshot) != before:
                raise RuntimeError('Frozen collection model changed')
            self.phase = 'replay'
            point('before_replay')
            evicted = self.replay.add_generation(generation, games)
            self.metadata[generation] = dict(schedule=self.schedule.to_dict(), simulations=c.self_play_simulations,
                                             protocol=PROTOCOL, games_count=len(games))
            for g in evicted:
                del self.metadata[g]
            point('after_replay')
            self.phase = 'training'
            metrics, addresses = [], []
            for _ in range(self.config.optimizer_updates):
                point('before_update')
                positions, examples = self.replay.sample(c.batch_size, self.sampling_rng)
                addresses.extend(self.replay.address(p) for p in positions)
                batch, _ = self.augmenter.batch(examples)
                metrics.append(self.trainer.step(batch))
                point('after_update')
            self.completed_generations = generation
            self.total_games += len(games)
            positions = sum(len(g.examples) for g in games)
            self.total_positions += positions
            summary = dict(generation=generation, games=len(games), new_positions=positions,
                           updates=len(metrics), trainer_steps=self.trainer.steps,
                           collection_weights_sha256=before, learner_weights_sha256=self.factory.hash(self.model),
                           schedule=self.schedule.to_dict(), protocol=PROTOCOL,
                           replay_digest=self.replay.digest(), evicted_generations=list(evicted),
                           metrics=metrics, collection=dict(forward_calls=stats.forward_calls,
                               neural_positions=stats.neural_positions, batch_sizes=stats.batch_sizes,
                               max_active=stats.max_active, search=observer.deterministic()),
                           diagnostics=dict(games=diagnostics.game_diagnostics(games),
                               replay=diagnostics.replay_diagnostics(self.replay),
                               sampling=diagnostics.sampling_diagnostics(addresses, generation),
                               gradients=diagnostics.gradient_diagnostics(metrics)))
            self.history.append(summary)
            self.global_rng = rng_payload()
            ready = time.perf_counter()
            self.resources.append(dict(generation=generation, collection_seconds=collected-start,
                training_seconds_before_persistence=ready-collected, wall_seconds_before_persistence=ready-start))
            self.phase = 'persisting'
            point('before_checkpoint')
            self.check_identity()
            sha = self.save(path, check=lambda: point('boundary_validation'))
            self.phase = 'idle'
            finished = time.perf_counter()
            return dict(summary, boundary_sha256=sha, resources=dict(collection_seconds=collected-start,
                        training_and_persistence_seconds=finished-collected, wall_seconds=finished-start))
        except BaseException:
            self.failed, self.phase = True, 'failed'
            raise


def load_experimental(path, *, expected_sha256=None, expected_config=None, restore_global_rng=True):
    """Strict only. Inspect/fork conversion is separate; no relaxed continuation flags."""
    if expected_sha256 is not None and file_sha256(path) != expected_sha256:
        raise ValueError('Experimental boundary file hash differs')
    p = torch.load(path, map_location='cpu', weights_only=True)
    keys = {'contract', 'config', 'schedule', 'lineage', 'arm', 'parent', 'counters', 'model', 'model_hash',
            'optimizer', 'replay', 'sampling', 'augmentation', 'global_rng', 'history', 'resources', 'runtime', 'source', 'integrity'}
    if not isinstance(p, dict) or set(p) != keys or p['contract'] != CONTRACT:
        raise ValueError('Experimental boundary schema differs')
    if digest({k: v for k, v in p.items() if k != 'integrity'}) != p['integrity']:
        raise ValueError('Experimental boundary integrity mismatch')
    if p['source']['execution'] != PROCESS_SOURCE or execution_source_identity() != PROCESS_SOURCE:
        raise ValueError('Experimental source identity differs')
    if p['runtime'] != runtime_identity():
        raise ValueError('Experimental runtime identity differs')
    config = ExperimentalConfig.from_dict(p['config'])
    if expected_config is not None and config != expected_config:
        raise ValueError('Declared experimental configuration differs')
    runner = ExperimentalRunner.__new__(ExperimentalRunner)
    runner.config = config
    runner._install(model_factory(config.architecture).restore(p['model']),
                    TemperatureSchedule.from_dict(p['schedule']), p['lineage'], p['arm'])
    counters = p['counters']
    if set(counters) != {'completed_generations', 'total_games', 'total_positions', 'trainer_steps'} or any(
            type(v) is not int or v < 0 for v in counters.values()):
        raise ValueError('Invalid training counters')
    for k, v in counters.items():
        if k != 'trainer_steps':
            setattr(runner, k, v)
    if runner.completed_generations > config.training.max_generations:
        raise ValueError('Generation counter exceeds declaration')
    runner.factory.restore_optimizer(runner.trainer, p['optimizer'], counters['trainer_steps'])
    runner.parent = deepcopy(p['parent'])
    stored = p['replay']['generations']
    if len(stored) > config.training.replay_generations:
        raise ValueError('Replay window exceeds declaration')
    for entry in stored:
        if set(entry) != {'generation', 'schedule', 'simulations', 'protocol', 'games_count', 'games'}:
            raise ValueError('Missing generation schedule/protocol metadata')
        g = entry['generation']
        schedule = TemperatureSchedule.from_dict(entry['schedule'])
        if type(entry['simulations']) is not int or entry['simulations'] < 1:
            raise ValueError('Invalid generation visit budget')
        if entry['protocol'] == LEGACY_PROTOCOL:
            if (runner.parent is None or g > runner.parent['counters']['completed_generations']
                    or schedule.to_dict() != runner.parent['legacy_schedule']
                    or entry['simulations'] != runner.parent['config']['self_play_simulations']
                    or entry['games_count'] != runner.parent['config']['games_per_generation']):
                raise ValueError('Historical generation metadata differs from verified parent')
        elif entry['protocol'] != PROTOCOL or (runner.parent and g <= runner.parent['counters']['completed_generations']):
            raise ValueError('Generation RNG protocol differs')
        elif entry['simulations'] != config.training.self_play_simulations or entry['games_count'] != config.training.games_per_generation:
            raise ValueError('Experimental generation budget differs')
        games = tuple(rebuild_scheduled_game(g, game, schedule, entry['simulations']) for game in entry['games'])
        if type(entry['games_count']) is not int or len(games) != entry['games_count']:
            raise ValueError('Generation game count differs')
        runner.replay.add_generation(g, games)
        runner.metadata[g] = {k: deepcopy(v) for k, v in entry.items() if k not in ('generation', 'games')}
    expected_window = tuple(range(max(1, runner.completed_generations-config.training.replay_generations+1),
                                  runner.completed_generations+1))
    if runner.replay.generations != expected_window or runner.replay.digest() != p['replay']['digest']:
        raise ValueError('Replay identity/content digest differs')
    runner.history = deepcopy(p['history'])
    runner.resources = deepcopy(p['resources'])
    if [r['generation'] for r in runner.resources] != [h['generation'] for h in runner.history]:
        raise ValueError('Resource history identities differ')
    import math
    for record in runner.resources:
        if set(record) != {'generation', 'collection_seconds', 'training_seconds_before_persistence', 'wall_seconds_before_persistence'} or any(
                type(v) not in (float, int) or not math.isfinite(v) or v < 0 for k,v in record.items() if k != 'generation'):
            raise ValueError('Invalid resource accounting')
    parent_count = runner.parent['counters']['completed_generations'] if runner.parent else 0
    if [h['generation'] for h in runner.history] != list(range(parent_count+1, runner.completed_generations+1)):
        raise ValueError('Experimental history identities differ')
    base = runner.parent['counters'] if runner.parent else dict(total_games=0, total_positions=0, trainer_steps=0)
    if any(counters[k] != base[k] + sum(h[field] for h in runner.history) for k, field in (
            ('total_games', 'games'), ('total_positions', 'new_positions'), ('trainer_steps', 'updates'))):
        raise ValueError('History and training counters differ')
    cumulative_steps = base['trainer_steps']
    previous_hash = (runner.parent['model_hash'] if runner.parent else
                     runner.factory.hash(runner.factory.construct(config.training.seed)))
    for h in runner.history:
        cumulative_steps += config.optimizer_updates
        if h['trainer_steps'] != cumulative_steps or h['collection_weights_sha256'] != previous_hash:
            raise ValueError('History training/model chain differs')
        previous_hash = h['learner_weights_sha256']
        if h['generation'] in runner.metadata and h['new_positions'] != runner.replay.counts()[h['generation']]['positions']:
            raise ValueError('History positions disagree with retained games')
        if h['games'] != config.training.games_per_generation or h['updates'] != config.optimizer_updates or h['protocol'] != PROTOCOL:
            raise ValueError('History budget/protocol differs')
        TemperatureSchedule.from_dict(h['schedule'])
        if h['generation'] in runner.metadata and h['schedule'] != runner.metadata[h['generation']]['schedule']:
            raise ValueError('History/replay schedule differs')
    if runner.history and previous_hash != p['model_hash']:
        raise ValueError('Final history model hash differs')
    runner.sampling_rng.setstate(_python_rng_tuple(p['sampling']))
    a = p['augmentation']
    runner.augmenter.rng.setstate(_python_rng_tuple(a['rng']))
    if any(type(a[k]) is not int or a[k] < 0 for k in ('reflected', 'unreflected')) or (
            a['reflected']+a['unreflected'] != runner.trainer.steps*config.training.batch_size):
        raise ValueError('Augmentation counters differ')
    runner.augmenter.reflected, runner.augmenter.unreflected = a['reflected'], a['unreflected']
    runner.global_rng = deepcopy(p['global_rng'])
    if runner.factory.hash(runner.model) != p['model_hash']:
        raise ValueError('Model weight hash differs')
    # Validate RNGs even when caller elects not to install global state.
    previous = rng_payload()
    try:
        restore_rng(runner.global_rng)
    finally:
        restore_rng(previous)
    if restore_global_rng:
        restore_rng(runner.global_rng)
    return runner


def evaluate_boundary(path, evaluator, *, expected_sha256):
    """Read-only evaluation hook on an independent frozen model; RNG-neutral."""
    previous = rng_payload()
    try:
        runner = load_experimental(path, expected_sha256=expected_sha256, restore_global_rng=False)
        snapshot = runner.factory.freeze(runner.model)
        before = runner.factory.hash(snapshot)
        result = evaluator(snapshot, deepcopy(runner.counters()))
        if runner.factory.hash(snapshot) != before:
            raise ValueError('Evaluation mutated checkpoint weights')
        return result
    finally:
        restore_rng(previous)
