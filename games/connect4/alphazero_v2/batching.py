"""Opt-in independent-game inference batching; no campaign integration.

One suspended traversal per game. Generators yield only nonterminal neural
requests; terminal leaves back up exact values without a forward. Each tree is
strictly sequential and discarded after its move, like historical PUCT.
"""
from dataclasses import dataclass, field
import hashlib
import math
import random
import re
import time

from ..connect4 import Connect4
from ..agents.mcts_nn_agent import Node, backup, terminal_value
from .bitboard import BitboardState
from .config import action_temperature
from .data import capture_example, finalize_game
from .search import (PUCTSearch, V2RootNoise, V2SearchResult, SelfPlayDecision,
                     SelfPlayer, select_action)
from .selfplay import play_game

PROTOCOL = 'connect4-az-independent-games-v1'


def game_seed(base_seed, generation, index, domain):
    if (type(base_seed) is not int or type(generation) is not int or generation < 0
            or type(index) is not int or index < 0 or domain not in ('search', 'action', 'noise')):
        raise ValueError('Invalid per-game seed identity/domain')
    return int.from_bytes(hashlib.sha256(
        f'{PROTOCOL}:{base_seed}:{generation}:{index}:{domain}'.encode('ascii')).digest(), 'big')


@dataclass
class GameStreams:
    search: random.Random
    action: random.Random
    noise: V2RootNoise

    def state(self):
        return dict(search=self.search.getstate(), action=self.action.getstate(), noise=self.noise.get_state())


def game_streams(config, generation, index):
    seeds = {d: game_seed(config.seed, generation, index, d) for d in ('search', 'action', 'noise')}
    return GameStreams(random.Random(seeds['search']), random.Random(seeds['action']),
                       V2RootNoise(config.root_noise_epsilon, config.root_dirichlet_alpha, seed=seeds['noise']))


def game_ids(indices):
    ids = tuple(indices)
    if any(type(i) is not int or i < 0 for i in ids) or len(set(ids)) != len(ids):
        raise ValueError('Expected distinct nonnegative integer game IDs')
    return ids


@dataclass(frozen=True)
class TemperatureSchedule:
    """Explicit piecewise-constant execution temperature by zero-based pre-move ply.

    ``segments`` are (first ply, temperature) steps: the first starts at ply 0,
    starts strictly increase below 42, and adjacent temperatures differ so each
    schedule has one representation. Only the executed action is affected;
    policy targets stay normalized raw visits and no extra RNG draw occurs.
    """
    name: str
    segments: tuple

    def __post_init__(self):
        if type(self.name) is not str or not re.fullmatch('[a-z0-9_]+', self.name):
            raise ValueError('Schedule name must be lowercase letters, digits and underscores')
        if type(self.segments) not in (tuple, list) or not self.segments:
            raise ValueError('Schedule needs at least one (first ply, temperature) segment')
        segments = []
        for segment in self.segments:
            if type(segment) not in (tuple, list) or len(segment) != 2:
                raise ValueError('Segments are (first ply, temperature) pairs')
            start, temperature = segment
            if type(start) is not int or not 0 <= start < 42:
                raise ValueError('Segment starts must be integer plies 0..41')
            if (isinstance(temperature, bool) or type(temperature) not in (int, float)
                    or not math.isfinite(temperature) or temperature < 0):
                raise ValueError('Temperatures must be finite and nonnegative')
            segments.append((start, float(temperature)))
        if segments[0][0] != 0 or any(a[0] >= b[0] or a[1] == b[1] for a, b in zip(segments, segments[1:])):
            raise ValueError('Segments start at ply 0, strictly increase and change temperature')
        object.__setattr__(self, 'segments', tuple(segments))

    def temperature(self, ply):
        if type(ply) is not int or not 0 <= ply < 42:
            raise ValueError('ply must be an integer pre-move occupancy 0..41')
        return next(t for start, t in reversed(self.segments) if start <= ply)

    def to_dict(self):
        return dict(name=self.name, segments=[list(s) for s in self.segments])

    @classmethod
    def from_dict(cls, data):
        if (not isinstance(data, dict) or set(data) != {'name', 'segments'}
                or type(data['segments']) not in (tuple, list)):
            raise ValueError('Schedule fields do not match TemperatureSchedule')
        return cls(data['name'], tuple(tuple(s) if type(s) is list else s for s in data['segments']))


@dataclass
class CollectionStats:
    """Optional diagnostics; observer retains trees only if explicitly requested."""
    forward_calls: int = 0
    neural_positions: int = 0
    batch_sizes: dict = field(default_factory=dict)
    inference_seconds: float = 0.0
    game_work_seconds: float = 0.0
    collection_seconds: float = 0.0
    simulations: int = 0
    terminal_leaves: int = 0
    plies: int = 0
    completion_seconds: dict = field(default_factory=dict)
    active_seconds: dict = field(default_factory=dict)
    rng_states: dict = field(default_factory=dict)
    max_active: int = 0


def _expand(node, prediction):
    for move in node.game_state.get_valid_moves():
        node.children[move] = Node(node.game_state.drop(move), parent=node, move=move,
                                   prior_p=prediction.policy[move])


def search_requests(state, simulations, streams, *, check=None):
    """Yield root then nonterminal leaves; send V2Prediction; return V2SearchResult.

    Exposes resumable search only to the optional scheduler. Selection/backup,
    move ordering and root noise operations match PUCTSearch exactly.
    """
    if type(simulations) is not int or simulations < 1:
        raise ValueError('simulations must be a positive integer')
    if not isinstance(state, BitboardState) or state.is_game_over():
        raise ValueError('Search requires a nonterminal bitboard')
    root = Node(state)
    prediction = yield state
    _expand(root, prediction)
    mixed, noise = streams.noise.mix(prediction.policy, root.children)
    search_prior = tuple(float(p) for p in mixed)
    for action, child in root.children.items():
        child.prior_p = search_prior[action]
    max_depth = depth_sum = terminal_leaves = 0
    for _ in range(simulations):
        if check is not None:
            check()
        node, depth = root, 0
        while node.children:
            node = PUCTSearch._select_child(node, streams.search)
            depth += 1
        max_depth, depth_sum = max(max_depth, depth), depth_sum + depth
        if node.game_state.is_game_over():
            terminal_leaves += 1
            backup(node, terminal_value(node.game_state))
        else:
            leaf_prediction = yield node.game_state
            _expand(node, leaf_prediction)
            backup(node, leaf_prediction.value)
    visits = tuple(root.children[a].visits if a in root.children else 0 for a in range(7))
    if sum(visits) != simulations or root.visits != simulations:
        raise RuntimeError('Root-edge visit accounting failed')
    return V2SearchResult(visits, simulations, prediction.logits, prediction.value,
                          prediction.policy, search_prior,
                          None if noise is None else tuple(float(x) for x in noise),
                          None if noise is None else streams.noise.draws, root,
                          max_depth, depth_sum, terminal_leaves)


def _game_requests(config, generation, index, streams, stats, *, check, observer, schedule=None):
    game = Connect4()
    state = BitboardState.from_game(game)
    moves, pending = [], []
    while not state.is_game_over():
        started = time.perf_counter()
        result = yield from search_requests(state, config.self_play_simulations, streams, check=check)
        temperature = (action_temperature(len(moves), config.exploratory_plies) if schedule is None
                       else schedule.temperature(len(moves)))
        action = select_action(result.visits, temperature, streams.action)
        decision = SelfPlayDecision(len(moves), result, action)
        pending.append(capture_example(game, decision))
        state = state.drop(action.move)
        if not game.make_move(action.move):
            raise RuntimeError('Engine rejected a searched move')
        if state != BitboardState.from_game(game):
            raise RuntimeError('Engine/bitboard game state mismatch')
        if observer is not None:
            observer(index, decision, time.perf_counter() - started)
        moves.append(action.move)
        stats.simulations += result.simulations
        stats.terminal_leaves += result.terminal_leaves
        stats.plies += 1
    return finalize_game(generation, index, moves, pending)


def collect_batched(inference, config, generation, indices, *, batch_size,
                    check=None, observer=None, on_game=None, stats=None, temperature_schedule=None):
    """Return complete games in increasing ID order, refill immediately on finish.

    ``temperature_schedule`` (Phase 4D.4D, experimental) replaces the action
    temperature by ply; None keeps ``config.exploratory_plies`` exactly.

    On ANY exception no dataset is returned; on_game may already have observed
    explicitly complete games. Check before every batch and simulation bounds
    cooperative cancellation to one forward/traversal/capture/finalization.
    No threads, shared trees, virtual loss, queue or RNG draws on waits.
    """
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('batch_size must be a positive integer')
    if temperature_schedule is not None and not isinstance(temperature_schedule, TemperatureSchedule):
        raise ValueError('temperature_schedule must be a TemperatureSchedule or None')
    ids = game_ids(indices)
    game_seed(config.seed, generation, 0, 'search')  # validate even an empty collection
    stats = CollectionStats() if stats is None else stats
    pending = iter(ids)
    active, completed = [], {}
    started = time.perf_counter()

    def refill():
        while len(active) < batch_size:
            index = next(pending, None)
            if index is None:
                break
            streams = game_streams(config, generation, index)
            gen = _game_requests(config, generation, index, streams, stats, check=check, observer=observer,
                                 schedule=temperature_schedule)
            active.append([index, gen, next(gen), streams, time.perf_counter()])
            stats.max_active = max(stats.max_active, len(active))

    try:
        if check is not None:
            check()
        refill()
        while active:
            if check is not None:
                check()
            tick = time.perf_counter()
            predictions = inference.predict_batch([entry[2] for entry in active])
            stats.inference_seconds += time.perf_counter() - tick
            n = len(active)
            if len(predictions) != n:
                raise RuntimeError('Inference returned the wrong batch length')
            stats.forward_calls += 1
            stats.neural_positions += n
            stats.batch_sizes[n] = stats.batch_sizes.get(n, 0) + 1
            continuing = []
            for entry, prediction in zip(active, predictions):
                index, gen, _, streams, entered = entry
                advanced = time.perf_counter()
                try:
                    entry[2] = gen.send(prediction)
                    continuing.append(entry)
                except StopIteration as done:
                    completed[index] = done.value
                    stats.completion_seconds[index] = time.perf_counter() - started
                    stats.active_seconds[index] = time.perf_counter() - entered
                    stats.rng_states[index] = streams.state()
                    if on_game is not None:
                        on_game(done.value)
                finally:
                    stats.game_work_seconds += time.perf_counter() - advanced
            active = continuing
            refill()
        if len(completed) != len(ids):
            raise RuntimeError('Incomplete collection')
        stats.collection_seconds = time.perf_counter() - started
        return tuple(completed[index] for index in sorted(ids))
    finally:
        for entry in active:
            entry[1].close()


def collect_serial_reference(inference, config, generation, indices, *, check=None,
                             observer=None, on_game=None, stats=None):
    """Unmodified Phase 4D.4B search with the NEW independent per-game RNG scheme."""
    ids = game_ids(indices)
    game_seed(config.seed, generation, 0, 'search')
    stats = CollectionStats() if stats is None else stats
    started = time.perf_counter()
    collected = []
    for index in ids:
        streams = game_streams(config, generation, index)
        player = SelfPlayer(inference, config, search_rng=streams.search, action_rng=streams.action,
                            root_noise=streams.noise, backend='bitboard')
        entered = time.perf_counter()
        def observed(i, decision, seconds):
            stats.simulations += decision.search.simulations
            stats.terminal_leaves += decision.search.terminal_leaves
            stats.plies += 1
            if observer is not None:
                observer(i, decision, seconds)
        collected.append(play_game(player, generation, index, check=check, observer=observed))
        stats.max_active = max(stats.max_active, 1)
        stats.completion_seconds[index] = time.perf_counter() - started
        stats.active_seconds[index] = time.perf_counter() - entered
        stats.rng_states[index] = streams.state()
        if on_game is not None:
            on_game(collected[-1])
    return tuple(sorted(collected, key=lambda game: game.index))
