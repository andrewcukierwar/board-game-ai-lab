"""Measured cost benchmarks: neural inference, pool-eight collection and batch-128 training.

Every number is a wall-clock or resident-memory measurement on this machine
with one thread and deterministic algorithms; nothing is derived from
parameter or multiply-accumulate counts. Memory comes from fresh child
processes, one architecture each, so one model's peak cannot hide another's.
Weights are the seed-determined initializations: no trained ResNet exists
before the fixed-data experiment, and dense convolution cost does not depend
on the weight values.
"""
import argparse
from copy import deepcopy
import json
import random
import statistics
import subprocess
import sys
import time

import torch

from .common import IDENTITY, NAMES, PYTHON, REPO, cpu_seconds, peak_rss_mib, runtime, text_sha, thread_environment

BENCH_SEED = 20261012
INFERENCE_BATCHES = (1, 8, 16, 32)
INFERENCE_CALLS = {1: 400, 8: 250, 16: 160, 32: 100}
WARMUP = 20


def child(arguments, timeout):
    """Run one benchmark stage in a new interpreter and return its JSON line."""
    completed = subprocess.run([PYTHON, '-B', '-m', 'research.alphazero_resnet_benchmark.bench', *map(str, arguments)],
                               cwd=REPO, env=thread_environment(), capture_output=True, text=True, timeout=timeout)
    if completed.returncode:
        raise RuntimeError(f'Benchmark child failed: {arguments}: {completed.stderr[-3000:]}')
    return json.loads(completed.stdout.strip().splitlines()[-1])


def model(name, seed=BENCH_SEED):
    from games.connect4.alphazero_v2.experimental_models import model_factory
    factory = model_factory(IDENTITY[name])
    return factory, factory.construct(seed)


def random_positions(count, seed=BENCH_SEED):
    """Seeded random-play nonterminal positions (bitboards) with a spread of plies; no replay data."""
    from games.connect4.connect4 import Connect4
    from games.connect4.alphazero_v2.bitboard import BitboardState
    rng = random.Random(seed)
    states = []
    while len(states) < count:
        game = Connect4()
        while not game.is_game_over() and len(states) < count:
            states.append(BitboardState.from_game(game))
            game.make_move(rng.choice(game.get_valid_moves()))
    return states


def encode(states):
    from games.connect4.alphazero_v2.network import encode_bitboard_current_player
    return torch.cat([encode_bitboard_current_player(s) for s in states])


def summary(seconds, batch):
    ordered = sorted(seconds)
    mean = statistics.fmean(ordered)
    return dict(calls=len(ordered), mean_ms=mean * 1e3, median_ms=statistics.median(ordered) * 1e3,
                p95_ms=ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))] * 1e3, min_ms=ordered[0] * 1e3,
                positions_per_second=batch / mean)


def time_calls(function, calls):
    for _ in range(WARMUP):
        function()
    seconds = []
    for _ in range(calls):
        tick = time.perf_counter()
        function()
        seconds.append(time.perf_counter() - tick)
    return seconds


def inference_timing(repetitions, scale=1.0, check=None):
    """Interleaved architectures; raw forward on a prepared tensor and the full predict_batch path."""
    states = random_positions(4096)
    rows = []
    built = {name: model(name) for name in NAMES}
    for repetition in range(repetitions):
        order = NAMES[repetition % 3:] + NAMES[:repetition % 3]
        for batch in INFERENCE_BATCHES:
            calls = max(10, int(INFERENCE_CALLS[batch] * scale))
            offset = (repetition * 997) % (len(states) - batch)
            chunk = states[offset:offset + batch]
            tensor = encode(chunk)
            for name in order:
                if check:
                    check()
                factory, net = built[name]
                inference = factory.inference(net)

                def raw():
                    with torch.inference_mode():
                        net(tensor)
                rows.append(dict(architecture=name, batch=batch, repetition=repetition, path='forward',
                                 **summary(time_calls(raw, calls), batch)))
                rows.append(dict(architecture=name, batch=batch, repetition=repetition, path='predict_batch',
                                 **summary(time_calls(lambda: inference.predict_batch(chunk), calls), batch)))
    return dict(positions_sha256=text_sha([list(s.pieces) for s in states]), rows=rows,
                weights={name: built[name][0].hash(built[name][1]) for name in NAMES})


def inference_memory(name, batch, calls=200):
    """Peak resident memory of one architecture at one batch size, in a process that does nothing else."""
    baseline = peak_rss_mib()
    factory, net = model(name)
    inference = factory.inference(net)
    loaded = peak_rss_mib()
    chunk = random_positions(batch)
    for _ in range(calls):
        inference.predict_batch(chunk)
    return dict(architecture=name, batch=batch, calls=calls, imported_peak_rss_mib=baseline,
                model_peak_rss_mib=loaded, peak_rss_mib=peak_rss_mib(),
                state_megabytes=sum(t.numel() * t.element_size() for t in net.state_dict().values()) / 2 ** 20)


def collection(name, first, count, simulations=256, seed=BENCH_SEED, max_forwards=None):
    """Pool-eight self-play on matched game identities (same seed, generation and game ids for every architecture)."""
    from games.connect4.alphazero_v2.batching import CollectionStats, collect_batched
    from games.connect4.alphazero_v2.config import V2Config
    factory, net = model(name, seed)
    inference = factory.inference(factory.freeze(net))
    config = V2Config(seed=seed, self_play_simulations=simulations)
    stats = CollectionStats()

    class Stop(Exception):
        pass

    def check():
        if max_forwards is not None and stats.forward_calls >= max_forwards:
            raise Stop
    cpu, tick, games, complete = cpu_seconds(), time.perf_counter(), (), True
    try:
        games = collect_batched(inference, config, 1, range(first, first + count), batch_size=8, stats=stats,
                                check=check if max_forwards is not None else None)
    except Stop:
        complete = False
    wall = time.perf_counter() - tick
    result = dict(architecture=name, complete=complete, game_ids=[first, first + count - 1], simulations_per_move=simulations,
                  wall_seconds=wall, cpu_seconds=cpu_seconds() - cpu, forward_calls=stats.forward_calls,
                  neural_positions=stats.neural_positions, inference_seconds=stats.inference_seconds,
                  game_work_seconds=stats.game_work_seconds, simulations=stats.simulations,
                  terminal_leaves=stats.terminal_leaves, plies=stats.plies,
                  batch_sizes={str(k): v for k, v in sorted(stats.batch_sizes.items())}, max_active=stats.max_active,
                  mean_batch=stats.neural_positions / max(1, stats.forward_calls), peak_rss_mib=peak_rss_mib(),
                  weights_sha256=factory.hash(net))
    if complete:
        result.update(games=len(games), game_lengths=[len(g.moves) for g in games],
                      winners=[g.winner for g in games], games_sha256=text_sha([list(g.moves) for g in games]),
                      collection_seconds=stats.collection_seconds)
    return result


def training_examples(count, seed=BENCH_SEED):
    from .validate import synthetic_examples
    return synthetic_examples(count, seed)


def without_batch_norm(net):
    """Same network with every batch norm replaced by the identity: the measured cost of normalization."""
    stripped = deepcopy(net)
    for parent in list(stripped.modules()):
        for child_name, module in list(parent.named_children()):
            if isinstance(module, torch.nn.BatchNorm2d):
                setattr(parent, child_name, torch.nn.Identity())
    return stripped


def staged_step(net, optimizer, parameters, max_norm, batch):
    """The trainer's update, statement for statement, with a clock between stages."""
    from games.connect4.alphazero_v2.training import batch_tensors, training_loss
    marks = [time.perf_counter()]
    states, policies, outcomes = batch_tensors(batch)
    marks.append(time.perf_counter())
    net.train()
    optimizer.zero_grad(set_to_none=True)
    outputs = net(states)
    marks.append(time.perf_counter())
    combined, _, _ = training_loss(*outputs, policies, outcomes)
    marks.append(time.perf_counter())
    combined.backward()
    marks.append(time.perf_counter())
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
        raise ValueError('Missing or nonfinite training gradients')
    torch.nn.utils.clip_grad_norm_(parameters, max_norm, error_if_nonfinite=True)
    torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(p.grad) for p in parameters]))
    marks.append(time.perf_counter())
    optimizer.step()
    marks.append(time.perf_counter())
    optimizer.zero_grad(set_to_none=True)
    net.eval()
    marks.append(time.perf_counter())
    return [b - a for a, b in zip(marks, marks[1:])]


STAGES = ('batch_tensors', 'forward', 'loss', 'backward', 'gradient_checks_and_clip', 'optimizer_step', 'reset')


def training(name, updates, strip=False):
    """Batch-128 update cost: the real trainer.step, a staged replica, and (ResNets) the batch-norm-free variant."""
    from games.connect4.alphazero_v2.config import V2Config
    from games.connect4.alphazero_v2.training import batch_tensors
    baseline = peak_rss_mib()
    factory, net = model(name)
    config = V2Config(seed=BENCH_SEED)
    examples = training_examples(128 * 16)
    batches = [examples[i * 128:(i + 1) * 128] for i in range(16)]
    result = dict(architecture=name, updates=updates, batch=128, imported_peak_rss_mib=baseline)
    if strip:
        # Timing only: the stripped network is not a registered architecture and is never trained for results.
        net = without_batch_norm(net)
        optimizer = torch.optim.AdamW(net.parameters(), lr=config.learning_rate, betas=config.adam_betas,
                                      eps=config.adam_eps, weight_decay=config.weight_decay)
    else:
        trainer = factory.trainer(net, config)
        for i in range(5):
            trainer.step(batches[i % 16])
        seconds = []
        for i in range(updates):
            tick = time.perf_counter()
            trainer.step(batches[i % 16])
            seconds.append(time.perf_counter() - tick)
        result['trainer_step'] = summary(seconds, 128)
        result['trainer_step']['examples_per_second'] = result['trainer_step'].pop('positions_per_second')
        optimizer = trainer.optimizer
    parameters = [p for p in net.parameters() if p.requires_grad]
    for i in range(5):
        staged_step(net, optimizer, parameters, config.max_grad_norm, batches[i % 16])
    stages = [staged_step(net, optimizer, parameters, config.max_grad_norm, batches[i % 16]) for i in range(updates)]
    result['staged'] = {stage: statistics.fmean(row[k] for row in stages) * 1e3 for k, stage in enumerate(STAGES)}
    result['staged_total_ms'] = statistics.fmean(sum(row) for row in stages) * 1e3
    # Forward-only cost of one 128 batch in each mode (train mode computes batch statistics).
    states = batch_tensors(batches[0])[0]
    net.eval()

    def evaluate():
        with torch.no_grad():
            net(states)
    result['eval_forward_ms'] = statistics.fmean(time_calls(evaluate, updates)) * 1e3
    probe = deepcopy(net).train()

    def train_forward():
        with torch.no_grad():
            probe(states)
    result['train_forward_ms'] = statistics.fmean(time_calls(train_forward, updates)) * 1e3
    result['peak_rss_mib'] = peak_rss_mib()
    result['stripped_batch_norm'] = strip
    return result


def evaluation_throughput(name, positions=2048, batch=512):
    """Eval-mode scoring speed at the evaluation batch size (used only to budget the experiment)."""
    _, net = model(name)
    tensor = encode(random_positions(positions))
    tick = time.perf_counter()
    with torch.inference_mode():
        for i in range(0, positions, batch):
            net(tensor[i:i + batch])
    return dict(architecture=name, positions_per_second=positions / (time.perf_counter() - tick))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    memory = commands.add_parser('inference-memory')
    memory.add_argument('name')
    memory.add_argument('batch', type=int)
    collect = commands.add_parser('collection')
    collect.add_argument('name')
    collect.add_argument('first', type=int)
    collect.add_argument('count', type=int)
    collect.add_argument('--max-forwards', type=int)
    train = commands.add_parser('training')
    train.add_argument('name')
    train.add_argument('updates', type=int)
    train.add_argument('--strip', action='store_true')
    args = parser.parse_args()
    identity = runtime()
    if args.command == 'inference-memory':
        result = inference_memory(args.name, args.batch)
    elif args.command == 'collection':
        result = collection(args.name, args.first, args.count, max_forwards=args.max_forwards)
    else:
        result = training(args.name, args.updates, strip=args.strip)
    result['runtime_sha256'] = text_sha(identity)
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
