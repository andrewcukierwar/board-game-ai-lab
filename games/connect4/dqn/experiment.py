"""Explicit offline CPU self-play; inert on import. Run with python -m ...experiment.

Bounds are cooperative, checked before collection and optimizer steps. An
in-flight Python/PyTorch operation may finish just after a wall-time deadline;
no new work starts after it. SIGINT/SIGTERM request a stop at a safe boundary.
Checkpoints are inference candidates, never complete training resume state.
"""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import signal
import subprocess
import time

import torch

from ..connect4 import Connect4
from .checkpoint import load_checkpoint, save_checkpoint
from .diagnostics import evaluate_random, predictions
from .encoding import encode_game
from .network import checked_q
from .training import DQNConfig, DQNTrainer, collect_transition, positive_int


@dataclass(frozen=True)
class Bounds:
    max_games: int = 200
    max_plies: int = 8400
    max_updates: int = 8400
    max_seconds: float = 600.0

    def __post_init__(self):
        for name in ('max_games', 'max_plies', 'max_updates'):
            positive_int(name, getattr(self, name))
        if (type(self.max_seconds) not in (int, float)
                or not math.isfinite(self.max_seconds) or self.max_seconds <= 0):
            raise ValueError('max_seconds must be positive and finite')


def finite_training_state(trainer):
    for network in (trainer.online, trainer.target):
        for value in (*network.parameters(), *network.buffers()):
            if value.device.type != 'cpu' or not torch.isfinite(value).all().item():
                raise ValueError('Nonfinite or non-CPU model state')
    for parameter in trainer.online.parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all().item():
            raise ValueError('Nonfinite gradients')


def run_training(trainer, bounds, *, clock=time.monotonic,
                 should_stop=lambda: False, emit=lambda event: None,
                 game_factory=Connect4):
    """One persistent trainer, one collected actor-relative ply, at most one update.

    Injectable clock/stop/game fixture makes limit and interruption tests cheap.
    Counts describe completed operations even if a later invariant fails.
    """
    if trainer.updates or len(trainer.memory):
        raise ValueError('Experiments must start with a fresh trainer')
    start = clock()
    report = dict(status='running', stop_reason=None, completed_games=0,
                  started_games=0, plies=0, updates=0, replay_size=0,
                  epsilon_start=trainer.epsilon, epsilon_final=trainer.epsilon,
                  wins_x=0, wins_o=0, draws=0, episodes=[], losses=[],
                  target_sync_updates=[], partial_episode=None)
    episode = None

    def stop_reason():
        if should_stop():
            return 'interrupted'
        if report['completed_games'] >= bounds.max_games:
            return 'max_games'
        if report['plies'] >= bounds.max_plies:
            return 'max_plies'
        if trainer.updates >= bounds.max_updates:
            return 'max_updates'
        if clock() - start >= bounds.max_seconds:
            return 'max_seconds'
        return None

    try:
        finite_training_state(trainer)
        while True:
            reason = stop_reason()
            if reason:
                report['stop_reason'] = reason
                break
            if episode is None:
                game = game_factory()
                if game.is_game_over():
                    raise ValueError('Episode must start nonterminal')
                report['started_games'] += 1
                episode = dict(index=report['started_games'], moves=[],
                               initial_state=list(encode_game(game)),
                               epsilon_start=trainer.epsilon, update_start=trainer.updates,
                               loss_start=len(report['losses']), start_seconds=clock() - start)
            # Check Q even on exploratory moves, which bypass greedy inference.
            with torch.inference_mode():
                q = checked_q(trainer.online, torch.tensor([encode_game(game)], dtype=torch.float32))
            action = trainer.choose_move(game)
            reason = stop_reason()
            if reason:
                report['stop_reason'] = reason
                break
            transition = collect_transition(game, action)
            report['plies'] += 1
            episode['moves'].append(action)
            trainer.memory.append(transition)
            previous_updates = trainer.updates
            loss = None
            # At the ply/deadline boundary, record the collected ply but skip
            # optimization. The first limit reached stops further training work.
            if stop_reason() is None:
                loss = trainer.optimize()
                if trainer.updates - previous_updates not in (0, 1):
                    raise ValueError('More than one optimizer update per ply')
                if loss is not None:
                    if not math.isfinite(loss):
                        raise ValueError('Nonfinite loss')
                    report['losses'].append(loss)
                    if trainer.updates % trainer.config.target_sync_interval == 0:
                        report['target_sync_updates'].append(trainer.updates)
                finite_training_state(trainer)
            grad_max = max((p.grad.abs().max().item() for p in trainer.online.parameters()
                            if p.grad is not None), default=0.0)
            emit(dict(event='ply', ply=report['plies'], episode=episode['index'],
                      actor=transition.actor, action=action, reward=transition.reward,
                      terminal=transition.done, updates=trainer.updates, loss=loss,
                      epsilon=trainer.epsilon, replay_size=len(trainer.memory),
                      q_min=float(q.min().item()), q_max=float(q.max().item()),
                      gradient_abs_max=grad_max,
                      target_sync=trainer.updates > previous_updates and
                      trainer.updates % trainer.config.target_sync_interval == 0,
                      elapsed_seconds=clock() - start))
            if transition.done:
                winner = game.check_winner()
                report['completed_games'] += 1
                report['draws' if winner == -1 else 'wins_x' if winner == 0 else 'wins_o'] += 1
                losses = report['losses'][episode.pop('loss_start'):]
                episode.update(winner=winner, plies=len(episode['moves']),
                               epsilon_end=trainer.epsilon, update_end=trainer.updates,
                               loss_mean=sum(losses) / len(losses) if losses else None,
                               end_seconds=clock() - start)
                report['episodes'].append(episode)
                emit(dict(event='episode', **episode))
                episode = None
        report['status'] = 'interrupted' if report['stop_reason'] == 'interrupted' else 'bounded_stop'
    except Exception as exc:
        report.update(status='invariant_failure', stop_reason='error',
                      error=f'{type(exc).__name__}: {exc}')
    finally:
        report.update(updates=trainer.updates, replay_size=len(trainer.memory),
                      epsilon_final=trainer.epsilon, elapsed_seconds=clock() - start,
                      partial_episode=episode)
    return report


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_identity(root, output):
    """Preserve dirty tracked diff and untracked source bytes, not just HEAD."""
    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args])

    patch = git('diff', '--binary', 'HEAD')
    (output / 'source.patch').write_bytes(patch)
    untracked = {}
    for raw in git('ls-files', '--others', '--exclude-standard', '-z').split(b'\0'):
        if not raw:
            continue
        name = raw.decode()
        source = root / name
        if source.is_symlink():
            raise ValueError('Untracked source symlinks cannot be archived')
        destination = output / 'untracked-source' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
        untracked[name] = sha256(destination)
    identity = dict(commit=git('rev-parse', 'HEAD').decode().strip(),
                    branch=git('branch', '--show-current').decode().strip(),
                    status=git('status', '--porcelain=v1').decode(),
                    tracked_diff_sha256=hashlib.sha256(patch).hexdigest(),
                    untracked_sha256=untracked)
    identity['dirty'] = bool(identity['status'])
    identity['identity_sha256'] = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return identity


def save_candidate(output, trainer, report):
    finite_training_state(trainer)
    path = output / 'candidate.pt'
    if path.exists():
        raise FileExistsError('Refusing to replace an existing candidate')
    save_checkpoint(path, trainer.online, training_metadata=report)
    reloaded, metadata = load_checkpoint(path)
    # Exact tensors plus all probe Q-values/actions; inference-only contract.
    if metadata != report or any(not torch.equal(value, reloaded.state_dict()[key])
                                 for key, value in trainer.online.state_dict().items()):
        raise ValueError('Reloaded checkpoint differs from saved model/metadata')
    if predictions(reloaded) != report['diagnostics']['final']:
        raise ValueError('Reloaded checkpoint predictions differ')
    return dict(path=str(path.resolve()), sha256=sha256(path),
                reload_exact=True, resumable=False)


def run_experiment(output, config, bounds, *, threads=1, interop_threads=1,
                   evaluation_games=24, evaluation_seconds=120.0,
                   should_stop=lambda: False):
    """One invocation, one training run. Never retry a failed or poor run."""
    positive_int('threads', threads)
    positive_int('interop_threads', interop_threads)
    if evaluation_games not in (0, 4, 8, 12, 16, 20, 24):
        raise ValueError('Evaluation games must be a multiple of four in 0..24')
    if not math.isfinite(evaluation_seconds) or not 0 < evaluation_seconds <= 120:
        raise ValueError('Evaluation time must be in (0, 120] seconds')
    output = Path(output).resolve()
    root = Path(__file__).resolve().parents[3]
    # Outputs inside the repository must be explicitly ignored, not source files.
    if output.is_relative_to(root):
        ignored = subprocess.run(['git', '-C', str(root), 'check-ignore', '-q', str(output)])
        if ignored.returncode:
            raise ValueError('Repository output directory must be Git-ignored')
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(interop_threads)
    torch.use_deterministic_algorithms(True)
    report = dict(schema_version=1, started_at_utc=datetime.now(timezone.utc).isoformat(),
                  config=asdict(config), bounds=asdict(bounds), source=source_identity(root, output),
                  runtime=dict(python=platform.python_version(), pytorch=str(torch.__version__),
                               platform=platform.platform(), machine=platform.machine(), device='cpu',
                               threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
                               deterministic_algorithms=torch.are_deterministic_algorithms_enabled()),
                  evaluation_config=dict(total_games=evaluation_games, total_seconds=evaluation_seconds,
                                         seed=config.seed, policy='greedy vs uniform legal Random',
                                         per_model_games=evaluation_games // 2),
                  checkpoint=None)
    write_json(output / 'configuration.json', report)
    trainer = DQNTrainer(config)
    experiment_start = time.monotonic()
    with (output / 'events.jsonl').open('x', buffering=1) as events:
        def emit(event):
            events.write(json.dumps(event, allow_nan=False) + '\n')
            if event['event'] == 'episode' and event['index'] % 25 == 0:
                print(f"Completed {event['index']} games; {event['update_end']} updates; "
                      f"epsilon={event['epsilon_end']:.6f}", flush=True)

        try:
            report['diagnostics'] = dict(initial=predictions(trainer.online))
            report['evaluation'] = dict(initial=evaluate_random(
                trainer.online, games=evaluation_games // 2, seed=config.seed,
                seconds=evaluation_seconds / 2, should_stop=should_stop))
            report['training'] = run_training(trainer, bounds, emit=emit, should_stop=should_stop)
            write_json(output / 'report.json', report)
            if report['training']['status'] != 'invariant_failure':
                report['diagnostics']['final'] = predictions(trainer.online)
                remaining = evaluation_seconds - report['evaluation']['initial']['elapsed_seconds']
                report['evaluation']['final'] = evaluate_random(
                    trainer.online, games=evaluation_games // 2, seed=config.seed,
                    seconds=max(0.000001, min(evaluation_seconds / 2, remaining)),
                    should_stop=should_stop)
                report['checkpoint'] = save_candidate(output, trainer, report)
        except Exception as exc:
            report['artifact_error'] = f'{type(exc).__name__}: {exc}'
        report['total_elapsed_seconds'] = time.monotonic() - experiment_start
        write_json(output / 'report.json', report)
    return report


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--output', type=Path, required=True, help='New external or Git-ignored directory')
    for name, default in asdict(Bounds()).items():
        cli.add_argument('--' + name.replace('_', '-'), type=type(default), default=default)
    defaults = DQNConfig(seed=42, epsilon_start=1.0, epsilon_min=0.10, epsilon_decay=0.9997)
    for name, default in asdict(defaults).items():
        cli.add_argument('--' + name.replace('_', '-'), type=type(default), default=default)
    cli.add_argument('--threads', type=int, default=1)
    cli.add_argument('--interop-threads', type=int, default=1)
    cli.add_argument('--evaluation-games', type=int, default=24)
    cli.add_argument('--evaluation-seconds', type=float, default=120.0)
    return cli


def main():
    args = vars(parser().parse_args())
    config = DQNConfig(**{name: args.pop(name) for name in DQNConfig.__dataclass_fields__})
    bounds = Bounds(**{name: args.pop(name) for name in Bounds.__dataclass_fields__})
    stopped = []
    def request_stop(signum, frame):
        stopped.append(signum)
    previous = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        report = run_experiment(config=config, bounds=bounds, should_stop=lambda: bool(stopped), **args)
        summary = {key: value for key, value in report.get('training', {}).items()
                   if key not in ('episodes', 'losses')}
        print(json.dumps(dict(training=summary, checkpoint=report.get('checkpoint'),
                              artifact_error=report.get('artifact_error')), indent=2))
        return int(report.get('artifact_error') is not None or
                   report.get('training', {}).get('status') == 'invariant_failure')
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    raise SystemExit(main())
