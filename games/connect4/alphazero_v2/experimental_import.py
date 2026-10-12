"""Read-only strict historical-code validation followed by an explicit fork.

The subprocess imports ONLY the pinned historical checkout; current code never
asks the historical loader to relax source/runtime verification. Transport is a
weights-only temporary file in a fresh directory, never an official artifact.
"""
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
import tempfile

import torch

from .artifacts import file_sha256
from .batching import TemperatureSchedule
from .experimental import (ExperimentalRunner, LEGACY_PROTOCOL, digest, validate_optimizer,
                           _python_rng_tuple)

HISTORICAL_COMMIT = '1894bc4d8f035cf738df8e3fc0ce234f1ce2c8be'
READER = '''
import sys, torch
from pathlib import Path
from games.connect4.alphazero_v2.provenance import configure_deterministic_runtime, REPO_ROOT
from games.connect4.alphazero_v2.generation import load_resume_boundary
if REPO_ROOT != Path(sys.argv[4]).resolve():
    raise ValueError('Historical import escaped checkout')
configure_deterministic_runtime(1)
r = load_resume_boundary(sys.argv[1], expected_sha256=sys.argv[2], restore_global_rng=False)
p = torch.load(sys.argv[1], map_location='cpu', weights_only=True)
if r.completed_generations != 20 or r.config.seed not in (42, 314159):
    raise ValueError('Expected a designated generation-20 seed')
if r.replay.generations != tuple(range(13,21)) or r.replay.games != 2048:
    raise ValueError('Historical replay identities/count differ')
if sum(x.numel() for x in r.model.parameters()) != 325896:
    raise ValueError('Historical model parameter count differs')
if r.state_sha256() != p['state_sha256']:
    raise ValueError('Historical state attestation differs')
torch.save(dict(original=p, verified_state=r.state_sha256(), verified_replay=r.replay.digest()), sys.argv[3])
'''


def read_historical(path, *, expected_sha256, historical_checkout):
    root, path = Path(historical_checkout).resolve(), Path(path).resolve()
    if file_sha256(path) != expected_sha256:
        raise ValueError('Historical parent file hash differs')
    revision = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], check=True,
                              capture_output=True, text=True).stdout.strip()
    if revision != HISTORICAL_COMMIT:
        raise ValueError('Historical reader checkout revision differs')
    # cwd and explicit sys.path insertion prevent PYTHONPATH/current checkout import.
    script = 'import sys; sys.path.insert(0, sys.argv[4]);\n' + READER
    with tempfile.TemporaryDirectory(prefix='az-historical-reader-') as temporary:
        output = Path(temporary)/'verified.pt'
        result = subprocess.run([sys.executable, '-B', '-c', script, str(path), expected_sha256,
                                 str(output), str(root)], cwd=root, capture_output=True, text=True, timeout=180)
        if result.returncode:
            raise ValueError(f'Strict historical reader rejected boundary: {result.stderr[-4000:]}')
        verified = torch.load(output, map_location='cpu', weights_only=True)
    if file_sha256(path) != expected_sha256:
        raise ValueError('Historical boundary changed during read')
    return verified


def fork_verified(verified, config, *, parent_sha256, schedule, lineage, arm):
    """Convert strict-reader output. Caller must obtain it through read_historical.

    Public split lets tests exercise tiny synthetic conversion fixtures; production
    fork_historical always invokes the isolated reader first.
    """
    p = verified['original']
    if verified['verified_state'] != p['state_sha256'] or verified['verified_replay'] != p['replay']['digest']:
        raise ValueError('Historical reader attestation differs')
    from .config import V2Config
    original = V2Config.from_dict(p['config'])
    # Only generation/game budgets are fork changes. Updates are separately declared.
    allowed = {'games_per_generation', 'max_generations'}
    if any(original.to_dict()[k] != config.training.to_dict()[k] for k in original.to_dict() if k not in allowed):
        raise ValueError('Warm start changed relevant training configuration')
    if config.training.max_generations <= p['counters']['completed_generations']:
        raise ValueError('Fork declaration leaves no new generations')
    runner = ExperimentalRunner.__new__(ExperimentalRunner)
    runner.config = config
    from .experimental_models import model_factory
    runner._install(model_factory(config.architecture).restore(p['model_state_dict']), schedule, lineage, arm)
    counters = p['counters']
    runner.factory.restore_optimizer(runner.trainer, p['optimizer_state_dict'], counters['trainer_steps'])
    for k in ('completed_generations', 'total_games', 'total_positions'):
        setattr(runner, k, counters[k])
    plies = original.exploratory_plies
    legacy = TemperatureSchedule('legacy_verified', ((0, 0),) if plies == 0 else
                                 ((0, 1),) if plies == 42 else ((0, 1), (plies, 0)))
    from .experimental import rebuild_scheduled_game
    for entry in p['replay']['generations']:
        games = tuple(rebuild_scheduled_game(entry['generation'], record, legacy,
                                            original.self_play_simulations) for record in entry['games'])
        runner.replay.add_generation(entry['generation'], games)
        runner.metadata[entry['generation']] = dict(schedule=legacy.to_dict(), protocol=LEGACY_PROTOCOL,
                         simulations=original.self_play_simulations, games_count=len(games))
    if runner.replay.digest() != verified['verified_replay']:
        raise ValueError('Converted replay differs from verified historical replay')
    runner.sampling_rng.setstate(_python_rng_tuple(p['rng']['sampling']))
    runner.augmenter.rng.setstate(_python_rng_tuple(p['rng']['augmentation']))
    runner.augmenter.reflected = p['augmentation_counts']['reflected']
    runner.augmenter.unreflected = p['augmentation_counts']['unreflected']
    runner.global_rng = {k: deepcopy(p['rng'][k]) for k in ('python', 'numpy', 'torch')}
    runner.parent = dict(file_sha256=parent_sha256, state_sha256=p['state_sha256'],
                         replay_digest=p['replay']['digest'], model_hash=p['learner_weights_sha256'],
                         source=deepcopy(p['source']), runtime=deepcopy(p['runtime']), config=p['config'],
                         counters=deepcopy(counters), legacy_schedule=legacy.to_dict(),
                         historical_rng=deepcopy(p['rng']), historical_history=deepcopy(p['history']),
                         conversion='isolated-pinned-historical-code-strict-reader-v1',
                         guarantee='validated fork; never official continuation',
                         new_protocol=config.rng_protocol,
                         changed_config={k: dict(before=original.to_dict()[k], after=config.training.to_dict()[k])
                                         for k in allowed if original.to_dict()[k] != config.training.to_dict()[k]},
                         optimizer_updates=config.optimizer_updates)
    if runner.factory.hash(runner.model) != p['learner_weights_sha256']:
        raise ValueError('Imported network hash differs')
    return runner


def fork_historical(path, config, *, expected_sha256, historical_checkout, schedule, lineage, arm):
    verified = read_historical(path, expected_sha256=expected_sha256, historical_checkout=historical_checkout)
    return fork_verified(verified, config, parent_sha256=expected_sha256, schedule=schedule, lineage=lineage, arm=arm)
