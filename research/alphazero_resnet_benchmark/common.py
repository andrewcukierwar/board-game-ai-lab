"""Shared paths, durable JSON, official-artifact preservation and the source freeze.

Everything here is read-only towards the official campaign, the historical
checkout and the Phase 4D.4G worktree. Output goes to the ignored
experiment-output/phase4d4h-b tree and to tracked results/ beside this file.
"""
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
HISTORICAL = REPO.parents[1]
OFFICIAL = HISTORICAL / 'experiment-output/phase4d3c-alphazero-v2-official-m4-20261008-110950'
PYTHON = '/Users/andrewcukierwar/.venvs/board-game-alphazero-m4/bin/python'
OUTPUT = REPO / 'experiment-output/phase4d4h-b'
RESULTS = ROOT / 'results'
THREADS = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')
SEEDS = (42, 314159)
# Trusted generation-20 resume boundaries (the hashes Phase 4D.4F/4D.4G forked from).
PARENTS = {42: '2866722b5e55726b75ae27d4bdb70e98d9b01b9ca503b0925fe9aa8b17d210b1',
           314159: '328022e5f4c45c11650b7fd2b76fbafbbe4734f1161a75d91b5d0b4cb274f9d6'}
DECLARATION_SHA256 = 'c8ab2227d4b2c707b3185da8695a991f0914faede88ecd5f1c7fc84fb017b716'
START_COMMIT = '7f5182daa10156e04a8b0f7727d9bec9cad39cf1'
SHORT = {'connect4-conv64-128-128-policy7-scalar-tanh-v2': 'cnn',
         'connect4-resnet3x64-bn-policy32-value32x64-scalar-tanh-v1': 'resnet3x64',
         'connect4-resnet5x64-bn-policy32-value32x64-scalar-tanh-v1': 'resnet5x64'}
ARCHITECTURES = tuple(SHORT)
NAMES = tuple(SHORT.values())
IDENTITY = {short: identity for identity, short in SHORT.items()}
PARAMETERS = {'cnn': 325_896, 'resnet3x64': 322_440, 'resnet5x64': 470_408}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def text_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    """Atomic, fsynced JSON; NaN/inf are rejected so every stored number is real."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, indent=1, sort_keys=True, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def peak_rss_mib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2 if sys.platform == 'darwin' else 1024)


def cpu_seconds():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


def inventory():
    return {p.relative_to(OFFICIAL).as_posix(): sha(p) for p in sorted(OFFICIAL.rglob('*')) if p.is_file()}


def preserve():
    """All 150 official files byte-identical to the Phase 4D.4F inventory."""
    actual = inventory()
    expected = read(REPO / 'research/alphazero_training/results/official-before.json')
    if len(actual) != 150 or actual != expected or actual['declaration.json'] != DECLARATION_SHA256:
        raise ValueError('Official inventory changed')
    return actual


def runtime():
    """One-thread deterministic CPU runtime; the launching shell must already set the thread variables."""
    from games.connect4.alphazero_v2.provenance import configure_deterministic_runtime, unavailable_runtime_fields
    identity = configure_deterministic_runtime(1)
    if any(v != '1' for v in identity['thread_environment'].values()) or unavailable_runtime_fields(identity):
        raise ValueError('Benchmark requires all five library thread controls set to 1')
    return identity


def thread_environment():
    return dict(os.environ, **{name: '1' for name in THREADS})


FROZEN = ('__init__.py', 'common.py', 'dataset.py', 'metrics.py', 'bench.py', 'stages.py', 'learn.py', 'manage.py',
          'validate.py', 'pilot.py', 'plan.py', 'analyze.py', 'DESIGN.md', 'results/dataset-freeze.json',
          'results/design.json')


def source_map():
    """Enforced package execution closure plus the frozen benchmark sources, design and dataset membership."""
    from games.connect4.alphazero_v2.provenance import execution_source_identity
    files = execution_source_identity()['files'].copy()
    for name in FROZEN:
        files[(ROOT / name).relative_to(REPO).as_posix()] = sha(ROOT / name)
    return files


def enforce():
    """Refuse to run measured work from sources, design, dataset or runtime other than the frozen ones."""
    from games.connect4.alphazero_v2.provenance import runtime_identity
    freeze = read(RESULTS / 'freeze.json')
    if source_map() != freeze['files'] or runtime_identity() != freeze['runtime']:
        raise ValueError('Frozen source/design/dataset/runtime differs; a new documented fork is required')
    return freeze


class Deadline:
    """Cooperative absolute deadline shared by every measured stage."""

    def __init__(self, epoch):
        self.epoch = epoch

    def __call__(self):
        if time.time() >= self.epoch:
            raise TimeoutError('Phase 4D.4H-B hard deadline reached')

    def remaining(self):
        return self.epoch - time.time()
