"""Shared paths, durable JSON, preservation inventories and the source freeze for Phase 4D.4H-C.

Everything here is read-only towards the official campaign, the historical
checkout and the Phase 4D.4G / 4D.4H-B worktrees. Checkpoints go to the ignored
experiment-output/phase4d4h-c tree; evidence goes to tracked results/ beside
this file.

Preservation follows the lesson of Phase 4D.4H-B's false-positive flag: what is
enforced is the BYTES of protected files (a file that existed at launch must
still exist with the same hash). Another worktree's HEAD, cleanliness and any
files added there by another session are recorded as observations and never,
on their own, fail a check.
"""
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
HISTORICAL = REPO.parents[1]
OFFICIAL = HISTORICAL / 'experiment-output/phase4d3c-alphazero-v2-official-m4-20261008-110950'
PHASE4D4G = REPO.parent / 'phase4d4g-worktree'
PHASE4D4HB = REPO.parent / 'phase4d4h-b-worktree'
PYTHON = '/Users/andrewcukierwar/.venvs/board-game-alphazero-m4/bin/python'
OUTPUT = REPO / 'experiment-output/phase4d4h-c'
RESULTS = ROOT / 'results'
PACKAGE = 'research.alphazero_resnet_selfplay'
BRANCH = 'phase4d4h-c-resnet-selfplay'
BASE_COMMIT = '6ef9148503a6e268cdcc2a51d9621c00e8e3a92e'
THREADS = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')
SEEDS = (42, 314159)
DECLARATION_SHA256 = 'c8ab2227d4b2c707b3185da8695a991f0914faede88ecd5f1c7fc84fb017b716'
FIXTURES_SHA256 = '80f6851838b18172d6f818ebd306ce0cd36e4796cac1e3a71df29694feb3ca2c'
IDENTITY = {'cnn': 'connect4-conv64-128-128-policy7-scalar-tanh-v2',
            'resnet3x64': 'connect4-resnet3x64-bn-policy32-value32x64-scalar-tanh-v1'}
NAMES = tuple(IDENTITY)
PARAMETERS = {'cnn': 325_896, 'resnet3x64': 322_440}
ARMS = tuple((seed, name) for seed in SEEDS for name in NAMES)

# Protected trees: (label, root, enforced). Bytes of every launch-time file are enforced.
PROTECTED = (
    ('phase4d4g-output', PHASE4D4G / 'experiment-output/phase4d4g-primary'),
    ('phase4d4g-results', PHASE4D4G / 'research/alphazero_training_ablation/results'),
    ('phase4d4h-b-output', PHASE4D4HB / 'experiment-output/phase4d4h-b'),
    ('phase4d4h-b-results', PHASE4D4HB / 'research/alphazero_resnet_benchmark/results'),
    ('historical-games-source', HISTORICAL / 'games'),
)
OBSERVED_WORKTREES = (('historical', HISTORICAL), ('phase4d4g', PHASE4D4G), ('phase4d4h-b', PHASE4D4HB))


def arm_label(seed, name):
    return f'{seed}-{name}'


def experimental_config(design, seed, name):
    """The one declared configuration of an arm; every strict load is checked against it."""
    from games.connect4.alphazero_v2.config import V2Config
    from games.connect4.alphazero_v2.experimental import ExperimentalConfig
    arm = design['arms'][name]
    training = V2Config(seed=seed, games_per_generation=arm['games_per_generation'],
                        max_generations=design['generations'], **design['training_overrides'])
    return ExperimentalConfig(training, optimizer_updates=arm['optimizer_updates'], architecture=IDENTITY[name])


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
    tmp = path.with_name(path.name + f'.tmp{os.getpid()}')
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


def thread_environment():
    return dict(os.environ, PYTHONDONTWRITEBYTECODE='1', **{name: '1' for name in THREADS})


def runtime():
    """One-thread deterministic CPU runtime; the launching environment must already set the thread variables."""
    from games.connect4.alphazero_v2.provenance import configure_deterministic_runtime, unavailable_runtime_fields
    identity = configure_deterministic_runtime(1)
    if any(v != '1' for v in identity['thread_environment'].values()) or unavailable_runtime_fields(identity):
        raise ValueError('Phase 4D.4H-C requires all five library thread controls set to 1')
    return identity


# Preservation ---------------------------------------------------------------------------------

def official_inventory():
    return {p.relative_to(OFFICIAL).as_posix(): sha(p) for p in sorted(OFFICIAL.rglob('*')) if p.is_file()}


def preserve():
    """All 150 official campaign files byte-identical to the Phase 4D.4F inventory."""
    actual = official_inventory()
    expected = read(REPO / 'research/alphazero_training/results/official-before.json')
    if len(actual) != 150 or actual != expected or actual['declaration.json'] != DECLARATION_SHA256:
        raise ValueError('Official campaign inventory changed')
    return actual


def tree_inventory(root):
    return {p.relative_to(root).as_posix(): sha(p) for p in sorted(Path(root).rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}


def protected_inventory():
    return {label: tree_inventory(root) for label, root in PROTECTED}


def compare_protected(before, after):
    """Enforced: every launch-time file still present with identical bytes. Additions are observations only."""
    report, intact = {}, True
    for label, _ in PROTECTED:
        old, new = before[label], after[label]
        changed = sorted(k for k in old if k in new and new[k] != old[k])
        missing = sorted(k for k in old if k not in new)
        added = sorted(k for k in new if k not in old)
        report[label] = dict(files_at_launch=len(old), files_now=len(new), changed=changed, missing=missing,
                             added_by_others=added, launch_inventory_sha256=text_sha(old),
                             intact=not changed and not missing)
        intact = intact and not changed and not missing
    return dict(intact=intact, trees=report)


def git(root, *arguments):
    done = subprocess.run(['git', '-C', str(root), *arguments], capture_output=True, text=True)
    return done.stdout.strip() if done.returncode == 0 else None


def worktree_observations():
    """Reported, never enforced: commits and cleanliness of worktrees other sessions may be using."""
    return {label: dict(head=git(root, 'rev-parse', 'HEAD'), branch=git(root, 'rev-parse', '--abbrev-ref', 'HEAD'),
                        clean=git(root, 'status', '--porcelain') == '')
            for label, root in OBSERVED_WORKTREES}


def describe_external_commits(before, after):
    """Separate record of commits other sessions made during the run; informational only."""
    rows = {}
    for label, root in OBSERVED_WORKTREES:
        a, b = before[label]['head'], after[label]['head']
        row = dict(head_at_launch=a, head_now=b, changed=a != b)
        if a != b and a and b:
            row['descends_from_launch_head'] = subprocess.run(
                ['git', '-C', str(root), 'merge-base', '--is-ancestor', a, b]).returncode == 0
            row['paths_changed'] = (git(root, 'diff', '--name-only', a, b) or '').splitlines()
        rows[label] = row
    return rows


# Source freeze --------------------------------------------------------------------------------

FROZEN = ('__init__.py', 'common.py', 'measures.py', 'evaluate.py', 'worker.py', 'manage.py', 'prepare.py',
          'validate.py', 'calibrate.py', 'analyze.py', 'DESIGN.md', 'fixtures.json', 'results/design.json',
          'results/calibration.json')


def source_map():
    """Enforced package execution closure plus this phase's frozen sources, protocol, fixtures and design."""
    from games.connect4.alphazero_v2.provenance import execution_source_identity
    files = execution_source_identity()['files'].copy()
    for name in FROZEN:
        files[(ROOT / name).relative_to(REPO).as_posix()] = sha(ROOT / name)
    return files


def enforce():
    """Refuse measured work from sources, design, fixtures or runtime other than the frozen ones."""
    from games.connect4.alphazero_v2.provenance import runtime_identity
    freeze = read(RESULTS / 'freeze.json')
    current = source_map()
    if current != freeze['files']:
        different = sorted(k for k in set(current) | set(freeze['files']) if current.get(k) != freeze['files'].get(k))
        raise ValueError(f'Frozen execution-source bytes differ: {different}')
    if runtime_identity() != freeze['runtime']:
        raise ValueError('Runtime identity differs from the frozen one')
    return freeze


class Deadline:
    """Cooperative absolute deadline plus a cancel file, shared by every measured task."""

    def __init__(self, epoch, cancel=None):
        self.epoch, self.cancel = epoch, cancel
        self._next = 0.0

    def __call__(self):
        now = time.time()
        if now >= self.epoch:
            raise TimeoutError('Phase 4D.4H-C hard deadline reached')
        if self.cancel is not None and now >= self._next:
            self._next = now + 1.0
            if Path(self.cancel).exists():
                raise InterruptedError('Cooperative cancellation requested')

    def remaining(self):
        return self.epoch - time.time()
