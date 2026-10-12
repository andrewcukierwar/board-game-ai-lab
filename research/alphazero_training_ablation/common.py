import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
HISTORICAL = REPO.parents[1]
OFFICIAL = HISTORICAL / 'experiment-output/phase4d3c-alphazero-v2-official-m4-20261008-110950'
PYTHON = '/Users/andrewcukierwar/.venvs/board-game-alphazero-m4/bin/python'
OUTPUT = REPO / 'experiment-output/phase4d4g-primary'
THREADS = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)
    directory=os.open(path.parent,os.O_RDONLY)
    try: os.fsync(directory)
    finally: os.close(directory)

def inventory():
    return {p.relative_to(OFFICIAL).as_posix(): sha(p) for p in sorted(OFFICIAL.rglob('*')) if p.is_file()}

def preserve():
    actual = inventory()
    expected = read(REPO / 'research/alphazero_training/results/official-before.json')
    if len(actual) != 150 or actual != expected:
        raise ValueError('Official inventory changed')
    return actual

def source_map():
    from games.connect4.alphazero_v2.provenance import execution_source_identity
    files = execution_source_identity()['files'].copy()
    for p in sorted(ROOT.rglob('*.py')):
        files[p.relative_to(REPO).as_posix()] = sha(p)
    for name in ('DESIGN.md', 'fixtures.json'):
        files[(ROOT / name).relative_to(REPO).as_posix()] = sha(ROOT / name)
    return files

def enforce():
    from games.connect4.alphazero_v2.provenance import runtime_identity
    freeze = read(ROOT / 'results/freeze.json')
    if source_map() != freeze['files'] or runtime_identity() != freeze['runtime']:
        raise ValueError('Frozen execution-source/runtime differs; new experimental fork required')
    return freeze
