"""Safe paths, frozen execution closure and no-overwrite publication."""
import json
import os
from pathlib import Path
import time
from research.alphazero_resnet_selfplay import common as hc
from research.alphazero_resnet_selfplay.worker import load as strict_parent_load
from games.connect4.alphazero_v2.provenance import execution_source_identity, runtime_identity

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
RESULTS = ROOT / 'results'
OUTPUT = REPO / 'experiment-output/phase4d4i-b'
HC = REPO.parent / 'phase4d4h-c-worktree'
IA = REPO.parent / 'phase4d4i-a-worktree'
BASE = '9b3e171b239ba6f25272f1cc5abc47907ad66249'
BRANCH = 'phase4d4i-b-value-regularization'
LINEAGE = 'phase4d4i-b-fixed-g5-value-output-v1'
PYTHON = hc.PYTHON
sha, read, runtime = hc.sha, hc.read, hc.runtime


def write(path, value):
    """Every new JSON evidence artifact is immutable, including prospective files."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=1, sort_keys=True, allow_nan=False) + '\n'
    with path.open('x') as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())


def output_path(path):
    path = Path(path).resolve()
    if not path.is_relative_to(OUTPUT.resolve()) or path == OUTPUT.resolve():
        raise ValueError('New checkpoint/evidence must be beneath the dedicated output root')
    if path.exists():
        raise FileExistsError('Historical or existing artifacts cannot be overwritten')
    return path


def parent_entry(seed, generation):
    ledger = HC / f'research/alphazero_resnet_selfplay/results/run-evidence/tasks/{seed}-cnn-boundaries.json'
    return next(x for x in read(ledger) if x['generation'] == generation)


def parent(seed, generation):
    e = parent_entry(seed, generation)
    design = read(REPO / 'research/alphazero_resnet_selfplay/results/design.json')
    runner = strict_parent_load(e['path'], e['sha256'], design, seed, 'cnn', restore_global_rng=False)
    if runner.completed_generations != generation:
        raise ValueError('Wrong original generation')
    return runner, e


def source_map():
    files = execution_source_identity()['files'].copy()
    # Entire new execution closure; reporting and independently authored review are frozen too.
    for p in sorted(ROOT.glob('*.py')):
        files[str(p.relative_to(REPO))] = sha(p)
    for rel in (
        'research/alphazero_resnet_selfplay/common.py',
        'research/alphazero_resnet_selfplay/worker.py',
        'research/alphazero_resnet_selfplay/evaluate.py',
        'research/alphazero_resnet_selfplay/measures.py',
        'research/alphazero_resnet_selfplay/results/design.json',
        'research/alphazero_resnet_selfplay/fixtures.json',
        'games/connect4/alphazero_v2/frozen/openings-development.json',
        'research/alphazero_value_diagnostics/evaluate.py',
        'research/alphazero_value_diagnostics/common.py',
        'research/alphazero_value_diagnostics/values.py',
        'research/alphazero_value_regularization/DESIGN.md',
        'research/alphazero_value_regularization/results/design.json',
        'research/alphazero_value_regularization/results/membership.json',
        'research/alphazero_value_regularization/results/pilot.json',
        'research/alphazero_value_regularization/results/protected-before.json'):
        files[rel] = sha(REPO / rel)
    return files


def enforce():
    f = read(RESULTS / 'freeze.json')
    if source_map() != f['files'] or runtime_identity() != f['runtime']:
        raise ValueError('Frozen source/design/runtime mismatch')
    for entry in f['parents']:
        if sha(entry['path']) != entry['sha256']:
            raise ValueError('Original checkpoint bytes changed')
    return f


def protected():
    # Hash-only byte inventories; never parse or display a sealed payload.
    trees = list(hc.PROTECTED) + [
        ('h-source', REPO.parent / 'phase4d4h-worktree/games'),
        ('h-docs', REPO.parent / 'phase4d4h-worktree/docs'),
        ('h-results', REPO.parent / 'phase4d4h-worktree/research/alphazero_resnet'),
        ('hc-output', HC / 'experiment-output/phase4d4h-c'),
        ('hc-results', HC / 'research/alphazero_resnet_selfplay'),
        ('ia-output', IA / 'experiment-output'),
        ('ia-results', IA / 'research/alphazero_value_diagnostics'),
        ('ia-report', IA / 'docs')]
    result = {label: hc.tree_inventory(path) for label, path in trees}
    if any(not v for v in result.values()):
        raise ValueError('Required preservation scope is empty')
    result['official'] = hc.preserve()
    # Assert original tracked game/model/UI/API/test bytes, including actual frozen paths.
    paths = hc.git(REPO, 'ls-tree', '-r', '--name-only', BASE, '--', 'games', 'models', 'tests', 'api', 'ui').splitlines()
    result['baseline-current'] = {p: sha(REPO / p) for p in paths}
    return result


def preservation():
    before, after = read(RESULTS / 'protected-before.json'), protected()
    rows = {k: dict(files=len(v), changed=[p for p, h in v.items() if after[k].get(p) != h])
            for k, v in before.items()}
    return dict(intact=all(not x['changed'] for x in rows.values()), trees=rows)


class Deadline:
    def __init__(self, epoch):
        self.epoch = epoch
    def __call__(self):
        if time.time() >= self.epoch:
            raise TimeoutError('Absolute experimental resource ceiling reached')
