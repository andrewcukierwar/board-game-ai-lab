"""Paths, strict freeze/enforcement, preservation inventory and lifecycle gates."""
from pathlib import Path
import time
from research.alphazero_resnet_selfplay import common as hc
from research.alphazero_tactical_guard import common as ja
from games.connect4.alphazero_v2.provenance import execution_source_identity, runtime_identity
ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
RESULTS = ROOT / 'results'
OUTPUT = REPO / 'experiment-output/phase4d4j-b'
BRANCH = 'phase4d4j-b-tactical-guard-confirmation'
BASE = '1420cc803bdfa882a3b68a7ae93017f1f0081dbd'
GUARD_SHA256 = 'd2ee54ba1fe9ddbcc4641538e2f5dd89a2435234ff194d6909bed72096632d04'
sha, read, write, runtime = hc.sha, hc.read, hc.write, hc.runtime
GUARD = REPO / 'research/alphazero_tactical_guard/guard.py'
JA = REPO.parent / 'phase4d4j-a-worktree'
SEEDS = (42, 314159)
TASK_ORIGIN_EPOCH = 1791950274  # 2026-10-14 03:57:54 UTC, first command of this phase
HARD_MAXIMUM_SECONDS = 4500     # 75 minutes, includes pilot, review, report and publication
load, state = ja.load, ja.state


def opponents():
    return (3, read(RESULTS / 'calibration.json')['opponent_b_depth'])


def protected():
    """Hash-only inventories (never parse sealed payloads)."""
    inventory = ja.protected()
    for name, path in [('output', JA / 'experiment-output'), ('research', JA / 'research/alphazero_tactical_guard'), ('docs', JA / 'docs')]:
        inventory['phase4d4j-a-' + name] = hc.tree_inventory(path)
    paths = hc.git(REPO, 'ls-tree', '-r', '--name-only', BASE, '--', 'games', 'models', 'tests', 'api', 'ui', 'docs', 'research').splitlines()
    inventory['base-current'] = {p: sha(REPO / p) for p in paths}
    inventory['official'] = hc.preserve()
    if any(not v for v in inventory.values()):
        raise ValueError('Empty protection scope')
    return inventory


def preservation():
    before, after = read(RESULTS / 'protected-before.json'), protected()
    scopes = {k: dict(files=len(v), changed=[p for p, h in v.items() if after[k].get(p) != h]) for k, v in before.items()}
    return dict(intact=all(not v['changed'] for v in scopes.values()), scopes=scopes, official_files=len(after['official']))


def source_map():
    files = ja.source_map()  # 4D.4J-A frozen execution sources (guard, evaluate, loaders) plus its own evidence
    for p in ROOT.glob('*.py'):
        files[str(p.relative_to(REPO))] = sha(p)
    for p in [ROOT / 'DESIGN.md', ROOT / 'independent-prelaunch-review.md', RESULTS / 'design.json', RESULTS / 'openings.json',
              RESULTS / 'opponent-selection-rule.json', RESULTS / 'calibration.json', RESULTS / 'validation.json',
              RESULTS / 'protected-before.json', RESULTS / 'prelaunch-review.json']:
        files[str(p.relative_to(REPO))] = sha(p)
    return files


def enforce():
    freeze = read(RESULTS / 'freeze.json')
    if sha(GUARD) != GUARD_SHA256 or source_map() != freeze['files'] or runtime_identity() != freeze['runtime']:
        raise ValueError('Frozen guard/source/runtime mismatch')
    for e in freeze['checkpoints']:
        if sha(e['path']) != e['sha256']:
            raise ValueError('Checkpoint bytes changed')
    return freeze


class Deadline:
    def __init__(self, epoch): self.epoch = epoch
    def __call__(self):
        if time.time() >= self.epoch: raise TimeoutError('Absolute resource ceiling')
        if (OUTPUT / 'CANCEL').exists(): raise InterruptedError('Cooperative cancellation')
