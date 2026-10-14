"""Read-only historical loaders and strict source/runtime/preservation gates."""
from pathlib import Path
import time
from research.alphazero_resnet_selfplay import common as hc
from research.alphazero_resnet_selfplay.worker import state
from research.alphazero_value_regularization.common import parent, parent_entry
from research.alphazero_value_regularization import common as ib
from games.connect4.alphazero_v2.provenance import execution_source_identity, runtime_identity
ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
RESULTS = ROOT / 'results'
OUTPUT = REPO / 'experiment-output/phase4d4j-a'
BRANCH = 'phase4d4j-a-tactical-root-guard'
BASE = '37ec500ce70bd2bbe63ac6578cd7bf576004def6'
sha, read, write, runtime = hc.sha, hc.read, hc.write, hc.runtime


def load(seed):
    runner, entry = parent(seed, 6)
    model = runner.factory.freeze(runner.model)
    return runner, model, runner.factory.inference(model), entry


def protected():
    inventory = ib.protected()
    for phase, package in [('phase4d4i-b', 'alphazero_value_regularization')]:
        historical = REPO.parent / (phase + '-worktree')
        for name, path in [('output', historical/'experiment-output'), ('research', historical/'research'/package), ('docs',historical/'docs')]:
            inventory[phase+'-'+name] = hc.tree_inventory(path)
    for phase in ('phase4d4g','phase4d4h-b','phase4d4h-c'):
        inventory[phase+'-docs']=hc.tree_inventory(REPO.parent/(phase+'-worktree')/'docs')
    # Canonical tracked content is hash-only, including sealed fixtures: never parse or print it.
    paths = hc.git(REPO,'ls-tree','-r','--name-only',BASE,'--','games','models','tests','api','ui','docs','research').splitlines()
    inventory['canonical-current'] = {p:sha(REPO/p) for p in paths}
    if any(not v for v in inventory.values()):
        raise ValueError('Empty protection scope')
    return inventory


def preservation():
    before, after = read(RESULTS/'protected-before.json'), protected()
    scopes = {k:dict(files=len(v),changed=[p for p,h in v.items() if after[k].get(p)!=h]) for k,v in before.items()}
    return dict(intact=all(not v['changed'] for v in scopes.values()),scopes=scopes,official_files=len(hc.preserve()))


def source_map():
    files = execution_source_identity()['files'].copy()
    for p in ROOT.glob('*.py'):
        files[str(p.relative_to(REPO))] = sha(p)
    for rel in ['research/alphazero_resnet_selfplay/common.py','research/alphazero_resnet_selfplay/worker.py',
                'research/alphazero_resnet_selfplay/evaluate.py','research/alphazero_resnet_selfplay/measures.py',
                'research/alphazero_resnet_selfplay/fixtures.json','research/alphazero_resnet_selfplay/results/design.json',
                'research/alphazero_value_regularization/common.py',
                'games/connect4/alphazero_v2/frozen/openings-development.json']:
        files[rel]=sha(REPO/rel)
    for p in [ROOT/'DESIGN.md', ROOT/'independent-prelaunch-review.md', RESULTS/'design.json',
              RESULTS/'pilot.json',RESULTS/'validation.json',RESULTS/'protected-before.json',RESULTS/'prelaunch-review.json']:
        files[str(p.relative_to(REPO))]=sha(p)
    return files


def enforce():
    freeze=read(RESULTS/'freeze.json')
    if source_map()!=freeze['files'] or runtime_identity()!=freeze['runtime']:
        raise ValueError('Frozen source/runtime mismatch')
    for e in freeze['checkpoints']:
        if sha(e['path'])!=e['sha256']:raise ValueError('Checkpoint bytes changed')
    return freeze


class Deadline:
    def __init__(self, epoch):self.epoch=epoch
    def __call__(self):
        if time.time()>=self.epoch:raise TimeoutError('Absolute resource ceiling')
        if (OUTPUT/'CANCEL').exists():raise InterruptedError('Cooperative cancellation')
