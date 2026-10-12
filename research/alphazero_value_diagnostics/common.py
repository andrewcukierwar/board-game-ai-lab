"""Strict read-only loading and independent source/runtime/preservation gates."""
from pathlib import Path
from research.alphazero_resnet_selfplay import common as prior
from research.alphazero_resnet_selfplay.worker import load as prior_load

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
RESULTS = ROOT / 'results'
HC = REPO.parent / 'phase4d4h-c-worktree'
PYTHON = prior.PYTHON
read, write, sha, runtime = prior.read, prior.write, prior.sha, prior.runtime
VARIANTS = {'A': (False, 1.0), 'B': (False, 0.0), 'C': (True, 1.0),
            'D': (True, 0.0), 'E': (False, 0.5)}
ARMS = ('42-cnn', '42-resnet3x64', '314159-cnn', '314159-resnet3x64')


def entries():
    return {arm: read(HC / f'research/alphazero_resnet_selfplay/results/run-evidence/tasks/{arm}-boundaries.json')
            for arm in ARMS}


def load(arm, generation, frozen=None):
    design = read(REPO / 'research/alphazero_resnet_selfplay/results/design.json')
    ledger = entries() if frozen is None else frozen['checkpoints']
    entry = next(e for e in ledger[arm] if e['generation'] == generation)
    seed, name = arm.split('-', 1)
    runner = prior_load(entry['path'], entry['sha256'], design, int(seed), name, restore_global_rng=False)
    if runner.completed_generations != generation:
        raise ValueError('Wrong checkpoint generation')
    model = runner.factory.freeze(runner.model)
    return runner, model, runner.factory.inference(model), entry


def protected():
    trees = list(prior.PROTECTED) + [
        ('hc-output', HC / 'experiment-output/phase4d4h-c'),
        ('hc-results', HC / 'research/alphazero_resnet_selfplay/results'),
        ('hc-source', HC / 'games'),
        ('hc-research-source', HC / 'research/alphazero_resnet_selfplay'),
        ('g-research-source', prior.PHASE4D4G / 'research/alphazero_training_ablation'),
        ('hb-research-source', prior.PHASE4D4HB / 'research/alphazero_resnet_benchmark'),
        ('canonical-benchmarks', prior.HISTORICAL / 'benchmarks')]
    # No fixture content is read here: hash only. Byte inventories protect sealed and nonsealed files alike.
    return {label: prior.tree_inventory(path) for label, path in trees}


def source_map():
    from games.connect4.alphazero_v2.provenance import execution_source_identity
    files = execution_source_identity()['files'].copy()
    for p in sorted(ROOT.glob('*.py')):
        if p.name.startswith(('report', 'independent')):
            continue
        files[str(p.relative_to(REPO))] = sha(p)
    for p in (ROOT / 'DESIGN.md', RESULTS / 'design.json', RESULTS / 'pilot.json'):
        files[str(p.relative_to(REPO))] = sha(p)
    for p in (REPO / 'games/connect4/alphazero_v2/frozen/openings-development.json',
              REPO / 'research/alphazero_resnet_selfplay/common.py',
              REPO / 'research/alphazero_resnet_selfplay/worker.py',
              REPO / 'research/alphazero_resnet_selfplay/evaluate.py',
              REPO / 'research/alphazero_resnet_selfplay/measures.py',
              REPO / 'research/alphazero_resnet_selfplay/fixtures.json',
              REPO / 'research/alphazero_resnet_selfplay/results/design.json'):
        files[str(p.relative_to(REPO))] = sha(p)
    return files


def enforce():
    from games.connect4.alphazero_v2.provenance import runtime_identity
    freeze = read(RESULTS / 'freeze.json')
    if (source_map() != freeze['files'] or runtime_identity() != freeze['runtime']
            or sha(RESULTS / 'protected-before.json') != freeze['protected_inventory_sha256']):
        raise ValueError('Source/runtime differs from prospective freeze')
    return freeze


def preservation(before):
    after = protected()
    result = {k: dict(files=len(v), changed=[p for p in v if after[k].get(p) != v[p]],
                      added=[p for p in after[k] if p not in v]) for k, v in before.items()}
    return dict(intact=all(not r['changed'] for r in result.values()), trees=result,
                official_files=len(prior.preserve()))
