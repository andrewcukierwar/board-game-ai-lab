"""Explicit new fixed-data boundary contract; original checkpoints stay unchanged."""
from copy import deepcopy
from pathlib import Path
import torch
import random
from games.connect4.alphazero_v2.experimental import digest, rng_payload, restore_rng
from games.connect4.alphazero_v2.generation import _python_rng_tuple, _python_rng_state
from games.connect4.alphazero_v2.artifacts import atomic_torch_save
from games.connect4.alphazero_v2.provenance import runtime_identity
from .common import LINEAGE, OUTPUT, RESULTS, parent, read, sha, output_path, source_map
from .objective import COEFFICIENT

CONTRACT = dict(format='connect4-fixed-replay-value-regularization', version=1,
    replay='immutable-original-g5-retained-complete-games-v1',
    boundary='integer-optimizer-update-count-no-collection-v1',
    objective='raw-policy-ce+realized-value-mse+coefficient*mean(value**2)-v1',
    exact='parent-strict-load+source+runtime+config+full-computational-state-v1')
STATE_KEYS = {'model', 'model_hash', 'optimizer', 'replay', 'sampling', 'augmentation', 'global_rng', 'counters'}


def state(runner):
    return dict(model=runner.factory.state(runner.model), model_hash=runner.factory.hash(runner.model),
        optimizer=deepcopy(runner.trainer.optimizer.state_dict()), replay=runner.replay_payload(),
        sampling=_python_rng_state(runner.sampling_rng.getstate()),
        augmentation=dict(rng=_python_rng_state(runner.augmenter.rng.getstate()),
            reflected=runner.augmenter.reflected, unreflected=runner.augmenter.unreflected),
        global_rng=rng_payload(), counters=runner.counters())


def initial(seed):
    runner, entry = parent(seed, 5)
    restore_rng(runner.global_rng)
    return runner, entry


def sample(runner):
    addresses, examples = runner.replay.sample(runner.config.training.batch_size, runner.sampling_rng)
    batch, flags = runner.augmenter.batch(examples)
    return batch, dict(addresses=[list(runner.replay.address(x)) for x in addresses], reflections=flags,
                       batch_digest=digest([e.record() for e in batch]))


def payload(runner, seed, arm, updates, trace):
    current_global = rng_payload()
    original, entry = parent(seed, 5)
    restore_rng(current_global)
    if runner.replay.digest() != original.replay.digest():
        raise ValueError('Fixed replay changed')
    p = dict(contract=CONTRACT, lineage=LINEAGE, seed=seed, arm=arm, coefficient=COEFFICIENT[arm],
        updates=updates, trace=trace, parent=entry, config=runner.config.to_dict(),
        source=source_map(), runtime=runtime_identity(), design_sha256=sha(RESULTS / 'design.json'),
        state=state(runner))
    return dict(p, integrity=digest(p))


def load(path, *, expected_sha256, seed, arm, updates):
    if sha(path) != expected_sha256:
        raise ValueError('New checkpoint file hash mismatch')
    p = torch.load(path, map_location='cpu', weights_only=True)
    keys = {'contract','lineage','seed','arm','coefficient','updates','trace','parent','config','source',
            'runtime','design_sha256','state','integrity'}
    if set(p) != keys or p['contract'] != CONTRACT or p['lineage'] != LINEAGE:
        raise ValueError('New checkpoint schema/lineage differs')
    if digest({k:v for k,v in p.items() if k != 'integrity'}) != p['integrity']:
        raise ValueError('New checkpoint integrity differs')
    if (p['seed'],p['arm'],p['updates'],p['coefficient']) != (seed,arm,updates,COEFFICIENT[arm]):
        raise ValueError('New checkpoint experiment identity differs')
    if p['source'] != source_map() or p['runtime'] != runtime_identity() or p['design_sha256'] != sha(RESULTS/'design.json'):
        raise ValueError('New checkpoint source/runtime/design differs')
    runner, entry = initial(seed)
    if p['parent'] != entry or p['config'] != runner.config.to_dict():
        raise ValueError('Parent/config differs')
    if not isinstance(p['trace'], list) or len(p['trace']) != updates or any(
            not isinstance(x,str) or len(x)!=64 or any(c not in '0123456789abcdef' for c in x) for x in p['trace']):
        raise ValueError('Trace length/schema differs from update budget')
    s = p['state']
    if set(s) != STATE_KEYS or s['replay'] != runner.replay_payload():
        raise ValueError('Fixed replay schema/content differs')
    if updates > read(RESULTS/'design.json')['updates_per_arm']:
        raise ValueError('Update budget exceeds declaration')
    if digest(s['global_rng']) != digest(runner.global_rng):
        raise ValueError('CNN objective consumes no global RNG; state differs from parent')
    expected = runner.counters()
    expected['trainer_steps'] += updates
    if type(updates) is not int or updates < 0 or s['counters'] != expected:
        raise ValueError('New update/training counters differ')
    a = s['augmentation']
    if set(a) != {'rng','reflected','unreflected'} or any(type(a[k]) is not int or a[k] < 0 for k in ('reflected','unreflected')):
        raise ValueError('Invalid augmentation state')
    if a['reflected'] + a['unreflected'] != expected['trainer_steps']*runner.config.training.batch_size:
        raise ValueError('Augmentation/update count differs')
    # Independently reconstruct RNG progression, without model work or augmented examples.
    sampling = random.Random(); sampling.setstate(runner.sampling_rng.getstate())
    augmentation = random.Random(); augmentation.setstate(runner.augmenter.rng.getstate())
    reflected = runner.augmenter.reflected
    for _ in range(updates):
        sampling.sample(range(len(runner.replay)),runner.config.training.batch_size)
        reflected += sum(augmentation.random() < runner.config.training.reflection_probability
                         for _ in range(runner.config.training.batch_size))
    if (s['sampling'] != _python_rng_state(sampling.getstate()) or a['rng'] != _python_rng_state(augmentation.getstate())
            or a['reflected'] != reflected):
        raise ValueError('Sampling/augmentation state does not match parent plus declared updates')
    runner.model = runner.factory.restore(s['model'])
    runner.trainer = runner.factory.trainer(runner.model, runner.config.training)
    runner.factory.restore_optimizer(runner.trainer, s['optimizer'], expected['trainer_steps'])
    if runner.factory.hash(runner.model) != s['model_hash']:
        raise ValueError('New model hash differs')
    runner.sampling_rng.setstate(_python_rng_tuple(s['sampling']))
    runner.augmenter.rng.setstate(_python_rng_tuple(a['rng']))
    runner.augmenter.reflected,runner.augmenter.unreflected = a['reflected'],a['unreflected']
    restore_rng(s['global_rng'])
    runner.global_rng = deepcopy(s['global_rng'])
    return runner, p


def save(runner, seed, arm, updates, trace, path):
    path = output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    p = payload(runner, seed, arm, updates, trace)
    def validate(tmp):
        loaded, q = load(tmp, expected_sha256=sha(tmp), seed=seed, arm=arm, updates=updates)
        if digest(state(loaded)) != digest(p['state']):
            raise ValueError('New checkpoint roundtrip differs')
    return atomic_torch_save(path, p, validate)
