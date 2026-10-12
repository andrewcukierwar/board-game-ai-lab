"""Architectural validation that must pass before any benchmark is measured.

Independent numerical cross-checks of both ResNets (a float64 unfold/matmul
reference that shares no code with resnet.py), checkpoint contracts,
fresh-process exact resume, batch-norm running statistics, optimizer
compatibility, CNN backward compatibility against the official generation-20
artifacts (read-only) and historical/4D.4G artifact isolation.
No benchmark timing and no fixed-data training happens here.
"""
from dataclasses import replace
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

from .common import (ARCHITECTURES, HISTORICAL, IDENTITY, OFFICIAL, PARAMETERS, PARENTS, REPO, RESULTS, SEEDS, SHORT,
                     preserve, read, runtime, sha, text_sha, thread_environment, write)

EPS, MOMENTUM = 1e-5, 0.1
BLOCKS = {'resnet3x64': 3, 'resnet5x64': 5}
PHASE4D4G = REPO.parent / 'phase4d4g-worktree'
PHASE4D4G_HEAD = 'b4b1d9f5a42db0889fb2eed2acea6252da8526c9'


def checks():
    record = []

    def check(name, condition, detail=None):
        record.append(dict(check=name, passed=bool(condition), detail=detail))
        if not condition:
            raise AssertionError(f'{name}: {detail}')
    return record, check


def boards(n, seed):
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(-1, 2, (n, 1, 6, 7), generator=generator).to(torch.float32)


# Independent float64 reference ---------------------------------------------------------

def conv_reference(x, weight):
    """Zero-padded stride-1 convolution by explicit patch extraction and one matrix product."""
    n, _, h, w = x.shape
    out, _, kh, kw = weight.shape
    pad = kh // 2
    padded = torch.zeros(n, x.shape[1], h + 2 * pad, w + 2 * pad, dtype=torch.float64)
    padded[:, :, pad:pad + h, pad:pad + w] = x
    patches = torch.stack([padded[:, :, i:i + h, j:j + w] for i in range(kh) for j in range(kw)], dim=2)
    return torch.einsum('nckhw,ock->nohw', patches, weight.double().reshape(out, x.shape[1], kh * kw))


def norm_reference(x, state, prefix, batch_statistics):
    """Batch norm from its definition; returns (output, batch mean, unbiased batch variance)."""
    mean_b = x.mean(dim=(0, 2, 3))
    count = x.shape[0] * x.shape[2] * x.shape[3]
    var_biased = ((x - mean_b[None, :, None, None]) ** 2).mean(dim=(0, 2, 3))
    mean, var = ((mean_b, var_biased) if batch_statistics
                 else (state[f'{prefix}.running_mean'].double(), state[f'{prefix}.running_var'].double()))
    scale = state[f'{prefix}.weight'].double() / torch.sqrt(var + EPS)
    out = (x - mean[None, :, None, None]) * scale[None, :, None, None] + state[f'{prefix}.bias'].double()[None, :, None, None]
    return out, mean_b, var_biased * count / (count - 1)


def reference_forward(state, blocks, x, batch_statistics=False):
    """(policy logits, value, {bn prefix: (batch mean, unbiased variance)}) in float64."""
    seen = {}

    def norm(tensor, prefix):
        out, mean, var = norm_reference(tensor, state, prefix, batch_statistics)
        seen[prefix] = (mean, var)
        return out
    relu = lambda t: torch.clamp(t, min=0)
    h = relu(norm(conv_reference(x.double(), state['stem_conv.weight']), 'stem_bn'))
    for b in range(blocks):
        branch = relu(norm(conv_reference(h, state[f'tower.{b}.conv1.weight']), f'tower.{b}.bn1'))
        branch = norm(conv_reference(branch, state[f'tower.{b}.conv2.weight']), f'tower.{b}.bn2')
        h = relu(h + branch)
    p = relu(norm(conv_reference(h, state['policy_conv.weight']), 'policy_bn')).flatten(1)
    policy = p @ state['policy_fc.weight'].double().T + state['policy_fc.bias'].double()
    v = relu(norm(conv_reference(h, state['value_conv.weight']), 'value_bn')).flatten(1)
    v = relu(v @ state['value_fc1.weight'].double().T + state['value_fc1.bias'].double())
    value = torch.tanh(v @ state['value_fc2.weight'].double().T + state['value_fc2.bias'].double())
    return policy, value, seen


def synthetic_examples(count, seed):
    """Legal labeled positions from seeded random play (no replay data, no network)."""
    import random
    from games.connect4.connect4 import Connect4
    from games.connect4.alphazero_v2.data import V2Example, encode_board, finalize_game
    rng = random.Random(seed)
    result, index = [], 0
    while len(result) < count:
        game, moves, pending = Connect4(), [], []
        while not game.is_game_over():
            legal = game.get_valid_moves()
            move = rng.choice(legal)
            visits = tuple(rng.randint(1, 9) if a in legal else 0 for a in range(7))
            pending.append(V2Example(encode_board(game), game.current_player, len(moves), visits, move, 1.0))
            game.make_move(move)
            moves.append(move)
        result.extend(finalize_game(1, index, moves, pending).examples)
        index += 1
    return result[:count]


def architecture_review(check):
    """Numerical review of both ResNets against the independent reference."""
    from games.connect4.alphazero_v2 import resnet as R
    from games.connect4.alphazero_v2.config import V2Config
    from games.connect4.alphazero_v2.experimental_models import model_factory
    summary = {}
    for short in ('resnet3x64', 'resnet5x64'):
        identity, blocks = IDENTITY[short], BLOCKS[short]
        factory = model_factory(identity)
        model = factory.construct(20261012)
        state = factory.state(model)
        parameters = sum(p.numel() for p in model.parameters())
        check(f'{short}: parameter count', parameters == PARAMETERS[short], parameters)
        norms = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
        check(f'{short}: batch-norm layer count and defaults',
              len(norms) == 2 * blocks + 3 and all(m.eps == EPS and m.momentum == MOMENTUM and m.affine
                                                   and m.track_running_stats for m in norms), len(norms))
        convs = [m for m in model.modules() if isinstance(m, torch.nn.Conv2d)]
        check(f'{short}: every convolution is bias-free and feeds a batch norm',
              len(convs) == len(norms) and all(c.bias is None for c in convs))
        x = boards(128, 1)
        # Eval mode at initialization, then with non-trivial running statistics and affine terms.
        with torch.no_grad():
            policy, value = model(x)
        ref_policy, ref_value, _ = reference_forward(state, blocks, x)
        init_gap = max(float((policy.double() - ref_policy).abs().max()), float((value.double() - ref_value).abs().max()))
        check(f'{short}: eval forward equals independent float64 reference at initialization', init_gap < 2e-5, init_gap)
        trainer = factory.trainer(model, V2Config(seed=1))
        examples = synthetic_examples(128 * 6, 3)
        captured = {}
        hooks = [module.register_forward_hook(lambda m, inputs, out, name=name: captured.__setitem__(name, inputs[0].detach().clone()))
                 for name, module in model.named_modules() if isinstance(module, torch.nn.BatchNorm2d)]
        expected = {name: (module.running_mean.clone().double(), module.running_var.clone().double())
                    for name, module in model.named_modules() if isinstance(module, torch.nn.BatchNorm2d)}
        metrics = []
        for step in range(6):
            metrics.append(trainer.step(examples[step * 128:(step + 1) * 128]))
            for name, (mean, var) in expected.items():
                inputs = captured[name].double()
                count = inputs.shape[0] * inputs.shape[2] * inputs.shape[3]
                batch_mean = inputs.mean(dim=(0, 2, 3))
                batch_var = ((inputs - batch_mean[None, :, None, None]) ** 2).sum(dim=(0, 2, 3)) / (count - 1)
                expected[name] = ((1 - MOMENTUM) * mean + MOMENTUM * batch_mean, (1 - MOMENTUM) * var + MOMENTUM * batch_var)
        for hook in hooks:
            hook.remove()
        modules = dict(model.named_modules())
        stat_gap = max(max(float((modules[n].running_mean.double() - mean).abs().max()),
                           float((modules[n].running_var.double() - var).abs().max())) for n, (mean, var) in expected.items())
        check(f'{short}: running statistics equal an independent exponential-average recomputation', stat_gap < 1e-5, stat_gap)
        check(f'{short}: every batch-norm counter equals the trainer step count',
              all(int(b) == trainer.steps == 6 for n, b in model.named_buffers() if n.endswith('num_batches_tracked')))
        check(f'{short}: trainer leaves the model in eval mode with finite loss', not model.training and
              all(torch.isfinite(torch.tensor(m['combined_loss'])) for m in metrics))
        state = factory.state(model)
        before = factory.hash(model)
        with torch.no_grad():
            policy, value = model(x)
        ref_policy, ref_value, _ = reference_forward(state, blocks, x)
        trained_gap = max(float((policy.double() - ref_policy).abs().max()), float((value.double() - ref_value).abs().max()))
        check(f'{short}: eval forward equals the reference after training (running statistics in use)', trained_gap < 2e-5, trained_gap)
        check(f'{short}: eval inference leaves weights and statistics unchanged', factory.hash(model) == before)
        # Train-mode forward uses batch statistics (checked on a disposable copy).
        from copy import deepcopy
        copy = deepcopy(model).train()
        with torch.no_grad():
            policy_t, value_t = copy(x)
        ref_policy_t, ref_value_t, _ = reference_forward(state, blocks, x, batch_statistics=True)
        train_gap = max(float((policy_t.double() - ref_policy_t).abs().max()), float((value_t.double() - ref_value_t).abs().max()))
        check(f'{short}: train-mode forward equals the batch-statistics reference', train_gap < 2e-5, train_gap)
        check(f'{short}: train- and eval-mode outputs differ (statistics source matters)',
              float((policy_t - policy).abs().max()) > 1e-4)
        # Per-row independence in eval mode: a row's output does not depend on its batch.
        with torch.no_grad():
            single = torch.cat([model(x[i:i + 1])[0] for i in range(8)])
        row_gap = float((single - policy[:8]).abs().max())
        check(f'{short}: eval-mode rows are batch-independent', row_gap < 1e-5, row_gap)
        # Receptive field: one corner cell reaches the opposite corner of the trunk output.
        probe = torch.zeros(1, 1, 6, 7)
        changed = probe.clone()
        changed[0, 0, 0, 0] = 1.0
        with torch.no_grad():
            trunk = lambda t: model.tower(torch.relu(model.stem_bn(model.stem_conv(t))))
            reach = float((trunk(changed) - trunk(probe))[0, :, 5, 6].abs().max())
        check(f'{short}: trunk receptive field covers the whole board', reach > 0, reach)
        # Mirror symmetry is not built in (it is learned via augmentation): record, do not require.
        decay, no_decay = R.parameter_groups(model)
        named = dict(model.named_parameters())
        check(f'{short}: decay groups partition the parameters; batch-norm affines and biases never decay',
              sorted(decay + no_decay) == sorted(named) and all(named[n].ndim > 1 for n in decay)
              and all(named[n].ndim == 1 for n in no_decay))
        summary[short] = dict(parameters=parameters, batch_norm_layers=len(norms), initialization_gap=init_gap,
                              trained_eval_gap=trained_gap, train_mode_gap=train_gap, statistic_gap=stat_gap,
                              row_independence_gap=row_gap, corner_to_corner_response=reach,
                              six_step_losses=[m['combined_loss'] for m in metrics])
    return summary


def checkpoint_contracts(check):
    """Inference-artifact contract of both ResNets and every cross-architecture rejection."""
    from games.connect4.alphazero_v2 import network, resnet as R
    from games.connect4.alphazero_v2.experimental_models import model_factory
    result = {}
    with tempfile.TemporaryDirectory(prefix='phase4d4h-b-contract-') as temporary:
        root = Path(temporary)
        paths = {}
        for identity in ARCHITECTURES:
            short = SHORT[identity]
            factory = model_factory(identity)
            model = factory.construct(5)
            paths[short] = root / f'{short}.pt'
            if short == 'cnn':
                digest = network.save_inference_checkpoint(paths[short], model)
                loaded = network.load_inference_checkpoint(paths[short])
            else:
                digest = R.save_inference_checkpoint(paths[short], model)
                loaded = R.load_inference_checkpoint(paths[short], identity)
            check(f'{short}: inference artifact round trip is bit-identical',
                  factory.hash(loaded.model) == factory.hash(model) and sha(paths[short]) == digest)
            try:
                (network if short == 'cnn' else R).save_inference_checkpoint(paths[short], model)
                overwritten = True
            except FileExistsError:
                overwritten = False
            check(f'{short}: artifact publication refuses to overwrite', not overwritten)
            result[short] = dict(artifact_sha256=digest, weights_sha256=factory.hash(model),
                                 state_tensors=len(factory.state(model)))
        rejected = []
        for source in paths:
            for target in ARCHITECTURES:
                if SHORT[target] == source:
                    continue
                try:
                    if SHORT[target] == 'cnn':
                        network.load_inference_checkpoint(paths[source])
                    else:
                        R.load_inference_checkpoint(paths[source], target)
                    rejected.append(False)
                except ValueError:
                    rejected.append(True)
                try:
                    model_factory(target).restore(model_factory(IDENTITY[source]).state(model_factory(IDENTITY[source]).construct(5)))
                    rejected.append(False)
                except ValueError:
                    rejected.append(True)
        check('all six ordered cross-architecture artifact loads and state restores are rejected',
              len(rejected) == 12 and all(rejected), rejected)
    return result


RESUME_SCRIPT = '''
import sys
from games.connect4.alphazero_v2.provenance import configure_deterministic_runtime
configure_deterministic_runtime(1)
from games.connect4.alphazero_v2.experimental import load_experimental
runner = load_experimental(sys.argv[1], expected_sha256=sys.argv[2])
runner.run_generation(sys.argv[3])
'''


def comparable(runner):
    from games.connect4.alphazero_v2 import experimental as E
    content = runner._content()
    content.pop('resources')
    return dict(content=E.digest(content), model=runner.factory.hash(runner.model),
                optimizer=E.digest(runner.trainer.optimizer.state_dict()), replay=runner.replay.digest(),
                counters=runner.counters(), global_rng=E.digest(runner.global_rng),
                sampling=text_sha(repr(runner.sampling_rng.getstate())),
                augmentation=text_sha(repr(runner.augmenter.rng.getstate())),
                buffers=text_sha({n: b.tolist() for n, b in runner.model.named_buffers()}))


def fresh_process_resume(check):
    """Generation 2 from a saved generation-1 boundary: in-process versus a brand-new interpreter."""
    from games.connect4.alphazero_v2 import experimental as E
    from games.connect4.alphazero_v2.artifacts import file_sha256
    from games.connect4.alphazero_v2.config import V2Config
    result = {}
    for identity in ARCHITECTURES:
        short = SHORT[identity]
        training = V2Config(seed=20261012, games_per_generation=8, self_play_simulations=16, batch_size=32,
                            replay_generations=2, replay_max_games=16, max_generations=3)
        config = E.ExperimentalConfig(training, optimizer_updates=8, architecture=identity)
        with tempfile.TemporaryDirectory(prefix=f'phase4d4h-b-resume-{short}-') as temporary:
            root = Path(temporary)
            runner = E.ExperimentalRunner(config, lineage='phase4d4h-b-validation', arm=short)
            first = runner.run_generation(root / 'generation-1.pt')
            boundary = file_sha256(root / 'generation-1.pt')
            runner.run_generation(root / 'expected.pt')
            completed = subprocess.run([sys.executable, '-B', '-c', RESUME_SCRIPT, str(root / 'generation-1.pt'),
                                        boundary, str(root / 'actual.pt')], cwd=REPO, env=thread_environment(),
                                       capture_output=True, text=True, timeout=300)
            check(f'{short}: fresh interpreter resumed and published generation 2', completed.returncode == 0,
                  completed.stderr[-2000:])
            expected = comparable(E.load_experimental(root / 'expected.pt', restore_global_rng=False))
            actual = comparable(E.load_experimental(root / 'actual.pt', restore_global_rng=False))
            check(f'{short}: fresh-process continuation is bit-identical to the in-process one', expected == actual,
                  dict(expected=expected, actual=actual))
            check(f'{short}: live in-process runner equals its reloaded boundary', comparable(runner) == expected)
            if short != 'cnn':
                steps = expected['counters']['trainer_steps']
                loaded = E.load_experimental(root / 'actual.pt', restore_global_rng=False)
                check(f'{short}: resumed batch-norm counters equal trainer steps',
                      steps == 16 and all(int(b) == steps for n, b in loaded.model.named_buffers()
                                          if n.endswith('num_batches_tracked')))
            result[short] = dict(generation1_boundary_sha256=boundary, generation1_positions=first['new_positions'],
                                 pool_batches=first['collection']['batch_sizes'], state=actual)
    return result


def optimizer_compatibility(check):
    from games.connect4.alphazero_v2.config import V2Config
    from games.connect4.alphazero_v2.experimental_models import model_factory
    config = V2Config(seed=9)
    examples = synthetic_examples(128 * 5, 11)
    result, states = {}, {}
    for identity in ARCHITECTURES:
        short = SHORT[identity]
        factory = model_factory(identity)
        straight = factory.trainer(factory.construct(9), config)
        for step in range(5):
            straight.step(examples[step * 128:(step + 1) * 128])
        # Interrupt after three updates, restore model + optimizer into new objects, finish.
        first = factory.trainer(factory.construct(9), config)
        for step in range(3):
            first.step(examples[step * 128:(step + 1) * 128])
        resumed = factory.trainer(factory.restore(factory.state(first.model)), config)
        factory.restore_optimizer(resumed, first.optimizer.state_dict(), 3)
        for step in range(3, 5):
            resumed.step(examples[step * 128:(step + 1) * 128])
        check(f'{short}: optimizer restore continues bit-identically',
              factory.hash(resumed.model) == factory.hash(straight.model) and resumed.steps == 5)
        groups = straight.optimizer.param_groups
        settings = [{k: v for k, v in g.items() if k != 'params'} for g in groups]
        check(f'{short}: AdamW with the declared settings and two groups',
              type(straight.optimizer) is torch.optim.AdamW and len(groups) == 2
              and [g['weight_decay'] for g in groups] == [config.weight_decay, 0.0]
              and all(g['lr'] == config.learning_rate and tuple(g['betas']) == config.adam_betas
                      and g['eps'] == config.adam_eps for g in groups))
        states[short] = (factory, straight)
        result[short] = dict(decay_tensors=len(groups[0]['params']), no_decay_tensors=len(groups[1]['params']),
                             decay_parameters=sum(p.numel() for p in groups[0]['params']),
                             no_decay_parameters=sum(p.numel() for p in groups[1]['params']),
                             settings=text_sha(settings))
    check('optimizer settings other than parameter membership are identical across architectures',
          len({r['settings'] for r in result.values()}) == 1)
    for source, (_, donor) in states.items():
        for target, (factory, _) in states.items():
            if source == target:
                continue
            trainer = factory.trainer(factory.construct(9), config)
            try:
                factory.restore_optimizer(trainer, donor.optimizer.state_dict(), 5)
                accepted = True
            except (ValueError, KeyError, RuntimeError):
                accepted = False
            check(f'{target}: rejects the {source} optimizer state', not accepted)
    return result


def cnn_backward_compatibility(check):
    """The unchanged CNN path still reads, reproduces and forks the official generation-20 artifacts."""
    from games.connect4.alphazero_v2 import experimental as E, network
    from games.connect4.alphazero_v2.config import ARCHITECTURE, V2Config
    from games.connect4.alphazero_v2.experimental_import import fork_verified, read_historical
    from games.connect4.alphazero_v2.experimental_models import CNNFactory, model_factory
    from games.connect4.alphazero_v2.generation import initial_model
    check('default experimental architecture is the CNN',
          E.ExperimentalConfig(V2Config(), 1).architecture == ARCHITECTURE and type(model_factory(ARCHITECTURE)) is CNNFactory)
    check('CNN parameter count', sum(p.numel() for p in initial_model(42).parameters()) == PARAMETERS['cnn'])
    result = {}
    for seed in SEEDS:
        base = OFFICIAL / f'runs/seed-{seed}/generations/generation-0020'
        summary = read(base / 'summary.json')
        inference = network.load_inference_checkpoint(base / 'generation-0020.inference.pt')
        verified = read_historical(base / 'generation-0020.resume.pt', expected_sha256=PARENTS[seed],
                                   historical_checkout=HISTORICAL)
        original = verified['original']
        check(f'seed {seed}: official inference artifact loads and matches the boundary weights',
              network.weights_sha256(inference.model) == original['learner_weights_sha256'])
        old = V2Config.from_dict(original['config'])
        config = E.ExperimentalConfig(replace(old, games_per_generation=64, max_generations=22), 64)
        fork = fork_verified(verified, config, parent_sha256=PARENTS[seed], schedule=E.ORIGINAL,
                             lineage='phase4d4h-b-validation', arm='cnn-compat')
        check(f'seed {seed}: strict historical read forks through the current CNN factory',
              fork.factory.hash(fork.model) == original['learner_weights_sha256']
              and fork.replay.digest() == original['replay']['digest'] and fork.trainer.steps == original['counters']['trainer_steps'])
        for short in ('resnet3x64', 'resnet5x64'):
            try:
                model_factory(IDENTITY[short]).restore(original['model_state_dict'])
                accepted = True
            except ValueError:
                accepted = False
            check(f'seed {seed}: {short} refuses the historical CNN weights', not accepted)
            try:
                bad = E.ExperimentalConfig(replace(old, games_per_generation=64, max_generations=22), 64, architecture=IDENTITY[short])
                fork_verified(verified, bad, parent_sha256=PARENTS[seed], schedule=E.ORIGINAL, lineage='x', arm='x')
                accepted = True
            except ValueError:
                accepted = False
            check(f'seed {seed}: a {short} fork of the historical boundary is refused', not accepted)
        result[str(seed)] = dict(resume_sha256=PARENTS[seed], summary_resume_sha256=summary['files']['resume']['sha256'],
                                 weights_sha256=original['learner_weights_sha256'], trainer_steps=original['counters']['trainer_steps'])
    return result


def phase4d4g_inventory():
    root = PHASE4D4G / 'experiment-output/phase4d4g-primary'
    files = {p.relative_to(root).as_posix(): sha(p) for p in sorted(root.rglob('*')) if p.is_file()}
    head = subprocess.run(['git', '-C', str(PHASE4D4G), 'rev-parse', 'HEAD'], check=True, capture_output=True,
                          text=True).stdout.strip()
    status = subprocess.run(['git', '-C', str(PHASE4D4G), 'status', '--porcelain'], check=True, capture_output=True,
                            text=True).stdout
    return dict(head=head, clean=not status.strip(), files=len(files), inventory_sha256=text_sha(files))


def isolation(check):
    from games.connect4.alphazero_v2.experimental import check_output_path
    try:
        check_output_path(OFFICIAL / 'runs/seed-42/anything.pt')
        refused = False
    except ValueError:
        refused = True
    check('experimental output is refused anywhere inside the official campaign', refused)
    head = subprocess.run(['git', '-C', str(HISTORICAL), 'rev-parse', 'HEAD'], check=True, capture_output=True,
                          text=True).stdout.strip()
    check('historical checkout is at the pinned reader commit', head == '1894bc4d8f035cf738df8e3fc0ce234f1ce2c8be', head)
    base = subprocess.run(['git', '-C', str(REPO), 'diff', '--stat', '7f5182daa10156e04a8b0f7727d9bec9cad39cf1', '--',
                           'games', 'api', 'ui', 'models'], check=True, capture_output=True, text=True).stdout
    check('no package, API, UI or model file differs from the Phase 4D.4H-A commit', not base.strip(), base)
    g = phase4d4g_inventory()
    check('Phase 4D.4G worktree is at its frozen commit and clean', g['head'] == PHASE4D4G_HEAD and g['clean'], g)
    return dict(historical_head=head, phase4d4g=g)


def main():
    identity = runtime()
    before = preserve()
    record, check = checks()
    from games.connect4.alphazero_v2.provenance import execution_source_identity
    result = dict(runtime=identity, execution_source_sha256=execution_source_identity()['sha256'])
    result['isolation'] = isolation(check)
    result['architecture_review'] = architecture_review(check)
    result['checkpoint_contracts'] = checkpoint_contracts(check)
    result['optimizer'] = optimizer_compatibility(check)
    result['fresh_process_resume'] = fresh_process_resume(check)
    result['cnn_backward_compatibility'] = cnn_backward_compatibility(check)
    after = preserve()
    check('all 150 official files are byte-identical before and after validation', before == after and len(after) == 150)
    check('Phase 4D.4G output inventory is unchanged by validation',
          phase4d4g_inventory() == result['isolation']['phase4d4g'])
    result['checks'] = record
    result['passed'] = len(record)
    write(RESULTS / 'validation.json', result)
    print(json.dumps(dict(passed=len(record), failed=sum(not r['passed'] for r in record)), indent=1))


if __name__ == '__main__':
    main()
