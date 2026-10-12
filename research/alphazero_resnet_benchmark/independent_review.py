"""Independent post-execution recomputation for Phase 4D.4H-B (reviewer's tool, written after the run).

Not part of the frozen source set. It reads the run directory, the tracked
results and the frozen dataset, and writes results/independent-recomputed.json.
It never trains, evaluates a model or reruns a benchmark stage. It imports
neither analyze.py nor metrics.py: every statistic below is recomputed from
the per-position tensors, the per-row suite records and the stage JSONs with
its own code and its own random stream. The only project code it calls is
common (paths, runtime, source hashes, the read-only official inventory) and
dataset.frozen (the strict historical read of the replay).
"""
import hashlib
import json
import random
import statistics
import subprocess
import time
from collections import Counter, defaultdict

import numpy as np
import torch

from . import common, dataset

RUN = common.OUTPUT / 'run'
RESULTS = common.RESULTS
NAMES = ('cnn', 'resnet3x64', 'resnet5x64')
RESNETS = NAMES[1:]
SEEDS = (42, 314159)
# Declared protocol, transcribed by hand from DESIGN.md (not read from design.json).
BUDGET = {'cnn': 6800, 'resnet3x64': 4700, 'resnet5x64': 3000}
MILESTONES = [0, 50, 100, 200, 400, 800, 1200, 1600, 2000, 2500, 3000, 3500, 4000, 4700, 5400, 6100, 6800]
COMMON = 3000
ORDER = {42: ['cnn', 'resnet3x64', 'resnet5x64'], 314159: ['resnet5x64', 'resnet3x64', 'cnn']}
TABLE = {42: dict(train_games=1739, train_positions=59094, held_games=309, held_positions=9762, novel=7616, families=180),
         314159: dict(train_games=1730, train_positions=55248, held_games=318, held_positions=9266, novel=7018, families=139)}
DESIGN_COMMIT = 'bc59dc19b6dd678d7dca788224750c3d17f575be'
G_BEFORE, G_AFTER = 'b4b1d9f5a42db0889fb2eed2acea6252da8526c9', '01526b813c226e1ab4d6ad6cb5fa453380b9b5c9'
REPLICATES = 10_000
REVIEW_SEED = 9_071_733           # the reviewer's stream; unrelated to the analysis seed 4404801
LOWER_BETTER = {'combined', 'policy_ce', 'value_mse', 'optimal_set_ce', 'mistake_rate', 'avoidable_rate'}


def load(path):
    return json.loads(path.read_text())


def file_sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def interval(samples):
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


def verdict(metric, per_seed, low, high):
    """DESIGN.md 'Uncertainty': both seeds on one side and the pooled 95% interval excludes zero on that side."""
    sign = -1.0 if metric in LOWER_BETTER else 1.0
    points = [sign * p for p in per_seed]
    a, b = sorted((sign * low, sign * high))
    if all(p > 0 for p in points) and a > 0:
        return 'BETTER'
    if all(p < 0 for p in points) and b < 0:
        return 'WORSE'
    return 'INCONCLUSIVE'


# 1. Dataset: split, leakage, labels -----------------------------------------------------------

def mirror_key(board):
    """My own canonical key: the lexicographically smaller of the board bytes and its left-right mirror."""
    a = np.asarray(board, dtype=np.int8)
    return min(a.tobytes(), a[:, ::-1].tobytes())


def boards_from_moves(moves):
    """Actor-relative pre-move boards rebuilt from the move list alone (row 0 is the top row, X moves first)."""
    board, heights, out = np.zeros((6, 7), dtype=np.int8), [0] * 7, []
    for ply, column in enumerate(moves):
        out.append(board.copy() if ply % 2 == 0 else -board)
        board[5 - heights[column], column] = 1 if ply % 2 == 0 else -1
        heights[column] += 1
    return out


def position_after(moves):
    """Actor-relative board after a move list (for the development-suite rows)."""
    board, heights = np.zeros((6, 7), dtype=np.int8), [0] * 7
    for ply, column in enumerate(moves):
        board[5 - heights[column], column] = 1 if ply % 2 == 0 else -1
        heights[column] += 1
    return board if len(moves) % 2 == 0 else -board


def dataset_checks(seed, data):
    games, train, held = data['games'], data['train'], data['held']
    train_games, held_games = sorted({g for _, g, _ in train}), sorted({g for _, g, _ in held})
    # (a) every stored observation equals the board rebuilt from the game's moves
    mismatched = 0
    for game in games:
        rebuilt = boards_from_moves(list(game.moves))
        if len(rebuilt) != len(game.examples):
            mismatched += 1
            continue
        for ply, (example, board) in enumerate(zip(game.examples, rebuilt)):
            if example.ply != ply or not np.array_equal(np.asarray(example.observation, dtype=np.int8), board):
                mismatched += 1
    # (b) my keys, my novelty mask
    train_keys = {mirror_key(e.observation) for e, _, _ in train}
    held_keys = [mirror_key(e.observation) for e, _, _ in held]
    novel = np.array([k not in train_keys for k in held_keys])
    plies = np.array([e.ply for e, _, _ in held])
    family = np.array([f for _, _, f in held])
    # (c) my own components over ALL games on shared keys at ply >= 8 (label propagation, not their union-find)
    by_key = defaultdict(set)
    for index, game in enumerate(games):
        for example in game.examples[8:]:
            by_key[mirror_key(example.observation)].add(index)
    neighbours = defaultdict(set)
    for members in by_key.values():
        if len(members) > 1:
            first = next(iter(members))
            for other in members:
                neighbours[first].add(other)
                neighbours[other].add(first)
    component, label = {}, 0
    for start in range(len(games)):
        if start in component:
            continue
        stack = [start]
        component[start] = label
        while stack:
            for other in neighbours[stack.pop()]:
                if other not in component:
                    component[other] = label
                    stack.append(other)
        label += 1
    theirs = data['family_of']
    pairs = {(component[i], theirs[i]) for i in range(len(games))}
    partition_equal = len(pairs) == label == len(set(theirs.values()))
    # (d) direct leakage checks that do not rely on any family label
    deep_train = {mirror_key(e.observation) for e, _, _ in train if e.ply >= 8}
    deep_shared = sum(1 for (e, _, _), k in zip(held, held_keys) if e.ply >= 8 and k in deep_train)
    train_families, held_families = {f for _, _, f in train}, {f for _, _, f in held}
    sizes = Counter(theirs.values())
    novel_family_positions = Counter(family[novel].tolist())
    counts = np.array(sorted(novel_family_positions.values(), reverse=True), dtype=float)
    owners = defaultdict(set)
    for key, fam, is_novel in zip(held_keys, family, novel):
        if is_novel:
            owners[key].add(int(fam))
    shared_keys = {k for k, v in owners.items() if len(v) > 1}
    outcomes = np.array([e.outcome for e, _, _ in held], dtype=float)
    visits = np.array([e.visits for e, _, _ in held])
    tied = (visits == visits.max(1, keepdims=True)).sum(1) > 1
    probe = data['probe']
    train_exact = {np.asarray(e.observation, dtype=np.int8).tobytes() for e, _, _ in train}
    exact = sum(1 for (e, _, _), n in zip(held, novel) if not n and np.asarray(e.observation, dtype=np.int8).tobytes() in train_exact)

    def prefix(game, n):
        moves = tuple(game.moves[:n])
        return min(moves, tuple(6 - m for m in moves))
    shared_prefix = {}
    for n in (6, 8):
        known = {prefix(games[i], n) for i in train_games}
        shared_prefix[str(n)] = sum(prefix(games[i], n) in known for i in held_games) / len(held_games)
    held_key_set = set(held_keys)
    suite_overlap = {}
    for suite in ('solved', 'tactical'):
        rows = load(common.REPO / f'games/connect4/alphazero_v2/frozen/{suite}-development.json')['rows']
        keys = [mirror_key(position_after(r['moves'])) for r in rows]
        suite_overlap[suite] = dict(in_training=sum(k in train_keys for k in keys), in_held_out=sum(k in held_key_set for k in keys))
    expected = TABLE[seed]
    measured = dict(train_games=len(train_games), train_positions=len(train), held_games=len(held_games),
                    held_positions=len(held), novel=int(novel.sum()), families=len(held_families))
    return dict(
        counts=measured, counts_match_design_table=measured == expected,
        observations_differing_from_boards_rebuilt_from_moves=mismatched,
        games_in_both_splits=len(set(train_games) & set(held_games)),
        families_in_both_splits=len(train_families & held_families),
        my_components_equal_their_families=partition_equal, my_components=label,
        held_out_positions_at_ply_ge_8_sharing_a_key_with_training=deep_shared,
        my_novel_mask_equals_theirs=bool(np.array_equal(novel, np.array(data['novel']))),
        novel_positions_whose_key_is_in_training=int(sum(k in train_keys for k, n in zip(held_keys, novel) if n)),
        held_out_positions_at_ply_ge_8=int((plies >= 8).sum()),
        held_out_positions_at_ply_ge_8_not_novel=int(((plies >= 8) & ~novel).sum()),
        novel_positions_before_ply_8=int(((plies < 8) & novel).sum()),
        largest_held_out_family_games=max(sizes[f] for f in held_families), oversized_threshold_games=0.05 * len(games),
        largest_families_games=sorted(sizes.values(), reverse=True)[:3],
        held_out_families_with_novel_positions=len(novel_family_positions),
        novel_positions_per_family_top5=[int(c) for c in counts[:5]],
        novel_share_of_five_largest_families=float(counts[:5].sum() / counts.sum()),
        effective_number_of_families=float(counts.sum() ** 2 / (counts ** 2).sum()),
        novel_keys_distinct=len(owners), novel_keys_shared_between_held_out_families=len(shared_keys),
        novel_positions_on_shared_keys=int(sum(1 for k, n in zip(held_keys, novel) if n and k in shared_keys)),
        novel_positions_with_tied_most_visited_move=int((tied & novel).sum()),
        probe_positions=len(probe), probe_within_training=bool(max(probe) < len(train) and len(set(probe)) == len(probe)),
        held_out_seen_exact=exact, held_out_seen_mirror_only=int((~novel).sum()) - exact,
        distinct_training_positions=len(train_keys),
        mean_game_length=dict(train=statistics.fmean(len(games[i].moves) for i in train_games),
                              held_out=statistics.fmean(len(games[i].moves) for i in held_games)),
        held_out_games_sharing_a_mirror_canonical_move_prefix_with_training=shared_prefix,
        development_suite_overlap=suite_overlap,
        constant_zero_value_mse_on_novel=float((outcomes[novel] ** 2).mean()),
        novel_outcome_mean=float(outcomes[novel].mean()),
    ), novel, family, outcomes


def sampling_stream(seed, count):
    """Regenerate the declared minibatch and reflection streams from their domains alone and hash them as learn.py does."""
    def stream(domain):
        return random.Random(int(hashlib.sha256(f'{domain}:{seed}'.encode()).hexdigest(), 16))
    sampler, reflector = stream('phase4d4h-b-fixed-data-sampling'), stream('connect4-alphazero-v2-horizontal-reflection')
    digest, hashes, reflected, seen = hashlib.sha256(), {0: hashlib.sha256().hexdigest()}, {0: 0}, set()
    total, coverage = 0, {}
    for update in range(1, max(BUDGET.values()) + 1):
        positions = sampler.sample(range(count), 128)
        flags = [reflector.random() < 0.5 for _ in positions]
        digest.update(json.dumps([positions, flags]).encode())
        total += sum(flags)
        seen.update(positions)
        if update in MILESTONES:
            hashes[update], reflected[update] = digest.hexdigest(), total
            coverage[update] = dict(distinct_positions_drawn=len(seen), passes_over_training_set=update * 128 / count)
    return hashes, reflected, coverage


# 2. Held-out contrasts from the per-position tensors ------------------------------------------------

def positions(seed, name, updates):
    raw = torch.load(RUN / f'learning-seed-{seed}' / f'{name}-positions-{updates:05d}.pt')
    return {k: v.double().numpy() for k, v in raw.items()}


def family_sums(scores, novel, family):
    """Per-family [positions, sum CE, sum SE, sum top-1] over novel positions; families in ascending label order."""
    labels = np.unique(family[novel])
    rows = np.zeros((len(labels), 4))
    for k, label in enumerate(labels):
        mask = novel & (family == label)
        rows[k] = [mask.sum(), scores['ce'][mask].sum(), scores['se'][mask].sum(), scores['top1'][mask].sum()]
    return labels, rows


METRICS = ('policy_ce', 'value_mse', 'top1', 'combined')


def contrast_statistics(sums_resnet, sums_cnn, index=None):
    """ResNet minus CNN, position-weighted over whole families. index: (replicates, families) resampled family rows."""
    delta = sums_resnet[:, 1:] - sums_cnn[:, 1:]
    weight = sums_cnn[:, 0]
    if index is None:
        d = delta.sum(0) / weight.sum()
        return np.array([d[0], d[1], d[2], d[0] + d[1]])
    d = delta[index].sum(1) / weight[index].sum(1)[:, None]
    return np.column_stack([d, d[:, 0] + d[:, 1]])


def pooled_contrast(sums, indices):
    """sums: {seed: (resnet rows, cnn rows)}. Equal seed weight; families resampled independently within each seed."""
    result, replicates = {}, []
    for seed in SEEDS:
        resnet, cnn = sums[seed]
        point = contrast_statistics(resnet, cnn)
        samples = contrast_statistics(resnet, cnn, indices[seed])
        replicates.append(samples)
        result[str(seed)] = {m: dict(difference=float(point[k]), low=interval(samples[:, k])[0], high=interval(samples[:, k])[1])
                             for k, m in enumerate(METRICS)}
    pooled = sum(replicates) / len(replicates)
    result['pooled'] = {m: dict(difference=float(np.mean([result[str(s)][m]['difference'] for s in SEEDS])),
                                low=interval(pooled[:, k])[0], high=interval(pooled[:, k])[1]) for k, m in enumerate(METRICS)}
    result['verdicts'] = {m: verdict(m, [result[str(s)][m]['difference'] for s in SEEDS], result['pooled'][m]['low'],
                                     result['pooled'][m]['high']) for m in METRICS}
    return result


def jackknife_se(sums_resnet, sums_cnn):
    """Delete-one-family jackknife standard error of the same ratio estimator, in METRICS order."""
    delta = sums_resnet[:, 1:] - sums_cnn[:, 1:]
    delta = np.column_stack([delta, delta[:, 0] + delta[:, 1]])
    weight = sums_cnn[:, 0]
    leave = (delta.sum(0) - delta) / (weight.sum() - weight)[:, None]
    families = len(weight)
    return np.sqrt((families - 1) / families * ((leave - leave.mean(0)) ** 2).sum(0))


def compare_with_analysis(mine, theirs, verdicts):
    """Largest point and interval-endpoint differences from results/comparisons.json and any verdict that differs."""
    point = endpoint = 0.0
    flips = {}
    for label in (*map(str, SEEDS), 'pooled'):
        for metric in METRICS:
            a, b = mine[label][metric], theirs[label][metric]
            point = max(point, abs(a['difference'] - b['difference']))
            endpoint = max(endpoint, abs(a['low'] - b['low']), abs(a['high'] - b['high']))
    for metric in METRICS:
        if mine['verdicts'][metric] != verdicts[metric]:
            flips[metric] = dict(mine=mine['verdicts'][metric], analysis=verdicts[metric])
    return dict(max_point_difference=point, max_interval_endpoint_difference=endpoint, verdict_differences=flips)


# 3. Development suites ----------------------------------------------------------------------------

def my_clusters(rows):
    """DESIGN.md: mirror families unioned with reflection-equivalent prefixes through ply 8. Own label propagation."""
    prefixes = [tuple(r['moves'][:8]) for r in rows]
    adjacency = defaultdict(set)
    for i in range(len(rows)):
        for j in range(i):
            n = min(len(prefixes[i]), len(prefixes[j]))
            p, q = prefixes[i][:n], prefixes[j][:n]
            if rows[i]['family'] == rows[j]['family'] or p == q or p == tuple(6 - c for c in q):
                adjacency[i].add(j)
                adjacency[j].add(i)
    labels, current = [-1] * len(rows), 0
    for start in range(len(rows)):
        if labels[start] >= 0:
            continue
        stack, labels[start] = [start], current
        while stack:
            for other in adjacency[stack.pop()]:
                if labels[other] < 0:
                    labels[other] = current
                    stack.append(other)
        current += 1
    return np.array(labels)


SUITE_FIELDS = {'solved': (('accuracy', 'correct', None), ('optimal_set_ce', 'ce', None), ('value_mse', 'se', None),
                           ('mistake_rate', 'mistake', None)),
                'tactical': (('accuracy', 'correct', None), ('unique_win_accuracy', 'correct', 'unique_win'),
                             ('unique_safe_accuracy', 'correct', 'unique_safe'), ('avoidable_rate', 'avoidable', None))}


def suite_contrasts(records, packages, rng):
    """records: {seed: {arch: per-row list}} for one suite at one checkpoint pairing; cluster bootstrap, rows paired."""
    out = {}
    for suite in ('solved', 'tactical'):
        rows = packages[suite]
        clusters = my_clusters(rows)
        stored = np.array([r['cluster'] for r in records[SEEDS[0]]['cnn'][suite]])
        same_partition = len({(a, b) for a, b in zip(clusters, stored)}) == len(set(clusters)) == len(set(stored))
        ids_ok = all([r['id'] for r in records[s][n][suite]] == [r['id'] for r in rows] for s in SEEDS for n in NAMES)
        # the recorded 'correct' flags against the development package itself
        flags_ok = True
        for s in SEEDS:
            for n in NAMES:
                for record, row in zip(records[s][n][suite], rows):
                    truth = (record['action'] in row['optimal_actions']) if suite == 'solved' else (record['action'] == row['expected_action'])
                    flags_ok &= int(truth) == record['correct']
        block = dict(my_clusters=int(clusters.max() + 1), my_clusters_equal_recorded=bool(same_partition),
                     row_ids_match_package_order=bool(ids_ok), correct_flags_match_package=bool(flags_ok), resnets={})
        for name in RESNETS:
            entry = {}
            for metric, field, category in SUITE_FIELDS[suite]:
                keep = np.array([category is None or r.get('category') == category for r in rows])
                per_seed = {s: np.array([a[field] - b[field] for a, b in zip(records[s][name][suite], records[s]['cnn'][suite])],
                                        dtype=float)[keep] for s in SEEDS}
                labels = clusters[keep]
                unique = np.unique(labels)
                member = (labels[None, :] == unique[:, None]).astype(float)        # clusters x rows
                index = rng.integers(0, len(unique), size=(REPLICATES, len(unique)))
                size = member.sum(1)
                cell = {}
                for label, values in [*((str(s), per_seed[s]) for s in SEEDS), ('pooled', np.mean([per_seed[s] for s in SEEDS], axis=0))]:
                    total = member @ values
                    samples = total[index].sum(1) / size[index].sum(1)
                    low, high = interval(samples)
                    cell[label] = dict(difference=float(values.mean()), low=low, high=high, rows=int(keep.sum()), clusters=len(unique))
                cell['verdict'] = verdict(metric, [cell[str(s)]['difference'] for s in SEEDS], cell['pooled']['low'], cell['pooled']['high'])
                entry[metric] = cell
            block['resnets'][name] = entry
        out[suite] = block
    return out


# 4. Checkpoint sensitivity from the per-milestone family sums -----------------------------------------

def milestone_sums(run, updates):
    row = next(r for r in run['milestones'] if r['updates'] == updates)
    families = row['held_out']['novel_families']
    return np.array([families[k] for k in sorted(families, key=int)], dtype=float)


def sweep(runs, indices, pairs):
    """pairs: [(label, resnet updates, cnn updates)] for one ResNet; returns per-pair contrasts and verdicts."""
    out = {}
    for name in RESNETS:
        rows = {}
        for label, resnet_updates, cnn_updates in pairs[name]:
            sums = {s: (milestone_sums(runs[s][name], resnet_updates), milestone_sums(runs[s]['cnn'], cnn_updates)) for s in SEEDS}
            c = pooled_contrast(sums, indices)
            rows[label] = dict(resnet_updates=resnet_updates, cnn_updates=cnn_updates,
                               per_seed={m: [c[str(s)][m]['difference'] for s in SEEDS] for m in METRICS},
                               pooled={m: [c['pooled'][m]['difference'], c['pooled'][m]['low'], c['pooled'][m]['high']] for m in METRICS},
                               verdicts=c['verdicts'])
        out[name] = rows
    return out


def interpolated_sums(run, seconds):
    """Per-family sums linearly interpolated in measured optimizer seconds between the two bracketing milestones."""
    rows = run['milestones']
    for a, b in zip(rows, rows[1:]):
        if a['train_seconds'] <= seconds <= b['train_seconds']:
            w = (seconds - a['train_seconds']) / (b['train_seconds'] - a['train_seconds'])
            return (1 - w) * milestone_sums(run, a['updates']) + w * milestone_sums(run, b['updates'])
    return None


def averaged_sums(run, updates):
    """Mean of the per-family sums over several checkpoints of one run (a checkpoint-averaged score per family)."""
    return np.mean([milestone_sums(run, u) for u in updates], axis=0)


# 5. Protocol conformance, timing, integrity ---------------------------------------------------------

def git(directory, *arguments):
    done = subprocess.run(['git', '--no-optional-locks', '-C', str(directory), *arguments], capture_output=True, text=True)
    return done.returncode, done.stdout.strip()


def conformance(runs, summaries, status, design, regenerated):
    inference, training, collection = (load(RUN / f'{k}.json') for k in ('inference', 'training', 'collection'))
    freeze = load(RESULTS / 'freeze.json')
    dataset_freeze = load(RESULTS / 'dataset-freeze.json')
    checks = {}
    checks['status.design_commit is bc59dc1'] = status['design_commit'] == DESIGN_COMMIT
    checks['worktree HEAD is the design commit'] = git(common.REPO, 'rev-parse', 'HEAD')[1] == DESIGN_COMMIT
    checks['design commit is on the pushed branch'] = git(common.REPO, 'rev-parse', 'origin/phase4d4h-b-resnet-benchmark')[1] == DESIGN_COMMIT
    commit_epoch = int(git(common.REPO, 'show', '-s', '--format=%ct', DESIGN_COMMIT)[1])
    checks['design commit precedes the run start'] = commit_epoch < status['start_epoch']
    checks['frozen files unmodified in the working tree'] = git(common.REPO, 'status', '--porcelain', '--', *(
        f'research/alphazero_resnet_benchmark/{n}' for n in common.FROZEN), 'games')[1] == ''
    checks['source_map() equals freeze.json files'] = common.source_map() == freeze['files']
    checks['stage runtimes equal the frozen runtime'] = all(x['runtime'] == freeze['runtime'] for x in (inference, training, collection)) \
        and all(summaries[s]['runtime'] == freeze['runtime'] for s in SEEDS)
    checks['stage order and outcomes'] = [(s['stage'], s['outcome'], s['returncode']) for s in status['stages']] == [
        (n, 'COMPLETE', 0) for n in ('inference', 'training', 'collection', 'learning-seed-42', 'learning-seed-314159')]
    checks['stages do not overlap'] = all(a['end_epoch'] <= b['start_epoch'] for a, b in zip(status['stages'], status['stages'][1:]))
    checks['run within its ceiling and the 75-minute maximum'] = status['end_epoch'] < status['deadline_epoch'] and \
        status['elapsed_seconds'] + design['budget']['prior_measured_seconds'] <= 75 * 60
    checks['design.json learning plan equals DESIGN.md'] = design['learning']['updates'] == BUDGET and \
        design['learning']['milestones'] == MILESTONES and design['learning']['common_updates'] == COMMON and \
        {int(k): v for k, v in design['learning']['order'].items()} == ORDER
    for seed in SEEDS:
        tag = f'seed {seed}: '
        checks[tag + 'order'] = summaries[seed]['order'] == ORDER[seed] and set(summaries[seed]['architectures']) == set(NAMES)
        finished = [(RUN / f'learning-seed-{seed}' / f'{n}.json').stat().st_mtime for n in ORDER[seed]]
        checks[tag + 'result files written in the declared order'] = finished == sorted(finished)
        checks[tag + 'dataset identity equals the freeze'] = summaries[seed]['dataset'] == dataset_freeze['seeds'][str(seed)]['identity']
        checks[tag + 'budgets'] = all(runs[seed][n]['updates'] == BUDGET[n] for n in NAMES)
        checks[tag + 'milestones'] = all([r['updates'] for r in runs[seed][n]['milestones']] == [m for m in MILESTONES if m <= BUDGET[n]]
                                         for n in NAMES)
        checks[tag + 'batch 128, AdamW 3e-4, betas, eps, decay 1e-4, clip 5, reflection 0.5'] = all(
            runs[seed][n]['batch_size'] == 128 and runs[seed][n]['optimizer'] == dict(
                type='AdamW', learning_rate=3e-4, betas=[0.9, 0.999], eps=1e-8, weight_decay=1e-4, max_grad_norm=5.0,
                reflection_probability=0.5) for n in NAMES)
        hashes, reflected, _ = regenerated[seed]
        checks[tag + 'sampling hash identical across architectures at every shared milestone'] = all(
            len({r['sampling_sha256'] for n in NAMES for r in runs[seed][n]['milestones'] if r['updates'] == m}) == 1 for m in MILESTONES)
        checks[tag + 'sampling hash equals my regeneration of the declared streams'] = all(
            r['sampling_sha256'] == hashes[r['updates']] and r['reflected'] == reflected[r['updates']]
            for n in NAMES for r in runs[seed][n]['milestones'])
        checks[tag + 'model files match their recorded SHA-256'] = all(
            file_sha(RUN / f'learning-seed-{seed}' / f'{n}-model-{int(u):05d}.pt') == digest
            for n in NAMES for u, digest in runs[seed][n]['models'].items())
        checks[tag + 'optimizer clock increases and excludes evaluation'] = all(
            all(a['train_seconds'] < b['train_seconds'] for a, b in zip(runs[seed][n]['milestones'], runs[seed][n]['milestones'][1:]))
            and runs[seed][n]['train_seconds'] + runs[seed][n]['evaluation_seconds'] <= runs[seed][n]['wall_seconds'] for n in NAMES)
    rows = inference['timing']['rows']
    cells = Counter((r['architecture'], r['batch'], r['path']) for r in rows)
    checks['inference: 7 repetitions of every cell with the declared call counts'] = len(cells) == 24 and set(cells.values()) == {7} and all(
        r['calls'] == {1: 400, 8: 250, 16: 160, 32: 100}[r['batch']] for r in rows)
    first = {}
    for r in rows:
        first.setdefault(r['repetition'], r['architecture'])
    checks['inference: architecture order rotated by repetition'] = all(first[k] == NAMES[k % 3] for k in range(7))
    checks['inference: 12 fresh-process memory cells'] = len(inference['memory']) == 12
    plain = [(r['architecture'], r['stripped_batch_norm']) for r in training['rows']]
    checks['training: two passes, reversed, 60 updates at batch 128'] = plain == [
        ('cnn', False), ('resnet3x64', False), ('resnet3x64', True), ('resnet5x64', False), ('resnet5x64', True),
        ('resnet5x64', False), ('resnet5x64', True), ('resnet3x64', False), ('resnet3x64', True), ('cnn', False)] and all(
        r['updates'] == 60 and r['batch'] == 128 for r in training['rows'])
    checks['collection: 2 x 16 games, reversed order, 256 simulations per move, complete'] = [
        (r['architecture'], r['game_ids']) for r in collection['rows']] == [
        ('cnn', [0, 15]), ('resnet3x64', [0, 15]), ('resnet5x64', [0, 15]), ('resnet5x64', [16, 31]), ('resnet3x64', [16, 31]),
        ('cnn', [16, 31])] and all(r['complete'] and r['games'] == 16 and r['simulations_per_move'] == 256
                                   and r['simulations'] == 256 * r['plies'] and r['max_active'] == 8 for r in collection['rows'])
    return checks, inference, training, collection


def timing(inference, training, collection, runs, status):
    rows = inference['timing']['rows']
    latency = {}
    for path in ('forward', 'predict_batch'):
        for batch in (1, 8, 16, 32):
            cell = {}
            for name in NAMES:
                means = [r['mean_ms'] for r in sorted((r for r in rows if (r['path'], r['batch'], r['architecture']) == (path, batch, name)),
                                                      key=lambda r: r['repetition'])]
                centre = statistics.median(means)
                cell[name] = dict(median_ms=centre, min_ms=min(means), max_ms=max(means), range_percent=100 * (max(means) - min(means)) / centre,
                                  ms_per_position=centre / batch, repetitions_more_than_5_percent_above_median=sum(m > 1.05 * centre for m in means))
            for name in RESNETS:
                ratios = [a['mean_ms'] / b['mean_ms'] for a in rows for b in rows
                          if (a['path'], a['batch'], a['architecture']) == (path, batch, name)
                          and (b['path'], b['batch'], b['architecture']) == (path, batch, 'cnn') and a['repetition'] == b['repetition']]
                cell[name].update(ratio_median=statistics.median(ratios), ratio_min=min(ratios), ratio_max=max(ratios))
            latency[f'{path}:{batch}'] = cell
    full = defaultdict(list)
    for r in training['rows']:
        if not r['stripped_batch_norm']:
            full[r['architecture']].append(r)
    update = {n: dict(mean_ms_by_pass=[r['trainer_step']['mean_ms'] for r in full[n]],
                      median_ms_by_pass=[r['trainer_step']['median_ms'] for r in full[n]],
                      p95_ms_by_pass=[r['trainer_step']['p95_ms'] for r in full[n]],
                      peak_rss_mib_by_pass=[r['peak_rss_mib'] for r in full[n]]) for n in NAMES}
    for n in RESNETS:
        update[n]['ratio_to_cnn_of_pass_means'] = statistics.fmean(update[n]['mean_ms_by_pass']) / statistics.fmean(update['cnn']['mean_ms_by_pass'])
        update[n]['ratio_to_cnn_of_pass_medians'] = statistics.fmean(update[n]['median_ms_by_pass']) / statistics.fmean(update['cnn']['median_ms_by_pass'])
        # pass 1 ran cnn, 3x64, 5x64; pass 2 ran 5x64, 3x64, cnn: list index 0 is pass 1 for every architecture
        update[n]['ratio_by_pass_medians'] = [a / b for a, b in zip(update[n]['median_ms_by_pass'], update['cnn']['median_ms_by_pass'])]
        update[n]['pass1_over_pass2_median'] = update[n]['median_ms_by_pass'][0] / update[n]['median_ms_by_pass'][1]
        bare = [r for r in training['rows'] if r['architecture'] == n and r['stripped_batch_norm']]
        update[n]['staged_total_ms_by_pass'] = [r['staged_total_ms'] for r in full[n]]
        update[n]['staged_total_without_batch_norm_ms_by_pass'] = [r['staged_total_ms'] for r in bare]
        update[n]['batch_norm_overhead_ms_by_pass'] = [a['staged_total_ms'] - b['staged_total_ms'] for a, b in zip(full[n], bare)]
    update['cnn']['pass1_over_pass2_median'] = update['cnn']['median_ms_by_pass'][0] / update['cnn']['median_ms_by_pass'][1]
    blocks = defaultdict(list)
    for r in collection['rows']:
        blocks[r['architecture']].append(r)                   # block order differs: sort by first game id
    collect = {}
    for n in NAMES:
        ordered = sorted(blocks[n], key=lambda r: r['game_ids'][0])
        wall, positions_ = sum(r['wall_seconds'] for r in ordered), sum(r['neural_positions'] for r in ordered)
        collect[n] = dict(ms_per_neural_position=1e3 * wall / positions_,
                          ms_per_neural_position_by_block=[1e3 * r['wall_seconds'] / r['neural_positions'] for r in ordered],
                          inference_ms_per_neural_position_by_block=[1e3 * r['inference_seconds'] / r['neural_positions'] for r in ordered],
                          mean_batch_by_block=[r['mean_batch'] for r in ordered],
                          cpu_over_wall_by_block=[r['cpu_seconds'] / r['wall_seconds'] for r in ordered],
                          mean_game_length_by_block=[statistics.fmean(r['game_lengths']) for r in ordered],
                          mean_batch=positions_ / sum(r['forward_calls'] for r in ordered),
                          inference_ms_per_neural_position=1e3 * sum(r['inference_seconds'] for r in ordered) / positions_,
                          inference_share_of_wall=sum(r['inference_seconds'] for r in ordered) / wall,
                          peak_rss_mib=max(r['peak_rss_mib'] for r in ordered))
    for n in RESNETS:
        collect[n]['ratio_to_cnn'] = collect[n]['ms_per_neural_position'] / collect['cnn']['ms_per_neural_position']
        collect[n]['inference_only_ratio_to_cnn'] = collect[n]['inference_ms_per_neural_position'] / collect['cnn']['inference_ms_per_neural_position']
        collect[n]['ratio_to_cnn_by_block'] = [a / b for a, b in zip(collect[n]['ms_per_neural_position_by_block'],
                                                                     collect['cnn']['ms_per_neural_position_by_block'])]
    learn = {}
    for seed in SEEDS:
        entry = {}
        for n in NAMES:
            ms = runs[seed][n]['milestones']
            per = [1e3 * (b['step_seconds'] - a['step_seconds']) / (b['updates'] - a['updates']) for a, b in zip(ms, ms[1:])]
            entry[n] = dict(step_ms_per_update=1e3 * runs[seed][n]['step_seconds'] / runs[seed][n]['updates'],
                            optimizer_ms_per_update=1e3 * runs[seed][n]['train_seconds'] / runs[seed][n]['updates'],
                            interval_step_ms_min=min(per), interval_step_ms_max=max(per),
                            interval_spread_percent=100 * (max(per) - min(per)) / statistics.median(per),
                            optimizer_seconds=runs[seed][n]['train_seconds'],
                            cpu_over_wall=runs[seed][n]['cpu_seconds'] / runs[seed][n]['wall_seconds'],
                            process_peak_rss_mib=runs[seed][n]['peak_rss_mib'])
        for n in RESNETS:
            entry[n]['ratio_to_cnn'] = entry[n]['step_ms_per_update'] / entry['cnn']['step_ms_per_update']
        learn[str(seed)] = entry
    learn['seed42_over_seed314159_step_ms'] = {n: learn['42'][n]['step_ms_per_update'] / learn['314159'][n]['step_ms_per_update'] for n in NAMES}
    pilot = load(RESULTS / 'pilot.json')
    reference = dict(pilot_update_ms={n: pilot['training'][n]['trainer_step']['mean_ms'] for n in NAMES},
                     pilot_ms_per_neural_position={n: 1e3 * pilot['collection'][n]['wall_seconds'] / pilot['collection'][n]['neural_positions']
                                                   for n in NAMES})
    cpu, previous = [], 0.0
    for s in status['stages']:
        cpu.append(dict(stage=s['stage'], children_cpu_over_wall=(s['children_cpu_seconds_so_far'] - previous) / s['wall_seconds']))
        previous = s['children_cpu_seconds_so_far']
    memory = defaultdict(dict)
    for r in inference['memory']:
        memory[str(r['batch'])][r['architecture']] = r['peak_rss_mib']
    return dict(inference=latency, training_update=update, collection=collect, learning=learn, pilot=reference,
                stage_cpu_over_wall=cpu, inference_peak_rss_mib=memory)


def integrity(status):
    """Read-only re-derivation of the Phase 4D.4G claim: the commit moved, the 151-file output inventory did not."""
    worktree = common.REPO.parent / 'phase4d4g-worktree'
    root = worktree / 'experiment-output/phase4d4g-primary'
    files = {p.relative_to(root).as_posix(): file_sha(p) for p in sorted(root.rglob('*')) if p.is_file()}
    digest = hashlib.sha256(json.dumps(files, sort_keys=True, allow_nan=False).encode()).hexdigest()
    newest = max(p.stat().st_mtime for p in root.rglob('*') if p.is_file())
    changed = git(worktree, 'diff', '--name-status', G_BEFORE, G_AFTER)[1].splitlines()
    paths = [line.split('\t')[-1] for line in changed]
    written = [(worktree / p).stat().st_mtime for p in paths if (worktree / p).exists()]
    commits = git(worktree, 'rev-list', f'{G_BEFORE}..{G_AFTER}')[1].split()
    official = common.preserve()                                     # raises if any of the 150 official files changed
    return dict(
        head_now=git(worktree, 'rev-parse', 'HEAD')[1], clean_now=git(worktree, 'status', '--porcelain')[1] == '',
        recorded_before=status['phase4d4g_before'], recorded_after=status['phase4d4g_after'],
        fields_differing_before_after=sorted(k for k in status['phase4d4g_before'] if status['phase4d4g_before'][k] != status['phase4d4g_after'][k]),
        after_descends_from_before=git(worktree, 'merge-base', '--is-ancestor', G_BEFORE, G_AFTER)[0] == 0,
        new_commits=len(commits), new_commit_epoch=int(git(worktree, 'show', '-s', '--format=%ct', G_AFTER)[1]),
        changed_paths=len(paths), changed_status_counts=dict(Counter(line.split('\t')[0] for line in changed)),
        changed_top_level=dict(Counter(p.split('/')[0] for p in paths)),
        changed_paths_under_games_or_experiment_output=sum(p.startswith(('games/', 'experiment-output/')) for p in paths),
        output_tree_tracked_files=len(git(worktree, 'ls-files', 'experiment-output')[1].split()),
        inventory_files=len(files), inventory_sha256_recomputed=digest,
        inventory_matches_before_and_after=digest == status['phase4d4g_before']['inventory_sha256'] == status['phase4d4g_after']['inventory_sha256'],
        newest_inventory_file_epoch=newest, newest_inventory_file_precedes_run=newest < status['start_epoch'],
        committed_files_written_between_epochs=[min(written), max(written)],
        committed_files_written_during_run=sum(status['start_epoch'] <= t <= status['end_epoch'] for t in written),
        run_start_epoch=status['start_epoch'], run_end_epoch=status['end_epoch'],
        stage_windows={s['stage']: [s['start_epoch'], s['end_epoch']] for s in status['stages']},
        official_files_identical_now=len(official))


def main():
    started = time.time()
    common.runtime()
    rng = np.random.default_rng(REVIEW_SEED)
    design, status = load(RESULTS / 'design.json'), load(RUN / 'status.json')
    theirs = load(RESULTS / 'comparisons.json')
    runs = {s: {n: load(RUN / f'learning-seed-{s}' / f'{n}.json') for n in NAMES} for s in SEEDS}
    summaries = {s: load(RUN / f'learning-seed-{s}' / 'summary.json') for s in SEEDS}
    out = dict(reviewer_seed=REVIEW_SEED, replicates=REPLICATES)

    # Dataset, leakage, sampling stream
    split, masks, regenerated, coverage = {}, {}, {}, {}
    for seed in SEEDS:
        data = dataset.frozen(seed)
        split[str(seed)], novel, family, outcomes = dataset_checks(seed, data)
        masks[seed] = (novel, family, outcomes)
        hashes, reflected, cover = sampling_stream(seed, len(data['train']))
        regenerated[seed] = (hashes, reflected, cover)
        coverage[str(seed)] = {str(u): cover[u] for u in (3000, 4700, 6800)}
        del data
    out['dataset'] = split
    out['sampling_coverage'] = coverage

    # Per-position tensors: alignment with the held-out list, then the two primary comparisons
    indices, sums, alignment = {}, {}, {}
    for seed in SEEDS:
        novel, family, outcomes = masks[seed]
        sums[seed], worst_se, worst_mean, worst_family = {}, 0.0, 0.0, 0.0
        for name in NAMES:
            for updates in sorted({0, COMMON, BUDGET[name]}):
                scores = positions(seed, name, updates)
                if len(scores['ce']) != len(novel):
                    raise ValueError('Per-position tensor length differs from the frozen held-out list')
                # se must equal (value - outcome)^2 with MY outcome vector: proves the tensors are in held-out list order
                worst_se = max(worst_se, float(np.abs(scores['se'] - (scores['value'] - outcomes) ** 2).max()))
                labels, rows = family_sums(scores, novel, family)
                sums[seed][(name, updates)] = rows
                row = next(r for r in runs[seed][name]['milestones'] if r['updates'] == updates)['held_out']
                recorded = row['novel']
                worst_mean = max(worst_mean, abs(scores['ce'][novel].mean() - recorded['policy_ce']),
                                 abs(scores['se'][novel].mean() - recorded['value_mse']), abs(scores['top1'][novel].mean() - recorded['top1']))
                stored = milestone_sums(runs[seed][name], updates)
                if [int(k) for k in sorted(row['novel_families'], key=int)] != labels.tolist():
                    raise ValueError('Recorded family labels differ from mine')
                worst_family = max(worst_family, float(np.abs(stored - rows).max()))
        families = len(np.unique(family[novel]))
        indices[seed] = rng.integers(0, families, size=(REPLICATES, families))
        alignment[str(seed)] = dict(max_abs_se_minus_squared_error_from_my_outcomes=worst_se,
                                    max_abs_novel_mean_minus_recorded=worst_mean,
                                    max_abs_family_sum_minus_recorded=worst_family, bootstrap_families=families)
    out['position_tensor_alignment'] = alignment

    primary = {}
    for label, updates_of in (('equal_updates_3000', lambda n: COMMON), ('equal_wall_final', lambda n: BUDGET[n])):
        primary[label] = {}
        for name in RESNETS:
            mine = pooled_contrast({s: (sums[s][(name, updates_of(name))], sums[s][('cnn', updates_of('cnn'))]) for s in SEEDS}, indices)
            block = theirs[label]['resnets'][name]
            mine['comparison_with_analysis'] = compare_with_analysis(mine, block['held_out'], block['verdicts'])
            # Monte Carlo spread of the pooled interval over 20 further independent streams of mine
            spread = {m: dict(low=[], high=[], verdicts=Counter()) for m in METRICS}   # 20 streams
            for _ in range(20):
                again = {s: rng.integers(0, indices[s].shape[1], size=indices[s].shape) for s in SEEDS}
                redo = pooled_contrast({s: (sums[s][(name, updates_of(name))], sums[s][('cnn', updates_of('cnn'))]) for s in SEEDS}, again)
                for m in METRICS:
                    spread[m]['low'].append(redo['pooled'][m]['low'])
                    spread[m]['high'].append(redo['pooled'][m]['high'])
                    spread[m]['verdicts'][redo['verdicts'][m]] += 1
            mine['monte_carlo_spread'] = {m: dict(low_min=min(v['low']), low_max=max(v['low']), high_min=min(v['high']),
                                                  high_max=max(v['high']), verdicts=dict(v['verdicts']),
                                                  streams_with_interval_excluding_zero=sum((a > 0) or (b < 0) for a, b in zip(v['low'], v['high'])))
                                          for m, v in spread.items()}
            # Leave-one-family-out range of each seed's combined-loss difference (sign stability of the seed condition)
            jack = {}
            for s in SEEDS:
                a, b = sums[s][(name, updates_of(name))], sums[s][('cnn', updates_of('cnn'))]
                d = (a[:, 1] + a[:, 2]) - (b[:, 1] + b[:, 2])
                leave = (d.sum() - d) / (b[:, 0].sum() - b[:, 0])
                jack[str(s)] = dict(min=float(leave.min()), max=float(leave.max()), sign_changes=bool(leave.min() < 0 < leave.max()))
            mine['leave_one_family_out_combined'] = jack
            # A second interval that does not use the percentile bootstrap: jackknife SE, normal and t(22) multipliers.
            # 22 is the rounded-down effective number of families in seed 314159, the smaller of the two seeds.
            errors = {s: jackknife_se(sums[s][(name, updates_of(name))], sums[s][('cnn', updates_of('cnn'))]) for s in SEEDS}
            pooled_se = 0.5 * np.sqrt(sum(errors[s] ** 2 for s in SEEDS))
            mine['jackknife'] = {m: dict(
                seed_se={str(s): float(errors[s][k]) for s in SEEDS}, pooled_se=float(pooled_se[k]),
                normal_interval=[mine['pooled'][m]['difference'] - 1.959964 * float(pooled_se[k]),
                                 mine['pooled'][m]['difference'] + 1.959964 * float(pooled_se[k])],
                t22_interval=[mine['pooled'][m]['difference'] - 2.073873 * float(pooled_se[k]),
                              mine['pooled'][m]['difference'] + 2.073873 * float(pooled_se[k])],
                bootstrap_interval=[mine['pooled'][m]['low'], mine['pooled'][m]['high']]) for k, m in enumerate(METRICS)}
            for m in METRICS:
                low, high = mine['jackknife'][m]['t22_interval']
                mine['jackknife'][m]['verdict_with_t22_interval'] = verdict(m, [mine[str(s)][m]['difference'] for s in SEEDS], low, high)
            primary[label][name] = mine
    out['held_out_primary'] = primary

    # Value baseline: how the models compare with predicting zero on novel positions
    baseline = {}
    for seed in SEEDS:
        novel = masks[seed][0]
        zero = split[str(seed)]['constant_zero_value_mse_on_novel']
        baseline[str(seed)] = dict(constant_zero=zero, models={
            f'{n}@{u}': float(sums[seed][(n, u)][:, 2].sum() / novel.sum()) for n in NAMES for u in sorted({COMMON, BUDGET[n]})})
    out['novel_value_mse_against_constant_zero'] = baseline

    # Suites at the two primary comparisons
    packages = {s: load(common.REPO / f'games/connect4/alphazero_v2/frozen/{s}-development.json')['rows'] for s in ('solved', 'tactical')}
    suites = {}
    for label, updates_of in (('equal_updates_3000', lambda n: COMMON), ('equal_wall_final', lambda n: BUDGET[n])):
        records = {s: {n: {suite: next(r for r in runs[s][n]['milestones'] if r['updates'] == updates_of(n))['suites'][suite]['rows']
                           for suite in ('solved', 'tactical')} for n in NAMES} for s in SEEDS}
        mine = suite_contrasts(records, packages, rng)
        differences = {}
        for suite in ('solved', 'tactical'):
            for name in RESNETS:
                for metric, cell in mine[suite]['resnets'][name].items():
                    other = theirs[label]['resnets'][name][suite][metric]
                    recorded = theirs[label]['resnets'][name]['verdicts'][f'{suite}_{metric}']
                    cell['max_point_difference_from_analysis'] = max(abs(cell[k]['difference'] - other[k]['difference']) for k in ('42', '314159', 'pooled'))
                    cell['max_endpoint_difference_from_analysis'] = max(abs(cell[k][e] - other[k][e]) for k in ('42', '314159', 'pooled') for e in ('low', 'high'))
                    if cell['verdict'] != recorded:
                        differences[f'{name}:{suite}_{metric}'] = dict(mine=cell['verdict'], analysis=recorded)
        mine['verdict_differences'] = differences
        suites[label] = mine
    out['suites_primary'] = suites

    # Checkpoint sensitivity (uses the recorded per-family sums, validated above at 0 / 3000 / final)
    equal_updates = {n: [(str(u), u, u) for u in MILESTONES if 50 <= u <= BUDGET[n]] for n in RESNETS}
    equal_time = {'resnet3x64': [(f'{r}v{c}', r, c) for r in (3500, 4000, 4700) for c in (5400, 6100, 6800)],
                  'resnet5x64': [(f'{r}v{c}', r, c) for r in (2000, 2500, 3000) for c in (5400, 6100, 6800)]}
    sensitivity = dict(equal_updates=sweep(runs, indices, equal_updates), equal_time_neighbours=sweep(runs, indices, equal_time))
    windows = dict(equal_updates_2000_2500_3000={n: ((2000, 2500, 3000), (2000, 2500, 3000)) for n in RESNETS},
                   last_three_checkpoints={'resnet3x64': ((3500, 4000, 4700), (5400, 6100, 6800)),
                                           'resnet5x64': ((2000, 2500, 3000), (5400, 6100, 6800))})
    averaged = {}
    for label, plan in windows.items():
        averaged[label] = {}
        for name in RESNETS:
            c = pooled_contrast({s: (averaged_sums(runs[s][name], plan[name][0]), averaged_sums(runs[s]['cnn'], plan[name][1])) for s in SEEDS}, indices)
            averaged[label][name] = dict(resnet_checkpoints=plan[name][0], cnn_checkpoints=plan[name][1],
                                         per_seed={m: [c[str(s)][m]['difference'] for s in SEEDS] for m in METRICS},
                                         pooled={m: [c['pooled'][m]['difference'], c['pooled'][m]['low'], c['pooled'][m]['high']] for m in METRICS},
                                         verdicts=c['verdicts'])
    sensitivity['checkpoint_averaged'] = averaged
    # The declared secondary comparisons (other equal-update points; interpolated equal-time points), recomputed
    secondary = {}
    plans = {f'equal_updates_{u}': {s: {n: milestone_sums(runs[s][n], u) for n in NAMES} for s in SEEDS} for u in (400, 800, 1600)}
    limit = {s: min(runs[s][n]['train_seconds'] for n in NAMES) for s in SEEDS}
    for point in (60, 120, 240, 'matched'):
        plans[f'equal_seconds_{point}'] = {s: {n: interpolated_sums(runs[s][n], limit[s] if point == 'matched' else point)
                                               for n in NAMES} for s in SEEDS}
    for label, plan in plans.items():
        secondary[label] = {}
        for name in RESNETS:
            c = pooled_contrast({s: (plan[s][name], plan[s]['cnn']) for s in SEEDS}, indices)
            block = theirs[label]['resnets'][name]
            secondary[label][name] = dict(verdicts=c['verdicts'], pooled={m: [c['pooled'][m]['difference'], c['pooled'][m]['low'],
                                                                              c['pooled'][m]['high']] for m in METRICS},
                                          comparison_with_analysis=compare_with_analysis(c, block['held_out'], block['verdicts']))
    out['held_out_secondary'] = secondary
    # Checkpoint-to-checkpoint movement of the novel value MSE and of the ResNet-minus-CNN contrast, 1,200 updates onward
    noise = {}
    for seed in SEEDS:
        entry = {}
        for name in NAMES:
            series = [(r['updates'], r['held_out']['novel']['value_mse'], r['held_out']['novel']['policy_ce'])
                      for r in runs[seed][name]['milestones'] if r['updates'] >= 1200]
            steps = [abs(b[1] - a[1]) for a, b in zip(series, series[1:])]
            steps_ce = [abs(b[2] - a[2]) for a, b in zip(series, series[1:])]
            entry[name] = dict(value_mse_min=min(v for _, v, _ in series), value_mse_max=max(v for _, v, _ in series),
                               value_mse_adjacent_change_mean=statistics.fmean(steps), value_mse_adjacent_change_max=max(steps),
                               policy_ce_adjacent_change_mean=statistics.fmean(steps_ce), policy_ce_adjacent_change_max=max(steps_ce))
        for name in RESNETS:
            shared = [u for u in MILESTONES if 1200 <= u <= min(BUDGET[name], BUDGET['cnn'])]
            value = lambda n, u: next(r for r in runs[seed][n]['milestones'] if r['updates'] == u)['held_out']['novel']
            delta = [value(name, u)['value_mse'] + value(name, u)['policy_ce'] - value('cnn', u)['value_mse'] - value('cnn', u)['policy_ce'] for u in shared]
            entry[name]['combined_contrast_by_shared_checkpoint'] = dict(zip(map(str, shared), delta))
            entry[name]['combined_contrast_min'], entry[name]['combined_contrast_max'] = min(delta), max(delta)
        noise[str(seed)] = entry
    sensitivity['checkpoint_noise'] = noise
    # Suite accuracy point differences at every shared checkpoint (summaries; per-row records exist only at 3000 / final)
    accuracy = {}
    for name in RESNETS:
        accuracy[name] = {}
        for u in [m for m in MILESTONES if 800 <= m <= BUDGET[name]]:
            get = lambda s, n, suite: next(r for r in runs[s][n]['milestones'] if r['updates'] == u)['suites'][suite]['accuracy']
            accuracy[name][str(u)] = {suite: [get(s, name, suite) - get(s, 'cnn', suite) for s in SEEDS] for suite in ('solved', 'tactical')}
    sensitivity['suite_accuracy_point_differences'] = accuracy
    out['checkpoint_sensitivity'] = sensitivity

    checks, inference, training, collection = conformance(runs, summaries, status, design, regenerated)
    out['protocol_conformance'] = dict(checks=checks, failed=[k for k, v in checks.items() if not v],
                                       supervisor_status=status['status'])
    out['timing'] = timing(inference, training, collection, runs, status)
    out['phase4d4g_integrity'] = integrity(status)
    out['seconds'] = time.time() - started
    path = RESULTS / 'independent-recomputed.json'
    path.write_text(json.dumps(out, indent=1, sort_keys=True, allow_nan=False, default=lambda o: o.item() if hasattr(o, 'item') else list(o)) + '\n')

    print(json.dumps(dict(
        seconds=round(out['seconds'], 1), failed_protocol_checks=out['protocol_conformance']['failed'],
        dataset={s: {k: v for k, v in split[s].items() if k in ('counts_match_design_table', 'families_in_both_splits',
                 'my_components_equal_their_families', 'my_novel_mask_equals_theirs', 'held_out_positions_at_ply_ge_8_not_novel',
                 'observations_differing_from_boards_rebuilt_from_moves')} for s in split},
        alignment=alignment,
        primary={label: {n: dict(verdicts=v['verdicts'], compare=v['comparison_with_analysis'],
                                 pooled={m: [round(x, 5) for x in (v['pooled'][m]['difference'], v['pooled'][m]['low'], v['pooled'][m]['high'])]
                                         for m in METRICS}) for n, v in block.items()} for label, block in primary.items()},
        secondary={label: {n: v['comparison_with_analysis'] for n, v in block.items()} for label, block in out['held_out_secondary'].items()},
        suite_verdict_differences={label: suites[label]['verdict_differences'] for label in suites}), indent=1))


if __name__ == '__main__':
    main()
