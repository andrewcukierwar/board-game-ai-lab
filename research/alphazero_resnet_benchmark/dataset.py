"""Read-only generation 13-20 replay, game-family split, leakage audit and freeze.

The replay is obtained only through the pinned historical checkout's strict
reader (experimental_import.read_historical): source and runtime checks of the
historical code are never relaxed and no official file is opened for writing.
No model takes part in any decision made here.
"""
from collections import Counter
import hashlib
import json

from .common import (HISTORICAL, OFFICIAL, PARENTS, REPO, RESULTS, SEEDS, preserve, read, runtime, sha,
                     text_sha, write)

SPLIT_DOMAIN = 'phase4d4h-b-heldout-family-order'
PROBE_DOMAIN = 'phase4d4h-b-train-probe'
HELD_OUT_FRACTION = 0.15          # of complete games, reached by whole families
MAX_FAMILY_FRACTION = 0.05        # larger families are never held out (see DESIGN.md)
FIRST_PLY = 8                     # families link on shared positions from the first greedy ply onward
LINK_PLIES = tuple(range(0, 17))  # audited linking plies (structure only)
PROBE_SIZE = 4096                 # fixed training positions scored at every milestone


def canonical(observation):
    """Mirror-invariant position key. The observation is actor-relative, so it also fixes the actor."""
    flat = bytes(c + 1 for row in observation for c in row)
    mirrored = bytes(c + 1 for row in observation for c in row[::-1])
    return min(flat, mirrored)


def load_games(seed):
    """Strictly validated complete games of generations 13-20 for one seed, plus provenance."""
    from games.connect4.alphazero_v2.batching import TemperatureSchedule
    from games.connect4.alphazero_v2.config import V2Config
    from games.connect4.alphazero_v2.data import GenerationReplay
    from games.connect4.alphazero_v2.experimental import rebuild_scheduled_game
    from games.connect4.alphazero_v2.experimental_import import read_historical
    base = OFFICIAL / f'runs/seed-{seed}/generations/generation-0020'
    parent = base / 'generation-0020.resume.pt'
    expected = read(base / 'summary.json')['files']['resume']['sha256']
    if expected != PARENTS[seed]:
        raise ValueError('Generation-20 boundary hash differs from the trusted Phase 4D.4F parent')
    verified = read_historical(parent, expected_sha256=expected, historical_checkout=HISTORICAL)
    p = verified['original']
    if verified['verified_state'] != p['state_sha256'] or verified['verified_replay'] != p['replay']['digest']:
        raise ValueError('Historical reader attestation differs')
    config = V2Config.from_dict(p['config'])
    if config.seed != seed or config.exploratory_plies != 8:
        raise ValueError('Unexpected historical configuration')
    legacy = TemperatureSchedule('legacy_verified', ((0, 1), (8, 0)))
    replay = GenerationReplay(config.replay_generations, config.replay_max_games)
    for entry in p['replay']['generations']:
        replay.add_generation(entry['generation'], tuple(
            rebuild_scheduled_game(entry['generation'], record, legacy, config.self_play_simulations)
            for record in entry['games']))
    if (replay.digest() != verified['verified_replay'] or replay.generations != tuple(range(13, 21))
            or replay.games != 2048):
        raise ValueError('Rebuilt replay differs from the verified historical replay')
    provenance = dict(seed=seed, parent=str(parent.relative_to(HISTORICAL)), parent_sha256=expected,
                      replay_digest=replay.digest(), state_sha256=p['state_sha256'], generations=list(range(13, 21)),
                      games=replay.games, positions=len(replay), historical_trainer_steps=p['counters']['trainer_steps'],
                      historical_config=p['config'], teacher='CNN self-play PUCT-256 root visits, generations 13-20',
                      per_generation={str(g): c for g, c in replay.counts().items()})
    return tuple(replay.iter_games()), provenance


def game_keys(game, first_ply=0):
    return {canonical(e.observation) for e in game.examples[first_ply:]}


def components(games, first_ply):
    """Union games that share any mirror-canonical pre-move position at ply >= first_ply."""
    parent = list(range(len(games)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    owner = {}
    for index, game in enumerate(games):
        for key in game_keys(game, first_ply):
            if key in owner:
                parent[find(index)] = find(owner[key])
            else:
                owner[key] = index
    groups = {}
    for index in range(len(games)):
        groups.setdefault(find(index), []).append(index)
    return sorted(groups.values(), key=lambda members: members[0])


def link_table(games):
    """Family structure for every candidate linking ply; the audit that selects the rule."""
    rows = []
    for ply in LINK_PLIES:
        sizes = sorted((len(c) for c in components(games, ply)), reverse=True)
        rows.append(dict(first_ply=ply, families=len(sizes), largest=sizes[0],
                         largest_share=sizes[0] / len(games), singletons=sum(s == 1 for s in sizes),
                         top5=sizes[:5]))
    return rows


def game_id(game):
    return f'g{game.generation:02d}-{game.index:03d}'


def split(seed, games, first_ply):
    """Whole families to held-out in a hash order fixed by seed and family identity only.

    A family larger than MAX_FAMILY_FRACTION of the games stays in training: holding
    it out would remove most mainline play from training, and splitting it would leak.
    """
    families = components(games, first_ply)

    def order(members):
        return hashlib.sha256(f'{SPLIT_DOMAIN}:{seed}:{game_id(games[members[0]])}'.encode()).hexdigest()
    target = HELD_OUT_FRACTION * len(games)
    held, family_of = [], {}
    for number, members in enumerate(families):
        for index in members:
            family_of[index] = number
    for members in sorted(families, key=order):
        if len(held) >= target:
            break
        if len(members) <= MAX_FAMILY_FRACTION * len(games):
            held.extend(members)
    held = sorted(held)
    chosen = set(held)
    train = [i for i in range(len(games)) if i not in chosen]
    return train, held, family_of


def flatten(games, indices, family_of):
    """Position rows in (game, ply) order: example, game index, family number."""
    return [(example, index, family_of[index]) for index in indices for example in games[index].examples]


def examples_sha(rows):
    digest = hashlib.sha256()
    for example, index, family in rows:
        digest.update(json.dumps([example.record(), index, family], sort_keys=True).encode())
    return digest.hexdigest()


def probe_indices(seed, count):
    """Fixed training-probe positions (eval-mode train loss and batch-norm probes); never used to sample updates."""
    import random
    rng = random.Random(int(hashlib.sha256(f'{PROBE_DOMAIN}:{seed}'.encode()).hexdigest(), 16))
    return sorted(rng.sample(range(count), min(PROBE_SIZE, count)))


def build(seed, first_ply=FIRST_PLY):
    """The complete frozen dataset for one seed, recomputed from the strict historical read."""
    games, provenance = load_games(seed)
    table = link_table(games)
    train_games, held_games, family_of = split(seed, games, first_ply)
    train = flatten(games, train_games, family_of)
    held = flatten(games, held_games, family_of)
    train_keys = {canonical(e.observation) for e, _, _ in train}
    novel = [canonical(e.observation) not in train_keys for e, _, _ in held]
    return dict(seed=seed, games=games, provenance=provenance, link_table=table, first_ply=first_ply,
                train_games=train_games, held_games=held_games, family_of=family_of, train=train, held=held,
                novel=novel, train_keys=train_keys, probe=probe_indices(seed, len(train)))


def membership(data):
    games = data['games']
    return dict(train_games=[game_id(games[i]) for i in data['train_games']],
                held_out_games=[game_id(games[i]) for i in data['held_games']])


def identity(data):
    """Every hash a later process must reproduce before it may train or evaluate."""
    return dict(parent_sha256=data['provenance']['parent_sha256'], replay_digest=data['provenance']['replay_digest'],
                first_ply=data['first_ply'], membership_sha256=text_sha(membership(data)),
                train_examples_sha256=examples_sha(data['train']), held_out_examples_sha256=examples_sha(data['held']),
                novel_mask_sha256=text_sha([int(x) for x in data['novel']]), probe_sha256=text_sha(data['probe']),
                train_positions=len(data['train']), held_out_positions=len(data['held']),
                held_out_novel_positions=sum(data['novel']), train_games=len(data['train_games']),
                held_out_games=len(data['held_games']))


def frozen(seed):
    """Rebuild one seed's dataset and require it to equal the committed freeze."""
    freeze = read(RESULTS / 'dataset-freeze.json')
    data = build(seed)
    if identity(data) != freeze['seeds'][str(seed)]['identity'] or membership(data) != freeze['seeds'][str(seed)]['membership']:
        raise ValueError('Dataset differs from the frozen membership and hashes')
    return data


def ply_bucket(ply):
    return 'p00_07' if ply < 8 else 'p08_15' if ply < 16 else 'p16_23' if ply < 24 else 'p24_41'


def leakage(data):
    """Exact, mirror and opening-prefix overlap between the frozen train and held-out sets."""
    def exact(e):
        return bytes(c + 1 for row in e.observation for c in row)
    train_exact = {exact(e) for e, _, _ in data['train']}
    train_keys = data['train_keys']
    held = data['held']
    by_bucket = Counter()
    leaked_bucket = Counter()
    exact_hits = mirror_only = 0
    for (example, _, _), novel in zip(held, data['novel']):
        bucket = ply_bucket(example.ply)
        by_bucket[bucket] += 1
        if not novel:
            leaked_bucket[bucket] += 1
            if exact(example) in train_exact:
                exact_hits += 1
            else:
                mirror_only += 1
    games = data['games']
    first = data['first_ply']
    # Opening-prefix relatedness that survives the family rule: mirror-canonical move prefixes.
    def prefix(game, n):
        moves = tuple(game.moves[:n])
        return min(moves, tuple(6 - m for m in moves))
    prefix_overlap = {}
    for n in (2, 4, 6, 8):
        train_prefixes = {prefix(games[i], n) for i in data['train_games']}
        shared = sum(prefix(games[i], n) in train_prefixes for i in data['held_games'])
        prefix_overlap[f'moves_{n}'] = dict(held_out_games_sharing_a_train_prefix=shared,
                                            share=shared / len(data['held_games']))
    deep_leak = sum(1 for (e, _, _), novel in zip(held, data['novel']) if not novel and e.ply >= first)
    held_keys = Counter(canonical(e.observation) for e, _, _ in held)
    train_counts = Counter(canonical(e.observation) for e, _, _ in data['train'])
    families = Counter(family for _, _, family in held)
    sizes = Counter(data['family_of'].values())
    oversized = sorted((n for n in sizes.values() if n > MAX_FAMILY_FRACTION * len(games)), reverse=True)
    return dict(
        first_ply=first, families_total=len(set(data['family_of'].values())),
        train_only_oversized_families=oversized, train_only_oversized_game_share=sum(oversized) / len(games),
        held_out_game_lengths=dict(mean=sum(len(games[i].moves) for i in data['held_games']) / len(data['held_games'])),
        train_game_lengths=dict(mean=sum(len(games[i].moves) for i in data['train_games']) / len(data['train_games'])),
        held_out_families=len(families), train_families=len({data['family_of'][i] for i in data['train_games']}),
        largest_held_out_family_positions=max(families.values()),
        held_out_positions=len(held), held_out_exact_duplicates_of_train=exact_hits,
        held_out_mirror_only_duplicates_of_train=mirror_only, held_out_novel_positions=sum(data['novel']),
        leaked_at_or_after_link_ply=deep_leak,
        by_ply_bucket={b: dict(positions=by_bucket[b], leaked=leaked_bucket[b], novel=by_bucket[b] - leaked_bucket[b])
                       for b in sorted(by_bucket)},
        distinct_train_positions=len(train_counts), distinct_held_out_positions=len(held_keys),
        distinct_held_out_novel=sum(1 for k in held_keys if k not in train_keys),
        train_duplicate_positions=sum(train_counts.values()) - len(train_counts),
        opening_prefix_overlap=prefix_overlap,
        outcome_balance=dict(train=dict(Counter(str(e.outcome) for e, _, _ in data['train'])),
                             held_out=dict(Counter(str(e.outcome) for e, _, _ in held))),
        generation_balance=dict(train=dict(Counter(str(games[i].generation) for i in data['train_games'])),
                                held_out=dict(Counter(str(games[i].generation) for i in data['held_games']))))


def suite_overlap(data):
    """Development solved300/tactical200 rows whose position (or mirror) occurs in the replay split."""
    from games.connect4.alphazero_v2.data import encode_board
    from games.connect4.alphazero_v2.oracle import engine_position
    held_keys = {canonical(e.observation) for e, _, _ in data['held']}
    result = {}
    for suite in ('solved', 'tactical'):
        package = read(REPO / f'games/connect4/alphazero_v2/frozen/{suite}-development.json')
        in_train, in_held = [], []
        for row in package['rows']:
            key = canonical(encode_board(engine_position(row['moves'])))
            if key in data['train_keys']:
                in_train.append(row['id'])
            if key in held_keys:
                in_held.append(row['id'])
        result[suite] = dict(rows=len(package['rows']), rows_sha256=package['rows_sha256'],
                             in_train=in_train, in_held_out=in_held,
                             in_train_count=len(in_train), in_held_out_count=len(in_held))
    return result


def main():
    identity_runtime = runtime()
    before = preserve()
    seeds, audit, built = {}, {}, {}
    for seed in SEEDS:
        data = built[seed] = build(seed)
        seeds[str(seed)] = dict(provenance=data['provenance'], identity=identity(data), membership=membership(data))
        audit[str(seed)] = dict(link_table=data['link_table'], leakage=leakage(data),
                                development_suite_overlap=suite_overlap(data))
    cross = {}
    a, b = (built[s] for s in SEEDS)
    cross['seed42_held_out_positions_in_seed314159_train'] = sum(
        canonical(e.observation) in b['train_keys'] for e, _, _ in a['held'])
    cross['seed314159_held_out_positions_in_seed42_train'] = sum(
        canonical(e.observation) in a['train_keys'] for e, _, _ in b['held'])
    cross['note'] = 'Informational: each seed trains and evaluates only on its own replay.'
    if preserve() != before:
        raise ValueError('Official inventory changed while reading')
    write(RESULTS / 'dataset-freeze.json', dict(
        rule=dict(split='complete games; whole opening families',
                  family='union of games sharing any mirror-canonical pre-move position at ply >= first_ply',
                  first_ply=FIRST_PLY, oversized='families above max_family_fraction of games are train-only',
                  max_family_fraction=MAX_FAMILY_FRACTION, held_out_fraction=HELD_OUT_FRACTION,
                  order=f'sha256({SPLIT_DOMAIN}:seed:first game id)', probe=f'{PROBE_SIZE} fixed training positions',
                  primary_held_out='held-out positions whose mirror-canonical key is absent from training',
                  sealed_inputs='never read'),
        seeds=seeds, runtime=identity_runtime, official_files=len(before),
        sources={'games/connect4/alphazero_v2/frozen/solved-development.json':
                 sha(REPO / 'games/connect4/alphazero_v2/frozen/solved-development.json'),
                 'games/connect4/alphazero_v2/frozen/tactical-development.json':
                 sha(REPO / 'games/connect4/alphazero_v2/frozen/tactical-development.json')}))
    write(RESULTS / 'dataset-audit.json', dict(seeds=audit, cross_seed=cross))
    print(json.dumps({s: dict(identity=seeds[s]['identity'],
                              suite={k: (v['in_train_count'], v['in_held_out_count'])
                                     for k, v in audit[s]['development_suite_overlap'].items()})
                      for s in seeds}, indent=1))


if __name__ == '__main__':
    main()
