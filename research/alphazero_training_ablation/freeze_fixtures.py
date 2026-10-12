"""Model-blind selection and opening-ancestry cluster freeze; no solver execution."""
from .common import ROOT, REPO, read, write, sha
from games.connect4.alphazero_v2.oracle import engine_position, mirror_moves, family_key, safe_moves_scan, winning_moves_scan

def clusters(rows):
    parents = list(range(len(rows)))
    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]; i = parents[i]
        return i
    prefixes = [tuple(r['moves'][:8]) for r in rows]
    for i, a in enumerate(rows):
        for j in range(i):
            p, q = prefixes[i], prefixes[j]
            n = min(len(p), len(q))
            same = p[:n] == q[:n] or p[:n] == tuple(6-c for c in q[:n])
            if a['family'] == rows[j]['family'] or same:
                parents[find(i)] = find(j)
    names = {v: f'cluster-{k:03d}' for k, v in enumerate(sorted({find(i) for i in range(len(rows))}))}
    return [dict(r, cluster=names[find(i)]) for i, r in enumerate(rows)]

def validate_solved(row):
    legal = set(engine_position(row['moves']).get_valid_moves())
    values = {int(a): v for a, v in row['action_values'].items()}
    if set(values) != legal or not set(values.values()) <= {-1, 0, 1}:
        raise ValueError('Incomplete exact legal-action vector')
    if max(values.values()) != row['value'] or set(row['optimal_actions']) != {a for a,v in values.items() if v == row['value']}:
        raise ValueError('Exact state/optimal labels inconsistent')

def main():
    frozen = REPO / 'games/connect4/alphazero_v2/frozen'
    solved = read(frozen / 'solved-development.json')['rows']
    tactical = read(frozen / 'tactical-development.json')['rows']
    for r in solved: validate_solved(r)
    for r in tactical:
        if r['safe_actions'] != safe_moves_scan(r['moves']) or r['immediate_wins'] != winning_moves_scan(r['moves']):
            raise ValueError('Independent tactical labels differ')
    # Full vectors only; selected-action and scalar proofs cannot qualify.
    proof_path = REPO / 'research/alphazero_exploration_confirmation/results/exact/proofs.json'
    queries = read(proof_path)['queries']
    candidates = {}
    for q in queries:
        moves = q['history']
        if q['status'] != 'exact' or len(moves) < 16 or winning_moves_scan(moves): continue
        values = {str(a): v for a,v in q['actions'].items()}
        if set(map(int, values)) != set(engine_position(moves).get_valid_moves()): continue
        if len(set(values.values())) < 2: continue
        fam = family_key(moves)
        candidates.setdefault(fam, dict(moves=moves, action_values=values, value=q['value'],
             optimal_actions=[int(a) for a,v in values.items() if v == q['value']], family=f'4e-{fam}',
             proof_query=q))
    selected = sorted(candidates.values(), key=lambda r: r['family'])[:64]
    deep = []
    for i, r in enumerate(selected):
        base = dict(r, id=f'deep-{i:03d}-base'); validate_solved(base); deep.append(base)
        mirror = dict(r, id=f'deep-{i:03d}-mirror', moves=mirror_moves(r['moves']),
                      action_values={str(6-int(a)): v for a,v in r['action_values'].items()},
                      optimal_actions=[6-a for a in r['optimal_actions']])
        validate_solved(mirror); deep.append(mirror)
    openings = [r for r in read(frozen / 'openings-development.json')['rows'] if r['id'].endswith('-base')][:32]
    if (len(solved),len(tactical),len(deep),len(openings)) != (300,200,128,32):
        raise ValueError('Fixture counts differ')
    sources = {str(p.relative_to(REPO)): sha(p) for p in [proof_path] + [frozen / f'{n}-development.json' for n in ('solved','tactical','openings')]}
    write(ROOT / 'fixtures.json', dict(solved=clusters(solved), tactical=clusters(tactical), deep=clusters(deep),
        openings=openings, sources=sources, search_seed=4404701, match_seed=4404702, bootstrap_seed=4404703,
        bootstrap_replicates=10000, cluster_rule='union mirror position families and reflection-equivalent prefix ancestry through ply 8'))

if __name__ == '__main__': main()
