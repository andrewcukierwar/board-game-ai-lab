"""Model-free, prospectively frozen confirmation and calibration opening packages.

Each candidate prefix is uniform random legal play from the empty board under a
domain-separated frozen RNG. Acceptance depends ONLY on Connect 4 rules and
(board, actor) mirror-canonical uniqueness: no model, guard, opponent or outcome.
The sealed package is never read; only the named development package is parsed.
"""
import hashlib, json
from pathlib import Path
from games.connect4.alphazero_v2.data import seeded_rng
from games.connect4.alphazero_v2.oracle import engine_position
from research.alphazero_tactical_guard.independent_verify import ReferenceBoard

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
DEVELOPMENT = REPO / 'games/connect4/alphazero_v2/frozen/openings-development.json'
SEED = 4420001
STRATA = (4, 6, 8, 10)
CONFIRMATION_PER_STRATUM = 24
CALIBRATION_PER_STRATUM = 4
MAX_ATTEMPTS = 20000


def key(board):
    text = lambda cells: '/'.join(''.join(r) for r in cells)
    mirrored = tuple(tuple(reversed(r)) for r in board.cells)
    return f'{board.actor}:' + min(text(board.cells), text(mirrored))


def self_symmetric(board):
    return board.cells == tuple(tuple(reversed(r)) for r in board.cells)


def threat(board):
    """Either side has an immediate winning drop (side to move, or opponent if it were to move)."""
    other = ReferenceBoard(board.cells, 1 - board.actor)
    return bool(board.classify()['immediate_winning_actions'] or other.classify()['immediate_winning_actions'])


def development_rows():
    package = json.loads(DEVELOPMENT.read_text())
    if package['split'] != 'development':
        raise ValueError('Only the development opening package may be read')
    return package['rows']


def sample(domain, length):
    rng, board, moves = seeded_rng(domain, SEED), ReferenceBoard.empty(), []
    for _ in range(length):
        if board.terminal():
            return None, None
        move = rng.choice(board.legal())
        moves.append(move)
        board = board.play(move)
    return moves, board


def generate(kind, per_stratum, excluded):
    rows, seen, stats = [], set(excluded), {}
    for length in STRATA:
        accepted, counts = 0, dict(attempts=0, prefix_terminal=0, terminal=0, threat=0, self_symmetric=0, duplicate=0)
        while accepted < per_stratum:
            if counts['attempts'] >= MAX_ATTEMPTS:
                raise RuntimeError(f'{kind} stratum {length} infeasible')
            domain = f'phase4d4j-b-{kind}-openings:ply{length}:attempt{counts["attempts"]}'
            counts['attempts'] += 1
            moves, board = sample(domain, length)
            if moves is None:counts['prefix_terminal'] += 1;continue
            if board.terminal():counts['terminal'] += 1;continue
            if threat(board):counts['threat'] += 1;continue
            if self_symmetric(board):counts['self_symmetric'] += 1;continue
            if key(board) in seen:counts['duplicate'] += 1;continue
            seen.add(key(board))
            game = engine_position(moves)  # second rules engine: legal, nonterminal
            assert not game.is_game_over() and game.current_player == board.actor
            rows.append(dict(id=f'{kind}-ply{length:02d}-{accepted:02d}', family=f'{kind}-ply{length:02d}-{accepted:02d}',
                             stratum=f'ply{length:02d}', moves=moves, canonical=key(board), rng_domain=domain))
            accepted += 1
        stats[f'ply{length:02d}'] = counts
    return rows, stats


def ancestry(rows, development):
    """Shared shorter-prefix ancestry: prefix positions equal (mirror-canonically) to other openings."""
    original = {key(ReferenceBoard.from_moves(r['moves'])): r['family'] for r in development}
    by_prefix = {}
    out = []
    for r in rows:
        board, hits = ReferenceBoard.empty(), []
        for ply, move in enumerate(r['moves'][:-1], 1):
            board = board.play(move)
            k = key(board)
            by_prefix.setdefault((ply, k), set()).add(r['family'])
            if k in original and ply >= 1:
                hits.append(dict(ply=ply, development_family=original[k]))
        out.append(dict(family=r['family'], development_prefix_matches=hits))
    shared = {f'{ply}:{k}': sorted(f) for (ply, k), f in by_prefix.items() if len(f) > 1}
    return out, shared


def build():
    development = development_rows()
    original = {key(ReferenceBoard.from_moves(r['moves'])) for r in development}
    confirmation, cstats = generate('confirmation', CONFIRMATION_PER_STRATUM, original)
    calibration, kstats = generate('calibration', CALIBRATION_PER_STRATUM, original | {r['canonical'] for r in confirmation})
    lineage, shared = ancestry(confirmation, development)
    payload = lambda rows: json.dumps([[r['id'], r['moves']] for r in rows], separators=(',', ':')).encode()
    return dict(contract='phase4d4j-b-opening-generation-v1', seed=SEED, strata=list(STRATA),
        rule='uniform random legal drops from empty board, RNG seeded_rng(domain, 4420001); reject prefix terminal, '
             'final terminal, any immediate win for side to move OR opponent, self-symmetric board, '
             'and (actor, mirror-canonical board) duplicates of the 100 development rows or earlier accepted rows',
        model_blind=True, outcome_blind=True, sealed_read=False,
        development_package_sha256=hashlib.sha256(DEVELOPMENT.read_bytes()).hexdigest(),
        confirmation=confirmation, calibration=calibration, stats=dict(confirmation=cstats, calibration=kstats),
        confirmation_sha256=hashlib.sha256(payload(confirmation)).hexdigest(),
        calibration_sha256=hashlib.sha256(payload(calibration)).hexdigest(),
        ancestry=dict(development_prefix_matches=lineage,
                      families_with_development_prefix=sum(bool(x['development_prefix_matches']) for x in lineage),
                      shared_new_prefixes=shared, shared_new_prefix_count=len(shared)))


if __name__ == '__main__':
    out = ROOT / 'results/openings.json'
    data = build()
    if out.exists() and json.loads(out.read_text())['confirmation_sha256'] != data['confirmation_sha256']:
        raise ValueError('Frozen opening package would change')
    out.write_text(json.dumps(data, indent=1, sort_keys=True) + '\n')
    print(data['confirmation_sha256'], len(data['confirmation']), data['stats'],
          data['ancestry']['families_with_development_prefix'], data['ancestry']['shared_new_prefix_count'])
