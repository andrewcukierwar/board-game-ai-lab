"""Pre-freeze synthetic dry run of analyze.py: rules-only fake games, real guard calls, random visits.

No model, no opponent, no confirmation match outcome. Output goes to a scratch directory only.
"""
import random, shutil, sys
from pathlib import Path
from games.connect4.connect4 import Connect4
from games.connect4.alphazero_v2.bitboard import BitboardState
from research.alphazero_tactical_guard.guard import guard
from research.alphazero_tactical_guard.independent_verify import ReferenceBoard
from .common import RESULTS, read, write, SEEDS
from . import analyze


def main(scratch):
    scratch = Path(scratch); shutil.rmtree(scratch, ignore_errors=True); scratch.mkdir(parents=True)
    for name in ('design.json', 'openings.json', 'calibration.json'): shutil.copy(RESULTS / name, scratch / name)
    rng = random.Random(1); rows = read(RESULTS / 'openings.json')['confirmation']; depth = read(RESULTS / 'calibration.json')['opponent_b_depth']
    for s in SEEDS:
        for d in (3, depth):
            out = scratch / 'tasks' / f'{s}-negamax{d}'; out.mkdir(parents=True)
            for v in ('control', 'guarded'):
                records = []
                for row in rows:
                    for color in (0, 1):
                        ref, moves, decisions = ReferenceBoard.from_moves(row['moves']), list(row['moves']), []
                        while not ref.terminal():
                            legal = list(ref.legal())
                            if ref.actor == color:
                                visits = [rng.randrange(0, 60) if a in legal else 0 for a in range(7)]; original = max(legal, key=lambda a: visits[a])
                                g = guard(BitboardState.from_game(Connect4([list(r) for r in ref.cells], ref.actor)), visits, original) if v == 'guarded' else None
                                move = g['final_action'] if g else original
                                decisions.append(dict(color=color, move=move, seconds=.04, original_action=original, final_action=move, visits=visits,
                                                      search_seconds=.04, total_seconds=.0401, guard=g, occupancy=len(moves)))
                            else:
                                move = rng.choice(legal); decisions.append(dict(color=1 - color, move=move, seconds=.001))
                            ref = ref.play(move); moves.append(move)
                        w = ref.winner(); result = 'draw' if w == -1 else ('win' if w == color else 'loss')
                        records.append(dict(family=row['family'], agent_color=color, moves=moves, decisions=decisions, result=result,
                                            score={'win': 1., 'draw': .5, 'loss': 0.}[result]))
                write(out / f'{v}-matches.json', dict(complete=True, records=records))
            if d == 3:
                for suite in ('tactical', 'solved', 'deep'):
                    shutil.copy(RESULTS.parents[1] / 'alphazero_tactical_guard/results/tasks' / str(s) / f'{suite}.json', out / f'{suite}.json')
    analyze.main(scratch)
    print('synthetic analysis OK (outcomes meaningless)')


if __name__ == '__main__':
    main(sys.argv[1])
