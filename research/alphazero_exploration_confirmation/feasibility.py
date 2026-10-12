"""Runtime-only feasibility using old 4D games, not confirmatory identities."""
import gzip
import json
from pathlib import Path
import time
from games.connect4.alphazero_v2.oracle import BitboardSolver, SolverBudgetExceeded, engine_position, legal_moves_scan
from .solver import NativeSession
from research.alphazero_baseline import audit

def main():
    started=time.perf_counter()
    output=Path('research/alphazero_exploration_confirmation/results/feasibility')
    output.mkdir(parents=True,exist_ok=False)
    prior=Path('research/alphazero_exploration/results/phase4d4d-20261011/main')
    histories=[]
    for file in sorted(prior.glob('*original.jsonl.gz')):
        games=[json.loads(l) for l in gzip.open(file,'rt')]
        for ply in (8,12,16,24,32,40):
            g=next((g for g in games if len(g['moves'])>ply),None)
            if g: histories.append(g['moves'][:ply])
    session=NativeSession(200_000,.3)
    rows=[]
    try:
        for h in histories:
            native=session.query(h)
            solver=BitboardSolver(20_000)
            tick=time.perf_counter()
            try: py=solver.value(h)
            except SolverBudgetExceeded: py=None
            if py is not None and native['status']=='exact': assert py==native['value']
            if native['status']=='exact':
                assert set(native['actions'])==set(legal_moves_scan(h))
                for c,v in native['actions'].items():
                    successor=engine_position(h+[c])
                    if successor.is_game_over(): assert v==(0 if successor.check_winner()==-1 else 1)
            rows.append(dict(history=h,ply=len(h),native=native,python_value=py,python_nodes=solver.nodes,
                             python_seconds=time.perf_counter()-tick))
    finally: session.close()
    audit.write(output/'benchmark.json',dict(rows=rows,elapsed_seconds=time.perf_counter()-started,
        native_source_sha256=audit.sha(Path(__file__).with_name('c4_oracle.c')),
        native_source_origin='8704142702da273e6571f0ca9df3fd4a793be285 tests/victor_validation/c4_oracle.c',
        prior_data_only=True,pilot_excluded=True))
    print(json.dumps(dict(positions=len(rows),native_exact=sum(r['native']['status']=='exact' for r in rows),
          python_exact=sum(r['python_value'] is not None for r in rows),seconds=time.perf_counter()-started)))
if __name__=='__main__': main()
