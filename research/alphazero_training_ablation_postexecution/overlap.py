"""Descriptive observed-replay overlap, using existing predictions only."""
from collections import defaultdict
import time
import numpy as np
from research.alphazero_training_ablation.common import ROOT, OUTPUT,read,write,enforce
from games.connect4.alphazero_v2.provenance import configure_deterministic_runtime
from games.connect4.alphazero_v2.experimental import load_experimental
from games.connect4.alphazero_v2.oracle import engine_position
from games.connect4.alphazero_v2.data import encode_board

def key(actor,observation):
    return (actor,min(observation,tuple(row[::-1] for row in observation)))

def main():
    started=time.monotonic(); configure_deterministic_runtime(1);enforce()
    fixture=read(ROOT/'fixtures.json'); audit=read(ROOT/'results/integrity-audit.json')
    keys={name:{r['id']:key(len(r['moves'])%2,encode_board(engine_position(r['moves']))) for r in fixture[name]}
          for name in ('solved','tactical','deep')}
    result={}
    for seed in (42,314159):
        for arm,generation in (('control',20),('control',22),('treatment',22)):
            identity=audit['boundaries']['records'][f'{seed}-{arm}-g{generation}']
            runner=load_experimental(identity['path'],expected_sha256=identity['file_sha256'],restore_global_rng=False)
            families=defaultdict(set)
            for game in runner.replay.iter_games():
                for example in game.examples: families[key(example.actor,example.observation)].add(game.generation)
            name=f'{seed}-{arm}-g{generation}'; table={}
            directory=OUTPUT/f'execution/parent-{seed}' if generation==20 else OUTPUT/f'execution/{seed}-{arm}-g22-evaluation'
            for suite in keys:
                rows=read(directory/f'{suite}.json')['rows']; matched=[r for r in rows if keys[suite][r['id']] in families]
                table[suite]=dict(matching_rows=len(matched),rows=len(rows),
                    identities={r['id']:sorted(families[keys[suite][r['id']]]) for r in matched},
                    observed_replay_absent_rows=len(rows)-len(matched))
                if suite=='solved':
                    for label,subrows in (('overlap',matched),('absent',[r for r in rows if keys[suite][r['id']] not in families])):
                        table[suite][label]=dict(n=len(subrows),metrics={f:float(np.mean([r[f] for r in subrows])) for f in ('ce','accuracy','mse')} if subrows else None)
            result[name]=table
    write(ROOT/'results/observed-replay-overlap.json',dict(results=result,wall_seconds=time.monotonic()-started,
        scope='Reflection/actor board overlap with retained parent generations13-20 or current generations15-22; historical generations1-12 are not available in these boundaries.',
        interpretation='Postexecution descriptive check only. Absence from observed replay does not establish unseen generalization; no unseen heldout split or comparable train-vs-test loss was prospectively defined.',
        model_queries=0,search_queries=0,optimizer_updates=0))
    print({name:{s:r['matching_rows'] for s,r in suites.items()} for name,suites in result.items()})

if __name__=='__main__':main()
