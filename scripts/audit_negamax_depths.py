"""Independent-engine evidence replay and result-excluded deterministic repeats."""
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

from scripts.evaluate_mcts_strength import atomic_json, digest, load_rows
from scripts.evaluate_negamax_depths import check_source, game_id, play, schedule, validate


def normalized(row):
    row=copy.deepcopy(row)
    row.pop('elapsed_seconds')
    for m in row['moves']:
        m.pop('wall_seconds')
        m.pop('cpu_seconds')
    return row


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    args=parser.parse_args()
    directory=args.directory
    config=json.loads((directory/'experiment.json').read_text())
    candidate=json.loads((directory/'candidate.json').read_text())
    check_source(config)
    rows=load_rows(directory/'results.jsonl')
    preflight=load_rows(directory/'preflight.jsonl')
    validate(config,rows)
    validate(candidate,preflight,True)
    if len(rows)!=len(list(schedule(config))):
        raise ValueError('Main run incomplete')
    originals={r['id']:r for r in rows}
    repeats=[]
    start=time.perf_counter()
    for m,o,c in schedule(config):
        if o['id']!=config['openings'][0]['id']:
            continue
        row=play(config,m,o,c)
        validate(config,[row])
        if normalized(row)!=normalized(originals[game_id(m,o,c)]):
            raise ValueError('Seeded repeat mismatch')
        repeats.append(dict(id=row['id'],moves=len(row['moves']),winner=row['winner'],
                            elapsed_seconds=row['elapsed_seconds'],match=True))
    result=dict(experiment_sha256=digest(config),source_hashes=config['source_hashes'],
                audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                independently_replayed_games=len(rows)+len(preflight),
                main_moves=sum(len(r['moves']) for r in rows),
                preflight_moves=sum(len(r['moves']) for r in preflight),
                repeated_games=repeats,excluded_from_inference=True,elapsed_seconds=time.perf_counter()-start)
    atomic_json(directory/'audit.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('source_hashes','repeated_games')}))


if __name__=='__main__':
    main()
