"""Post-main cost diagnostic, selected by maximum nodes/time, excluded from claims."""
import argparse
import hashlib
import json
from pathlib import Path

from scripts.benchmark_public_agents import position
from scripts.evaluate_mcts_strength import atomic_json, digest, load_rows
from scripts.evaluate_negamax_depths import check_source, labels, validate
from scripts.profile_negamax_depths import assert_same, decision, memory_run, profile_run


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    args=parser.parse_args()
    directory=args.directory
    output=directory/'tail-diagnostic.json'
    if output.exists():
        raise ValueError('Refusing to overwrite evidence')
    config=json.loads((directory/'experiment.json').read_text())
    check_source(config)
    rows=load_rows(directory/'results.jsonl')
    validate(config,rows)
    choices=[(r,m) for r in rows for m in r['moves'] if r[m['role']+'_config'].get('depth')==10]
    selected={}
    for rule,key in [('max_nodes',lambda x:x[1]['negamax_stats']['nodes']),
                     ('max_elapsed',lambda x:x[1]['wall_seconds'])]:
        r,m=max(choices,key=key)
        identity=f'{r["id"]}/{m["ply"]}'
        selected.setdefault(identity,dict(r=r,m=m,rules=[]))['rules'].append(rule)
    result=dict(experiment_sha256=digest(config),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                selection='Post hoc largest observed depth-10 node count and elapsed time; no outcome filter',
                excluded_from_strength_and_predeclared_profile_aggregates=True,rows=[])
    for identity,item in selected.items():
        r,m=item['r'],item['m']
        history=r['complete_history'][:m['ply']]
        game=position(history)
        normal=decision(game,10)
        if normal['stats']!=m['negamax_stats'] or normal['move']!=m['column']:
            raise ValueError('Recorded search replay mismatch')
        memory,profile=memory_run(game,10),profile_run(game,10)
        assert_same(normal,memory)
        assert_same(normal,profile)
        result['rows'].append(dict(id=identity,selection_rules=item['rules'],history=history,
                                  labels=labels(history),original_move=m,normal=normal,memory=memory,profile=profile))
    atomic_json(output,result)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))


if __name__=='__main__':
    main()
