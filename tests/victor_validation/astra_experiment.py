"""Bounded, disjoint Astra development/held-out research (outputs never feed a book)."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from time import perf_counter

from . import performance_benchmark as bench
from .native_oracle import solve_histories

OUT = Path('docs/victor-astra')


def write(path, data):
    """One complete record per line keeps generated result diffs reviewable."""
    parts = []
    for key, value in data.items():
        if isinstance(value, list):
            body = '[\n' + ',\n'.join(
                '    ' + json.dumps(row, separators=(',', ':')) for row in value) + '\n  ]'
        else:
            body = json.dumps(value, indent=2)
        parts.append('  ' + json.dumps(key) + ': ' + body)
    path.write_text('{\n' + ',\n'.join(parts) + '\n}\n')


def histories(obj):
    if isinstance(obj, dict):
        h = obj.get('history')
        if isinstance(h, list) and h and all(type(c) is int and 0 <= c < 7 for c in h):
            yield h
        for v in obj.values():
            yield from histories(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from histories(v)


def sample(seed):
    # 2/3 in the target frontier; 1/6 each later phase. Alternate colours.
    bench.PHASES = {'early': (9, 18), 'middle': (9, 18), 'late': (19, 33)}
    return bench._sample_game(seed)


def solve_chunk(rows):
    outputs = solve_histories([r['history'] for r in rows], node_limit=20_000_000, timeout=180)
    return [dict(r, oracle=dict(status=o.status, value=o.value,
                                move_values=o.move_values, nodes=o.nodes))
            for r, o in zip(rows, outputs)]


def build():
    old = list(Path('docs/victor-performance').glob('*.json')) + list(Path('docs/victor-opening').glob('*.json'))
    old += [Path('games/connect4/victor/data/opening_book.json')]
    seen = set()
    for p in old:
        for h in histories(json.loads(p.read_text())):
            seen.add(bench.canonical(h))
    # The book's compact rows store 1-based histories at index 4, not under
    # a named history key. Explicitly decode them as well.
    book=json.loads(Path('games/connect4/victor/data/opening_book.json').read_text())
    for entry in book['entries']:
        seen.add(bench.canonical([int(c)-1 for c in entry[4]]))
    prior = len(seen)
    rows = []
    with ProcessPoolExecutor(4) as pool:
        for r in pool.map(sample, range(913000, 914500), chunksize=8):
            if r is None:
                continue
            key = bench.canonical(r['history'])
            if key in seen:
                continue
            game = bench.game_from(r['history'])
            if bench.immediate(game, game.current_player) or bench.immediate(game, 1-game.current_player):
                continue
            seen.add(key)
            r.update(plies=len(r['history']), mover=game.current_player,
                     canonical_sha256=hashlib.sha256(repr(key).encode()).hexdigest())
            rows.append(r)
            if len(rows) == 600:
                break
    # Assignment fixed before obtaining any oracle labels or tuning search.
    rows.sort(key=lambda r:r['canonical_sha256'])
    for i,r in enumerate(rows):
        r['split'] = 'dev' if i % 2 == 0 else 'heldout'
    out = dict(schema='astra-precommitted-split-v1', excluded_prior_states=prior,
               seed_range=[913000,914500], quiet_only=True, positions=rows)
    OUT.mkdir(exist_ok=True)
    (OUT/'split.json').write_text(json.dumps(out, indent=2)+'\n')
    print('Frozen split:',len(rows), 'excluded:',prior, flush=True)
    with ProcessPoolExecutor(4) as pool:
        solved = [r for chunk in pool.map(solve_chunk, [rows[i::16] for i in range(16)]) for r in chunk]
    for split in ('dev','heldout'):
        selected = [r for r in solved if r['split']==split]
        write(OUT/f'{split}.json', dict(positions=selected))
        print(split, len(selected), 'exact', sum(r['oracle']['status']=='exact' for r in selected), flush=True)


def evaluate_one(task):
    config, row = task
    from games.connect4.victor.native import NativeBudget, prove_moves
    from games.connect4.victor import Position, analyze_position
    game=bench.game_from(row['history'])
    start=perf_counter()
    if config=='baseline' or config.startswith(('hybrid:', 'hybrid_nocl:')):
        budget=bench.CONFIGS['victor_full']
        if config.startswith(('hybrid:', 'hybrid_nocl:')):
            budget=replace(budget,native=NativeBudget(nodes=int(config.split(':')[1]),seconds=None,
                                                   claimeven=not config.startswith('hybrid_nocl')))
        result=analyze_position(game.board,game.current_player,budget)
        if result.move_proof and result.move_proof.intervals:
            truth = row['oracle']['move_values']
            assert all(lo <= truth[str(c)] <= hi for c, lo, hi in result.move_proof.intervals)
        return dict(hash=row['canonical_sha256'],move=result.move,kind=result.move_kind,
                    exact_status=result.exact.status,nodes=result.exact.nodes,
                    seconds=perf_counter()-start,
                    proof=None if result.move_proof is None else dict(
                        status=result.move_proof.status, intervals=result.move_proof.intervals,
                        nodes=result.move_proof.nodes, seconds=result.move_proof.elapsed,
                        optimal_proved=bool(result.move_proof.optimal_moves)))
    scores=bench.NegamaxAgent(4).score_moves(game)
    order=tuple(sorted(range(7),key=lambda c:(-scores.get(c,-1e9), (3,2,4,1,5,0,6).index(c))))
    count=int(config.split(':')[1])
    proof=prove_moves(Position.from_board(game.board,game.current_player),
                       NativeBudget(nodes=count,seconds=None,serial=config.startswith('serial'),claimeven=config.startswith('cl')),order)
    truth={int(c):v for c,v in row['oracle']['move_values'].items()}
    assert all(lo<=truth[c]<=hi for c,lo,hi in proof.intervals), (row,proof)
    options=proof.optimal_moves or proof.admissible_moves
    return dict(hash=row['canonical_sha256'],move=next(c for c in order if c in options),
                kind=proof.status,intervals=proof.intervals,nodes=proof.nodes,
                seconds=perf_counter()-start,search_seconds=proof.elapsed,
                optimal_proved=bool(proof.optimal_moves),lower=proof.lower,upper=proof.upper,
                bound_hits=proof.bound_hits)


def evaluate(split, configs):
    rows=json.loads((OUT/f'{split}.json').read_text())['positions'] if split != 'frozen' else [
        dict(r, canonical_sha256=r['id'], oracle=dict(status='exact',value=r['value'],move_values=r['move_values']))
        for r in json.loads(Path('docs/victor-performance/suite.json').read_text())['positions']]
    rows=[r for r in rows if r['oracle']['status']=='exact']
    results={}
    for config in configs:
        with ProcessPoolExecutor(4) as pool:
            decisions=list(pool.map(evaluate_one,[(config,r) for r in rows],chunksize=4))
        results[config]=decisions
        print(config, 'optimal', sum(r['oracle']['move_values'][str(d['move'])]==r['oracle']['value']
                                    for r,d in zip(rows,decisions)), '/',len(rows),
              'proved',sum(d.get('optimal_proved',False) or bool(d.get('proof') and d['proof']['optimal_proved']) for d in decisions),flush=True)
    tag='-'.join(c.replace(':','_') for c in configs)
    write(OUT/f'{split}-{tag}.json', results)


def play_game(task):
    from games.connect4.victor import VictorSolver
    from games.connect4.victor.native import NativeBudget
    from .opening_app_benchmark import make_opponent
    config,opponent,opening,color,seed=task
    budget=bench.CONFIGS['victor_full']
    if config=='hybrid':
        budget=replace(budget,native=NativeBudget(seconds=None))
    victor=VictorSolver(budget)
    other=make_opponent(opponent,seed)
    game=bench.game_from(opening)
    history=list(opening)
    decisions=[]
    while not game.is_game_over():
        t=perf_counter()
        if game.current_player==color:
            c=victor.choose_move(game)
            r=victor.last_result
            decisions.append(dict(ply=len(history),move=c,kind=r.move_kind,
                seconds=perf_counter()-t,native_nodes=r.move_proof.nodes if r.move_proof else 0,
                optimal_proved=bool(r.move_proof and r.move_proof.optimal_moves)))
        else:
            c=other(game)
        assert c in game.get_valid_moves()
        game.make_move(c)
        history.append(c)
    winner=game.check_winner()
    return dict(config=config,opponent=opponent,opening=list(opening),color=color,seed=seed,
                history=history,winner=winner,score=.5 if winner==-1 else float(winner==color),
                decisions=decisions)


def games():
    tasks=[]
    for config in ('baseline','hybrid'):
        for opponent in ('negamax:6','mcts:400'):
            for i,opening in enumerate(bench.openings(16,77431)):
                for color in (0,1):
                    tasks.append((config,opponent,opening,color,55400+i*2+color))
        for opponent in ('epsnegamax:6:0.1','mcts:400'):
            for i in range(8):
                for color in (0,1):
                    tasks.append((config,opponent,(),color,65400+i*2+color))
    with ProcessPoolExecutor(4) as pool:
        results=[]
        for i,r in enumerate(pool.map(play_game,tasks,chunksize=1)):
            results.append(r)
            if (i+1)%16==0: print('games',i+1,'/',len(tasks),flush=True)
    write(OUT/'games.json', dict(games=results))
    unique={tuple(r['history'][:d['ply']]) for r in results for d in r['decisions'] if d['ply']>=9}
    rows=[dict(history=list(h)) for h in sorted(unique)]
    with ProcessPoolExecutor(4) as pool:
        judged=[r for chunk in pool.map(solve_chunk,[rows[i::32] for i in range(32)]) for r in chunk]
    write(OUT/'game-oracle.json', dict(positions=judged))
    print('adjudicated',len(judged),flush=True)


def latency():
    from games.connect4.agents.victor_research_agent import PUBLIC_BUDGET
    from games.connect4.victor import analyze_position
    rows=json.loads((OUT/'heldout.json').read_text())['positions']
    results={}
    for config in ('baseline','hybrid'):
        budget=PUBLIC_BUDGET if config=='hybrid' else replace(PUBLIC_BUDGET,native=None)
        decisions=[]
        for row in rows:
            g=bench.game_from(row['history'])
            start=perf_counter()
            r=analyze_position(g.board,g.current_player,budget)
            assert r.move in g.get_valid_moves()
            decisions.append(dict(hash=row['canonical_sha256'],move=r.move,kind=r.move_kind,
                seconds=perf_counter()-start,deadline_reached=r.deadline_reached,
                native_status=r.move_proof.status if r.move_proof else None,
                nodes=r.move_proof.nodes if r.move_proof else r.exact.nodes))
        results[config]=decisions
        times=sorted(d['seconds'] for d in decisions)
        print(config,'latency p50/p95/max',times[len(times)//2],times[int(.95*len(times))],times[-1],flush=True)
    write(OUT/'public-latency.json', results)


def summarize():
    from collections import Counter
    import itertools
    import statistics

    def stats(rows, decisions):
        ds={d['hash']:d for d in decisions}
        errors=Counter()
        kinds=Counter()
        proofs=Counter()
        times=[]
        nodes=0
        for r in rows:
            d=ds[r['canonical_sha256']]
            o=r['oracle']
            errors[bench.classify(o['value'],o['move_values'][str(d['move'])])]+=1
            kinds[d['kind']]+=1
            times.append(d['seconds'])
            q=d.get('proof')
            if q:
                assert all(lo<=o['move_values'][str(c)]<=hi for c,lo,hi in q['intervals'])
                proofs[q['status']]+=1
                nodes+=q['nodes']
        return dict(n=len(rows),errors=dict(errors),kinds=dict(kinds),proofs=dict(proofs),
                    native_nodes=nodes,median_seconds=statistics.median(times),
                    p95_seconds=sorted(times)[int(.95*len(times))],max_seconds=max(times))

    output={}
    for split in ('dev','heldout','frozen'):
        if split=='frozen':
            rows=[dict(r,canonical_sha256=r['id'],oracle=dict(status='exact',value=r['value'],move_values=r['move_values']))
                  for r in json.loads(Path('docs/victor-performance/suite.json').read_text())['positions']]
        else:
            rows=json.loads((OUT/f'{split}.json').read_text())['positions']
        unknown=sum(r['oracle']['status']!='exact' for r in rows)
        rows=[r for r in rows if r['oracle']['status']=='exact']
        if split=='dev':
            baseline=json.loads((OUT/'dev-baseline-native_200000-native_2000000-serial_2000000.json').read_text())['baseline']
            improved=json.loads((OUT/'dev-hybrid_10000000-cl_10000000-native_10000000.json').read_text())['hybrid:10000000']
        else:
            raw=json.loads((OUT/f'{split}-baseline-hybrid_10000000.json').read_text())
            baseline,improved=raw['baseline'],raw['hybrid:10000000']
        groups={'all':rows,'9-18':[r for r in rows if 9<=r['plies']<=18],
                '19-25':[r for r in rows if 19<=r['plies']<=25],
                '26+':[r for r in rows if r['plies']>=26],
                'white':[r for r in rows if r['mover']==0],
                'black':[r for r in rows if r['mover']==1],
                'decisive':[r for r in rows if len(set(r['oracle']['move_values'].values()))>1],
                'winning':[r for r in rows if r['oracle']['value']==1]}
        output[split]=dict(oracle_unknown=unknown,groups={name:{'baseline':stats(rs,baseline),
                            'hybrid':stats(rs,improved)} for name,rs in groups.items()})
    gs=json.loads((OUT/'games.json').read_text())['games']
    oracle={tuple(r['history']):r['oracle'] for r in json.loads((OUT/'game-oracle.json').read_text())['positions']}
    pairs={}
    game_summary={}
    for config in ('baseline','hybrid'):
        group=[g for g in gs if g['config']==config]
        errors=Counter()
        unknown=0
        for g in group:
            key=(g['opponent'],tuple(g['opening']),g['color'],g['seed'])
            pairs.setdefault(key,{})[config]=g['score']
            for d in g['decisions']:
                truth=oracle.get(tuple(g['history'][:d['ply']]))
                if truth and truth['status']=='exact':
                    errors[bench.classify(truth['value'],truth['move_values'][str(d['move'])])]+=1
                elif truth:
                    unknown+=1
        game_summary[config]=dict(wdl=dict(Counter(g['score'] for g in group)),
            score=bench.mean_ci([g['score'] for g in group]),errors_from_ply9=dict(errors),
            oracle_unknown_decisions=unknown)
    diffs=[p['hybrid']-p['baseline'] for p in pairs.values()]
    nz=[d for d in diffs if d]
    p_value=sum(abs(sum(a*b for a,b in zip(sign,nz)))>=abs(sum(nz))-1e-9
                for sign in itertools.product([-1,1],repeat=len(nz)))/2**len(nz)
    game_summary['paired']=dict(**bench.mean_ci(diffs),better=sum(d>0 for d in diffs),
                               worse=sum(d<0 for d in diffs),sign_flip_p=p_value)
    output['games']=game_summary
    (OUT/'summary.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(game_summary,indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['build','eval','games','latency','summary'])
    parser.add_argument('--split',default='dev',choices=['dev','heldout','frozen'])
    parser.add_argument('--configs',nargs='+',default=['baseline'])
    args=parser.parse_args()
    if args.command=='build': build()
    elif args.command=='eval': evaluate(args.split,args.configs)
    elif args.command=='games': games()
    elif args.command=='latency': latency()
    else: summarize()
