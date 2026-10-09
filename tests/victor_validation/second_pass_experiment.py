"""Bounded second-pass experiments; never edits earlier artifacts or oracle code."""
import argparse
import ctypes
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from random import Random
import subprocess
import tempfile
import statistics
import resource
import sys
from collections import Counter
from time import perf_counter, process_time

from . import performance_benchmark as bench
from .astra_experiment import histories, write
from .native_oracle import solve_histories
from games.connect4.victor import Position, analyze_position
from games.connect4.victor import native

OUT = Path('docs/victor-second-pass')
BASE = 'ac02f237ee2c2518119e75255e359e1f08e7cbf5'


def library(config):
    source = subprocess.check_output(['git', 'show', BASE+':games/connect4/victor/native_search.c'], text=True)
    if config == 'current':
        source = native.SOURCE.read_text()
    elif config != 'baseline':
        source = variant(source, config)
    digest = hashlib.sha256(source.encode()).hexdigest()
    path = Path(tempfile.gettempdir()) / ('victor-second-'+digest[:16]+'.so')
    if not path.exists():
        cpath = path.with_suffix('.c')
        cpath.write_text(source)
        subprocess.run(['cc','-O3','-std=c11','-shared','-fPIC',str(cpath),'-o',str(path)],check=True)
    lib=ctypes.CDLL(str(path))
    lib.victor_prove.argtypes=[ctypes.c_uint64,ctypes.c_uint64,ctypes.c_int,ctypes.c_uint64,
        ctypes.c_double,ctypes.c_uint64,ctypes.POINTER(ctypes.c_int),ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_uint64)]
    lib.victor_prove.restype=ctypes.c_int
    native._library=lambda:lib
    return digest


def variant(source, config):
    if config == 'cl_win':
        return source.replace('if(s->claimeven && !(played&1) && alpha>=0 && victor_claimeven(p,mask)) {\n        s->bound_hits++; return 0;',
            '''if(s->claimeven && !(played&1) && victor_claimeven(p,mask)) {
        Bits uppers=(bottom*42) & ~mask & ~(mask<<1);
        if(four((p^mask)|uppers)) { s->bound_hits++; return -1; }
        if(alpha>=0) { s->bound_hits++; return 0; }''')
    if config in ('safe_first','adaptive'):
        return source.replace('int alpha=hi[c]==1 ? -1 : 0;',
            'int alpha=lo[c]==-1 ? 0 : -1;' if config=='safe_first' else
            'int alpha=best<0 && lo[c]==-1 ? 0 : hi[c]==1 ? -1 : 0;')
    if config not in ('focus','bucket','focus_bucket','depth','focus_depth','resume','focus_resume','resume_bucket','focus_resume_bucket'):
        raise ValueError(config)
    if config.startswith('focus'):
        source=source.replace('if(lo[c]==hi[c]) continue;',
            'if(lo[c]==hi[c] || (!mode && hi[c]<=best)) continue;')
        source=source.replace('if(!mode && lo[c]==1)',
            'if(lo[c]>best) best=lo[c];\n            if(!mode && lo[c]==1)')
    if 'bucket' in config or 'depth' in config:
        source=source.replace('int8_t lower, upper;', 'int8_t lower, upper; uint32_t work;')
        source=source.replace('    if (!enter(s)) return 0;',
            '    uint64_t started=s->nodes;\n    if (!enter(s)) return 0;')
        source=source.replace('    Entry *e=&s->tt[((key*11400714819323198485ULL)>>32)%s->size];',
            '''    uint64_t index=((key*11400714819323198485ULL)>>32)%s->size;
    Entry *first=&s->tt[index], *second=&s->tt[(index ^ 1)<s->size ? (index ^ 1) : index];
    Entry *e=first->key==key ? first : second;''')
        source=source.replace('    /* Descendants may replace this slot: recheck the full key before merging. */',
            '''    /* Re-probe both slots after descendants may have replaced either. */
    e=first->key==key ? first : second->key==key ? second :
        first->work<=second->work ? first : second;
    uint64_t spent=s->nodes-started;
    uint32_t work=spent>UINT32_MAX ? UINT32_MAX : (uint32_t)spent;''')
        source=source.replace('e->lower=-1; e->upper=1; }',
            'e->lower=-1; e->upper=1; e->work=0; }\n    if(work>e->work) e->work=work;')
        if 'depth' in config:
            source=source.replace('uint64_t spent=s->nodes-started;', 'uint64_t spent=43-played;\n    (void)started;')
    if 'resume' in config:
        source=source.replace('int8_t lower, upper;', 'int8_t lower, upper; uint8_t refuted[2];')
        source=source.replace('    if(reflected<key) key=reflected;',
            '    int flipped=reflected<key;\n    if(flipped) key=reflected;')
        source=source.replace('    int a0=alpha,b0=beta;',
            '    int a0=alpha,b0=beta;\n    uint8_t refuted=0;')
        source=source.replace('        s->hits++;',
            '        s->hits++;\n        refuted=e->refuted[beta==1];')
        source=source.replace('        if(!m) continue;',
            '        if(!m || (refuted & (1 << (flipped ? 6-centre[i] : centre[i])))) continue;')
        source=source.replace('    int best=-2;', '    int best=refuted ? alpha : -2;')
        source=source.replace('        if(s->stopped) return 0;', '''        if(s->stopped) {
            /* Only completed child refutations survive cancellation. */
            if(refuted) {
                if(e->key!=key) { e->key=key; e->lower=-1; e->upper=1;
                    e->refuted[0]=e->refuted[1]=0; }
                e->refuted[b0==1] |= refuted;
            }
            return 0;
        }
        if(value<beta) {
            int c=__builtin_ctzll(moves[i])/7;
            refuted |= (uint8_t)(1 << (flipped ? 6-c : c));
        }''')
        source=source.replace('e->lower=-1; e->upper=1; }',
            'e->lower=-1; e->upper=1; e->refuted[0]=e->refuted[1]=0; }')
        source=source.replace('    if(best>a0', '    e->refuted[b0==1] |= refuted;\n    if(best>a0')
        # Bucket variants initialize additional priority bookkeeping on replacement.
        source=source.replace('e->work=0; }','e->work=0; e->refuted[0]=e->refuted[1]=0; }')
        if 'bucket' in config:
            source=source.replace('            if(refuted) {', '''            if(refuted) {
                e=first->key==key ? first : second->key==key ? second :
                    first->work<=second->work ? first : second;''')
            source=source.replace('e->refuted[0]=e->refuted[1]=0; }\n                e->refuted',
                'e->refuted[0]=e->refuted[1]=0; e->work=0; }\n                e->refuted')
    return source


def candidate(seed):
    rng=Random(seed)
    style=seed%3
    target=rng.randint(6,12) if seed%5<3 else rng.randint(13,24)
    g=bench.Connect4(); history=[]
    depth=6 if style==2 else 4
    agent=bench.NegamaxAgent(depth)
    for ply in range(target):
        options=[]
        for c in g.get_valid_moves():
            child=bench.Connect4(g.board,g.current_player); child.make_move(c)
            if not child.is_game_over(): options.append(c)
        if not options:return None
        if ply<4 or rng.random() < (0.35 if style==0 else 0.08): c=rng.choice(options)
        else:
            scores=agent.score_moves(g); best=max(scores[c] for c in options)
            c=rng.choice([c for c in options if scores[c]==best])
        g.make_move(c);history.append(c)
    if bench.immediate(g,0) or bench.immediate(g,1): return None
    p=Position.from_board(g.board,g.current_player)
    scores=bench.NegamaxAgent(4).score_moves(g)
    order=tuple(sorted(range(7),key=lambda c:(-scores.get(c,-1e9),(3,2,4,1,5,0,6).index(c))))
    proof=native.prove_moves(p,native.NativeBudget(nodes=200_000,seconds=None),order)
    key=bench.canonical(history)
    return dict(seed=seed,style=['epsilon4','opening4','opening6'][style],history=history,
                plies=target,mover=g.current_player,order=order,
                canonical_sha256=hashlib.sha256(repr(key).encode()).hexdigest(),
                probe_nodes=proof.nodes,probe_proved=bool(proof.optimal_moves))


def build():
    old=[p for folder in ('victor-astra','victor-performance','victor-opening')
         for p in Path('docs',folder).glob('*.json')]
    seen={bench.canonical(h) for p in old for h in histories(json.loads(p.read_text()))}
    book=json.loads(Path('games/connect4/victor/data/opening_book.json').read_text())
    seen.update(bench.canonical([int(c)-1 for c in row[4]]) for row in book['entries'])
    prior=len(seen); rows=[]
    library('baseline')
    with ProcessPoolExecutor(4,initializer=library,initargs=('baseline',)) as pool:
        for i,r in enumerate(pool.map(candidate,range(1029000,1030200),chunksize=4)):
            if r is not None:
                key=bench.canonical(r['history'])
                if key not in seen: seen.add(key);rows.append(r)
            if i%100==99: print('candidates',i+1,'unique quiet',len(rows),flush=True)
    # Equal colour quotas; difficulty selection uses ONLY original runtime,
    # never labels or candidate algorithms. Split within each stratum by hash.
    selected=[]
    for mover in (0,1):
        for hard,quota in ((True,80),(False,40)):
            group=sorted([r for r in rows if r['mover']==mover and (not r['probe_proved'])==hard],
                         key=lambda r:r['canonical_sha256'])[:quota]
            for i,r in enumerate(group):r.update(split='dev' if i%2==0 else 'heldout',targeted=hard)
            selected.extend(group)
    write(OUT/'split.json',dict(schema='second-pass-frozen-v1',base=BASE,seed_range=[1029000,1030200],
        prior_states=prior,candidate_count=len(rows),quiet_only=True,probe_nodes=200_000,positions=selected))
    print('FROZEN',len(selected),'positions',flush=True)


def label_chunk(rows):
    truth=solve_histories([r['history'] for r in rows],node_limit=100_000_000,timeout=180)
    return [dict(r,oracle=dict(status=o.status,value=o.value,move_values=o.move_values,nodes=o.nodes))
            for r,o in zip(rows,truth)]


def label():
    rows=json.loads((OUT/'split.json').read_text())['positions']
    with ProcessPoolExecutor(4) as pool:
        results=[]
        for chunk in pool.map(label_chunk,[rows[i:i+8] for i in range(0,len(rows),8)]):
            results.extend(chunk);print('labelled',len(results),flush=True)
    for split in ('dev','heldout'):
        write(OUT/(split+'.json'),dict(positions=[r for r in results if r['split']==split]))


def evaluate_row(task):
    row,nodes,public=task
    g=bench.game_from(row['history'])
    budget=replace(bench.CONFIGS['victor_full'],native=native.NativeBudget(nodes=nodes,seconds=None))
    if public:
        from games.connect4.agents.victor_research_agent import PUBLIC_BUDGET
        budget=PUBLIC_BUDGET
    start=perf_counter(); cpu=process_time()
    result=analyze_position(g.board,g.current_player,budget)
    q=result.move_proof
    o=row['oracle']
    if o['status']=='exact' and q:
        assert all(lo<=o['move_values'][str(c)]<=hi for c,lo,hi in q.intervals),(row,q)
        if q.optimal_moves:assert o['move_values'][str(result.move)]==o['value']
    return dict(hash=row['canonical_sha256'],move=result.move,kind=result.move_kind,
                seconds=perf_counter()-start,cpu_seconds=process_time()-cpu,
                search_seconds=q.elapsed if q else 0,
                status=q.status if q else None, intervals=q.intervals if q else [],
                nodes=q.nodes if q else 0,hits=q.cache_hits if q else 0,
                bound_hits=q.bound_hits if q else 0,proved=bool(q and q.optimal_moves),
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),
                deadline=result.deadline_reached)


def get_rows(split):
    if split=='frozen':
        return [dict(r,canonical_sha256=r['id'],oracle=dict(status='exact',value=r['value'],move_values=r['move_values']))
                for r in json.loads(Path('docs/victor-performance/suite.json').read_text())['positions']]
    path=Path('docs/victor-astra')/split[6:] if split.startswith('prior_') else OUT/split
    return json.loads(path.with_suffix('.json').read_text())['positions']


def evaluate(args):
    rows=get_rows(args.split)
    for config in args.configs:
        library(config)
        start=perf_counter()
        with ProcessPoolExecutor(1 if args.public else 4,initializer=library,initargs=(config,)) as pool:
            results=list(pool.map(evaluate_row,[(r,args.nodes,args.public) for r in rows],chunksize=1))
        exact=[(r,d) for r,d in zip(rows,results) if r['oracle']['status']=='exact']
        summary=dict(n=len(rows),labelled=len(exact),optimal=sum(r['oracle']['move_values'][str(d['move'])]==r['oracle']['value'] for r,d in exact),
            proved=sum(d['proved'] for r,d in exact),all_moves=sum(d['status']=='all_moves' for r,d in exact),
            nodes=sum(d['nodes'] for d in results),wall=perf_counter()-start)
        suffix=('-'+args.tag) if args.tag else ''
        write(OUT/f'{args.split}-{config}-{args.nodes}{"-public" if args.public else ""}{suffix}.json',dict(summary=summary,decisions=results))
        print(config,summary,flush=True)


def summarize():
    output={}
    for path in sorted(OUT.glob('*-*.json')):
        data=json.loads(path.read_text())
        if 'decisions' not in data:continue
        split=path.name.split('-')[0]
        rows=get_rows(split)
        decisions={d['hash']:d for d in data['decisions']}
        groups={'all':rows}
        if split in ('dev','heldout'):
            groups.update({
                'targeted':[r for r in rows if r['targeted']],
                'control':[r for r in rows if not r['targeted']],
                '6-8':[r for r in rows if r['plies']<=8],
                '9-12':[r for r in rows if 9<=r['plies']<=12],
                '13-18':[r for r in rows if 13<=r['plies']<=18],
                '19+':[r for r in rows if r['plies']>=19],
                'white':[r for r in rows if r['mover']==0],
                'black':[r for r in rows if r['mover']==1],
                'winning':[r for r in rows if r['oracle']['value']==1],
                'drawn':[r for r in rows if r['oracle']['value']==0],
                'lost':[r for r in rows if r['oracle']['value']==-1],
                'decisive':[r for r in rows if r['oracle']['status']=='exact' and len(set(r['oracle']['move_values'].values()))>1]})
            baseline=json.loads((OUT/f'{split}-baseline-10000000.json').read_text())['decisions']
            exhausted={d['hash'] for d in baseline if d['nodes']==10_000_000 and not d['proved']}
            groups['baseline_exhausted']=[r for r in rows if r['canonical_sha256'] in exhausted]
        result={}
        for name,rs in groups.items():
            if not rs:continue
            ds=[decisions[r['canonical_sha256']] for r in rs]
            known=[(r,decisions[r['canonical_sha256']]) for r in rs if r['oracle']['status']=='exact']
            times=sorted(d['seconds'] for d in ds)
            search_times=sorted(d['search_seconds'] for d in ds if 'search_seconds' in d)
            result[name]=dict(n=len(rs),labelled=len(known),unknown=len(rs)-len(known),
                optimal=sum(r['oracle']['move_values'][str(d['move'])]==r['oracle']['value'] for r,d in known),
                errors=dict(Counter(bench.classify(r['oracle']['value'],r['oracle']['move_values'][str(d['move'])]) for r,d in known)),
                proved=sum(d['proved'] for r,d in known),
                all_moves=sum(d['status']=='all_moves' for r,d in known),
                kinds=dict(Counter(d['kind'] for d in ds)),
                nodes=sum(d['nodes'] for d in ds),hits=sum(d['hits'] for d in ds),
                bound_hits=sum(d['bound_hits'] for d in ds),
                cpu_seconds=sum(d['cpu_seconds'] for d in ds),
                peak_rss_bytes=max(d.get('peak_rss_bytes',0) for d in ds) or None,
                median_search_seconds=statistics.median(search_times) if search_times else None,
                p95_search_seconds=search_times[int(.95*len(search_times))] if search_times else None,
                max_search_seconds=max(search_times) if search_times else None,
                median_seconds=statistics.median(times),p95_seconds=times[int(.95*len(times))],max_seconds=max(times),
                deadlines=sum(d['deadline'] for d in ds))
        output[path.stem]=result
    (OUT/'summary.json').write_text(json.dumps(output,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['build','label','eval','summary'])
    p.add_argument('--split',default='dev');p.add_argument('--configs',nargs='+',default=['baseline'])
    p.add_argument('--nodes',type=int,default=10_000_000);p.add_argument('--public',action='store_true')
    p.add_argument('--tag',default='',help='Optional artifact suffix for repeated timing trials')
    args=p.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.command=='build':build()
    elif args.command=='label':label()
    elif args.command=='summary':summarize()
    else:evaluate(args)
