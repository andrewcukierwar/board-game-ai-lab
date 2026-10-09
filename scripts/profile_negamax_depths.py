"""Production latency separated from cProfile, line tracing and TT retention."""
import argparse
import cProfile
import gc
import inspect
import json
import pstats
import statistics
import sys
import time
import tracemalloc
from pathlib import Path
from unittest.mock import patch

from games.connect4.agents import negamax_agent as search
from scripts.benchmark_negamax_ordering import table_bytes
from scripts.benchmark_public_agents import position
from scripts.evaluate_mcts_strength import atomic_json, digest
from scripts.evaluate_negamax_depths import check_source


def decision(game, depth):
    agent = search.NegamaxAgent(depth)
    wall, cpu = time.perf_counter(), time.process_time()
    move = agent.choose_move(game)
    wall, cpu = time.perf_counter()-wall, time.process_time()-cpu
    return dict(move=move, scores=agent.last_scores, stats=agent.last_stats,
                wall_seconds=wall, cpu_seconds=cpu)


def memory_run(game, depth):
    held, factory = [], search.SearchTable
    def capture():
        table = factory()
        held.append(table)
        return table
    gc.collect()
    tracemalloc.start()
    try:
        with patch.object(search, 'SearchTable', capture):
            result = decision(game, depth)
        current, peak = tracemalloc.get_traced_memory()
        sites = [dict(line=s.traceback[0].lineno, file=Path(s.traceback[0].filename).name,
                      bytes=s.size, count=s.count) for s in tracemalloc.take_snapshot().statistics('lineno')[:10]]
    finally:
        tracemalloc.stop()
    result.update(retained_bytes=current, peak_bytes=peak, tt_bytes=table_bytes(held[0]),
                  entries=len(held[0].entries), top_allocations=sites,
                  retention='Diagnostic reference retains TT; production releases TT after each decision')
    return result


def profile_run(game, depth):
    profiler = cProfile.Profile()
    profiler.enable()
    result = decision(game, depth)
    profiler.disable()
    stats = pstats.Stats(profiler)
    functions = [dict(file=Path(file).name, line=line, function=name, primitive_calls=cc,
                      calls=nc, self_seconds=tt, cumulative_seconds=ct)
                 for (file,line,name),(cc,nc,tt,ct,callers) in stats.stats.items()]
    def cumulative(names):
        return sum(f['cumulative_seconds'] for f in functions if f['function'] in names)
    categories = dict(heuristic=cumulative(['heuristic']), terminal=cumulative(['terminal_value']),
                      legal_ordering=cumulative(['ordered_moves']), state_play_undo=cumulative(['play','undo']))
    # The remainder includes TT, recursive body, root work and measurement glue.
    categories['tt_and_recursive_other'] = stats.total_tt-sum(categories.values())
    result.update(profile_total_seconds=stats.total_tt, functions=functions, categories_seconds=categories,
                  leaf_evaluations=sum(f['calls'] for f in functions if f['function']=='heuristic'))
    return result


def line_run(game, depth=6):
    """Intrusive line-event attribution, separate from normal timing.

    Charge each interval to the last active source line; child call/return
    events switch ownership, avoiding inclusive recursion double counting.
    Callback overhead is deliberately excluded but tracing changes costs.
    """
    filename = search.__file__
    times, stack = {}, []
    previous_time, previous_key = time.perf_counter(), None
    def trace(frame, event, arg):
        nonlocal previous_time, previous_key
        now = time.perf_counter()
        if previous_key:
            times[previous_key] = times.get(previous_key,0)+now-previous_time
        if event == 'call':
            stack.append(previous_key)
        if event in ('call','line'):
            previous_key = (frame.f_code.co_name, frame.f_lineno) if frame.f_code.co_filename == filename else None
        elif event == 'return':
            previous_key = stack.pop() if stack else None
        previous_time = time.perf_counter()
        return trace
    sys.settrace(trace)
    try:
        result = decision(game, depth)
    finally:
        sys.settrace(None)
    source_lines, first = inspect.getsourcelines(search.negamax)
    tt_lines = {first+i for i,s in enumerate(source_lines) if any(token in s for token in (
        'key =', 'alpha_original,', 'hint =', 'key in table.entries', 'table.entries[key]',
        'table.tt_moves', 'table.hits', 'flag ==', 'return value', 'alpha = max(alpha, value)',
        'beta = min(beta, value)', 'if alpha >= beta:', 'flag = UPPER'))}
    categories = dict(heuristic=0., terminal=0., legal_ordering=0., tt=0., state_play_undo=0., recursive_other=0.)
    for (name,line), elapsed in times.items():
        category = ('heuristic' if name=='heuristic' else 'terminal' if name in ('terminal_value','has_four')
                    else 'legal_ordering' if name in ('legal','ordered_moves','winning_squares','<listcomp>','<lambda>')
                    else 'state_play_undo' if name in ('play','undo') else 'tt' if name=='negamax' and line in tt_lines
                    else 'recursive_other')
        categories[category] += elapsed
    result.update(depth=depth, categories_seconds=categories, tt_lines=sorted(tt_lines),
                  lines=[dict(function=n,line=l,seconds=t) for (n,l),t in sorted(times.items())],
                  method='intrusive exclusive line-event attribution; depth 6 diagnostic, not latency')
    return result


def assert_same(a,b):
    for key in ('move','scores','stats'):
        if a[key] != b[key]:
            raise RuntimeError(f'Diagnostic changed {key}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['preflight-memory','profile'])
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads((args.directory / 'candidate.json').read_text())
    check_source(config)
    output = args.directory / ('preflight-memory.json' if args.command=='preflight-memory' else 'profiling.json')
    if output.exists():
        raise ValueError('Refusing to overwrite evidence')
    result = dict(candidate_sha256=digest(config), rows=[], depths=[4,6,8,10],
                  samples=3, warmups=1, line_depth=6)
    positions = config['preflight_openings'] if args.command=='preflight-memory' else config['profile_positions']
    atomic_json(output,result)
    start = time.perf_counter()
    for opening in positions:
        game = position(opening['history'])
        if args.command=='preflight-memory':
            for depth in (6,8,10):
                memory = memory_run(game,depth)
                result['rows'].append(dict(position=opening['id'], depth=depth, memory=memory))
        else:
            samples = {d:[] for d in result['depths']}
            for repetition in range(4):
                depths=result['depths']
                offset=repetition%4
                for depth in depths[offset:]+depths[:offset]:
                    run=decision(game,depth)
                    if repetition:
                        samples[depth].append(run)
            line = line_run(game)
            assert_same(samples[6][0],line)
            for depth in result['depths']:
                memory, profile = memory_run(game,depth), profile_run(game,depth)
                for run in samples[depth]+[memory,profile]:
                    assert_same(samples[depth][0],run)
                wall=statistics.median(r['wall_seconds'] for r in samples[depth])
                result['rows'].append(dict(position=opening['id'], depth=depth, samples=samples[depth],
                    median_wall_seconds=wall, maximum_wall_seconds=max(r['wall_seconds'] for r in samples[depth]),
                    median_cpu_seconds=statistics.median(r['cpu_seconds'] for r in samples[depth]),
                    nodes_per_second=samples[depth][0]['stats']['nodes']/wall,
                    memory=memory, profile=profile, line_profile=line if depth==6 else None))
        result['elapsed_seconds']=time.perf_counter()-start
        atomic_json(output,result)
        print(f'{args.command} {opening["id"]}: complete ({result["elapsed_seconds"]:.1f}s)',flush=True)


if __name__=='__main__':
    main()
