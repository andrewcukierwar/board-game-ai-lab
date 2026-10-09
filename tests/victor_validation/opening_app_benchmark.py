"""Benchmarks added for the opening-book / app-readiness milestone.

  white-policy  White refutation-avoidance policies on a FRESH development set
                (new seed; never the frozen 371-position suite), oracle-judged
  book-games    games from the empty board vs seeded stochastic opponents
  composite-recheck  composite covers re-audited with and without retiring spares
                (the stored 1,512 covers and/or a fresh sample with a new seed)
  stress        public-profile latency under CPU contention

  PYTHONPATH=.:tests .venv/bin/python -m victor_validation.opening_app_benchmark <command> ...

The frozen suite, its methodology and the positions/games/latency commands stay
in ``performance_benchmark``; this module only adds measurements.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import json
import multiprocessing
import os
import platform
from random import Random
from time import perf_counter

from games.connect4.connect4 import Connect4
from games.connect4.agents.negamax_agent import WIN_SCORE, NegamaxAgent
from games.connect4.victor import Position
from games.connect4.victor.coverage import SearchStatus
from games.connect4.victor.exact import Bits
from games.connect4.victor.white import search_white_covers, white_evaluation_contexts

from .native_oracle import solve_histories
from .performance_benchmark import (
    CONFIGS, configure, game_from, hardware, make_player, mean_ci, wilson, write,
)

POLICIES = ('off', 'first_unrefuted', 'certified_only', 'positive_evidence')


# --------------------------------------------------------- white-policy

def _dev_sample(seed):
    """Same style mix as the frozen suite sampler; White to move, >24 empty cells."""
    rng = Random(seed)
    style = ('random', 'eps_negamax2', 'eps_negamax4')[seed % 3]
    epsilon = {'random': 1.0, 'eps_negamax2': 0.3, 'eps_negamax4': 0.15}[style]
    depth = {'eps_negamax2': 2, 'eps_negamax4': 4}.get(style)
    target = rng.choice(range(6, 17, 2))
    game, history = Connect4(), []
    while len(history) < target:
        moves = []
        for c in game.get_valid_moves():
            g = Connect4(game.board, game.current_player)
            g.make_move(c)
            if not g.is_game_over():
                moves.append(c)
        if not moves:
            return None
        if rng.random() < epsilon:
            c = rng.choice(moves)
        else:
            scores = NegamaxAgent(depth).score_moves(game)
            best = max(scores[c] for c in moves)
            c = rng.choice([c for c in moves if scores[c] == best])
        game.make_move(c)
        history.append(c)
    return history


def _scan(history):
    """Every candidate's refutation status and White-cover evidence (no early exit)."""
    from games.connect4.victor.solver import _refutation
    budget = replace(CONFIGS['victor_full'], opening_book=False)
    game = game_from(history)
    p = Position.from_board(game.board, 0)
    bits = Bits.from_position(p)
    if bits.winning(0) or bits.winning(1):
        return None  # tactical: refutation avoidance is never consulted
    scores = NegamaxAgent(budget.fallback_depth).score_moves(game)
    safe = tuple(sorted((c for c in bits.legal() if not bits.drop(c).winning(1)),
                        key=lambda c: -scores[c]))
    if not safe or scores[safe[0]] >= WIN_SCORE:
        return None
    candidates = tuple(c for c in safe if scores[c] > -WIN_SCORE)[:budget.strategic_children]
    scan = []
    for c in candidates:
        child = bits.drop(c)
        status = _refutation(child, budget, lambda: False)
        cover = False
        position = Position.from_board(child.board, 1)
        if white_evaluation_contexts(position):
            cover = any(w.status is SearchStatus.FOUND for w in search_white_covers(
                position, node_budget=budget.cover_nodes, context_budget=budget.white_contexts,
                rules=budget.rules))
        scan.append((c, status, cover))
    return dict(history=history, fallback=safe[0], scan=scan)


def run_white_policy(args):
    from games.connect4.victor.solver import white_choice_from_scan
    start = perf_counter()
    with ProcessPoolExecutor(args.workers) as pool:
        histories = [h for h in pool.map(_dev_sample, range(args.seed, args.seed + args.pool))
                     if h is not None]
    frozen = {tuple(p['history']) for p in json.load(open(args.suite))['positions']}
    distinct, seen = [], set()
    for h in histories:
        key = tuple(''.join(r) for r in game_from(h).board)
        if key not in seen and tuple(h) not in frozen:
            seen.add(key)
            distinct.append(h)
    with ProcessPoolExecutor(args.workers) as pool:
        scans = [s for s in pool.map(_scan, distinct, chunksize=2) if s is not None]
    chunks = [scans[i::args.workers] for i in range(args.workers)]
    with ProcessPoolExecutor(args.workers) as pool:
        solved = list(pool.map(_solve, [[s['history'] for s in c] for c in chunks]))
    rows = []
    for chunk, out in zip(chunks, solved):
        for s, o in zip(chunk, out):
            if o['status'] != 'exact':
                continue
            values = {int(c): v for c, v in o['move_values'].items()}
            choices = {name: white_choice_from_scan(s['scan'], s['fallback'], name)[0]
                       for name in POLICIES}
            rows.append(dict(s, value=o['value'], move_values=values,
                             decisive=min(values.values()) < o['value'],
                             choices=choices,
                             optimal={k: values[c] == o['value'] for k, c in choices.items()}))
    summary = {}
    for group, members in (('all', rows), ('decisive', [r for r in rows if r['decisive']])):
        block = {}
        for name in POLICIES:
            k = sum(r['optimal'][name] for r in members)
            changed = [r for r in members if r['choices'][name] != r['choices']['off']]
            better = sum(r['move_values'][r['choices'][name]] > r['move_values'][r['choices']['off']]
                         for r in changed)
            worse = sum(r['move_values'][r['choices'][name]] < r['move_values'][r['choices']['off']]
                        for r in changed)
            by_value = Counter((r['value'], r['optimal'][name]) for r in members)
            block[name] = dict(n=len(members), optimal=k, accuracy_ci=wilson(k, len(members)),
                               changed_vs_off=len(changed), better=better, worse=worse,
                               by_root_value={f'{v}:{"opt" if o else "err"}': n
                                              for (v, o), n in sorted(by_value.items())})
        summary[group] = block
    write(args.output, dict(schema='victor-white-policy-dev-v1', seed=args.seed, pool=args.pool,
                            sampled=len(histories), distinct=len(distinct), scanned=len(scans),
                            solved=len(rows), policies=POLICIES,
                            seconds=round(perf_counter() - start, 1), summary=summary, rows=rows))
    print(json.dumps(summary, indent=1))


def _solve(histories):
    return [dict(status=o.status, value=o.value, move_values=o.move_values, nodes=o.nodes)
            for o in solve_histories(histories, node_limit=2_000_000_000)]


# ----------------------------------------------------------- book-games

def make_opponent(name, seed):
    """'epsnegamax:<depth>:<epsilon>' is seeded epsilon-greedy Negamax; else the shared factory."""
    if name.startswith('epsnegamax:'):
        _, depth, epsilon = name.split(':')
        rng, agent = Random(seed), NegamaxAgent(int(depth))

        def move(game):
            if rng.random() < float(epsilon):
                return rng.choice(game.get_valid_moves())
            return agent.choose_move(game)
        return move
    return make_player(name, seed)


def _book_game(task):
    config, opponent, color, seed = task
    configure(24)
    budget = CONFIGS[config]
    game = Connect4()
    players = [None, None]
    from games.connect4.victor import VictorSolver
    players[color] = VictorSolver(budget)
    players[1 - color] = make_opponent(opponent, seed)
    history, decisions = [], []
    while not game.is_game_over():
        mover = game.current_player
        t = perf_counter()
        if mover == color:
            column = players[mover].choose_move(Connect4(game.board, mover))
            r = players[mover].last_result
            decisions.append(dict(ply=len(history), move=column, kind=r.move_kind,
                                  seconds=round(perf_counter() - t, 4)))
        else:
            column = players[mover](Connect4(game.board, mover))
        assert game.make_move(column)
        history.append(column)
    winner = game.check_winner()
    return dict(config=config, opponent=opponent, victor_color=color, seed=seed, moves=history,
                score=0.5 if winner == -1 else float(winner == color), decisions=decisions)


def run_book_games(args):
    """Empty-board games: in-book play is exercised; stochastic opponents vary the lines."""
    configure(24)
    tasks = [(config, opponent, color, args.seed + 1000 * i + color)
             for config in args.configs for opponent in args.opponents
             for i in range(args.games) for color in (0, 1)]
    start = perf_counter()
    with ProcessPoolExecutor(args.workers) as pool:
        games = list(pool.map(_book_game, tasks, chunksize=1))
    # Oracle-adjudicate Victor decisions with at most --adjudicate-plies stones played.
    positions = sorted({tuple(g['moves'][:d['ply']]) for g in games for d in g['decisions']
                        if d['ply'] >= args.adjudicate_from})
    chunks = [positions[i::args.workers] for i in range(args.workers)]
    with ProcessPoolExecutor(args.workers) as pool:
        values = {}
        for chunk, out in zip(chunks, pool.map(_solve, chunks)):
            values.update(zip(chunk, out))
    for g in games:
        for d in g['decisions']:
            o = values.get(tuple(g['moves'][:d['ply']]))
            if o and o['status'] == 'exact':
                d['value'], d['move_value'] = o['value'], o['move_values'][d['move']]
    summary = {}
    for config in args.configs:
        for opponent in args.opponents + ['all']:
            gs = [g for g in games if g['config'] == config and opponent in (g['opponent'], 'all')]
            judged = [d for g in gs for d in g['decisions'] if 'move_value' in d]
            early = [d for d in judged if d['ply'] <= 13]
            summary.setdefault(config, {})[opponent] = dict(
                games=len(gs), wins=sum(g['score'] == 1 for g in gs),
                draws=sum(g['score'] == 0.5 for g in gs), losses=sum(g['score'] == 0 for g in gs),
                score=mean_ci([g['score'] for g in gs]),
                as_white=mean_ci([g['score'] for g in gs if g['victor_color'] == 0]),
                as_black=mean_ci([g['score'] for g in gs if g['victor_color'] == 1]),
                judged=len(judged), errors=sum(d['move_value'] < d['value'] for d in judged),
                early_judged=len(early), early_errors=sum(d['move_value'] < d['value'] for d in early),
                book_moves=sum(d['kind'] == 'opening_book' for g in gs for d in g['decisions']),
                max_move_seconds=max((d['seconds'] for g in gs for d in g['decisions']), default=None))
    write(args.output, dict(schema='victor-book-games-v1', configs=args.configs,
                            opponents=args.opponents, games_per_colour=args.games, seed=args.seed,
                            adjudicate_from=args.adjudicate_from,
                            seconds=round(perf_counter() - start, 1), python=platform.python_version(),
                            summary=summary, games=games))
    print(json.dumps({c: {o: {k: v for k, v in x.items() if k in ('wins', 'draws', 'losses', 'score',
                                                                     'errors', 'judged', 'book_moves')}
                          for o, x in b.items()} for c, b in summary.items()}, indent=1))


# ---------------------------------------------------- composite-recheck

def _recheck(task):
    """Legacy (no retiring spare) audit, then the current policy's audit + playouts."""
    from games.connect4.victor import SearchBudget
    from games.connect4.victor.execution import NineRulePolicy
    from games.connect4.victor.nine_rules import search_nine_rule_cover
    from games.connect4.victor.rules import RuleName
    from .performance_benchmark import _composite_check
    history, rules, audit_remaining, audit_nodes, playouts = task
    position = Position.from_board(game_from(history).board, 0)
    cover = search_nine_rule_cover(position, node_budget=10_000,
                                   rules=tuple(RuleName(r) for r in rules))
    legacy = NineRulePolicy(cover.witness, retiring_spares=False).audit(SearchBudget(
        nodes=audit_nodes, seconds=None, max_remaining=audit_remaining, table_entries=audit_nodes))
    new = _composite_check(task)
    return dict(legacy=legacy.status, legacy_detail=legacy.detail, audit=new['audit'],
                detail=new['detail'], counterexample=new['counterexample'],
                playouts=dict(Counter(p['result'] for p in new['playouts'])))


def run_composite_recheck(args):
    from games.connect4.victor.rules import RuleName
    from .performance_benchmark import COMPOSITES, _composite_sample, _solve_chunk, canonical
    start = perf_counter()
    sets = {}
    if args.stored:
        sets['stored'] = json.load(open(args.stored))['covers']
    if args.fresh_pool:
        with ProcessPoolExecutor(args.workers) as pool:
            sampled = [s for s in pool.map(_composite_sample,
                                           range(args.seed, args.seed + args.fresh_pool),
                                           chunksize=16) if s]
        seen, fresh = set(), []
        for found in sampled:
            for s in found:
                key = (canonical(s['history']), s['search'])
                if key not in seen and set(s['rules']) & set(COMPOSITES):
                    seen.add(key)
                    fresh.append(s)
        chunks = [fresh[i::args.workers] for i in range(args.workers)]
        with ProcessPoolExecutor(args.workers) as pool:
            for chunk, out in zip(chunks, pool.map(_solve_chunk, [[s['history'] for s in c]
                                                                   for c in chunks],
                                                    [400_000_000] * len(chunks))):
                for s, o in zip(chunk, out):
                    s['oracle_value'], s['oracle_status'] = o['value'], o['status']
        sets['fresh'] = fresh
    out = dict(schema='victor-composite-recheck-v1', audit_remaining=args.audit_remaining,
               audit_nodes=args.audit_nodes, playouts=args.playouts, fresh_seed=args.seed,
               fresh_pool=args.fresh_pool, sets={})
    for name, covers in sets.items():
        tasks = [(s['history'], s['rules'] if s['search'] != 'all_rules' else
                  [r.value for r in RuleName], args.audit_remaining, args.audit_nodes,
                  args.playouts) for s in covers]
        with ProcessPoolExecutor(args.workers) as pool:
            checks = list(pool.map(_recheck, tasks, chunksize=1))
        transitions = Counter((c['legacy'], c['audit']) for c in checks)
        playouts = sum((Counter(c['playouts']) for c in checks), Counter())
        changed = [dict(history=s['history'], rules=s['rules'], search=s['search'],
                        oracle_value=s.get('oracle_value'), **c)
                   for s, c in zip(covers, checks) if c['legacy'] != c['audit']]
        out['sets'][name] = dict(
            covers=len(covers),
            oracle_white_wins=sum(s.get('oracle_status') == 'exact' and s.get('oracle_value') == 1
                                  for s in covers),
            transitions={f'{a} -> {b}': n for (a, b), n in sorted(transitions.items())},
            playouts=dict(playouts), changed=changed,
            refuted_now=[dict(history=s['history'], **c) for s, c in zip(covers, checks)
                         if c['audit'] == 'refuted'])
        print(name, json.dumps({k: v for k, v in out['sets'][name].items()
                                if k not in ('changed', 'refuted_now')}), flush=True)
    out['seconds'] = round(perf_counter() - start, 1)
    write(args.output, out)


# --------------------------------------------------------------- stress

def _busy(stop_at):
    x = 0
    while perf_counter() < stop_at:
        x += 1
    return x


def run_stress(args):
    """Public-profile decision latency with N competing CPU-bound processes.

    Approximates a weaker or shared host locally. These are NOT Render
    measurements. Also checks that every decision is legal.
    """
    from games.connect4.agents.victor_research_agent import VictorResearchAgent
    suite = json.load(open(args.suite))
    rng = Random(args.seed)
    histories = [p['history'] for p in suite['positions']]
    rng.shuffle(histories)
    histories = [[]] + [[3], [3, 3], [2, 3], [3, 2, 3]] + histories[:args.positions]
    results = {}
    for load in args.loads:
        # Plain processes, terminated explicitly: no executor joins a busy loop.
        burners = [multiprocessing.Process(target=_busy, args=(perf_counter() + 3600,), daemon=True)
                   for _ in range(load)]
        for proc in burners:
            proc.start()
        agent, rows = VictorResearchAgent(), []
        try:
            for history in histories:
                game = game_from(history)
                t = perf_counter()
                move = agent.choose_move(game)
                rows.append(dict(plies=len(history), seconds=round(perf_counter() - t, 4),
                                 kind=agent.last_decision['kind'],
                                 legal=move in game.get_valid_moves(),
                                 deadline_reached=agent.last_decision['deadline_reached']))
        finally:
            for proc in burners:
                proc.terminate()
            for proc in burners:
                proc.join()
        xs = sorted(r['seconds'] for r in rows)
        results[str(load)] = dict(n=len(xs), median=xs[len(xs) // 2],
                                  p95=xs[int(0.95 * (len(xs) - 1))], max=xs[-1],
                                  over_1_2s=sum(x > 1.2 for x in xs),
                                  all_legal=all(r['legal'] for r in rows),
                                  deadline_reached=sum(r['deadline_reached'] for r in rows),
                                  kinds=dict(Counter(r['kind'] for r in rows)))
        print(load, json.dumps(results[str(load)]), flush=True)
    write(args.output, dict(schema='victor-public-stress-v1', loads=args.loads,
                            cpu_count=os.cpu_count(), hardware=hardware(),
                            python=platform.python_version(), note='local contention, not Render',
                            results=results))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    w = sub.add_parser('white-policy')
    w.add_argument('--output', required=True)
    w.add_argument('--suite', default='docs/victor-performance/suite.json')
    w.add_argument('--seed', type=int, default=20261101)
    w.add_argument('--pool', type=int, default=1500)
    w.add_argument('--workers', type=int, default=8)
    g = sub.add_parser('book-games')
    g.add_argument('--output', required=True)
    g.add_argument('--configs', nargs='+', required=True)
    g.add_argument('--opponents', nargs='+', default=['mcts:400', 'epsnegamax:4:0.1'])
    g.add_argument('--games', type=int, default=8, help='games per colour per opponent')
    g.add_argument('--seed', type=int, default=20261102)
    g.add_argument('--adjudicate-from', type=int, default=6)
    g.add_argument('--workers', type=int, default=8)
    r = sub.add_parser('composite-recheck')
    r.add_argument('--output', required=True)
    r.add_argument('--stored', default='docs/victor-performance/composite.json')
    r.add_argument('--fresh-pool', type=int, default=0)
    r.add_argument('--seed', type=int, default=20261103)
    r.add_argument('--audit-remaining', type=int, default=16)
    r.add_argument('--audit-nodes', type=int, default=3_000_000)
    r.add_argument('--playouts', type=int, default=4)
    r.add_argument('--workers', type=int, default=8)
    s = sub.add_parser('stress')
    s.add_argument('--output', required=True)
    s.add_argument('--suite', default='docs/victor-performance/suite.json')
    s.add_argument('--positions', type=int, default=80)
    s.add_argument('--loads', nargs='+', type=int, default=[0, 4, 10, 16])
    s.add_argument('--seed', type=int, default=3)
    args = parser.parse_args(argv)
    {'white-policy': run_white_policy, 'book-games': run_book_games,
     'composite-recheck': run_composite_recheck, 'stress': run_stress}[args.command](args)


if __name__ == '__main__':
    main()
