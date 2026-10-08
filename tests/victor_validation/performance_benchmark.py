"""Reproducible Victor ablation benchmark against independent exact ground truth.

Run from the repository root; each subcommand writes only the --output it names
(see docs/victor-performance-and-integration.md for the exact recorded commands):

  suite           sample, solve (native oracle) and stratify decisive positions
  positions       every configuration on every suite position, oracle-judged
  games           paired-colour games vs Random/Negamax/MCTS, oracle-adjudicated
  composite       targeted composite-rule covers: oracle check, policy audit, playouts
  latency         serial per-configuration timing on a fixed suite subset
  public-latency  serial sweep of the opt-in API agent's public profile
  report          markdown tables regenerated from saved artifacts

  PYTHONPATH=tests .venv/bin/python -m victor_validation.performance_benchmark <command> ...

Ground truth comes from the native C oracle (``native_oracle``), which shares no
code with Victor. Every Victor configuration uses identical node-only budgets
(no wall-clock cutoffs), so decisions are reproducible on any host; only the
recorded latencies depend on the machine and load.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import json
import math
import platform
from random import Random
import statistics
import subprocess
import sys
from time import perf_counter

from games.connect4.connect4 import Connect4
from games.connect4.agents.mcts_agent import MCTSAgent
from games.connect4.agents.negamax_agent import NegamaxAgent
from games.connect4.victor import (
    Position, SearchBudget, SolverBudget, VictorSolver, analyze_position,
)
from games.connect4.victor.execution import NineRulePolicy
from games.connect4.victor.nine_rules import analyze_nine_rules, search_nine_rule_cover
from games.connect4.victor.rules import RuleName

from .native_oracle import solve_histories

THREE_RULES = (RuleName.CLAIMEVEN, RuleName.BASEINVERSE, RuleName.VERTICAL)
PHASES = {'early': (6, 13), 'middle': (14, 25), 'late': (26, 33)}  # plies played
AUDIT = SearchBudget(nodes=20_000, seconds=None, max_remaining=10, table_entries=20_000)
CONFIGS = {}


def configure(exact_remaining):
    """Same exact search, fallback depth, cover node budget and child limit throughout."""
    exact = SearchBudget(nodes=200_000, seconds=None, max_remaining=exact_remaining,
                         table_entries=200_000)
    base = SolverBudget(exact=exact, policy_audit=AUDIT)
    CONFIGS.clear()
    CONFIGS.update({
        'negamax4': None,
        'exact_fallback': replace(base, cover_nodes=0, white_contexts=0),
        'three_rule': replace(base, rules=THREE_RULES, white_contexts=0),
        'nine_rule': replace(base, white_contexts=0),
        'victor_full': base,
    })


configure(24)
STRATEGIC_KINDS = ('strategic_nonloss', 'verified_policy_nonloss', 'exploratory_unrefuted',
                   'exploratory_nine_rule', 'exploratory_white_context')


def game_from(history):
    game = Connect4()
    for c in history:
        if not game.make_move(c):
            raise ValueError(f'illegal history {history}')
    return game


def canonical(history):
    game = game_from(history)
    rows = tuple(''.join(r) for r in game.board)
    return min(rows, tuple(r[::-1] for r in rows))


def immediate(game, player):
    wins = []
    for c in game.get_valid_moves():
        g = Connect4(game.board, player)
        g.make_move(c)
        if g.check_winner() == player:
            wins.append(c)
    return wins


def wilson(k, n, z=1.96):
    if not n:
        return [None, None]
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(centre - half, 4), round(centre + half, 4)]


def mean_ci(values, z=1.96):
    if not values:
        return dict(mean=None, ci=[None, None], n=0)
    m = statistics.fmean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    half = z * sd / math.sqrt(len(values))
    return dict(mean=round(m, 4), ci=[round(m - half, 4), round(m + half, 4)], n=len(values))


# ---------------------------------------------------------------- suite

def _sample_game(seed):
    """One stochastic game prefix: random, eps-Negamax-2 or eps-Negamax-4 style."""
    rng = Random(seed)
    style = ('random', 'eps_negamax2', 'eps_negamax4')[seed % 3]
    epsilon = {'random': 1.0, 'eps_negamax2': 0.3, 'eps_negamax4': 0.15}[style]
    depth = {'eps_negamax2': 2, 'eps_negamax4': 4}.get(style)
    phase = ('early', 'middle', 'late')[(seed // 3) % 3]
    lo, hi = PHASES[phase]
    target = rng.randint(lo, hi)
    game, history = Connect4(), []
    while len(history) < target:
        legal = game.get_valid_moves()
        nonterminal = []
        for c in legal:
            g = Connect4(game.board, game.current_player)
            g.make_move(c)
            if not g.is_game_over():
                nonterminal.append(c)
        if not nonterminal:
            return None
        if rng.random() < epsilon:
            c = rng.choice(nonterminal)
        else:
            scores = NegamaxAgent(depth).score_moves(game)
            best = max(scores[c] for c in nonterminal)
            c = rng.choice([c for c in nonterminal if scores[c] == best])
        game.make_move(c)
        history.append(c)
    return dict(seed=seed, style=style, history=history)


def _negamax4(history):
    return NegamaxAgent(4).choose_move(game_from(history))


def build_suite(args):
    start = perf_counter()
    with ProcessPoolExecutor(args.workers) as pool:
        sampled = [s for s in pool.map(_sample_game, range(args.seed, args.seed + args.pool))
                   if s is not None]
    seen, pool_rows = set(), []
    for s in sampled:
        key = canonical(s['history'])
        if key not in seen:
            seen.add(key)
            pool_rows.append(s)
    # Solve in parallel chunks; each worker process gets its own oracle table.
    chunks = [pool_rows[i::args.workers] for i in range(args.workers)]
    with ProcessPoolExecutor(args.workers) as pool:
        solved = list(pool.map(_solve_chunk, [[r['history'] for r in c] for c in chunks],
                               [args.oracle_nodes] * len(chunks)))
    results = {}
    for chunk, out in zip(chunks, solved):
        for row, oracle in zip(chunk, out):
            results[tuple(row['history'])] = oracle
    with ProcessPoolExecutor(args.workers) as pool:
        nm4 = dict(zip([tuple(r['history']) for r in pool_rows],
                       pool.map(_negamax4, [r['history'] for r in pool_rows], chunksize=8)))
    rows, unknown = [], 0
    for r in pool_rows:
        h = tuple(r['history'])
        o = results[h]
        if o['status'] != 'exact':
            unknown += 1
            continue
        game = game_from(h)
        mover = game.current_player
        own, other = immediate(game, mover), immediate(game, 1 - mover)
        values = {int(c): v for c, v in o['move_values'].items()}
        phase = next(p for p, (lo, hi) in PHASES.items() if lo <= len(h) <= hi)
        rows.append(dict(history=list(h), plies=len(h), remaining=42 - len(h), mover=mover,
                         phase=phase, style=r['style'], seed=r['seed'], value=o['value'],
                         move_values=values, optimal=[c for c, v in values.items() if v == o['value']],
                         decisive=min(values.values()) < o['value'],
                         quiet=not own and not other, own_wins=own, opponent_wins=other,
                         oracle_nodes=o['nodes'], negamax4=nm4[h],
                         negamax4_optimal=values[nm4[h]] == o['value']))
    rng = Random(args.seed)
    rng.shuffle(rows)
    # Natural: decisive positions in sampled order. Hard: disjoint, quiet positions
    # where Negamax-4 (Victor's fallback) is not optimal; selection is adversarial
    # to Negamax-4 by construction, so its accuracy there is 0% by definition.
    selected = []
    for phase in PHASES:
        for mover in (0, 1):
            stratum = [r for r in rows if r['phase'] == phase and r['mover'] == mover and r['decisive']]
            natural = stratum[:args.natural]
            hard = [r for r in stratum[args.natural:]
                    if not r['negamax4_optimal'] and r['quiet']][:args.hard]
            selected += [dict(r, subset='natural') for r in natural]
            selected += [dict(r, subset='negamax_hard') for r in hard]
    for i, r in enumerate(selected):
        r['id'] = f"{r['phase'][0]}{r['mover']}-{i:03d}"
    pool_summary = Counter((r['phase'], r['mover'], r['decisive'], r['quiet'], r['negamax4_optimal'])
                           for r in rows)
    out = dict(schema='victor-benchmark-suite-v1', oracle='tests/victor_validation/c4_oracle.c',
               seed=args.seed, pool_games=args.pool, sampled=len(sampled), distinct=len(pool_rows),
               oracle_unknown=unknown, oracle_node_limit=args.oracle_nodes, phases=PHASES,
               natural_per_stratum=args.natural, hard_per_stratum=args.hard,
               pool_strata=[dict(phase=k[0], mover=k[1], decisive=k[2], quiet=k[3],
                                 negamax4_optimal=k[4], count=v) for k, v in sorted(pool_summary.items())],
               seconds=round(perf_counter() - start, 2), positions=selected)
    write(args.output, out)


def _solve_chunk(histories, node_limit):
    return [dict(status=o.status, value=o.value, move_values=o.move_values, nodes=o.nodes)
            for o in solve_histories(histories, node_limit=node_limit)]


# ------------------------------------------------------------ positions

def decide(config, history):
    game = game_from(history)
    start = perf_counter()
    if CONFIGS[config] is None:
        agent = NegamaxAgent(4)
        move = agent.choose_move(game)
        return dict(move=move, kind='heuristic', seconds=perf_counter() - start,
                    negamax_nodes=agent.last_stats['nodes'])
    r = analyze_position(game.board, game.current_player, CONFIGS[config])
    seconds = perf_counter() - start
    out = dict(move=r.move, kind=r.move_kind, seconds=seconds, bound=r.bound,
               exact_status=r.exact.status, exact_nodes=r.exact.nodes)
    if r.black_cover is not None:
        out['root_black_cover'] = r.black_cover.status.value
        if r.black_cover.witness:
            out['root_cover_rules'] = sorted({e.candidate.rule.value
                                              for e in r.black_cover.witness.evidence})
    if r.continuation_cover is not None and r.continuation_cover.witness:
        out['child_cover_rules'] = sorted({e.candidate.rule.value
                                           for e in r.continuation_cover.witness.evidence})
    if r.policy_audit is not None:
        out['policy_audit'] = r.policy_audit.status
    if r.continuation_white_covers:
        found = [c for c in r.continuation_white_covers if c.status.value.startswith('compatible')]
        out['white_cover_kinds'] = sorted({c.context.kind for c in found})
    return out


def _decide_task(task):
    config, pid, history, remaining = task
    configure(remaining)
    return config, pid, decide(config, history)


def run_positions(args):
    suite = json.load(open(args.suite))
    configs = args.configs or list(CONFIGS)
    tasks = [(c, p['id'], p['history'], args.exact_remaining)
             for p in suite['positions'] for c in configs]
    start = perf_counter()
    decisions = defaultdict(dict)
    with ProcessPoolExecutor(args.workers) as pool:
        for config, pid, out in pool.map(_decide_task, tasks, chunksize=2):
            decisions[pid][config] = out
    rows = []
    for p in suite['positions']:
        values = {int(c): v for c, v in p['move_values'].items()}
        row = dict(id=p['id'], phase=p['phase'], mover=p['mover'], subset=p['subset'],
                   quiet=p['quiet'], value=p['value'], remaining=p['remaining'], decisions={})
        for config in configs:
            d = decisions[p['id']][config]
            d['move_value'] = values[d['move']]
            d['optimal'] = d['move_value'] == p['value']
            d['seconds'] = round(d['seconds'], 5)
            row['decisions'][config] = d
        rows.append(row)
    out = dict(schema='victor-benchmark-positions-v1', suite=args.suite, configs=configs,
               exact_remaining=args.exact_remaining,
               budgets=describe_configs(), workers=args.workers,
               seconds=round(perf_counter() - start, 2), python=platform.python_version(),
               platform=platform.platform(), rows=rows, summary=summarize_positions(rows, configs))
    write(args.output, out)


def describe_configs():
    out = {}
    for name, b in CONFIGS.items():
        if b is None:
            out[name] = dict(agent='NegamaxAgent', depth=4)
        else:
            out[name] = dict(exact=vars(b.exact), cover_nodes=b.cover_nodes,
                             white_contexts=b.white_contexts, strategic_children=b.strategic_children,
                             fallback_depth=b.fallback_depth, policy_audit=vars(b.policy_audit),
                             rules=[r.value for r in b.rules],
                             **({k: getattr(b, k) for k in ('deadline',) if hasattr(b, k)}))
    return out


def classify(value, move_value):
    if move_value == value:
        return 'optimal'
    if value == 1:
        return 'missed_win_to_draw' if move_value == 0 else 'win_to_loss'
    return 'draw_to_loss'


def summarize_positions(rows, configs):
    summary = {}
    groups = {'all': rows}
    for key in ('phase', 'subset'):
        for r in rows:
            groups.setdefault(f'{key}={r[key]}', []).append(r)
    for r in rows:
        groups.setdefault(f"mover={'white' if r['mover'] == 0 else 'black'}", []).append(r)
        groups.setdefault('quiet' if r['quiet'] else 'tactical', []).append(r)
    for name, group in groups.items():
        block = {}
        for config in configs:
            ds = [r['decisions'][config] for r in group]
            k = sum(d['optimal'] for d in ds)
            errors = Counter(classify(r['value'], r['decisions'][config]['move_value']) for r in group)
            kinds = Counter(d['kind'] for d in ds)
            strategic = [d for d in ds if d['kind'] in STRATEGIC_KINDS]
            block[config] = dict(n=len(ds), optimal=k, accuracy=round(k / len(ds), 4) if ds else None,
                                 accuracy_ci=wilson(k, len(ds)), errors=dict(errors),
                                 kinds=dict(kinds), strategic=len(strategic),
                                 strategic_optimal=sum(d['optimal'] for d in strategic),
                                 median_seconds=round(statistics.median(d['seconds'] for d in ds), 5),
                                 p95_seconds=round(sorted(d['seconds'] for d in ds)[int(0.95 * (len(ds) - 1))], 5),
                                 max_seconds=round(max(d['seconds'] for d in ds), 5))
            if config != 'exact_fallback' and 'exact_fallback' in configs:
                changed = [r for r in group
                           if r['decisions'][config]['move'] != r['decisions']['exact_fallback']['move']]
                better = sum(r['decisions'][config]['move_value'] > r['decisions']['exact_fallback']['move_value']
                             for r in changed)
                worse = sum(r['decisions'][config]['move_value'] < r['decisions']['exact_fallback']['move_value']
                            for r in changed)
                block[config]['vs_exact_fallback'] = dict(changed=len(changed), better=better,
                                                          worse=worse, neutral=len(changed) - better - worse)
        summary[name] = block
    return summary


# ---------------------------------------------------------------- games

def make_player(name, seed):
    if name == 'random':
        rng = Random(seed)
        return lambda game: rng.choice(game.get_valid_moves())
    if name.startswith('negamax:'):
        agent = NegamaxAgent(int(name.split(':')[1]))
        return agent.choose_move
    if name.startswith('mcts:'):
        agent = MCTSAgent(int(name.split(':')[1]), rng=Random(seed))
        return agent.choose_move
    if name == 'negamax4':
        return NegamaxAgent(4).choose_move
    if name in CONFIGS:
        return VictorSolver(CONFIGS[name])
    raise ValueError(name)


def play(task):
    config, opponent, opening, victor_color, seed, remaining = task
    configure(remaining)
    game = game_from(opening)
    players = [None, None]
    players[victor_color] = make_player(config, seed)
    players[1 - victor_color] = make_player(opponent, seed + 7919)
    history, decisions = list(opening), []
    start = perf_counter()
    while not game.is_game_over():
        mover = game.current_player
        t = perf_counter()
        if isinstance(players[mover], VictorSolver):
            column = players[mover].choose_move(Connect4(game.board, game.current_player))
            r = players[mover].last_result
            decisions.append(dict(ply=len(history), move=column, kind=r.move_kind,
                                  seconds=round(perf_counter() - t, 4), exact_nodes=r.exact.nodes))
        else:
            column = players[mover](Connect4(game.board, game.current_player))
            if mover == victor_color:
                decisions.append(dict(ply=len(history), move=column, kind='heuristic',
                                      seconds=round(perf_counter() - t, 4)))
        if column not in game.get_valid_moves():
            raise RuntimeError(f'{config if mover == victor_color else opponent} illegal move')
        assert game.make_move(column)
        history.append(column)
    winner = game.check_winner()
    result = 0.5 if winner == -1 else 1.0 if winner == victor_color else 0.0
    return dict(config=config, opponent=opponent, opening=list(opening), victor_color=victor_color,
                seed=seed, moves=history, winner=winner, score=result,
                seconds=round(perf_counter() - start, 3), decisions=decisions)


def openings(count, seed):
    rng = Random(seed)
    all_two = [(a, b) for a in range(7) for b in range(7)]
    rng.shuffle(all_two)
    return all_two[:count]


def run_games(args):
    configs = args.configs or list(CONFIGS)
    opens = openings(args.openings, args.seed)
    tasks = []
    for config in configs:
        for opponent in args.opponents:
            n = args.random_openings if opponent == 'random' else len(opens)
            for i, opening in enumerate(opens[:n]):
                for color in (0, 1):
                    tasks.append((config, opponent, opening, color, args.seed + 1000 * i + color,
                                  args.exact_remaining))
    start = perf_counter()
    with ProcessPoolExecutor(args.workers) as pool:
        games = list(pool.map(play, sorted(tasks, key=lambda t: t[0] == 'negamax4'), chunksize=1))
    if args.adjudicate:
        adjudicate(games, args)
    out = dict(schema='victor-benchmark-games-v1', configs=configs, opponents=args.opponents,
               exact_remaining=args.exact_remaining,
               openings=[list(o) for o in opens], seed=args.seed, budgets=describe_configs(),
               workers=args.workers, seconds=round(perf_counter() - start, 2),
               python=platform.python_version(), platform=platform.platform(),
               summary=summarize_games(games), games=games)
    write(args.output, out)


def adjudicate(games, args):
    """Oracle-check each Victor decision: did it lower the exact game value?"""
    positions = {}
    for g in games:
        for d in g['decisions']:
            if 42 - d['ply'] <= args.adjudicate_remaining:
                positions[tuple(g['moves'][:d['ply']])] = None
    keys = list(positions)
    chunks = [keys[i::args.workers] for i in range(args.workers)]
    with ProcessPoolExecutor(args.workers) as pool:
        for chunk, out in zip(chunks, pool.map(_solve_chunk, chunks,
                                                [args.oracle_nodes] * len(chunks))):
            positions.update(zip(chunk, out))
    for g in games:
        for d in g['decisions']:
            o = positions.get(tuple(g['moves'][:d['ply']]))
            if o and o['status'] == 'exact':
                d['value'] = o['value']
                d['move_value'] = o['move_values'][d['move']]


def summarize_games(games):
    summary = {}
    by = defaultdict(list)
    for g in games:
        by[(g['config'], g['opponent'])].append(g)
        by[(g['config'], 'all')].append(g)
    for (config, opponent), gs in sorted(by.items()):
        w = sum(g['score'] == 1 for g in gs)
        d = sum(g['score'] == 0.5 for g in gs)
        decisions = [x for g in gs for x in g['decisions']]
        judged = [x for x in decisions if 'move_value' in x]
        errors = Counter(classify(x['value'], x['move_value']) for x in judged)
        summary.setdefault(config, {})[opponent] = dict(
            games=len(gs), wins=w, draws=d, losses=len(gs) - w - d,
            score=mean_ci([g['score'] for g in gs]),
            as_white=mean_ci([g['score'] for g in gs if g['victor_color'] == 0]),
            as_black=mean_ci([g['score'] for g in gs if g['victor_color'] == 1]),
            decisions=len(decisions), judged_decisions=len(judged), errors=dict(errors),
            error_rate=round(sum(v for k, v in errors.items() if k != 'optimal') / len(judged), 4)
            if judged else None,
            kinds=dict(Counter(x['kind'] for x in decisions)),
            median_move_seconds=round(statistics.median(x['seconds'] for x in decisions), 4)
            if decisions else None,
            max_move_seconds=max((x['seconds'] for x in decisions), default=None),
            mean_game_seconds=round(statistics.fmean(g['seconds'] for g in gs), 3))
    return summary


# ------------------------------------------------------------ composite

COMPOSITES = ('aftereven', 'lowinverse', 'highinverse', 'baseclaim', 'before', 'specialbefore')


def _composite_sample(seed):
    """A White-to-move position with 8-30 empty cells and a nine-rule cover."""
    rng = Random(seed)
    target = rng.choice(range(12, 35, 2))  # even ply count: White to move
    epsilon = (1.0, 0.5, 0.25)[seed % 3]
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
            scores = NegamaxAgent(2).score_moves(game)
            c = max(moves, key=lambda m: (scores[m], rng.random()))
        game.make_move(c)
        history.append(c)
    position = Position.from_board(game.board, 0)
    found = []
    cover = search_nine_rule_cover(position, node_budget=10_000)
    if cover.witness is not None:
        used = sorted({e.candidate.rule.value for e in cover.witness.evidence})
        essential = []
        for rule in used:
            if rule in COMPOSITES:
                others = tuple(r for r in RuleName if r.value != rule)
                if search_nine_rule_cover(position, node_budget=10_000, rules=others).witness is None:
                    essential.append(rule)
        found.append(dict(history=history, search='all_rules', rules=used, essential=essential))
    # Targeted exposure: CL/BI/VE plus ONE composite rule forces witnesses that
    # actually execute that rule whenever it is needed for a cover.
    useful = {e.candidate.rule.value for e in analyze_nine_rules(position).evidence
              if e.conditional_solved_groups}
    for rule in ('highinverse', 'baseclaim', 'lowinverse', 'specialbefore'):
        if rule not in useful or any(rule in f['rules'] for f in found):
            continue
        subset = THREE_RULES + (RuleName(rule),)
        targeted = search_nine_rule_cover(position, node_budget=10_000, rules=subset)
        if targeted.witness is not None:
            used = sorted({e.candidate.rule.value for e in targeted.witness.evidence})
            if rule in used:
                essential = [rule] if search_nine_rule_cover(
                    position, node_budget=10_000, rules=THREE_RULES).witness is None else []
                found.append(dict(history=history, search='targeted:' + rule, rules=used,
                                  essential=essential))
    return found


def _composite_check(task):
    """Complete policy replay where feasible, plus fixed adversarial playouts."""
    history, rules, audit_remaining, audit_nodes, playouts = task
    game = game_from(history)
    position = Position.from_board(game.board, 0)
    cover = search_nine_rule_cover(position, node_budget=10_000,
                                   rules=tuple(RuleName(r) for r in rules))
    policy = NineRulePolicy(cover.witness)
    audit = policy.audit(SearchBudget(nodes=audit_nodes, seconds=None,
                                      max_remaining=audit_remaining, table_entries=audit_nodes))
    out = dict(history=history, audit=audit.status, audit_nodes=audit.nodes,
               counterexample=list(audit.counterexample), detail=audit.detail, playouts=[])
    for k in range(playouts):
        rng = Random(1000 * len(history) + k)
        white = NegamaxAgent(6) if k == 0 else NegamaxAgent(4)
        epsilon = 0.0 if k == 0 else 0.3
        g, continuation, status = game_from(history), (), 'draw'
        while not g.is_game_over():
            if g.current_player == 0:
                c = (rng.choice(g.get_valid_moves()) if rng.random() < epsilon
                     else white.choose_move(Connect4(g.board, 0)))
            else:
                d = policy.select(continuation)
                if d.column is None:
                    status = 'unsupported:' + d.status
                    break
                c = d.column
            g.make_move(c)
            continuation += (c,)
        if status == 'draw' and g.check_winner() != -1:
            status = 'white_win' if g.check_winner() == 0 else 'black_win'
        out['playouts'].append(dict(result=status, continuation=list(continuation)))
    return out


def run_composite(args):
    start = perf_counter()
    with ProcessPoolExecutor(args.workers) as pool:
        sampled = list(pool.map(_composite_sample, range(args.seed, args.seed + args.pool),
                                chunksize=16))
    samples = [s for s in sampled if s is not None]
    seen, covers = set(), []
    for found in samples:
        for s in found:
            key = (canonical(s['history']), s['search'])
            if key not in seen:
                seen.add(key)
                covers.append(s)
    composite = [s for s in covers if set(s['rules']) & set(COMPOSITES)]
    chunks = [composite[i::args.workers] for i in range(args.workers)]
    with ProcessPoolExecutor(args.workers) as pool:
        solved = list(pool.map(_solve_chunk, [[s['history'] for s in c] for c in chunks],
                               [args.oracle_nodes] * len(chunks)))
    for chunk, out in zip(chunks, solved):
        for s, o in zip(chunk, out):
            s['oracle_value'] = o['value']  # White to move, White-relative
            s['oracle_status'] = o['status']
    tasks = [(s['history'], s['rules'] if s['search'] != 'all_rules' else [r.value for r in RuleName],
              args.audit_remaining, args.audit_nodes, args.playouts) for s in composite]
    with ProcessPoolExecutor(args.workers) as pool:
        checks = list(pool.map(_composite_check, tasks, chunksize=1))
    for s, c in zip(composite, checks):
        s.update(audit=c['audit'], audit_nodes=c['audit_nodes'], detail=c['detail'],
                 counterexample=c['counterexample'],
                 playouts=Counter(p['result'] for p in c['playouts']),
                 failing_playouts=[p for p in c['playouts'] if p['result'] != 'draw'
                                   and p['result'] != 'black_win'])
    summary = {}
    for rule in COMPOSITES:
        used = [s for s in composite if rule in s['rules']]
        summary[rule] = dict(
            targeted=sum(s['search'] == 'targeted:' + rule for s in used),
            covers_using=len(used), essential=sum(rule in s['essential'] for s in used),
            oracle_white_wins=sum(s['oracle_status'] == 'exact' and s['oracle_value'] == 1 for s in used),
            oracle_checked=sum(s['oracle_status'] == 'exact' for s in used),
            audit=dict(Counter(s['audit'] for s in used)),
            playouts=dict(sum((Counter(s['playouts']) for s in used), Counter())))
    out = dict(schema='victor-benchmark-composite-v1', seed=args.seed, pool=args.pool,
               sampled=len(samples), distinct_covers=len(covers),
               default_search_covers=sum(s['search'] == 'all_rules' for s in covers),
               cover_rule_sets=dict(Counter(','.join(s['rules']) for s in covers
                                            if s['search'] == 'all_rules').most_common(25)),
               audit_remaining=args.audit_remaining, audit_nodes=args.audit_nodes,
               playouts_per_cover=args.playouts, seconds=round(perf_counter() - start, 2),
               summary=summary,
               contradictions=[s for s in composite if s['oracle_status'] == 'exact'
                               and s['oracle_value'] == 1],
               refuted=[s for s in composite if s['audit'] == 'refuted' or s['failing_playouts']],
               covers=[{k: v for k, v in s.items() if k not in ('failing_playouts',)}
                       for s in composite])
    write(args.output, out)


# -------------------------------------------------------------- latency

def run_latency(args):
    """Serial timing on a fixed subset: no parallel-worker contention."""
    suite = json.load(open(args.suite))
    rng = Random(args.seed)
    positions = [p for p in suite['positions'] if p['subset'] == 'natural']
    rng.shuffle(positions)
    by_phase = defaultdict(list)
    for p in positions:
        if len(by_phase[p['phase']]) < args.per_phase:
            by_phase[p['phase']].append(p)
    configs = args.configs or list(CONFIGS)
    configure(args.exact_remaining)
    rows = []
    for phase, ps in by_phase.items():
        for p in ps:
            for config in configs:
                d = decide(config, p['history'])
                rows.append(dict(id=p['id'], phase=phase, config=config, seconds=round(d['seconds'], 5),
                                 kind=d['kind'], exact_nodes=d.get('exact_nodes')))
    summary = {}
    for config in configs:
        for phase in list(by_phase) + ['all']:
            xs = [r['seconds'] for r in rows if r['config'] == config and phase in (r['phase'], 'all')]
            summary.setdefault(config, {})[phase] = dict(
                n=len(xs), median=round(statistics.median(xs), 4),
                p90=round(sorted(xs)[int(0.9 * (len(xs) - 1))], 4), max=round(max(xs), 4))
    write(args.output, dict(schema='victor-benchmark-latency-v1', suite=args.suite,
                            exact_remaining=args.exact_remaining,
                            per_phase=args.per_phase, python=platform.python_version(),
                            platform=platform.platform(), hardware=hardware(),
                            summary=summary, rows=rows))


def run_public_latency(args):
    """Serial sweep of the opt-in API agent's exact public profile (deadline included)."""
    from itertools import product
    from games.connect4.agents.victor_research_agent import PUBLIC_BUDGET, VictorResearchAgent
    suite = json.load(open(args.suite))
    rng = Random(args.seed)
    openings = [list(h) for n in range(4) for h in product(range(7), repeat=n)
                if n <= 2 or rng.random() < args.opening_fraction]
    agent, rows = VictorResearchAgent(), []
    for history in [p['history'] for p in suite['positions']] + openings:
        game = Connect4()
        if not all(game.make_move(c) for c in history) or game.is_game_over():
            continue
        start = perf_counter()
        move = agent.choose_move(game)
        rows.append(dict(plies=len(history), seconds=round(perf_counter() - start, 4),
                         kind=agent.last_decision['kind'], move=move,
                         deadline_reached=agent.last_decision['deadline_reached']))
    xs = sorted(r['seconds'] for r in rows)
    write(args.output, dict(
        schema='victor-public-agent-latency-v1', suite=args.suite, seed=args.seed,
        budget=dict(exact=vars(PUBLIC_BUDGET.exact), deadline=PUBLIC_BUDGET.deadline,
                    cover_nodes=PUBLIC_BUDGET.cover_nodes, white_contexts=PUBLIC_BUDGET.white_contexts,
                    policy_audit=vars(PUBLIC_BUDGET.policy_audit)),
        hardware=hardware(), python=platform.python_version(),
        summary=dict(n=len(xs), median=xs[len(xs) // 2], p95=xs[int(0.95 * (len(xs) - 1))],
                     p99=xs[int(0.99 * (len(xs) - 1))], max=xs[-1],
                     deadline_reached=sum(r['deadline_reached'] for r in rows),
                     kinds=dict(Counter(r['kind'] for r in rows))),
        rows=rows))


# --------------------------------------------------------------- report

def _pct(x):
    return '-' if x is None else f'{100 * x:.1f}%'


def _ci(ci):
    return f'[{_pct(ci[0])}, {_pct(ci[1])}]'


def render_report(args):
    """Markdown tables from saved artifacts; every report number is regenerable."""
    load = lambda path: json.load(open(path))
    lines = []
    for label, path in (('Baseline (original solver)', args.baseline_positions),
                        ('Improved solver', args.positions)):
        if not path:
            continue
        data = load(path)
        lines += [f'### {label}: positions (exact cap {data.get("exact_remaining", 14)})', '',
                  '| Group | ' + ' | '.join(data['configs']) + ' |',
                  '| --- |' + ' ---: |' * len(data['configs'])]
        for group in ('all', 'phase=early', 'phase=middle', 'phase=late', 'subset=natural',
                      'subset=negamax_hard', 'mover=white', 'mover=black', 'quiet', 'tactical'):
            block = data['summary'][group]
            n = block[data['configs'][0]]['n']
            lines.append(f'| {group} (n={n}) | ' + ' | '.join(
                f"{_pct(block[c]['optimal'] / block[c]['n'])} {_ci(block[c]['accuracy_ci'])}"
                for c in data['configs']) + ' |')
        block = data['summary']['all']
        lines += ['', '| Config | Optimal | Win→draw | Win→loss | Draw→loss | Strategic (optimal) '
                  '| Changed vs exact+fallback (better/worse/neutral) | Median s | p95 s | Max s |',
                  '| --- |' + ' ---: |' * 9]
        for c in data['configs']:
            b = block[c]
            e = b['errors']
            v = b.get('vs_exact_fallback')
            changed = '-' if v is None or c == 'negamax4' else (
                f"{v['changed']} ({v['better']}/{v['worse']}/{v['neutral']})")
            lines.append(f"| {c} | {b['optimal']}/{b['n']} | {e.get('missed_win_to_draw', 0)} | "
                         f"{e.get('win_to_loss', 0)} | {e.get('draw_to_loss', 0)} | "
                         f"{b['strategic']} ({b['strategic_optimal']}) | {changed} | "
                         f"{b['median_seconds']:.3f} | {b['p95_seconds']:.3f} | {b['max_seconds']:.2f} |")
        lines.append('')
    for label, path in (('Baseline (original solver)', args.baseline_games),
                        ('Improved solver', args.games)):
        if not path:
            continue
        data = load(path)
        lines += [f'### {label}: games (exact cap {data.get("exact_remaining", 14)})', '',
                  '| Config | Opponent | Games | W/D/L | Score [95% CI] | As White | As Black '
                  '| Judged decisions | Oracle error rate | Median move s | Max move s |',
                  '| --- | --- |' + ' ---: |' * 9]
        for config, block in data['summary'].items():
            for opponent, x in block.items():
                lines.append(
                    f"| {config} | {opponent} | {x['games']} | {x['wins']}/{x['draws']}/{x['losses']} | "
                    f"{x['score']['mean']:.3f} [{x['score']['ci'][0]:.3f}, {x['score']['ci'][1]:.3f}] | "
                    f"{x['as_white']['mean']:.3f} | {x['as_black']['mean']:.3f} | {x['judged_decisions']} | "
                    f"{_pct(x['error_rate'])} | {x['median_move_seconds']:.3f} | {x['max_move_seconds']:.2f} |")
        lines.append('')
    if args.composite:
        data = load(args.composite)
        lines += [f"### Composite rules ({data['sampled']} samples, {data['distinct_covers']} covers)", '',
                  '| Rule | Covers using | Targeted | Essential | Oracle-checked | Oracle White wins '
                  '| Audit verified | Audit unknown | Audit refuted/unsupported | Playouts B/D/W |',
                  '| --- |' + ' ---: |' * 9]
        for rule, x in data['summary'].items():
            a, pl = x['audit'], x['playouts']
            bad = sum(v for k, v in a.items() if k in ('refuted', 'unsupported_policy'))
            unknown = sum(v for k, v in a.items() if k.startswith('unknown'))
            lines.append(f"| {rule} | {x['covers_using']} | {x['targeted']} | {x['essential']} | "
                         f"{x['oracle_checked']} | {x['oracle_white_wins']} | "
                         f"{a.get('verified_policy_nonloss', 0)} | {unknown} | {bad} | "
                         f"{pl.get('black_win', 0)}/{pl.get('draw', 0)}/{pl.get('white_win', 0)} |")
        lines += ['', f"Contradictions: {len(data['contradictions'])}; refuted/failing policies: "
                      f"{len(data['refuted'])}.", '']
    for label, path in (('Baseline', args.baseline_latency), ('Improved', args.latency)):
        if not path:
            continue
        data = load(path)
        lines += [f'### {label}: serial latency, seconds (median / p90 / max)', '',
                  '| Config | ' + ' | '.join(next(iter(data['summary'].values()))) + ' |',
                  '| --- |' + ' ---: |' * len(next(iter(data['summary'].values())))]
        for config, phases in data['summary'].items():
            lines.append(f'| {config} | ' + ' | '.join(
                f"{x['median']:.3f} / {x['p90']:.3f} / {x['max']:.3f}" for x in phases.values()) + ' |')
        lines.append('')
    text = '\n'.join(lines) + '\n'
    sys.stdout.write(text) if args.output == '-' else open(args.output, 'w').write(text)


def hardware():
    if platform.system() != 'Darwin':
        return [platform.processor()]
    text = subprocess.run(['system_profiler', 'SPHardwareDataType'], capture_output=True,
                          text=True).stdout
    return [line.strip() for line in text.splitlines()
            if any(k in line for k in ('Model Identifier:', 'Chip:', 'Total Number of Cores:', 'Memory:'))]


def write(path, data):
    text = json.dumps(data, separators=(',', ':'), default=str) + '\n'
    if path == '-':
        sys.stdout.write(text)
    else:
        with open(path, 'w') as f:
            f.write(text)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    s = sub.add_parser('suite')
    s.add_argument('--output', required=True)
    s.add_argument('--seed', type=int, default=20261009)
    s.add_argument('--pool', type=int, default=4500)
    s.add_argument('--natural', type=int, default=40)
    s.add_argument('--hard', type=int, default=25)
    s.add_argument('--oracle-nodes', type=int, default=400_000_000)
    s.add_argument('--workers', type=int, default=6)
    p = sub.add_parser('positions')
    p.add_argument('--suite', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--configs', nargs='+', choices=list(CONFIGS))
    p.add_argument('--workers', type=int, default=6)
    g = sub.add_parser('games')
    g.add_argument('--output', required=True)
    g.add_argument('--configs', nargs='+', choices=list(CONFIGS))
    g.add_argument('--opponents', nargs='+', default=['random', 'negamax:4', 'mcts:400'])
    g.add_argument('--openings', type=int, default=16)
    g.add_argument('--random-openings', type=int, default=8)
    g.add_argument('--seed', type=int, default=20261010)
    g.add_argument('--workers', type=int, default=6)
    g.add_argument('--adjudicate', action='store_true')
    g.add_argument('--adjudicate-remaining', type=int, default=36)
    g.add_argument('--oracle-nodes', type=int, default=100_000_000)
    lat = sub.add_parser('latency')
    lat.add_argument('--suite', required=True)
    lat.add_argument('--output', required=True)
    lat.add_argument('--configs', nargs='+', choices=list(CONFIGS))
    lat.add_argument('--per-phase', type=int, default=12)
    lat.add_argument('--seed', type=int, default=7)
    pub = sub.add_parser('public-latency')
    pub.add_argument('--suite', required=True)
    pub.add_argument('--output', required=True)
    pub.add_argument('--seed', type=int, default=1)
    pub.add_argument('--opening-fraction', type=float, default=0.35)
    rep = sub.add_parser('report')
    rep.add_argument('--output', default='-')
    for name in ('positions', 'baseline-positions', 'games', 'baseline-games', 'composite',
                 'latency', 'baseline-latency'):
        rep.add_argument('--' + name)
    comp = sub.add_parser('composite')
    comp.add_argument('--output', required=True)
    comp.add_argument('--seed', type=int, default=20261011)
    comp.add_argument('--pool', type=int, default=8000)
    comp.add_argument('--audit-remaining', type=int, default=16)
    comp.add_argument('--audit-nodes', type=int, default=3_000_000)
    comp.add_argument('--playouts', type=int, default=4)
    comp.add_argument('--oracle-nodes', type=int, default=400_000_000)
    comp.add_argument('--workers', type=int, default=6)
    for command in (p, g, lat):
        command.add_argument('--exact-remaining', type=int, default=24,
                             help='exact search cell cap for every Victor configuration')
    args = parser.parse_args(argv)
    dict(suite=build_suite, positions=run_positions, games=run_games, latency=run_latency,
         composite=run_composite, report=render_report,
         **{'public-latency': run_public_latency})[args.command](args)


if __name__ == '__main__':
    main()
