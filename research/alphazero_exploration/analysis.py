"""Diversity, tactical and uncertainty statistics over COMPLETE recorded games.

Games are the sampling units everywhere; positions inside a game are never
treated as independent. Every loaded game is replayed through the engine and
checked against the collector contracts before any statistic uses it.
"""
from collections import Counter
import gzip
import json
import math
from pathlib import Path
import statistics

from games.connect4.agents.mcts_nn_agent import visit_policy
from games.connect4.alphazero_v2.bitboard import BitboardState
from games.connect4.alphazero_v2.data import outcome_for_actor
from games.connect4.alphazero_v2.diagnostics import entropy
from games.connect4.alphazero_v2.oracle import (_opponent_threats, engine_position, legal_moves_scan,
                                               safe_moves_scan, winning_moves_scan)
from . import schedules as s

_TACTICS = {}


def stage_of(ply):
    return next(name for name, low, high in s.STAGES if low <= ply < high)


def mirror_bits(bits):
    """Reverse the seven 7-bit columns of a bitboard."""
    return sum(((bits >> (7 * col)) & 127) << (7 * (6 - col)) for col in range(7))


def canonical(key):
    """Smaller of a position and its horizontal reflection."""
    return min(key, (mirror_bits(key[0]), mirror_bits(key[1])))


def tactics(key, history):
    """One-reply oracle sets (W, S, T, L) for a position; history only locates it."""
    found = _TACTICS.get(key)
    if found is None:
        found = _TACTICS[key] = (frozenset(winning_moves_scan(history)), frozenset(safe_moves_scan(history)),
                                 frozenset(_opponent_threats(history)), frozenset(legal_moves_scan(history)))
    return found


def build_game(identity, schedule, moves, winner, plies, *, outcomes=None, verify=True, record_sha256=None):
    """Replay one record through the engine and attach positions, tactics and checks.

    ``plies`` rows need visits, action, temperature; distribution/root_prior are
    optional (archived games have neither). Raises on any contract violation.
    """
    moves = tuple(moves)
    game, state = engine_position([]), BitboardState.from_game(engine_position([]))
    if len(plies) != len(moves) or not moves:
        raise ValueError('Every move needs exactly one searched example')
    built = []
    for ply, (move, row) in enumerate(zip(moves, plies)):
        if game.is_game_over():
            raise ValueError('History continues after termination')
        key = state.pieces
        if state != BitboardState.from_game(game) or state.occupied.bit_count() != ply:
            raise ValueError('Recorded example is not the pre-move position')
        wins, safe, threats, legal = tactics(key, moves[:ply])
        visits, temperature = tuple(row['visits']), float(row['temperature'])
        best = max(visits)
        distribution = row.get('distribution')
        if verify:
            if row.get('ply', ply) != ply or row['action'] != move:
                raise ValueError('Recorded ply/action disagrees with the trajectory')
            if (sum(visits) != s.SIMULATIONS or any(type(v) is not int or v < 0 for v in visits)
                    or any(visits[a] for a in range(7) if a not in legal) or visits[move] < 1):
                raise ValueError('Visits are not a legal 256-simulation root count')
            if schedule is not None and temperature != schedule.temperature(ply):
                raise ValueError('Temperature disagrees with the schedule at this ply')
            if temperature == 0 and visits[move] != best:
                raise ValueError('Temperature-0 action is not a maximum-visit action')
            if distribution is not None and tuple(distribution) != tuple(float(p) for p in visit_policy(visits, temperature)):
                raise ValueError('Recorded execution distribution is not visit_policy(visits, temperature)')
            if outcomes is not None and outcomes[ply] != outcome_for_actor(winner, ply % 2):
                raise ValueError('Outcome is not relative to the pre-move actor')
            prior = row.get('root_prior')
            if prior is not None and (abs(sum(prior) - 1) > 1e-9 or any(prior[a] for a in range(7) if a not in legal)):
                raise ValueError('Root prior is not a legal probability vector')
        if distribution is None:
            distribution = tuple(float(p) for p in visit_policy(visits, temperature))
        built.append(dict(ply=ply, key=key, canon=canonical(key), visits=visits, action=move,
                          temperature=temperature, distribution=tuple(distribution),
                          prior=None if row.get('root_prior') is None else tuple(row['root_prior']),
                          value=row.get('root_value'), z=outcome_for_actor(winner, ply % 2),
                          wins=wins, safe=safe, threats=threats, legal=legal,
                          top=frozenset(a for a in range(7) if visits[a] == best)))
        if not game.make_move(move):
            raise ValueError('Engine rejected a recorded move')
        state = state.drop(move)
    if not game.is_game_over() or game.check_winner() != winner:
        raise ValueError('Recorded game is incomplete or its winner is wrong')
    result = dict(id=tuple(identity), schedule=None if schedule is None else schedule.name, moves=moves,
                  winner=winner, plies=built, record_sha256=record_sha256)
    result['tactics'] = game_tactics(result)
    result['sums'] = game_sums(result)
    return result


def load_collection(directory):
    """All games of the complete cells of one invocation, plus cell metadata.

    Incomplete cells are returned separately and their games are never loaded.
    """
    directory = Path(directory)
    cells = json.loads((directory / 'cells.json').read_text())
    games, skipped = [], []
    for cell in cells:
        if not cell['complete']:
            skipped.append(cell)
            continue
        schedule = s.BY_NAME[cell['schedule']['name']]
        if schedule.to_dict() != cell['schedule']:
            raise ValueError('Cell schedule is not a declared schedule')
        path = directory / cell['file']
        # Committed evidence is gzip-compressed; raw experiment folders are not.
        text = path.read_text() if path.exists() else gzip.open(f'{path}.gz', 'rt').read()
        lines = [json.loads(line) for line in text.splitlines()]
        if sorted(line['index'] for line in lines) != cell['ids']:
            raise ValueError('Complete cell does not contain exactly its declared identities')
        for line in sorted(lines, key=lambda r: r['index']):
            games.append(build_game((cell['seed'], cell['generation'], cell.get('cohort', 0), line['index']),
                                    schedule, line['moves'], line['winner'], line['plies'],
                                    outcomes=line['outcomes'], record_sha256=line['record_sha256']))
    return games, cells, skipped


# Per-game sufficient statistics -------------------------------------------------------------

def _attribution(top, errors):
    return 'search' if top <= errors else 'sampling' if not top & errors else 'tie'


def game_tactics(game):
    """Counts for one game; see DESIGN.md for every definition."""
    c = Counter(games=1)
    last = len(game['plies']) - 1
    for p in game['plies']:
        wins, safe, threats, legal, action, top = p['wins'], p['safe'], p['threats'], p['legal'], p['action'], p['top']
        stage, tau = stage_of(p['ply']), f"tau{p['temperature']:g}"
        c['positions'] += 1
        c['safe_available'] += bool(safe)
        c['lost_states'] += not safe
        c['tactical_states'] += bool(wins or threats or len(safe) < len(legal))
        if wins:
            errors = legal - wins
            missed = action not in wins
            c['win_opps'] += 1
            c[f'win_opps@{stage}'] += 1
            c[f'win_opps@{tau}'] += 1
            c['greedy_missed_wins'] += len(top & errors) / len(top)
            c['expected_missed_wins'] += sum(p['distribution'][a] for a in errors)
            if missed:
                c['missed_wins'] += 1
                c[f'missed_wins@{stage}'] += 1
                c[f'missed_wins@{tau}'] += 1
                c['missed_wins:' + _attribution(top, errors)] += 1
                c['result_changing_missed_wins'] += p['z'] != 1
        elif safe and len(safe) < len(legal):
            errors = legal - safe
            unsafe = action not in safe
            c['choice_states'] += 1
            c[f'choice_states@{stage}'] += 1
            c[f'choice_states@{tau}'] += 1
            c['greedy_unsafe'] += len(top & errors) / len(top)
            c['expected_unsafe'] += sum(p['distribution'][a] for a in errors)
            if unsafe:
                c['unsafe'] += 1
                c[f'unsafe@{stage}'] += 1
                c[f'unsafe@{tau}'] += 1
                c['unsafe:' + _attribution(top, errors)] += 1
        if not wins and threats:
            c['block_states'] += 1
            c['blockable_states'] += bool(safe)
            c['missed_blocks'] += bool(safe) and action not in safe
        if safe and action not in safe:
            c['avoidable_unsafe_moves'] += 1
            # The opponent took the immediate win this move allowed: the game ended on it.
            if p['ply'] == last - 1 and game['winner'] == 1 - p['ply'] % 2:
                c['game_ending_errors'] += 1
                c['game_ending_errors:' + _attribution(top, legal - safe)] += 1
    c['errors'] = c['missed_wins'] + c['unsafe']
    return c


def game_sums(game):
    """Per-stage sums of per-position policy statistics (ratio-of-sums means)."""
    sums = Counter()
    for p in game['plies']:
        target = [v / s.SIMULATIONS for v in p['visits']]
        for scope in (stage_of(p['ply']), 'all') + (('8+',) if p['ply'] >= 8 else ()):
            sums[scope, 'n'] += 1
            sums[scope, 'target_entropy'] += entropy(target)
            sums[scope, 'execution_entropy'] += entropy(p['distribution'])
            sums[scope, 'max_target'] += max(target)
            sums[scope, 'visited_actions'] += sum(v > 0 for v in p['visits'])
            sums[scope, 'executed_top'] += p['action'] in p['top']
            if p['prior'] is not None:
                sums[scope, 'n_prior'] += 1
                sums[scope, 'prior_entropy'] += entropy(p['prior'])
                sums[scope, 'max_prior'] += max(p['prior'])
                sums[scope, 'prior_on_action'] += p['prior'][p['action']]
                sums[scope, 'prior_top_is_visit_top'] += max(range(7), key=p['prior'].__getitem__) in p['top']
    return sums


def total(games, field):
    result = Counter()
    for game in games:
        result.update(game[field])
    return result


def rate(games, numerator, denominator):
    counts = total(games, 'tactics')
    return counts[numerator] / counts[denominator] if counts[denominator] else float('nan')


PRIOR_FIELDS = ('prior_entropy', 'max_prior', 'prior_on_action', 'prior_top_is_visit_top')
POLICY_FIELDS = ('target_entropy', 'execution_entropy', 'max_target', 'visited_actions', 'executed_top') + PRIOR_FIELDS


def stage_mean(games, scope, name):
    """Mean over examples in a stage scope ('0-7', '8-15', '16-23', '24+', '8+', 'all')."""
    sums = total(games, 'sums')
    count = sums[scope, 'n_prior' if name in PRIOR_FIELDS else 'n']
    return sums[scope, name] / count if count else float('nan')


# Dataset-level diversity --------------------------------------------------------------------

def examples(games, low=0, high=42):
    for game in games:
        yield from game['plies'][low:high]


def duplicate_fraction(games, low=0, high=42, field='key'):
    keys = [p[field] for p in examples(games, low, high)]
    return 1 - len(set(keys)) / len(keys) if keys else float('nan')


def collision_probability(games, low=0, high=42, field='key'):
    """P(two distinct games share the position at a common ply); free of the game count."""
    counts = Counter(p[field] for p in examples(games, low, high))
    reached = Counter(p['ply'] for p in examples(games, low, high))
    pairs = sum(n * (n - 1) for n in reached.values())
    return sum(n * (n - 1) for n in counts.values()) / pairs if pairs else float('nan')


def prefixes(games, length):
    eligible = [g['moves'][:length] for g in games if len(g['moves']) >= length]
    return dict(eligible_games=len(eligible), unique=len(set(eligible)),
                unique_per_eligible=len(set(eligible)) / len(eligible) if eligible else None)


def concentration(counter):
    """Top shares, Herfindahl index and effective numbers of a multiset."""
    n = sum(counter.values())
    if not n:
        return dict(items=0, unique=0, top1_share=None, top5_share=None, herfindahl=None,
                    effective_number_simpson=None, effective_number_shannon=None)
    ordered = sorted(counter.values(), reverse=True)
    simpson = sum(c * c for c in ordered) / (n * n)
    return dict(items=n, unique=len(ordered), top1_share=ordered[0] / n, top5_share=sum(ordered[:5]) / n,
                herfindahl=simpson, effective_number_simpson=1 / simpson,
                effective_number_shannon=math.exp(entropy([c / n for c in ordered])))


def families(games, ply):
    """Mirror-canonical positions before ``ply`` among games that have that example."""
    return concentration(Counter(g['plies'][ply]['canon'] for g in games if len(g['plies']) > ply))


def describe(values):
    values = sorted(values)
    if not values:
        return dict(n=0)
    quantile = lambda q: values[min(len(values) - 1, max(0, math.ceil(q * len(values)) - 1))]
    return dict(n=len(values), mean=statistics.fmean(values), min=values[0], p10=quantile(.1), p25=quantile(.25),
                median=quantile(.5), p75=quantile(.75), p90=quantile(.9), max=values[-1],
                stdev=statistics.stdev(values) if len(values) > 1 else None)


def mirror_moves(moves):
    return tuple(6 - m for m in moves)


def diversity(games):
    """Full descriptive diversity profile of one dataset (one checkpoint x schedule)."""
    lengths = [len(g['moves']) for g in games]
    trajectories = Counter(g['moves'] for g in games)
    canonical_trajectories = Counter(min(g['moves'], mirror_moves(g['moves'])) for g in games)
    all_examples = list(examples(games))
    positions = Counter(p['key'] for p in all_examples)
    canon = Counter(p['canon'] for p in all_examples)
    stages = {}
    for name, low, high in s.STAGES + (('8+', 8, 42), ('all', 0, 42)):
        rows = list(examples(games, low, high))
        stages[name] = dict(
            positions=len(rows), unique_positions=len({p['key'] for p in rows}),
            unique_mirror_canonical=len({p['canon'] for p in rows}),
            duplicate_fraction=duplicate_fraction(games, low, high),
            mirror_canonical_duplicate_fraction=duplicate_fraction(games, low, high, 'canon'),
            collision_probability=collision_probability(games, low, high),
            mirror_canonical_collision_probability=collision_probability(games, low, high, 'canon'),
            **{field: stage_mean(games, name, field) for field in POLICY_FIELDS})
    policy = {}
    for name, low, high in (('all', 0, 42), ('0-7', 0, 8), ('8+', 8, 42)):
        rows = list(examples(games, low, high))
        targets = [[v / s.SIMULATIONS for v in p['visits']] for p in rows]
        policy[name] = dict(
            target_entropy=describe([entropy(t) for t in targets]),
            max_target=describe([max(t) for t in targets]),
            near_one_hot_target_share=sum(max(t) >= .9 for t in targets) / len(targets) if targets else None,
            execution_entropy=describe([entropy(p['distribution']) for p in rows]),
            prior_entropy=describe([entropy(p['prior']) for p in rows if p['prior'] is not None]),
            max_prior=describe([max(p['prior']) for p in rows if p['prior'] is not None]))
    tactical = [p for p in all_examples if p['wins'] or p['threats'] or len(p['safe']) < len(p['legal'])]
    return dict(
        games=len(games), positions=len(all_examples), length=describe(lengths),
        length_histogram={f'{low}-{low + 5}': sum(low <= n <= low + 5 for n in lengths) for low in range(7, 43, 6)},
        games_reaching={str(k): sum(n > k for n in lengths) for k in (8, 16, 24, 30)},
        results=dict(x_wins=sum(g['winner'] == 0 for g in games), o_wins=sum(g['winner'] == 1 for g in games),
                     draws=sum(g['winner'] == -1 for g in games)),
        unique_trajectories=len(trajectories), unique_mirror_canonical_trajectories=len(canonical_trajectories),
        max_trajectory_multiplicity=max(trajectories.values()),
        most_repeated_trajectories=[dict(count=n, length=len(m), moves=''.join(map(str, m)))
                                    for m, n in trajectories.most_common(5) if n > 1],
        unique_positions=len(positions), unique_mirror_canonical=len(canon),
        duplicate_fraction=1 - len(positions) / len(all_examples),
        mirror_canonical_duplicate_fraction=1 - len(canon) / len(all_examples),
        prefixes={str(k): prefixes(games, k) for k in (8, 12, 16, 20, 24)},
        families={str(k): families(games, k) for k in (8, 16, 24)},
        stages=stages, policy=policy,
        tactical_positions=len(tactical), unique_tactical_positions=len({p['key'] for p in tactical}),
        unique_mirror_canonical_tactical_positions=len({p['canon'] for p in tactical}))


TACTICAL_RATES = dict(
    missed_win_rate=('missed_wins', 'win_opps'), unsafe_rate=('unsafe', 'choice_states'),
    errors_per_game=('errors', 'games'), game_ending_error_share=('game_ending_errors', 'games'),
    missed_block_rate=('missed_blocks', 'blockable_states'),
    greedy_missed_win_rate=('greedy_missed_wins', 'win_opps'), greedy_unsafe_rate=('greedy_unsafe', 'choice_states'),
    expected_missed_win_rate=('expected_missed_wins', 'win_opps'),
    expected_unsafe_rate=('expected_unsafe', 'choice_states'),
    tactical_state_share=('tactical_states', 'positions'), win_opps_per_game=('win_opps', 'games'),
    choice_states_per_game=('choice_states', 'games'))


def tactical_profile(games):
    counts = total(games, 'tactics')
    return dict(counts={k: counts[k] for k in sorted(counts)},
                rates={name: (counts[n] / counts[d] if counts[d] else None)
                       for name, (n, d) in TACTICAL_RATES.items()})


# Uncertainty --------------------------------------------------------------------------------

def jackknife(units, statistic, *, replicates=False):
    """Delete-one jackknife over sampling units; 95% = estimate +/- 1.96 SE (descriptive).

    ``replicates`` also returns the delete-one values, for sensitivity checks.
    """
    n = len(units)
    estimate = statistic(units)
    leave_one_out = [statistic(units[:i] + units[i + 1:]) for i in range(n)]
    usable = [x for x in leave_one_out if not math.isnan(x)]
    if math.isnan(estimate) or len(usable) < n:
        return dict(estimate=None if math.isnan(estimate) else estimate, se=None, low=None, high=None, units=n,
                    **(dict(leave_one_out=leave_one_out) if replicates else {}))
    mean = sum(usable) / n
    se = math.sqrt((n - 1) / n * sum((x - mean) ** 2 for x in usable))
    result = dict(estimate=estimate, se=se, low=estimate - 1.96 * se, high=estimate + 1.96 * se, units=n)
    if replicates:
        result['leave_one_out'] = leave_one_out
    return result


def pair(reference, candidate):
    """Matched (reference game, candidate game) units by identity; identities must agree."""
    by_id = {g['id']: g for g in candidate}
    if len(by_id) != len(candidate) or sorted(by_id) != sorted(g['id'] for g in reference):
        raise ValueError('Conditions do not contain identical game identities')
    return [(g, by_id[g['id']]) for g in sorted(reference, key=lambda g: g['id'])]


def paired_statistic(reference, candidate, statistic, *, replicates=False):
    """``statistic(reference games, candidate games)`` with a paired delete-one-identity jackknife."""
    return jackknife(pair(reference, candidate),
                     lambda units: statistic([u[0] for u in units], [u[1] for u in units]), replicates=replicates)


def paired_difference(reference, candidate, statistic, *, replicates=False):
    """Candidate minus reference with a paired delete-one-identity jackknife."""
    return paired_statistic(reference, candidate, lambda r, c: statistic(c) - statistic(r), replicates=replicates)


def stratified(statistic):
    """Mean over checkpoints of a per-checkpoint dataset statistic (never pools datasets)."""
    def combined(games):
        groups = {}
        for game in games:
            groups.setdefault(game['id'][:2], []).append(game)
        return sum(statistic(group) for group in groups.values()) / len(groups)
    return combined
