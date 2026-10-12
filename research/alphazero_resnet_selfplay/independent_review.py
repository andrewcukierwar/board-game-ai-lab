"""Independent recomputation of the Phase 4D.4H-C results from raw run evidence.

Written by the independent results reviewer. Imports NOTHING from
research.alphazero_resnet_selfplay: the game rules, tactical labels, metrics,
bootstrap, verdict rules and accounting are all re-implemented here. Project
imports are limited to games.connect4: the engine/oracle as a second opinion
on game replays and (time-capped) on exact solved labels, and, in the
re-execution step, the frozen loader, search and arena to re-run the
generation-6 endpoints from the hashed checkpoints.

Reads   research/alphazero_resnet_selfplay/results/run-evidence/**, fixtures.json,
        results/{design,freeze,analysis,evidence-manifest}.json, the development
        opening package, and the local binary checkpoints (read-only).
Writes  research/alphazero_resnet_selfplay/results/independent-recomputed.json

Run from the worktree root, one thread:
    python -B research/alphazero_resnet_selfplay/independent_review.py
Options: --no-networks skips the re-execution (about five CPU minutes);
         --reuse-reexecution keeps that section from an earlier full run.
"""
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
RESULTS = ROOT / 'results'
EVIDENCE = RESULTS / 'run-evidence'
TASKS = EVIDENCE / 'tasks'
LOCAL_RUN = REPO / 'experiment-output/phase4d4h-c/run'
SEEDS = (42, 314159)
ARCHS = ('cnn', 'resnet3x64')
ARMS = [(s, a) for s in SEEDS for a in ARCHS]
EVAL_GENERATIONS = (0, 2, 4, 6)
REPLICATES = 10_000
REVIEW_SEED = 20261012  # the reviewer's own stream; unrelated to the frozen 4404903
EXPECTED_FIXTURES = '80f6851838b18172d6f818ebd306ce0cd36e4796cac1e3a71df29694feb3ca2c'
DESIGN_COMMIT = '18ba3b56f8fc517a96ae0d34247abb407c2922ce'
W, H = 7, 6


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def label(seed, arch):
    return f'{seed}-{arch}'


def clean(x):
    """JSON-safe copy (numpy scalars and arrays to Python)."""
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return [clean(v) for v in x.tolist()]
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


# ---------------------------------------------------------------------------------------------
# Own Connect 4 rules (column stacks; player 0 moves at even plies)
# ---------------------------------------------------------------------------------------------

class Board:
    def __init__(self):
        self.cols = [[] for _ in range(W)]
        self.count = 0
        self.winner = None

    def legal(self):
        return [c for c in range(W) if len(self.cols[c]) < H]

    def cell(self, c, r):
        return self.cols[c][r] if 0 <= c < W and 0 <= r < len(self.cols[c]) else None

    def completes(self, c, player):
        """Would a stone of `player` dropped in column c complete four in a row?"""
        r = len(self.cols[c])
        if r >= H:
            return False
        for dc, dr in ((1, 0), (0, 1), (1, 1), (1, -1)):
            run = 1
            for sign in (1, -1):
                x, y = c + sign * dc, r + sign * dr
                while self.cell(x, y) == player:
                    run += 1
                    x, y = x + sign * dc, y + sign * dr
            if run >= 4:
                return True
        return False

    def over(self):
        return self.winner is not None or self.count == W * H

    def play(self, c):
        player = self.count % 2
        won = self.completes(c, player)
        self.cols[c].append(player)
        self.count += 1
        if won:
            self.winner = player

    def copy(self):
        other = Board()
        other.cols = [list(c) for c in self.cols]
        other.count, other.winner = self.count, self.winner
        return other


def replay(moves):
    """(board, problem). problem is None when every move was legal and none followed a finished game."""
    board = Board()
    for ply, move in enumerate(moves):
        if type(move) is not int or not 0 <= move < W:
            return board, f'non-column move at ply {ply}'
        if board.over():
            return board, f'move after the game ended at ply {ply}'
        if move not in board.legal():
            return board, f'full column at ply {ply}'
        board.play(move)
    return board, None


def tactical_derivation(moves):
    """Own model-blind one-reply labels for a position."""
    board, problem = replay(moves)
    if problem or board.over():
        raise ValueError('fixture position is not a legal nonterminal history')
    mover = board.count % 2
    legal = board.legal()
    wins = [c for c in legal if board.completes(c, mover)]
    threats = [c for c in legal if board.completes(c, 1 - mover)]
    safe = []
    for c in legal:
        if c in wins:
            safe.append(c)
            continue
        after = board.copy()
        after.play(c)
        if after.count == W * H or not any(after.completes(x, 1 - mover) for x in after.legal()):
            safe.append(c)
    if len(wins) == 1:
        category, expected = 'unique_win', wins[0]
    elif not wins and threats and len(safe) == 1:
        category, expected = 'unique_safe', safe[0]
    else:
        category, expected = None, None
    return dict(legal=legal, immediate_wins=wins, opponent_threats=threats, safe_actions=safe, category=category,
                expected_action=expected)


# ---------------------------------------------------------------------------------------------
# Statistics (own implementation)
# ---------------------------------------------------------------------------------------------

def stream(*parts):
    digest = hashlib.sha256(':'.join(str(p) for p in ('independent-review',) + parts).encode()).hexdigest()
    return np.random.default_rng(int(digest[:16], 16))


def cluster_contrast(rows_by_seed, rng, replicates=REPLICATES):
    """rows_by_seed[seed] = [(cluster, resnet, cnn)]. Paired cluster bootstrap within seed, independent across
    seeds, pooled = equal-weight mean of the seeds. Statistic is sum of differences over number of rows."""
    per_seed, draws, points = {}, [], []
    for seed in SEEDS:
        rows = rows_by_seed[seed]
        names = sorted({c for c, _, _ in rows})
        index = {c: i for i, c in enumerate(names)}
        count, diff = np.zeros(len(names)), np.zeros(len(names))
        for c, a, b in rows:
            count[index[c]] += 1
            diff[index[c]] += a - b
        point = diff.sum() / count.sum()
        picks = rng.integers(0, len(names), size=(replicates, len(names)))
        boot = diff[picks].sum(1) / count[picks].sum(1)
        low, high = np.quantile(boot, [0.025, 0.975])
        per_seed[str(seed)] = dict(resnet=float(np.mean([a for _, a, _ in rows])), cnn=float(np.mean([b for _, _, b in rows])),
                                   difference=float(point), interval95=[float(low), float(high)], rows=len(rows),
                                   clusters=len(names), share_negative=float((boot < 0).mean()),
                                   share_positive=float((boot > 0).mean()))
        draws.append(boot)
        points.append(point)
    pooled = np.mean(draws, axis=0)
    low, high = np.quantile(pooled, [0.025, 0.975])
    return dict(per_seed=per_seed, pooled=dict(difference=float(np.mean(points)), interval95=[float(low), float(high)],
                                               share_negative=float((pooled < 0).mean())))


def supporting_rule(result, higher_is_better):
    sign = 1 if higher_is_better else -1
    seeds = [sign * v['difference'] for v in result['per_seed'].values()]
    low, high = sorted(sign * x for x in result['pooled']['interval95'])
    if all(s > 0 for s in seeds) and low > 0:
        return 'BETTER'
    if all(s < 0 for s in seeds) and high < 0:
        return 'WORSE'
    return 'INCONCLUSIVE'


def strength_rule(final, middle_pooled, mean_final_score, rule):
    seeds = [v['difference'] for v in final['per_seed'].values()]
    pooled = final['pooled']['difference']
    low, high = final['pooled']['interval95']
    if not rule['saturation'][0] <= mean_final_score <= rule['saturation'][1]:
        return 'INCONCLUSIVE'
    if all(s > 0 for s in seeds) and pooled >= rule['practical_margin'] and low > 0 and middle_pooled >= 0:
        return 'PASS'
    if all(s < 0 for s in seeds) and high < 0:
        return 'FAIL'
    return 'INCONCLUSIVE'


def tactical_rule(categories, rule):
    if any(c['pooled']['difference'] <= rule['fail_point'] and c['pooled']['interval95'][1] < 0 for c in categories):
        return 'FAIL'
    if all(c['pooled']['difference'] >= rule['pass_point'] and c['pooled']['interval95'][0] >= rule['pass_lower']
           for c in categories):
        return 'PASS'
    return 'INCONCLUSIVE'


def compute_band(ratio, rule):
    if rule['approximately_equal'][0] <= ratio <= rule['approximately_equal'][1]:
        return 'APPROXIMATELY_EQUAL'
    if rule['qualified'][0] <= ratio <= rule['qualified'][1]:
        return 'ROUGHLY_COMPARABLE'
    return 'NOT_EQUAL'


def recommendation_rule(strength, tactical, claim):
    if strength == 'FAIL' or tactical == 'FAIL':
        return 'KEEP_CNN'
    if strength == 'PASS' and claim != 'NOT_EQUAL':
        return 'REPLACE_WITH_RESNET3X64'
    return 'NO_DEMONSTRATED_ADVANTAGE_KEEP_CNN_DEFAULT'


def _betacf(a, b, x):
    tiny = 1e-300
    c, d = 1.0, 1.0 - (a + b) * x / (a + 1.0)
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 400):
        m2 = 2 * m
        aa = m * (b - m) * x / ((a + m2 - 1) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (a + b + m) * x / ((a + m2) * (a + m2 + 1))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return h


def betainc(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x))
    if x < (a + 1) / (a + b + 2):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1 - x) / b


def t_two_sided_p(t, df):
    return betainc(df / 2.0, 0.5, df / (df + t * t))


def t_critical(df, alpha=0.05):
    low, high = 0.0, 1e4
    for _ in range(200):
        mid = (low + high) / 2
        if t_two_sided_p(mid, df) > alpha:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def t_interval(values):
    values = np.asarray(values, dtype=float)
    n, mean = len(values), float(values.mean())
    se = float(values.std(ddof=1) / math.sqrt(n))
    crit = t_critical(n - 1)
    t = mean / se if se else float('inf')
    return dict(n=n, mean=mean, se=se, df=n - 1, t=t, p_two_sided=t_two_sided_p(t, n - 1) if se else 0.0,
                interval95=[mean - crit * se, mean + crit * se])


def exact_sign_flip(units):
    """Exact sign-flip (paired permutation) distribution of the sum of integer `units`. Returns two-sided and
    lower-tail p-values for the observed sum, counting every one of the 2^n sign assignments."""
    units = [int(u) for u in units]
    observed = sum(units)
    dist = {0: 1}
    for u in units:
        nxt = {}
        for s, ways in dist.items():
            for t in ((s + u, s - u) if u else (s,)):
                nxt[t] = nxt.get(t, 0) + (ways if u else ways * 2)
        dist = nxt
    total = sum(dist.values())
    assert total == 2 ** len(units)
    two = sum(w for s, w in dist.items() if abs(s) >= abs(observed)) / total
    lower = sum(w for s, w in dist.items() if s <= observed) / total
    return dict(n=len(units), nonzero=sum(1 for u in units if u), observed_sum_units=observed, p_two_sided=two,
                p_lower_tail=lower)


def binomial_sign_test(negative, positive):
    n, k = negative + positive, min(negative, positive)
    if not n:
        return dict(negative=negative, positive=positive, p_two_sided=1.0)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return dict(negative=negative, positive=positive, p_two_sided=min(1.0, 2 * tail))


class Comparison:
    """Largest absolute discrepancy between this review and analysis.json, by group."""

    def __init__(self):
        self.groups = {}

    def add(self, group, name, mine, theirs):
        g = self.groups.setdefault(group, dict(compared=0, max_abs_discrepancy=0.0, worst=None))
        g['compared'] += 1
        d = abs(float(mine) - float(theirs))
        if d > g['max_abs_discrepancy']:
            g['max_abs_discrepancy'], g['worst'] = d, name


# ---------------------------------------------------------------------------------------------
# 1. Matches
# ---------------------------------------------------------------------------------------------

def check_matches(design, status, problems):
    package = load(REPO / 'games/connect4/alphazero_v2/frozen/openings-development.json')
    base = [r for r in package['rows'] if r['id'].endswith('-base')]
    by_id = {r['id']: r for r in base}
    declared = design['evaluation']['openings']
    facts = dict(package_split=package['split'], base_rows=len(base), base_families=len({r['family'] for r in base}),
                 declared_equals_package_base=[r['id'] for r in base] == declared,
                 prefix_lengths=sorted({len(r['moves']) for r in base}),
                 odd_prefix_openings=sum(len(r['moves']) % 2 for r in base))
    try:
        sys.path.insert(0, str(REPO))
        from games.connect4.alphazero_v2.oracle import engine_position
    except Exception as error:  # the project engine is only a second opinion
        engine_position = None
        facts['project_engine_unavailable'] = repr(error)
    games, arms = {}, {}
    replayed = engine_checked = 0
    for seed, arch in ARMS:
        for generation in EVAL_GENERATIONS:
            name = f'{label(seed, arch)}-g{generation}'
            data = load(TASKS / f'{name}-matches' / 'matches.json')
            records = data['records']
            issues = []
            agent, opponent = data['agent'], data['opponent']
            expected_weights = status['boundaries'][label(seed, arch)][str(generation)]['weights_sha256']
            if agent['weights_sha256'] != expected_weights:
                issues.append('agent weights hash differs from the published boundary')
            if (agent['simulations'], agent['root_noise'], agent['tactical_guard'], agent['temperature']) != (
                    design['evaluation']['simulations'], False, False, 0):
                issues.append('agent search settings differ from the design')
            if opponent['depth'] != design['evaluation']['opponent_depth'] or opponent['version'] != 'connect4-reference-negamax-v1':
                issues.append('opponent differs from the design')
            if not data['complete'] or len(records) != 80:
                issues.append(f'{len(records)} records')
            slots = sorted((r['opening_id'], r['game_index']) for r in records)
            if slots != sorted((o, i) for o in declared for i in (0, 1)):
                issues.append('games are not exactly the 40 declared base openings twice')
            table = {}
            for r in records:
                opening = by_id.get(r['opening_id'])
                slot = (r['opening_id'], r['game_index'])
                if opening is None:
                    issues.append(f'unknown opening {r["opening_id"]}')
                    continue
                moves = r['moves']
                if (r['family'] != opening['family'] or r['prefix_length'] != len(opening['moves'])
                        or moves[:len(opening['moves'])] != opening['moves'] or r['stratum'] != 'prefix'):
                    issues.append(f'{slot}: opening identity or prefix differs')
                if r['agent_color'] != r['game_index'] or r['agent_color'] not in (0, 1):
                    issues.append(f'{slot}: color differs from the paired-game rule')
                if r.get('abandoned'):
                    issues.append(f'{slot}: abandoned')
                board, problem = replay(moves)
                replayed += 1
                if problem:
                    issues.append(f'{slot}: {problem}')
                    continue
                if not board.over():
                    issues.append(f'{slot}: record ends before the game is over')
                winner = -1 if board.winner is None else board.winner
                result = 'draw' if winner == -1 else ('win' if winner == r['agent_color'] else 'loss')
                score = {'win': 1.0, 'draw': 0.5, 'loss': 0.0}[result]
                if (winner, result, score) != (r['winner'], r['result'], r['score']):
                    issues.append(f'{slot}: recorded outcome {r["winner"], r["result"], r["score"]} differs from replay '
                                  f'{winner, result, score}')
                decisions = r.get('decisions') or []
                tail = moves[r['prefix_length']:]
                if ([d['move'] for d in decisions] != tail
                        or [d['color'] for d in decisions] != [(r['prefix_length'] + i) % 2 for i in range(len(tail))]):
                    issues.append(f'{slot}: decision list differs from the move list')
                if engine_position is not None:
                    game = engine_position(list(moves))
                    engine_checked += 1
                    if game.check_winner() != winner or not game.is_game_over():
                        issues.append(f'{slot}: project engine disagrees with the reviewer engine')
                    if engine_position(list(moves[:-1])).is_game_over():
                        issues.append(f'{slot}: project engine says the game ended earlier')
                table[slot] = dict(family=r['family'], color=r['agent_color'], score=score, result=result,
                                   moves=tuple(moves), plies=len(tail))
            games[(seed, arch, generation)] = table
            rows = list(table.values())
            arms[name] = dict(
                games=len(rows), wins=sum(g['result'] == 'win' for g in rows), draws=sum(g['result'] == 'draw' for g in rows),
                losses=sum(g['result'] == 'loss' for g in rows), score=sum(g['score'] for g in rows) / len(rows),
                score_as_first=float(np.mean([g['score'] for g in rows if g['color'] == 0])),
                score_as_second=float(np.mean([g['score'] for g in rows if g['color'] == 1])),
                mean_plies=sum(g['plies'] for g in rows) / len(rows), issues=issues)
            problems.extend(f'matches {name}: {i}' for i in issues)
    identical = {}
    for generation in EVAL_GENERATIONS:
        for i, a in enumerate(ARMS):
            for b in ARMS[i + 1:]:
                x, y = games[a + (generation,)], games[b + (generation,)]
                identical[f'g{generation}:{label(*a)}=={label(*b)}'] = sum(x[s]['moves'] == y[s]['moves'] for s in x)
    facts.update(games_replayed=replayed, games_cross_checked_with_project_engine=engine_checked,
                 identical_game_records_between_arms=identical)
    return games, arms, facts


def family_scores(table):
    out = {}
    for g in table.values():
        out.setdefault(g['family'], []).append(g['score'])
    if any(len(v) != 2 for v in out.values()) or len(out) != 40:
        raise ValueError('expected 40 families with two games each')
    return {f: sum(v) / 2 for f, v in out.items()}


def match_rows(games, seed, cnn_generation, resnet_generation, color=None):
    a, b = games[(seed, 'resnet3x64', resnet_generation)], games[(seed, 'cnn', cnn_generation)]
    return [(a[s]['family'], a[s]['score'], b[s]['score']) for s in sorted(a) if color is None or a[s]['color'] == color]


def family_differences(games, seed, cnn_generation, resnet_generation):
    a = family_scores(games[(seed, 'resnet3x64', resnet_generation)])
    b = family_scores(games[(seed, 'cnn', cnn_generation)])
    return {f: a[f] - b[f] for f in sorted(a)}


# ---------------------------------------------------------------------------------------------
# 3. Robustness of the primary verdict
# ---------------------------------------------------------------------------------------------

def robustness(games, arms, design, primary, middle):
    rule = design['analysis']['strength']
    families = sorted(family_differences(games, 42, 6, 6))
    d = {s: np.array([family_differences(games, s, 6, 6)[f] for f in families]) for s in SEEDS}
    d4 = {s: np.array([family_differences(games, s, 4, 4)[f] for f in families]) for s in SEEDS}
    mean_final = float(np.mean([arms[f'{label(s, a)}-g6']['score'] for s, a in ARMS]))
    out = {}

    # (a) many independent bootstrap streams
    verdicts, uppers, lowers, seed_low_upper = {}, [], [], []
    for k in range(200):
        rng = stream('strength-robustness', k)
        final = cluster_contrast({s: match_rows(games, s, 6, 6) for s in SEEDS}, rng)
        verdict = strength_rule(final, middle['pooled']['difference'], mean_final, rule)
        verdicts[verdict] = verdicts.get(verdict, 0) + 1
        lowers.append(final['pooled']['interval95'][0])
        uppers.append(final['pooled']['interval95'][1])
        seed_low_upper.append(final['per_seed']['314159']['interval95'][1])
    out['a_bootstrap_streams'] = dict(streams=200, replicates_each=REPLICATES, verdict_counts=verdicts,
                                      pooled_upper_bound_range=[min(uppers), max(uppers)],
                                      pooled_lower_bound_range=[min(lowers), max(lowers)],
                                      seed_314159_upper_bound_range=[min(seed_low_upper), max(seed_low_upper)])

    # (b) leave one opening family out (same family removed in both seeds), and worst single (seed, family)
    loo, loo_upper, loo_seed = {}, {}, {}
    for i, f in enumerate(families):
        keep = np.arange(len(families)) != i
        per_seed = [float(d[s][keep].mean()) for s in SEEDS]
        loo[f] = float(np.mean(per_seed))
        loo_seed[f] = per_seed
        rng = stream('strength-loo', f)
        draws = [d[s][keep][rng.integers(0, 39, size=(REPLICATES, 39))].mean(1) for s in SEEDS]
        loo_upper[f] = float(np.quantile(np.mean(draws, axis=0), 0.975))
    worst = max(loo, key=loo.get)
    order = {s: np.sort(d[s]) for s in SEEDS}
    removal = {}
    for s in SEEDS:
        values, removed = list(order[s]), 0
        while values and sum(values) < 0:
            values.pop(0)  # drop the family most favourable to the CNN
            removed += 1
        removal[str(s)] = removed
    out['b_leave_one_family_out'] = dict(
        pooled_difference_range=[min(loo.values()), max(loo.values())], least_negative_when_removing=worst,
        pooled_upper_bound_range=[min(loo_upper.values()), max(loo_upper.values())],
        any_pooled_upper_bound_at_or_above_zero=max(loo_upper.values()) >= 0,
        seed_42_difference_range=[min(v[0] for v in loo_seed.values()), max(v[0] for v in loo_seed.values())],
        seed_314159_difference_range=[min(v[1] for v in loo_seed.values()), max(v[1] for v in loo_seed.values())],
        seed_314159_sign_flips_in_any_leave_one_out=max(v[1] for v in loo_seed.values()) >= 0,
        families_most_favourable_to_cnn_to_remove_before_seed_difference_is_nonnegative=removal)

    # (c) intervals that do not use the percentile bootstrap
    per_seed_t = {str(s): t_interval(d[s]) for s in SEEDS}
    v = [per_seed_t[str(s)]['se'] ** 2 for s in SEEDS]
    pooled_mean = float(np.mean([per_seed_t[str(s)]['mean'] for s in SEEDS]))
    pooled_se = math.sqrt(sum(v)) / 2
    welch_df = sum(v) ** 2 / sum(x ** 2 / 39 for x in v)
    crit = t_critical(welch_df)
    joint = (d[42] + d[314159]) / 2
    correlation = float(np.corrcoef(d[42], d[314159])[0, 1])
    seed_values = [float(d[s].mean()) for s in SEEDS]
    between = t_interval(seed_values)
    rng = stream('strength-joint-family')
    picks = rng.integers(0, 40, size=(REPLICATES, 40))
    joint_boot = joint[picks].mean(1)
    out['c_alternative_intervals'] = dict(
        per_seed_t_on_40_family_differences=per_seed_t,
        pooled_welch_t=dict(mean=pooled_mean, se=pooled_se, df=welch_df, t=pooled_mean / pooled_se,
                            p_two_sided=t_two_sided_p(pooled_mean / pooled_se, welch_df),
                            interval95=[pooled_mean - crit * pooled_se, pooled_mean + crit * pooled_se]),
        family_as_unit_across_both_seeds_t=t_interval(joint),
        family_difference_correlation_between_seeds=correlation,
        joint_family_bootstrap_pooled_interval95=[float(x) for x in np.quantile(joint_boot, [0.025, 0.975])],
        seeds_as_the_unit_of_replication_t=between,
        note='the last row treats the two seeds as the sample (n=2, df=1); it is the only one that includes '
             'training-run variation')

    # (d) by colour
    colour = {}
    for c, name in ((0, 'agent_owns_first_colour_x'), (1, 'agent_owns_second_colour_o')):
        result = cluster_contrast({s: match_rows(games, s, 6, 6, color=c) for s in SEEDS}, stream('strength-colour', c))
        colour[name] = dict(per_seed={k: dict(resnet=v['resnet'], cnn=v['cnn'], difference=v['difference'],
                                               interval95=v['interval95']) for k, v in result['per_seed'].items()},
                            pooled=result['pooled'])
    out['d_by_colour'] = colour

    # (e) exact paired sign-flip and sign tests on family differences (units of 0.25 score)
    def units(values):
        scaled = np.asarray(values) * 4
        if not np.allclose(scaled, np.round(scaled)):
            raise ValueError('family differences are not multiples of 0.25')
        return [int(x) for x in np.round(scaled)]
    exact = {str(s): dict(sign_flip=exact_sign_flip(units(d[s])),
                          sign_test=binomial_sign_test(int((d[s] < 0).sum()), int((d[s] > 0).sum())),
                          families_negative=int((d[s] < 0).sum()), families_zero=int((d[s] == 0).sum()),
                          families_positive=int((d[s] > 0).sum())) for s in SEEDS}
    exact['pooled_80_seed_family_pairs_flipped_independently'] = exact_sign_flip(units(d[42]) + units(d[314159]))
    exact['pooled_40_families_flipped_jointly_across_seeds'] = exact_sign_flip(units(d[42] + d[314159]))
    slots = {}
    for s in SEEDS:
        a, b = games[(s, 'resnet3x64', 6)], games[(s, 'cnn', 6)]
        worse = sum(a[k]['score'] < b[k]['score'] for k in a)
        better = sum(a[k]['score'] > b[k]['score'] for k in a)
        slots[str(s)] = dict(resnet_lower=worse, resnet_higher=better, equal=80 - worse - better,
                             sign_test_ignoring_family_clustering=binomial_sign_test(worse, better))
    exact['game_slots'] = slots
    out['e_exact_tests'] = exact

    # (f) checkpoint-to-checkpoint swings of a single arm against the between-architecture difference
    swings, magnitudes = {}, []
    for seed, arch in ARMS:
        scores = {g: arms[f'{label(seed, arch)}-g{g}']['score'] for g in EVAL_GENERATIONS}
        entry = dict(scores=scores)
        for g0, g1 in ((2, 4), (4, 6)):
            a, b = games[(seed, arch, g1)], games[(seed, arch, g0)]
            rows = [(a[s]['family'], a[s]['score'], b[s]['score']) for s in sorted(a)]
            names = sorted({r[0] for r in rows})
            diff = np.array([sum(x - y for f, x, y in rows if f == n) / 2 for n in names])
            rng = stream('swing', seed, arch, g0, g1)
            boot = diff[rng.integers(0, 40, size=(REPLICATES, 40))].mean(1)
            entry[f'g{g0}_to_g{g1}'] = dict(change=float(diff.mean()),
                                            interval95=[float(x) for x in np.quantile(boot, [0.025, 0.975])])
            magnitudes.append(abs(float(diff.mean())))
        swings[label(seed, arch)] = entry
    change = {}
    for s in SEEDS:
        dd = d[s] - d4[s]
        rng = stream('difference-change', s)
        boot = dd[rng.integers(0, 40, size=(REPLICATES, 40))].mean(1)
        change[str(s)] = dict(d_g4=float(d4[s].mean()), d_g6=float(d[s].mean()), change=float(dd.mean()),
                              interval95=[float(x) for x in np.quantile(boot, [0.025, 0.975])])
    averaged = {str(s): float(((d[s] + d4[s]) / 2).mean()) for s in SEEDS}
    out['f_checkpoint_swings'] = dict(
        arms=swings, absolute_swings=sorted(magnitudes), mean_absolute_swing=float(np.mean(magnitudes)),
        root_mean_square_swing=float(np.sqrt(np.mean(np.square(magnitudes)))), largest_absolute_swing=max(magnitudes),
        pooled_g6_difference=primary['pooled']['difference'],
        difference_g4_to_g6=change,
        descriptive_mean_of_g4_and_g6_difference=dict(per_seed=averaged, pooled=float(np.mean(list(averaged.values())))),
        note='not prespecified; shown only to size checkpoint noise against the generation-6 contrast')

    # (g) is the FAIL driven by one seed
    gap = float(d[42].mean() - d[314159].mean())
    gap_se = math.sqrt(sum(per_seed_t[str(s)]['se'] ** 2 for s in SEEDS))
    out['g_seed_dependence'] = dict(
        per_seed={k: dict(difference=v['difference'], interval95=v['interval95'], share_of_replicates_negative=v['share_negative'])
                  for k, v in primary['per_seed'].items()},
        heterogeneity_between_seeds=dict(
            difference_of_seed_differences=gap, se=gap_se, z=gap / gap_se,
            p_two_sided_normal=math.erfc(abs(gap / gap_se) / math.sqrt(2)),
            note='tests whether the two seeds estimate the same architecture difference given opening sampling alone'),
        points_behind_out_of_80={str(s): float(-d[s].sum() * 2) for s in SEEDS},
        share_of_pooled_difference_from_seed_42=float(d[42].mean() / (d[42].mean() + d[314159].mean())),
        pooled_interval_if_both_seeds_resembled_314159=[float(x) for x in np.quantile(
            np.mean([d[314159][stream('twin', k).integers(0, 40, size=(REPLICATES, 40))].mean(1) for k in (0, 1)], axis=0),
            [0.025, 0.975])],
        seed_314159_alone_excludes_zero=primary['per_seed']['314159']['interval95'][1] < 0)
    return out


# ---------------------------------------------------------------------------------------------
# 4-5. Position suites
# ---------------------------------------------------------------------------------------------

def audit_fixtures(fixtures, problems):
    """Re-derive tactical labels with the reviewer's rules and check solved/deep labels for internal consistency."""
    report = dict(sha256=sha(ROOT / 'fixtures.json'), sources=sorted(fixtures['sources']))
    report['sha256_matches_declared'] = report['sha256'] == EXPECTED_FIXTURES
    mismatches = []
    for row in fixtures['tactical']:
        mine = tactical_derivation(row['moves'])
        for key in ('legal', 'immediate_wins', 'opponent_threats', 'safe_actions', 'category', 'expected_action'):
            if (sorted(mine[key]) if isinstance(mine[key], list) else mine[key]) != (
                    sorted(row[key]) if isinstance(row[key], list) else row[key]):
                mismatches.append(f'{row["id"]}:{key}')
    report['tactical'] = dict(rows=len(fixtures['tactical']), label_fields_rederived=6 * len(fixtures['tactical']),
                              mismatches=mismatches,
                              categories={c: sum(r['category'] == c for r in fixtures['tactical'])
                                          for c in ('unique_win', 'unique_safe')},
                              clusters=len({r['cluster'] for r in fixtures['tactical']}),
                              clusters_by_category={c: len({r['cluster'] for r in fixtures['tactical'] if r['category'] == c})
                                                    for c in ('unique_win', 'unique_safe')})
    for suite in ('solved', 'deep'):
        bad = []
        for row in fixtures[suite]:
            board, problem = replay(row['moves'])
            values = {int(k): v for k, v in row['action_values'].items()}
            mover = board.count % 2
            if problem or board.over() or sorted(values) != board.legal():
                bad.append(f'{row["id"]}:legal')
                continue
            if row['value'] != max(values.values()) or sorted(row['optimal_actions']) != sorted(
                    a for a, v in values.items() if v == row['value']):
                bad.append(f'{row["id"]}:optimal')
            if any(values[c] != 1 for c in board.legal() if board.completes(c, mover)):
                bad.append(f'{row["id"]}:immediate-win')
        report[suite] = dict(rows=len(fixtures[suite]), clusters=len({r['cluster'] for r in fixtures[suite]}),
                             internal_inconsistencies=bad,
                             value_counts={str(v): sum(r['value'] == v for r in fixtures[suite]) for v in (-1, 0, 1)},
                             constant_zero_value_mse=sum(r['value'] ** 2 for r in fixtures[suite]) / len(fixtures[suite]))
    # Exact values re-derived with the project's alpha-beta solver, time capped (not an independent solver).
    solved = dict(attempted=0, agreed=0, disagreed=[], budget_exceeded=0, seconds_cap=60)
    try:
        from games.connect4.alphazero_v2.oracle import BitboardSolver, SolverBudgetExceeded
        started = time.perf_counter()
        for suite in ('solved', 'deep'):
            for row in sorted(fixtures[suite], key=lambda r: -len(r['moves'])):
                if time.perf_counter() - started > solved['seconds_cap']:
                    break
                solved['attempted'] += 1
                try:
                    values = BitboardSolver(max_nodes=300_000).action_values(row['moves'])
                except SolverBudgetExceeded:
                    solved['budget_exceeded'] += 1
                    continue
                if {str(k): v for k, v in values.items()} == row['action_values']:
                    solved['agreed'] += 1
                else:
                    solved['disagreed'].append(row['id'])
        solved['seconds'] = time.perf_counter() - started
    except Exception as error:
        solved['error'] = repr(error)
    report['exact_action_values_resolved_with_project_solver'] = solved
    for key, value in (('tactical label mismatches', mismatches), ('solved inconsistencies', report['solved']['internal_inconsistencies']),
                       ('deep inconsistencies', report['deep']['internal_inconsistencies']), ('solver disagreements', solved['disagreed'])):
        if value:
            problems.append(f'fixtures: {key}: {value[:5]}')
    if not report['sha256_matches_declared']:
        problems.append('fixtures: SHA-256 differs from the declared one')
    return report


def argmax_set(values):
    best = max(values)
    return [i for i, v in enumerate(values) if v == best]


def derive_row(suite, row, fx, counters):
    """Every per-row measure from stored outputs (policy, value, actions, visits) and fixture labels only."""
    policy, action = row['policy'], row['policy_action']
    board, _ = replay(fx['moves'])
    legal = board.legal()
    counters['rows'] += 1
    if action not in argmax_set(policy):
        counters['policy_action_not_argmax'] += 1
    if len(argmax_set(policy)) > 1:
        counters['policy_argmax_ties'] += 1
    if abs(sum(policy) - 1) > 1e-9 or any(policy[a] != 0 for a in range(W) if a not in legal) or action not in legal:
        counters['policy_not_legal_masked'] += 1
    out = dict(cluster=fx['cluster'], value=row['value'], category=fx.get('category'))
    puct = row.get('puct_action')
    if puct is not None:
        counters['puct_rows'] += 1
        visits = row['visits']
        if sum(visits) != 256 or any(visits[a] for a in range(W) if a not in legal) or puct not in argmax_set(visits):
            counters['puct_visit_problems'] += 1
        if len(argmax_set(visits)) > 1:
            counters['puct_visit_ties'] += 1
    if suite in ('solved', 'deep'):
        mass = sum(policy[a] for a in fx['optimal_actions'])
        out.update(ce=-math.log(max(mass, 1e-300)), optimal_mass=mass, accuracy=float(action in fx['optimal_actions']),
                   se=(row['value'] - fx['value']) ** 2,
                   policy_deep_mistake=float(fx['action_values'][str(action)] < fx['value']))
        stored = dict(ce=row['ce'], optimal_mass=row['optimal_mass'], accuracy=row['accuracy'], se=row['se'],
                      policy_deep_mistake=row['policy_deep_mistake'])
        if puct is not None:
            out.update(puct_accuracy=float(puct in fx['optimal_actions']),
                       puct_deep_mistake=float(fx['action_values'][str(puct)] < fx['value']))
            stored.update(puct_accuracy=row['puct_accuracy'], puct_deep_mistake=row['puct_deep_mistake'])
    else:
        out.update(policy_correct=float(action == fx['expected_action']), policy_avoidable=float(action not in fx['safe_actions']),
                   policy_missed_win=float(bool(fx['immediate_wins']) and action not in fx['immediate_wins']))
        stored = dict(policy_correct=row['policy_correct'], policy_avoidable=row['policy_avoidable'],
                      policy_missed_win=row['policy_missed_win'])
        if row.get('category') != fx['category']:
            counters['category_mismatch'] += 1
        if puct is not None:
            out.update(puct_correct=float(puct == fx['expected_action']), puct_avoidable=float(puct not in fx['safe_actions']),
                       puct_missed_win=float(bool(fx['immediate_wins']) and puct not in fx['immediate_wins']))
            stored.update(puct_correct=row['puct_correct'], puct_avoidable=row['puct_avoidable'],
                          puct_missed_win=row['puct_missed_win'])
    for key, value in stored.items():
        if abs(float(value) - out[key]) > 1e-12:
            counters['stored_flag_or_value_mismatch'] += 1
    if row['cluster'] != fx['cluster']:
        counters['cluster_mismatch'] += 1
    return out


RAW_METRICS = (  # name, suite, key, category, higher is better
    ('solved_optimal_set_ce', 'solved', 'ce', None, False),
    ('solved_optimal_accuracy', 'solved', 'accuracy', None, True),
    ('solved_optimal_probability', 'solved', 'optimal_mass', None, True),
    ('solved_exact_value_mse', 'solved', 'se', None, False),
    ('solved_policy_deep_mistake', 'solved', 'policy_deep_mistake', None, False),
    ('deep_optimal_set_ce', 'deep', 'ce', None, False),
    ('deep_optimal_accuracy', 'deep', 'accuracy', None, True),
    ('deep_exact_value_mse', 'deep', 'se', None, False),
    ('deep_policy_deep_mistake', 'deep', 'policy_deep_mistake', None, False),
    ('tactical_policy_immediate_win', 'tactical', 'policy_correct', 'unique_win', True),
    ('tactical_policy_safe_response', 'tactical', 'policy_correct', 'unique_safe', True),
    ('tactical_policy_avoidable_one_ply_error', 'tactical', 'policy_avoidable', None, False),
    ('tactical_policy_missed_win', 'tactical', 'policy_missed_win', 'unique_win', False))
PUCT_METRICS = (
    ('tactical_puct_immediate_win', 'tactical', 'puct_correct', 'unique_win', True),
    ('tactical_puct_safe_response', 'tactical', 'puct_correct', 'unique_safe', True),
    ('tactical_puct_avoidable_one_ply_error', 'tactical', 'puct_avoidable', None, False),
    ('tactical_puct_missed_win', 'tactical', 'puct_missed_win', 'unique_win', False),
    ('solved_puct_optimal_accuracy', 'solved', 'puct_accuracy', None, True),
    ('solved_puct_deep_mistake', 'solved', 'puct_deep_mistake', None, False),
    ('deep_puct_optimal_accuracy', 'deep', 'puct_accuracy', None, True),
    ('deep_puct_deep_mistake', 'deep', 'puct_deep_mistake', None, False))


def load_positions(fixtures, problems):
    """raw[(seed, arch, generation)][suite] = {id: derived row} for generations 0-6 (dynamics.json);
    puct[(seed, arch, generation)][suite] likewise for the evaluated generations (positions/*.json)."""
    labels = {suite: {r['id']: r for r in fixtures[suite]} for suite in ('solved', 'tactical', 'deep')}
    counters = {k: 0 for k in ('rows', 'puct_rows', 'policy_action_not_argmax', 'policy_argmax_ties', 'policy_not_legal_masked',
                               'puct_visit_problems', 'puct_visit_ties', 'stored_flag_or_value_mismatch', 'category_mismatch',
                               'cluster_mismatch', 'membership_problems')}
    raw, puct, hashes = {}, {}, {}
    max_raw_gap = 0.0
    for seed, arch in ARMS:
        dynamics = load(TASKS / f'{label(seed, arch)}-dynamics' / 'dynamics.json')
        for generation, entry in enumerate(dynamics['rows']):
            if entry['generation'] != generation:
                problems.append(f'dynamics {label(seed, arch)}: row order')
            hashes[(seed, arch, generation)] = dict(weights=entry['weights_sha256'], boundary=entry['boundary_sha256'],
                                                    counters=entry['counters'])
            table = {}
            for suite in labels:
                rows = entry['raw'][suite]
                if [r['id'] for r in rows] != [r['id'] for r in fixtures[suite]]:
                    counters['membership_problems'] += 1
                table[suite] = {r['id']: derive_row(suite, r, labels[suite][r['id']], counters) for r in rows}
            raw[(seed, arch, generation)] = table
        for generation in EVAL_GENERATIONS:
            table = {}
            for suite in labels:
                data = load(TASKS / f'{label(seed, arch)}-g{generation}-positions' / f'{suite}.json')
                rows = data['rows']
                if [r['id'] for r in rows] != [r['id'] for r in fixtures[suite]] or data['simulations'] != 256:
                    counters['membership_problems'] += 1
                table[suite] = {r['id']: derive_row(suite, r, labels[suite][r['id']], counters) for r in rows}
                reference = dynamics['rows'][generation]['raw'][suite]
                for r, q in zip(rows, reference):
                    max_raw_gap = max(max_raw_gap, abs(r['value'] - q['value']),
                                      max(abs(x - y) for x, y in zip(r['policy'], q['policy'])))
                    if r['policy_action'] != q['policy_action']:
                        counters['membership_problems'] += 1
            puct[(seed, arch, generation)] = table
    counters['max_raw_output_gap_between_positions_and_dynamics_tasks'] = max_raw_gap
    for key in ('policy_action_not_argmax', 'policy_not_legal_masked', 'puct_visit_problems', 'stored_flag_or_value_mismatch',
                'category_mismatch', 'cluster_mismatch', 'membership_problems'):
        if counters[key]:
            problems.append(f'positions: {key} = {counters[key]}')
    return raw, puct, hashes, counters


def metric_mean(table, suite, key, category):
    values = [r[key] for r in table[suite].values() if category is None or r['category'] == category]
    return sum(values) / len(values)


def metric_rows(a, b, suite, key, category):
    ids = sorted(i for i, r in a[suite].items() if category is None or r['category'] == category)
    return [(a[suite][i]['cluster'], a[suite][i][key], b[suite][i][key]) for i in ids]


# ---------------------------------------------------------------------------------------------
# 6. Computation accounting and scheduling
# ---------------------------------------------------------------------------------------------

def accounting(design, status, problems):
    arms = {}
    for seed, arch in ARMS:
        rows, totals = [], dict(games=0, updates=0, positions=0, neural_positions=0, collection_seconds=0.0,
                                update_seconds=0.0, compute_seconds=0.0, collection_cpu_seconds=0.0, task_wall_seconds=0.0,
                                task_cpu_seconds=0.0, generation_wall_seconds=0.0)
        budget = design['arms'][arch]
        for generation in range(1, 7):
            done = load(TASKS / f'{label(seed, arch)}-g{generation}-train' / 'complete.json')
            result, timing = done['result'], done['timing']
            task = load(TASKS / f'{label(seed, arch)}-g{generation}-train' / 'task-complete.json')
            if (result['games'], result['updates'], len(result['metrics'])) != (
                    budget['games_per_generation'], budget['optimizer_updates'], budget['optimizer_updates']):
                problems.append(f'accounting {label(seed, arch)} g{generation}: work differs from the frozen budget')
            counters = done['after']['counters']
            if (counters['completed_generations'], counters['total_games'], counters['trainer_steps']) != (
                    generation, generation * budget['games_per_generation'], generation * budget['optimizer_updates']):
                problems.append(f'accounting {label(seed, arch)} g{generation}: counters differ from the frozen budget')
            if done['events']['game_finalized'] != budget['games_per_generation'] or done['events']['after_update'] != budget['optimizer_updates']:
                problems.append(f'accounting {label(seed, arch)} g{generation}: event counts differ')
            neural = result['collection']['neural_positions']
            row = dict(generation=generation, games=result['games'], updates=result['updates'], positions=result['new_positions'],
                       neural_positions=neural, collection_seconds=timing['collection_seconds'],
                       update_seconds=timing['update_seconds'],
                       compute_seconds=timing['collection_seconds'] + timing['update_seconds'],
                       collection_cpu_over_wall=timing['collection_cpu_seconds'] / timing['collection_seconds'],
                       task_cpu_over_wall=task['cpu_seconds'] / task['wall_seconds'],
                       microseconds_per_neural_position=1e6 * timing['collection_seconds'] / neural,
                       seconds_per_game=timing['collection_seconds'] / result['games'],
                       ms_per_update=1e3 * timing['update_seconds'] / result['updates'],
                       mean_game_length=result['new_positions'] / result['games'],
                       clipped_fraction=sum(bool(m['clipped']) for m in result['metrics']) / len(result['metrics']),
                       gradient_norm_mean=sum(m['gradient_norm'] for m in result['metrics']) / len(result['metrics']),
                       nonfinite_updates=sum(not all(math.isfinite(float(m[k])) for k in ('combined_loss', 'gradient_norm', 'policy_loss', 'value_mse'))
                                             for m in result['metrics']))
            for key, value in (('games', result['games']), ('updates', result['updates']), ('positions', result['new_positions']),
                               ('neural_positions', neural), ('collection_seconds', timing['collection_seconds']),
                               ('update_seconds', timing['update_seconds']), ('compute_seconds', row['compute_seconds']),
                               ('collection_cpu_seconds', timing['collection_cpu_seconds']),
                               ('task_wall_seconds', task['wall_seconds']), ('task_cpu_seconds', task['cpu_seconds']),
                               ('generation_wall_seconds', timing['generation_wall_seconds'])):
                totals[key] += value
            rows.append(row)
        per_position = [r['microseconds_per_neural_position'] for r in rows]
        per_update = [r['ms_per_update'] for r in rows]
        arms[label(seed, arch)] = dict(
            generations=rows, totals=totals,
            games_per_generation=sorted({r['games'] for r in rows}), updates_per_generation=sorted({r['updates'] for r in rows}),
            clipped_fraction_by_generation=[r['clipped_fraction'] for r in rows],
            gradient_norm_mean_by_generation=[r['gradient_norm_mean'] for r in rows],
            clipped_fraction_all_updates=sum(r['clipped_fraction'] * r['updates'] for r in rows) / totals['updates'],
            collection_cpu_over_wall_range=[min(r['collection_cpu_over_wall'] for r in rows), max(r['collection_cpu_over_wall'] for r in rows)],
            task_cpu_over_wall_range=[min(r['task_cpu_over_wall'] for r in rows), max(r['task_cpu_over_wall'] for r in rows)],
            microseconds_per_neural_position=dict(min=min(per_position), max=max(per_position),
                                                  spread_over_mean=(max(per_position) - min(per_position)) / float(np.mean(per_position))),
            ms_per_update=dict(min=min(per_update), max=max(per_update),
                               spread_over_mean=(max(per_update) - min(per_update)) / float(np.mean(per_update))))
        if (totals['games'], totals['updates']) != (6 * budget['games_per_generation'], 6 * budget['optimizer_updates']):
            problems.append(f'accounting {label(seed, arch)}: totals differ from the frozen budget')
    bands = design['analysis']['equal_compute']
    ratios = {}
    for seed in SEEDS:
        a, b = arms[label(seed, 'resnet3x64')]['totals'], arms[label(seed, 'cnn')]['totals']
        ratios[str(seed)] = dict(compute=a['compute_seconds'] / b['compute_seconds'],
                                 collection=a['collection_seconds'] / b['collection_seconds'],
                                 update=a['update_seconds'] / b['update_seconds'],
                                 task_wall=a['task_wall_seconds'] / b['task_wall_seconds'],
                                 task_cpu=a['task_cpu_seconds'] / b['task_cpu_seconds'],
                                 neural_positions=a['neural_positions'] / b['neural_positions'],
                                 training_positions=a['positions'] / b['positions'],
                                 claim=compute_band(a['compute_seconds'] / b['compute_seconds'], bands))
    by_arch = {arch: sum(arms[label(s, arch)]['totals']['compute_seconds'] for s in SEEDS) for arch in ARCHS}
    pooled = by_arch['resnet3x64'] / by_arch['cnn']
    # equal-games pairing: ResNet through generation 6 against the CNN through generation 4 (576 games each)
    cnn_g4 = {s: sum(r['compute_seconds'] for r in arms[label(s, 'cnn')]['generations'][:4]) for s in SEEDS}
    equal_games = dict(
        per_seed={str(s): arms[label(s, 'resnet3x64')]['totals']['compute_seconds'] / cnn_g4[s] for s in SEEDS},
        pooled=by_arch['resnet3x64'] / sum(cnn_g4.values()),
        cnn_games_through_generation_4=sorted({sum(r['games'] for r in arms[label(s, 'cnn')]['generations'][:4]) for s in SEEDS}),
        cumulative_ratio_by_generation={str(s): [
            sum(r['compute_seconds'] for r in arms[label(s, 'resnet3x64')]['generations'][:g])
            / sum(r['compute_seconds'] for r in arms[label(s, 'cnn')]['generations'][:g]) for g in range(1, 7)] for s in SEEDS})
    cost = {arch: float(np.mean([r['microseconds_per_neural_position'] for s in SEEDS for r in arms[label(s, arch)]['generations']]))
            for arch in ARCHS}
    update_cost = {arch: float(np.mean([r['ms_per_update'] for s in SEEDS for r in arms[label(s, arch)]['generations']]))
                   for arch in ARCHS}

    # scheduling
    tasks = status['tasks']
    train = [t for t in tasks if t['kind'] == 'train']
    others = [t for t in tasks if t['kind'] != 'train']
    declared = [f'{label(seed, arch)}-g{g + 1}-train' for g, block in enumerate(design['order']) for seed, arch in block]
    ordered = sorted(train, key=lambda t: t['start_epoch'])
    gaps = [b['start_epoch'] - a['end_epoch'] for a, b in zip(ordered, ordered[1:])]
    last_train_end = max(t['end_epoch'] for t in train)
    events = sorted([(t['start_epoch'], 1) for t in others] + [(t['end_epoch'], -1) for t in others], key=lambda e: (e[0], e[1]))
    running = peak = 0
    for _, step in events:
        running += step
        peak = max(peak, running)
    schedule = dict(
        tasks=len(tasks), by_kind={k: sum(t['kind'] == k for t in tasks) for k in ('train', 'matches', 'positions', 'dynamics')},
        all_complete=all(t['state'] == 'COMPLETE' and t['returncode'] == 0 for t in tasks), retries=status['retries'],
        distinct_labels=len({t['label'] for t in tasks}),
        train_order_equals_declared=[t['label'] for t in ordered] == declared,
        ledger_order_equals_declared=[t['label'] for t in train] == declared,
        minimum_gap_between_consecutive_train_tasks_seconds=min(gaps), train_tasks_overlap=min(gaps) < 0,
        first_evaluation_start_minus_last_train_end_seconds=min(t['start_epoch'] for t in others) - last_train_end,
        evaluation_overlaps_training=min(t['start_epoch'] for t in others) < last_train_end,
        peak_concurrent_evaluation_tasks=peak, status=status['status'], integrity=status['integrity'],
        elapsed_seconds=status['elapsed_seconds'], deadline_margin_seconds=status['deadline_epoch'] - status['end_epoch'],
        external_commits=status['external_commits'],
        protected_trees_intact={k: v['intact'] for k, v in status['protected_trees'].items()})
    for flag, text in ((not schedule['train_order_equals_declared'], 'training order differs from the declared ABBA order'),
                       (schedule['train_tasks_overlap'], 'training tasks overlap'),
                       (schedule['evaluation_overlaps_training'], 'evaluation overlaps training'),
                       (not schedule['all_complete'], 'a task is not COMPLETE'), (schedule['retries'] != 0, 'retries recorded'),
                       (schedule['tasks'] != 60 or schedule['distinct_labels'] != 60, 'task count is not 60')):
        if flag:
            problems.append(f'schedule: {text}')
    return dict(arms=arms, ratio_resnet_over_cnn=ratios, compute_seconds_by_architecture=by_arch, pooled_compute_ratio=pooled,
                mean_of_seed_ratios=float(np.mean([r['compute'] for r in ratios.values()])), claim=compute_band(pooled, bands),
                expected_ratio_in_design=design['budget']['expected_ratio_resnet_over_cnn'],
                equal_games_compute_ratio_resnet_g6_over_cnn_g4=equal_games,
                mean_microseconds_per_neural_position=cost,
                per_neural_position_cost_ratio=cost['resnet3x64'] / cost['cnn'],
                mean_ms_per_update=update_cost, per_update_cost_ratio=update_cost['resnet3x64'] / update_cost['cnn'],
                schedule=schedule)


# ---------------------------------------------------------------------------------------------
# 7. Integrity and protocol conformance
# ---------------------------------------------------------------------------------------------

def git(*arguments):
    done = subprocess.run(['git', '--no-optional-locks', '-C', str(REPO), *arguments], capture_output=True, text=True)
    return done.stdout.strip() if done.returncode == 0 else f'ERROR: {done.stderr.strip()}'


def integrity(design, status, position_hashes, problems):
    freeze = load(RESULTS / 'freeze.json')
    launch = load(EVIDENCE / 'launch.json')
    out = {}

    # (a) local checkpoints
    checked, different, missing = 0, [], []
    for arm, generations in status['boundaries'].items():
        for generation, entry in generations.items():
            path = Path(entry['path'])
            if REPO not in path.parents:
                different.append(f'{arm} g{generation}: path outside the worktree')
            elif not path.exists():
                missing.append(f'{arm} g{generation}')
            else:
                checked += 1
                if sha(path) != entry['sha256']:
                    different.append(f'{arm} g{generation}')
    out['a_checkpoints'] = dict(hashed=checked, different=different, missing=missing)

    # (b) model chain
    chain = []
    for seed, arch in ARMS:
        arm = label(seed, arch)
        initial = freeze['initial'][arm]
        boundaries = status['boundaries'][arm]
        if (initial['sha256'], initial['weights_sha256']) != (boundaries['0']['sha256'], boundaries['0']['weights_sha256']):
            chain.append(f'{arm}: generation 0 differs from the freeze record')
        if launch['initial'][arm] != initial:
            chain.append(f'{arm}: launch record differs from the freeze record')
        if (initial['trainer_steps'], initial['replay_positions'], initial['optimizer_state_entries']) != (0, 0, 0):
            chain.append(f'{arm}: generation 0 is not an empty start')
        if initial['architecture'] != design['configurations'][arm]['architecture']:
            chain.append(f'{arm}: architecture identity')
        listed = load(TASKS / f'{arm}-boundaries.json')
        if [(e['generation'], e['sha256']) for e in listed] != [(g, boundaries[str(g)]['sha256']) for g in range(7)]:
            chain.append(f'{arm}: boundary list given to the dynamics task differs from the ledger')
        previous_weights, previous_boundary, previous_state = initial['weights_sha256'], initial['sha256'], None
        for generation in range(1, 7):
            done = load(TASKS / f'{arm}-g{generation}-train' / 'complete.json')
            result = done['result']
            checks = (
                (result['collection_weights_sha256'] == previous_weights, 'collection weights are not the previous learner weights'),
                (done['before']['model'] == previous_weights, 'loaded model is not the previous learner'),
                (done['parent_boundary_sha256'] == previous_boundary, 'parent boundary hash'),
                (result['learner_weights_sha256'] == done['after']['model'], 'learner weights hash'),
                (done['sha256'] == result['boundary_sha256'] == boundaries[str(generation)]['sha256'], 'boundary hash against the ledger'),
                (done['after']['model'] == boundaries[str(generation)]['weights_sha256'], 'weights hash against the ledger'),
                (result['learner_weights_sha256'] != previous_weights, 'weights did not change'),
                (done['seed'] == seed and done['architecture'] == arch and done['generation'] == generation, 'identity'),
                (result['schedule']['name'] == 'original' and result['protocol'] == design['configurations'][arm]['rng_protocol'],
                 'schedule or protocol'),
                (previous_state is None or all(done['before'][k] == previous_state[k] for k in
                                               ('model', 'optimizer', 'replay', 'sampling', 'augmentation', 'counters')),
                 'state before differs from the previous state after'),
                (result['evicted_generations'] == [], 'replay eviction'))
            chain.extend(f'{arm} g{generation}: {text}' for ok, text in checks if not ok)
            previous_weights, previous_boundary, previous_state = done['after']['model'], done['sha256'], done['after']
    out['b_model_chain'] = dict(links_checked=24, problems=chain)

    # (c) design copies
    out['c_design_copy'] = dict(run_evidence_sha256=sha(EVIDENCE / 'design.json'), results_sha256=sha(RESULTS / 'design.json'),
                                byte_identical=(EVIDENCE / 'design.json').read_bytes() == (RESULTS / 'design.json').read_bytes(),
                                equals_freeze_record=sha(RESULTS / 'design.json') == freeze['files'][
                                    'research/alphazero_resnet_selfplay/results/design.json'])

    # (d) frozen sources
    changed = sorted(name for name, digest in freeze['files'].items()
                     if not (REPO / name).exists() or sha(REPO / name) != digest)
    out['d_frozen_sources'] = dict(files=len(freeze['files']), changed=changed)

    # (e) git: design commit before the run, and on origin
    commit_epoch = git('log', '-1', '--format=%ct', DESIGN_COMMIT)
    reflog = git('reflog', 'show', '--date=unix', 'origin/phase4d4h-c-resnet-selfplay')
    tracked_changes = git('diff', '--name-only', DESIGN_COMMIT, '--')
    frozen_in_commit = [name for name in freeze['files']
                        if git('rev-parse', f'{DESIGN_COMMIT}:{name}').startswith('ERROR')]
    out['e_git'] = dict(
        design_commit=git('rev-parse', DESIGN_COMMIT), head=git('rev-parse', 'HEAD'),
        status_design_commit=status['design_commit'], launch_design_commit=launch['design_commit'],
        commit_epoch=int(commit_epoch) if commit_epoch.isdigit() else commit_epoch, run_start_epoch=launch['start_epoch'],
        seconds_from_commit_to_run_start=(launch['start_epoch'] - int(commit_epoch)) if commit_epoch.isdigit() else None,
        remote_branches_containing_commit=git('branch', '-r', '--contains', DESIGN_COMMIT).split(),
        origin_reflog=reflog.splitlines(),
        tracked_files_changed_since_design_commit=tracked_changes.splitlines(),
        frozen_files_absent_from_design_commit=frozen_in_commit,
        commit_files=git('show', '--stat', '--format=', DESIGN_COMMIT).splitlines()[-1:])

    # (g) evaluation identities
    identity_problems, identities = [], 0
    for seed, arch in ARMS:
        arm = label(seed, arch)
        for generation in EVAL_GENERATIONS:
            expected = status['boundaries'][arm][str(generation)]
            for kind in ('matches', 'positions'):
                identity = load(TASKS / f'{arm}-g{generation}-{kind}' / 'identity.json')
                identities += 1
                budget = design['arms'][arch]
                if ((identity['boundary_sha256'], identity['weights_sha256']) != (expected['sha256'], expected['weights_sha256'])
                        or (identity['seed'], identity['architecture'], identity['generation']) != (seed, arch, generation)
                        or identity['counters']['total_games'] != generation * budget['games_per_generation']
                        or identity['counters']['trainer_steps'] != generation * budget['optimizer_updates']):
                    identity_problems.append(f'{arm}-g{generation}-{kind}')
        for generation in range(7):
            expected = status['boundaries'][arm][str(generation)]
            seen = position_hashes[(seed, arch, generation)]
            identities += 1
            if (seen['boundary'], seen['weights']) != (expected['sha256'], expected['weights_sha256']):
                identity_problems.append(f'{arm}-dynamics-g{generation}')
    out['g_evaluation_identities'] = dict(checked=identities, problems=identity_problems)

    # run-evidence copies against the local originals the frozen analysis read
    copied, differing, absent = 0, [], []
    for path in sorted(EVIDENCE.rglob('*')):
        if path.is_file():
            original = LOCAL_RUN / path.relative_to(EVIDENCE)
            if not original.exists():
                absent.append(str(path.relative_to(EVIDENCE)))
            else:
                copied += 1
                if sha(original) != sha(path):
                    differing.append(str(path.relative_to(EVIDENCE)))
    out['run_evidence_copies'] = dict(files_compared=copied, differing=differing, without_local_original=absent)
    manifest = load(RESULTS / 'evidence-manifest.json')
    manifest_differing = sorted(name for name, digest in manifest['files'].items()
                                if not (ROOT / name).exists() or sha(ROOT / name) != digest)
    manifest_boundaries = sorted(
        name for name, entry in manifest['boundaries'].items()
        if (entry['sha256'], entry['weights_sha256']) != tuple(
            status['boundaries'][name.split('/')[0]][str(int(name.rsplit('-', 1)[1]))][k] for k in ('sha256', 'weights_sha256')))
    logs = sorted((EVIDENCE / 'logs').glob('*.log'))
    out['evidence_manifest'] = dict(files=len(manifest['files']), differing=manifest_differing,
                                    boundaries=len(manifest['boundaries']), boundaries_differing=manifest_boundaries,
                                    task_logs=len(logs), task_logs_empty=sum(p.stat().st_size == 0 for p in logs))
    if manifest_differing or manifest_boundaries:
        problems.append(f'integrity: evidence manifest: {(manifest_differing + manifest_boundaries)[:5]}')

    # (h) sealed packages: text scan of this phase's sources and the fixtures' declared sources
    mentions = {}
    for path in sorted(ROOT.glob('*.py')):
        if path.name == 'independent_review.py':
            continue
        lines = [f'{n}: {line.strip()}' for n, line in enumerate(path.read_text().splitlines(), 1) if 'sealed' in line.lower()]
        if lines:
            mentions[path.name] = lines
    out['h_sealed_references'] = dict(lines_mentioning_sealed_in_phase_sources=mentions,
                                      frozen_package_paths_in_phase_sources=sorted({
                                          token.strip('\'"(),') for path in ROOT.glob('*.py') if path.name != 'independent_review.py'
                                          for token in path.read_text().split() if 'alphazero_v2/frozen/' in token}))
    for key, value in (('checkpoint hashes', different + missing), ('model chain', chain), ('frozen sources', changed),
                       ('evaluation identities', identity_problems), ('run-evidence copies', differing)):
        if value:
            problems.append(f'integrity: {key}: {value[:5]}')
    if not out['c_design_copy']['byte_identical']:
        problems.append('integrity: design copies differ')
    return out


# ---------------------------------------------------------------------------------------------
# Re-execution of the generation-6 endpoints from the hashed local checkpoints
# ---------------------------------------------------------------------------------------------

def reexecute(design, status, fixtures, problems):
    """Load each generation-6 boundary strictly (hash and declared configuration), then re-run, with the
    project's frozen search and arena code but none of this phase's evaluation harness:
      - the raw network on solved300 and tactical200 (policy and value against the stored rows),
      - raw PUCT at 256 simulations on tactical200 (action and visits against the stored rows),
      - all 80 match games against reference negamax depth 3 (move lists against the stored records).
    Stream names and seeds are the declared ones (design.json seeds; namespaces from the frozen evaluate.py)."""
    sys.path.insert(0, str(REPO))
    from games.connect4.alphazero_v2.provenance import configure_deterministic_runtime
    configure_deterministic_runtime(1)
    from games.connect4.alphazero_v2.arena import NegamaxOpponent, paired_game
    from games.connect4.alphazero_v2.config import V2Config
    from games.connect4.alphazero_v2.data import seeded_rng
    from games.connect4.alphazero_v2.experimental import ExperimentalConfig, load_experimental
    from games.connect4.alphazero_v2.oracle import engine_position
    from games.connect4.alphazero_v2.search import EvaluationAgent

    plan = design['evaluation']

    class Agent:
        def __init__(self, inference, name):
            self.inference, self.name = inference, name

        def choose(self, game, rng):
            result, selection = EvaluationAgent(self.inference, plan['simulations'], rng=rng, backend='bitboard').search(game)
            return selection.move, {}

    package = load(REPO / 'games/connect4/alphazero_v2/frozen/openings-development.json')
    openings = [r for r in package['rows'] if r['id'] in set(plan['openings'])]
    report = {}
    for seed, arch in ARMS:
        arm, started = label(seed, arch), time.perf_counter()
        boundary = status['boundaries'][arm]['6']
        budget = design['arms'][arch]
        config = ExperimentalConfig(
            V2Config(seed=seed, games_per_generation=budget['games_per_generation'], max_generations=design['generations']),
            optimizer_updates=budget['optimizer_updates'], architecture=design['configurations'][arm]['architecture'])
        runner = load_experimental(boundary['path'], expected_sha256=boundary['sha256'], expected_config=config,
                                   restore_global_rng=False)
        model = runner.factory.freeze(runner.model)
        inference = runner.factory.inference(model)
        entry = dict(weights_hash_matches=runner.factory.hash(model) == boundary['weights_sha256'],
                     completed_generations=runner.completed_generations)
        stored = load(TASKS / f'{arm}-dynamics' / 'dynamics.json')['rows'][6]['raw']
        gap = 0.0
        for suite in ('solved', 'tactical'):
            for row, old in zip(fixtures[suite], stored[suite]):
                prediction = inference.predict(engine_position(row['moves']))
                gap = max(gap, abs(prediction.value - old['value']),
                          max(abs(x - y) for x, y in zip(prediction.policy, old['policy'])))
        entry['raw_outputs_compared'] = len(fixtures['solved']) + len(fixtures['tactical'])
        entry['max_abs_raw_output_difference'] = gap
        stored = load(TASKS / f'{arm}-g6-positions' / 'tactical.json')['rows']
        different = 0
        for row, old in zip(fixtures['tactical'], stored):
            rng = seeded_rng(f'phase4d4h-c-evaluation:tactical:{row["id"]}', plan['search_seed'])
            result, selection = EvaluationAgent(inference, plan['simulations'], rng=rng, backend='bitboard').search(
                engine_position(row['moves']))
            different += selection.move != old['puct_action'] or list(result.visits) != old['visits']
        entry['tactical_puct_rows_compared'] = len(stored)
        entry['tactical_puct_rows_different'] = different
        stored = {(r['opening_id'], r['game_index']): r for r in load(TASKS / f'{arm}-g6-matches' / 'matches.json')['records']}
        agent, opponent, mismatched, score = Agent(inference, f'{arm}-g6'), NegamaxOpponent(plan['opponent_depth']), [], 0.0
        for row in openings:
            for game_index in (0, 1):
                record = paired_game(agent, opponent, row, game_index, namespace='phase4d4h-c-strength', seed=plan['match_seed'],
                                     check=None, keep_decisions=False)
                old = stored[(row['id'], game_index)]
                if record['moves'] != old['moves'] or record['result'] != old['result']:
                    mismatched.append(f'{row["id"]}:{game_index}')
                score += {'win': 1.0, 'draw': 0.5, 'loss': 0.0}[record['result']]
        entry.update(match_games_replayed=2 * len(openings), match_games_different=mismatched, replayed_match_score=score / (2 * len(openings)),
                     weights_unchanged_after=runner.factory.hash(model) == boundary['weights_sha256'],
                     seconds=time.perf_counter() - started)
        report[arm] = entry
        if (not entry['weights_hash_matches'] or mismatched or different or gap > 1e-6):
            problems.append(f're-execution {arm}: stored evidence differs from a fresh run of the hashed checkpoint')
    return report


# ---------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------

def main():
    started = time.perf_counter()
    design = load(RESULTS / 'design.json')
    status = load(EVIDENCE / 'status.json')
    analysis = load(RESULTS / 'analysis.json')
    fixtures = load(ROOT / 'fixtures.json')
    config = design['analysis']
    problems, compare = [], Comparison()
    out = dict(reviewer='independent results reviewer; same host; not an external replication',
               bootstrap=dict(replicates=REPLICATES, reviewer_seed=REVIEW_SEED,
                              note='streams are derived from the reviewer seed and differ from the frozen analysis'))

    # ---- 1. matches
    games, arm_matches, match_facts = check_matches(design, status, problems)
    out['matches'] = dict(facts=match_facts, arms=arm_matches)
    for seed, arch in ARMS:
        for generation in EVAL_GENERATIONS:
            mine = arm_matches[f'{label(seed, arch)}-g{generation}']
            theirs = analysis['trajectories'][label(seed, arch)][generation]['match']
            for key in ('score', 'wins', 'draws', 'losses', 'score_as_first', 'score_as_second', 'mean_plies', 'games'):
                compare.add('match_summaries', f'{label(seed, arch)}-g{generation}:{key}', mine[key], theirs[key])

    # ---- 2. match contrasts
    def match_contrast(cnn_generation, resnet_generation, tag):
        return cluster_contrast({s: match_rows(games, s, cnn_generation, resnet_generation) for s in SEEDS},
                                stream(REVIEW_SEED, 'match', tag))
    strength = {}
    for generation in EVAL_GENERATIONS:
        strength[f'equal_compute_g{generation}'] = match_contrast(generation, generation, f'g{generation}')
    strength['equal_games_cnn_g4_resnet_g6'] = match_contrast(4, 6, 'cnn4-resnet6')
    interval_gaps = {}
    for name, theirs in ([(f'equal_compute_g{g}', analysis['equal_compute_contrasts'][str(g)]['match_score']) for g in EVAL_GENERATIONS]
                         + [('equal_games_cnn_g4_resnet_g6', analysis['equal_games_contrasts']['cnn-g4-resnet-g6']['match_score'])]):
        mine = strength[name]
        for s in SEEDS:
            compare.add('match_contrast_points', f'{name}:{s}', mine['per_seed'][str(s)]['difference'], theirs['per_seed'][str(s)]['difference'])
        compare.add('match_contrast_points', f'{name}:pooled', mine['pooled']['difference'], theirs['pooled']['difference'])
        interval_gaps[name] = dict(
            reviewer=mine['pooled']['interval95'], frozen=theirs['pooled']['interval95'],
            end_differences=[mine['pooled']['interval95'][i] - theirs['pooled']['interval95'][i] for i in (0, 1)],
            per_seed_end_differences={str(s): [mine['per_seed'][str(s)]['interval95'][i] - theirs['per_seed'][str(s)]['interval95'][i]
                                               for i in (0, 1)] for s in SEEDS})
        mine['supporting_rule'] = supporting_rule(mine, True)
        mine['frozen_supporting_rule'] = theirs['verdict']
    family_table = {str(s): family_differences(games, s, 6, 6) for s in SEEDS}
    theirs_family = analysis['equal_compute_contrasts']['6']['match_score']['by_opening_family']
    for s in SEEDS:
        for f, value in family_table[str(s)].items():
            compare.add('match_family_differences', f'{s}:{f}', value, theirs_family[str(s)][f]['difference'])
    final, middle = strength['equal_compute_g6'], strength['equal_compute_g4']
    mean_final = float(np.mean([arm_matches[f'{label(s, a)}-g6']['score'] for s, a in ARMS]))
    strength_verdict = strength_rule(final, middle['pooled']['difference'], mean_final, config['strength'])
    rule_by_generation = {}
    for generation in EVAL_GENERATIONS:
        scores = float(np.mean([arm_matches[f'{label(s, a)}-g{generation}']['score'] for s, a in ARMS]))
        rule_by_generation[f'g{generation}'] = dict(
            mean_score_of_four_arms=scores,
            fail_conditions_met=(all(v['difference'] < 0 for v in strength[f'equal_compute_g{generation}']['per_seed'].values())
                                 and strength[f'equal_compute_g{generation}']['pooled']['interval95'][1] < 0),
            inside_saturation_band=config['strength']['saturation'][0] <= scores <= config['strength']['saturation'][1])
    gain = {}
    for s in SEEDS:
        a = {f: v for f, v in family_differences(games, s, 6, 6).items()}
        b = {f: v for f, v in family_differences(games, s, 0, 0).items()}
        dd = np.array([a[f] - b[f] for f in sorted(a)])
        boot = dd[stream(REVIEW_SEED, 'gain', s).integers(0, 40, size=(REPLICATES, 40))].mean(1)
        gain[str(s)] = dict(resnet_gain=arm_matches[f'{label(s, "resnet3x64")}-g6']['score'] - arm_matches[f'{label(s, "resnet3x64")}-g0']['score'],
                            cnn_gain=arm_matches[f'{label(s, "cnn")}-g6']['score'] - arm_matches[f'{label(s, "cnn")}-g0']['score'],
                            difference_in_gain=float(dd.mean()), interval95=[float(x) for x in np.quantile(boot, [0.025, 0.975])])
    out['strength'] = dict(contrasts=strength, interval_end_differences_from_frozen_analysis=interval_gaps,
                           family_differences_g6=family_table, mean_final_score=mean_final,
                           verdict=strength_verdict, frozen_verdict=analysis['verdicts']['strength']['verdict'],
                           declared_fail_conditions_applied_at_each_evaluated_generation=rule_by_generation,
                           gain_from_generation_0_to_6=dict(per_seed=gain, pooled_difference_in_gain=float(np.mean(
                               [g['difference_in_gain'] for g in gain.values()])),
                               note='not prespecified; descriptive'))
    compare.add('verdict_inputs', 'mean_final_score', mean_final, analysis['verdicts']['strength']['mean_final_score'])

    # ---- 3. robustness
    out['strength_robustness'] = robustness(games, arm_matches, design, final, middle)

    # ---- 4/5. fixtures and position suites
    out['fixtures'] = audit_fixtures(fixtures, problems)
    raw, puct, position_hashes, counters = load_positions(fixtures, problems)
    out['position_rows'] = counters
    trajectories = {}
    for seed, arch in ARMS:
        rows = []
        for generation in range(7):
            entry = dict(generation=generation, raw={m: metric_mean(raw[(seed, arch, generation)], s, k, c)
                                                     for m, s, k, c, _ in RAW_METRICS})
            theirs = analysis['trajectories'][label(seed, arch)][generation]
            for m, value in entry['raw'].items():
                compare.add('raw_network_means', f'{label(seed, arch)}-g{generation}:{m}', value, theirs['raw'][m])
            if generation in EVAL_GENERATIONS:
                entry['puct'] = {m: metric_mean(puct[(seed, arch, generation)], s, k, c) for m, s, k, c, _ in PUCT_METRICS}
                for m, value in entry['puct'].items():
                    compare.add('raw_puct_means', f'{label(seed, arch)}-g{generation}:{m}', value, theirs['puct'][m])
            rows.append(entry)
        trajectories[label(seed, arch)] = rows
    out['trajectories'] = trajectories

    def position_contrasts(cnn_generation, resnet_generation, tag, theirs, with_puct):
        result, disagreements = {}, []
        sources = [(RAW_METRICS, raw)] + ([(PUCT_METRICS, puct)] if with_puct else [])
        for metrics, table in sources:
            for metric, suite, key, category, better in metrics:
                rows = {s: metric_rows(table[(s, 'resnet3x64', resnet_generation)], table[(s, 'cnn', cnn_generation)],
                                       suite, key, category) for s in SEEDS}
                mine = cluster_contrast(rows, stream(REVIEW_SEED, tag, metric))
                mine['supporting_rule'] = supporting_rule(mine, better)
                mine['frozen_supporting_rule'] = theirs[metric]['verdict']
                for s in SEEDS:
                    compare.add('position_contrast_points', f'{tag}:{metric}:{s}', mine['per_seed'][str(s)]['difference'],
                                theirs[metric]['per_seed'][str(s)]['difference'])
                    if mine['per_seed'][str(s)]['clusters'] != theirs[metric]['per_seed'][str(s)]['clusters']:
                        disagreements.append(f'{metric}: cluster count')
                compare.add('position_contrast_points', f'{tag}:{metric}:pooled', mine['pooled']['difference'],
                            theirs[metric]['pooled']['difference'])
                for i in (0, 1):
                    compare.add('position_contrast_interval_ends_other_stream', f'{tag}:{metric}:pooled:{i}',
                                mine['pooled']['interval95'][i], theirs[metric]['pooled']['interval95'][i])
                if mine['supporting_rule'] != mine['frozen_supporting_rule']:
                    disagreements.append(f'{metric}: reviewer {mine["supporting_rule"]} vs frozen {mine["frozen_supporting_rule"]} '
                                         f'(reviewer {mine["pooled"]["interval95"]}, frozen {theirs[metric]["pooled"]["interval95"]})')
                result[metric] = mine
        return result, disagreements
    contrasts, rule_disagreements = {}, {}
    for generation in range(7):
        tag = f'equal_compute_g{generation}'
        contrasts[tag], rule_disagreements[tag] = position_contrasts(
            generation, generation, tag, analysis['equal_compute_contrasts'][str(generation)], generation in EVAL_GENERATIONS)
    contrasts['equal_games_cnn_g4_resnet_g6'], rule_disagreements['equal_games_cnn_g4_resnet_g6'] = position_contrasts(
        4, 6, 'equal_games_cnn_g4_resnet_g6', analysis['equal_games_contrasts']['cnn-g4-resnet-g6'], True)
    contrasts['equal_games_cnn_g2_resnet_g3'], rule_disagreements['equal_games_cnn_g2_resnet_g3'] = position_contrasts(
        2, 3, 'equal_games_cnn_g2_resnet_g3', analysis['equal_games_contrasts']['cnn-g2-resnet-g3'], False)
    out['position_contrasts'] = contrasts
    out['supporting_rule_differences_between_bootstrap_streams'] = {k: v for k, v in rule_disagreements.items() if v}
    tally = {}
    for tag, table in contrasts.items():
        counts = {}
        for metric, result in table.items():
            counts.setdefault(result['supporting_rule'], []).append(metric)
        if tag in ('equal_compute_g0', 'equal_compute_g2', 'equal_compute_g4', 'equal_compute_g6', 'equal_games_cnn_g4_resnet_g6'):
            counts.setdefault(strength[tag]['supporting_rule'], []).append('match_score')
        tally[tag] = dict(contrasts=sum(len(v) for v in counts.values()), better=sorted(counts.get('BETTER', [])),
                          worse=sorted(counts.get('WORSE', [])))
    out['supporting_rule_tally'] = dict(
        by_pairing=tally, total_contrasts=sum(t['contrasts'] for t in tally.values()),
        total_better=sum(len(t['better']) for t in tally.values()), total_worse=sum(len(t['worse']) for t in tally.values()),
        note='generation 0 compares two untrained networks; any BETTER or WORSE there is not a learning effect')

    decisive = contrasts['equal_compute_g6']
    tactical = {}
    for method in ('puct', 'policy'):
        names = [f'tactical_{method}_immediate_win', f'tactical_{method}_safe_response']
        verdict = tactical_rule([decisive[n] for n in names], config['tactical'])
        counts = {}
        for k in range(50):
            trial = []
            for n, category in zip(names, ('unique_win', 'unique_safe')):
                table = puct if method == 'puct' else raw
                rows = {s: metric_rows(table[(s, 'resnet3x64', 6)], table[(s, 'cnn', 6)], 'tactical', f'{method}_correct', category)
                        for s in SEEDS}
                trial.append(cluster_contrast(rows, stream('tactical-robustness', method, n, k)))
            v = tactical_rule(trial, config['tactical'])
            counts[v] = counts.get(v, 0) + 1
        tactical[method] = dict(verdict=verdict, frozen_verdict=analysis['verdicts'][f'tactical_{method}']['verdict'],
                                binding=method == 'puct', categories={n: decisive[n] for n in names},
                                verdict_counts_over_50_streams=counts)
    accuracy_table = {}
    for seed, arch in ARMS:
        accuracy_table[label(seed, arch)] = {
            f'g{g}': dict(puct_immediate_win=trajectories[label(seed, arch)][g]['puct']['tactical_puct_immediate_win'],
                          puct_safe_response=trajectories[label(seed, arch)][g]['puct']['tactical_puct_safe_response'],
                          policy_immediate_win=trajectories[label(seed, arch)][g]['raw']['tactical_policy_immediate_win'],
                          policy_safe_response=trajectories[label(seed, arch)][g]['raw']['tactical_policy_safe_response'],
                          puct_wrong_rows=sum(1 - r['puct_correct'] for r in puct[(seed, arch, g)]['tactical'].values()))
            for g in EVAL_GENERATIONS}
    # generation-0 oddity: what do the untrained networks output, and how decisive is the search on tactical rows?
    untrained = {}
    for seed, arch in ARMS:
        rows = load(TASKS / f'{label(seed, arch)}-g0-positions' / 'tactical.json')['rows']
        labels = {r['id']: r for r in fixtures['tactical']}
        deviation, values, share = [], [], []
        for r in rows:
            legal = labels[r['id']]['legal']
            deviation.append(max(abs(r['policy'][a] - 1 / len(legal)) for a in legal))
            values.append(r['value'])
            share.append(r['visits'][labels[r['id']]['expected_action']] / 256)
        trained = load(TASKS / f'{label(seed, arch)}-g6-positions' / 'tactical.json')['rows']
        wrong = [r for r in trained if r['puct_action'] != labels[r['id']]['expected_action']]
        untrained[label(seed, arch)] = dict(
            g0_max_policy_deviation_from_uniform_over_legal=max(deviation), g0_mean_policy_deviation=float(np.mean(deviation)),
            g0_value_mean=float(np.mean(values)), g0_value_standard_deviation=float(np.std(values)),
            g0_value_min=min(values), g0_value_max=max(values),
            g0_mean_visit_share_on_expected_action=float(np.mean(share)), g0_min_visit_share_on_expected_action=min(share),
            g6_wrong_puct_rows=len(wrong),
            g6_wrong_rows_mean_prior_on_expected_action=float(np.mean([r['policy'][labels[r['id']]['expected_action']] for r in wrong])) if wrong else None,
            g6_wrong_rows_mean_abs_raw_value=float(np.mean([abs(r['value']) for r in wrong])) if wrong else None,
            g6_wrong_rows_by_category={c: sum(labels[r['id']]['category'] == c for r in wrong) for c in ('unique_win', 'unique_safe')},
            g6_wrong_row_ids=[r['id'] for r in wrong])
    out['tactical'] = dict(gate=tactical, accuracy_by_arm_and_generation=accuracy_table, generation_0_and_error_rows=untrained)
    for n in ('tactical_puct_immediate_win', 'tactical_puct_safe_response', 'tactical_policy_immediate_win', 'tactical_policy_safe_response'):
        method = 'puct' if '_puct_' in n else 'policy'
        compare.add('verdict_inputs', n, decisive[n]['pooled']['difference'],
                    analysis['verdicts'][f'tactical_{method}']['categories'][n]['difference'])

    value = decisive['solved_exact_value_mse']
    value_counts = {}
    for k in range(50):
        rows = {s: metric_rows(raw[(s, 'resnet3x64', 6)], raw[(s, 'cnn', 6)], 'solved', 'se', None) for s in SEEDS}
        v = supporting_rule(cluster_contrast(rows, stream('value-robustness', k)), False)
        value_counts[v] = value_counts.get(v, 0) + 1
    zero = out['fixtures']['solved']['constant_zero_value_mse']
    mse = {label(s, a): [trajectories[label(s, a)][g]['raw']['solved_exact_value_mse'] for g in range(7)] for s, a in ARMS}
    solved_values = {}
    for seed, arch in ARMS:
        rows = raw[(seed, arch, 6)]['solved']
        labels = {r['id']: r['value'] for r in fixtures['solved']}
        v = np.array([rows[i]['value'] for i in sorted(rows)])
        t = np.array([labels[i] for i in sorted(rows)])
        solved_values[label(seed, arch)] = dict(
            mse=float(((v - t) ** 2).mean()), mean_abs_value=float(np.abs(v).mean()),
            share_abs_value_above_0_95=float((np.abs(v) > 0.95).mean()), correlation_with_exact=float(np.corrcoef(v, t)[0, 1]),
            sign_agreement_on_decided_positions=float((np.sign(v[t != 0]) == t[t != 0]).mean()),
            mean_value=float(v.mean()), mean_value_by_exact_value={str(x): float(v[t == x].mean()) for x in (-1, 0, 1)},
            mse_by_exact_value={str(x): float(((v[t == x] - x) ** 2).mean()) for x in (-1, 0, 1)})
    out['value'] = dict(verdict=supporting_rule(value, False), frozen_verdict=analysis['verdicts']['value']['verdict'],
                        contrast=value, verdict_counts_over_50_streams=value_counts, constant_zero_baseline=zero,
                        solved_exact_value_mse_by_generation=mse,
                        generation_6_mse_over_constant_zero={k: v[6] / zero for k, v in mse.items()},
                        indicator_generation_6_mse_at_or_above_0_9_constant_zero_in_both_architectures=all(
                            v[6] >= 0.9 * zero for v in mse.values()),
                        generation_6_value_description=solved_values)
    compare.add('verdict_inputs', 'solved_exact_value_mse', value['pooled']['difference'],
                analysis['verdicts']['value']['pooled']['difference'])
    for seed, arch in ARMS:
        for generation in range(7):
            compare.add('constant_zero_baseline', f'{label(seed, arch)}-g{generation}', zero,
                        analysis['trajectories'][label(seed, arch)][generation]['raw']['solved_constant_zero_value_mse'])

    # ---- 6. computation and scheduling
    compute = accounting(design, status, problems)
    out['computation'] = compute
    theirs = analysis['computation']
    compare.add('computation', 'pooled_ratio', compute['pooled_compute_ratio'], theirs['pooled_compute_ratio'])
    for s in SEEDS:
        for key in ('compute', 'collection', 'update', 'task_wall'):
            compare.add('computation', f'{s}:{key}', compute['ratio_resnet_over_cnn'][str(s)][key], theirs['ratio_resnet_over_cnn'][str(s)][key])
    for seed, arch in ARMS:
        for key in ('games', 'updates', 'positions', 'collection_seconds', 'update_seconds', 'compute_seconds', 'neural_positions'):
            compare.add('computation', f'{label(seed, arch)}:{key}', compute['arms'][label(seed, arch)]['totals'][key],
                        theirs['arms'][label(seed, arch)]['totals'][key])

    # ---- 7. integrity
    out['integrity'] = integrity(design, status, position_hashes, problems)

    # ---- re-execution of the generation-6 endpoints from the hashed checkpoints (several CPU minutes)
    target = RESULTS / 'independent-recomputed.json'
    if '--no-networks' in sys.argv:
        out['reexecution_from_checkpoints'] = dict(skipped='--no-networks')
    elif '--reuse-reexecution' in sys.argv:  # keep the section an earlier full run of this script wrote
        out['reexecution_from_checkpoints'] = dict(load(target)['reexecution_from_checkpoints'],
                                                   reused_from_an_earlier_full_run_of_this_script=True)
    else:
        try:
            out['reexecution_from_checkpoints'] = reexecute(design, status, fixtures, problems)
        except Exception as error:
            out['reexecution_from_checkpoints'] = dict(error=repr(error))
            problems.append(f're-execution could not run: {error!r}')

    # ---- verdicts under the declared rules
    claim = compute['claim']
    out['verdicts'] = dict(
        strength=strength_verdict, tactical_puct=tactical['puct']['verdict'], tactical_policy_not_binding=tactical['policy']['verdict'],
        value=out['value']['verdict'], equal_compute_claim=claim,
        recommendation_rule=recommendation_rule(strength_verdict, tactical['puct']['verdict'], claim),
        frozen=dict(strength=analysis['verdicts']['strength']['verdict'], tactical_puct=analysis['verdicts']['tactical_puct']['verdict'],
                    tactical_policy=analysis['verdicts']['tactical_policy']['verdict'], value=analysis['verdicts']['value']['verdict'],
                    equal_compute_claim=analysis['verdicts']['equal_compute_claim']['claim'],
                    recommendation_rule=analysis['verdicts']['recommendation_rule']))
    out['verdicts']['all_agree_with_frozen_analysis'] = all(
        out['verdicts'][k] == out['verdicts']['frozen'][f] for k, f in (
            ('strength', 'strength'), ('tactical_puct', 'tactical_puct'), ('tactical_policy_not_binding', 'tactical_policy'),
            ('value', 'value'), ('equal_compute_claim', 'equal_compute_claim'), ('recommendation_rule', 'recommendation_rule')))
    out['comparison_with_analysis_json'] = compare.groups
    out['problems_found_by_automatic_checks'] = problems
    out['seconds'] = time.perf_counter() - started
    target.write_text(json.dumps(clean(out), indent=1, sort_keys=True, allow_nan=False) + '\n')

    print('verdicts', json.dumps(out['verdicts'], indent=1))
    print('comparison', json.dumps(clean(compare.groups), indent=1))
    print('problems', json.dumps(problems, indent=1))
    print('wrote', target, 'in', round(out['seconds'], 1), 's')


if __name__ == '__main__':
    main()
