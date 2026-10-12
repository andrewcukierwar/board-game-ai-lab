"""Replacement clean review: independent board/arithmetic audit and bounded reproduction.

No sealed path, old review import, training, or fixture selection occurs here.
Arithmetic is model-free. Reproduction is an explicit postlaunch operation.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RESULTS = HERE / 'results'
ARMS = ('42-cnn', '42-resnet3x64', '314159-cnn', '314159-resnet3x64')
CONTRASTS = {
    'zero_value_learned_prior': {'B': 1, 'A': -1},
    'zero_value_uniform_prior': {'D': 1, 'C': -1},
    'uniform_prior_learned_value': {'C': 1, 'A': -1},
    'uniform_prior_zero_value': {'D': 1, 'B': -1},
    'interaction': {'A': 1, 'B': -1, 'C': -1, 'D': 1},
    'half_value': {'E': 1, 'A': -1},
}


def read(path):
    return json.loads(Path(path).read_text())


def save(name, value):
    from .common import write
    write(RESULTS / name, value)


def winning_player(board):
    for row in range(6):
        for column in range(7):
            owner = board[row][column]
            if owner < 0:
                continue
            for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
                if 0 <= row + 3 * dy < 6 and 0 <= column + 3 * dx < 7:
                    if all(board[row + n * dy][column + n * dx] == owner for n in (1, 2, 3)):
                        return owner
    return None


def legal_columns(board):
    return [c for c in range(7) if board[5][c] == -1]


def played(board, column, actor):
    if type(column) is not int or column not in legal_columns(board):
        raise ValueError('Independent board rejected move')
    result = [row.copy() for row in board]
    height = next(r for r in range(6) if result[r][column] == -1)
    result[height][column] = actor
    return result


def board_from_history(history, terminal=False):
    board = [[-1] * 7 for _ in range(6)]
    for index, action in enumerate(history):
        assert winning_player(board) is None and legal_columns(board)
        board = played(board, action, index % 2)
    assert (winning_player(board) is not None or not legal_columns(board)) == terminal
    return board, len(history) % 2


def fixture_audit(fixtures):
    answer = {}
    for suite, count in (('tactical', 200), ('solved', 300), ('deep', 128)):
        rows = fixtures[suite]
        assert len(rows) == count and len({r['id'] for r in rows}) == count
        for row in rows:
            board, actor = board_from_history(row['moves'])
            legal = legal_columns(board)
            if 'actor' in row:
                assert row['actor'] == actor
            wins = [c for c in legal if winning_player(played(board, c, actor)) == actor]
            if suite == 'tactical':
                safe = []
                for action in legal:
                    child = played(board, action, actor)
                    replies_win = any(winning_player(played(child, reply, 1 - actor)) == 1 - actor
                                      for reply in legal_columns(child))
                    if winning_player(child) == actor or not replies_win:
                        safe.append(action)
                assert row['legal'] == legal and row['immediate_wins'] == wins
                assert row['safe_actions'] == safe
                assert row['opponent_threats'] == [c for c in legal if winning_player(played(board, c, 1 - actor)) == 1 - actor]
                if row['category'] == 'unique_win':
                    assert len(wins) == 1 and row['expected_action'] == wins[0]
                else:
                    assert row['category'] == 'unique_safe' and not wins and len(safe) == 1
                    assert row['expected_action'] == safe[0]
            else:
                labels = {int(c): v for c, v in row['action_values'].items()}
                assert set(labels) == set(legal) and set(labels.values()) <= {-1, 0, 1}
                assert max(labels.values()) == row['value']
                assert set(row['optimal_actions']) == {c for c in legal if labels[c] == row['value']}
                assert all(labels[c] == 1 for c in wins)
        answer[suite] = dict(rows=count, clusters=len({r['cluster'] for r in rows}),
                            composition=dict(Counter(str(r.get('value', r.get('category'))) for r in rows)))
    return answer


def independent_interval(rows, observations):
    assert len(rows) == len(observations) and rows
    grouped = defaultdict(lambda: [0., 0])
    for row, observation in zip(rows, observations):
        grouped[row['cluster']][0] += observation
        grouped[row['cluster']][1] += 1
    totals = np.asarray([grouped[k] for k in sorted(grouped)], dtype=float)
    # Multinomial cluster multiplicities independently implement sampling with replacement.
    multiplicities = np.random.default_rng(20261012).multinomial(
        len(totals), np.full(len(totals), 1 / len(totals)), size=10000)
    distribution = (multiplicities @ totals[:, 0]) / (multiplicities @ totals[:, 1])
    return dict(mean=float(sum(observations) / len(observations)), n=len(rows), clusters=len(totals),
                interval95=np.quantile(distribution, [.025, .975]).tolist())


def expected_jobs():
    return {f'{a}-g6-{v}-{kind}' for a in ARMS for v in 'ABCDE' for kind in ('positions', 'matches')} | {
        f'{a}-g0-A-positions' for a in ARMS} | {f'{a}-values' for a in ARMS}


def require_complete():
    status = read(RESULTS / 'status.json')
    assert status['status'] == 'COMPLETE' and not status['failed']
    assert len(status['completed']) == 48 and set(status['completed']) == expected_jobs()
    assert status['preservation']['intact']
    for label in expected_jobs():
        assert read(RESULTS / 'tasks' / label / 'complete.json')['complete']


def position_audit(rows, labels, variant):
    assert len(rows) == len(labels) and len({r['id'] for r in rows}) == len(rows)
    assert {r['id'] for r in rows} == set(labels)
    for row in rows:
        fixture = labels[row['id']]
        board, _ = board_from_history(fixture['moves'])
        legal = legal_columns(board)
        action = row['action']
        assert row['cluster'] == fixture['cluster'] and action in legal
        assert len(row['visits']) == 7 and all(type(n) is int and n >= 0 for n in row['visits'])
        assert sum(row['visits']) == 256 and row['visits'][action] == max(row['visits'])
        assert len(row['root_prior']) == 7 and abs(sum(row['root_prior']) - 1) < 1e-12
        assert all(math.isfinite(p) and p >= 0 for p in row['root_prior'])
        assert all(row['root_prior'][c] == row['visits'][c] == 0 for c in range(7) if c not in legal)
        if variant in 'CD':
            assert all(abs(row['root_prior'][c] - 1 / len(legal)) < 1e-12 for c in legal)
        assert math.isfinite(row['root_value']) and abs(row['root_value']) <= 1
        if variant in 'BD':
            assert row['root_value'] == 0
        if 'category' in fixture:
            assert row['category'] == fixture['category']
            assert row['correct'] == int(action == fixture['expected_action'])
            assert row['avoidable'] == int(bool(fixture['safe_actions']) and action not in fixture['safe_actions'])
            assert row['missed_win'] == int(bool(fixture['immediate_wins']) and action not in fixture['immediate_wins'])
        else:
            best, chosen = fixture['value'], fixture['action_values'][str(action)]
            assert row['correct'] == int(chosen == best) and row['mistake'] == int(chosen < best)
            assert row['exact'] == best and row['chosen_value'] == chosen
            for field, before, after in (('win_draw', 1, 0), ('win_loss', 1, -1), ('draw_loss', 0, -1)):
                assert row[field] == int(best == before and chosen == after)


def arithmetic():
    require_complete()
    fixtures = read(REPO / 'research/alphazero_resnet_selfplay/fixtures.json')
    package = read(REPO / 'games/connect4/alphazero_v2/frozen/openings-development.json')
    assert package['split'] == 'development'
    openings = {r['id']: r for r in package['rows'] if r['id'].endswith('-base')}
    assert len(openings) == 40 and len({r['family'] for r in openings.values()}) == 40
    output = dict(fixture_audit=fixture_audit(fixtures), positions={}, matches={}, values={}, pooled={})
    tables = {}
    discrepancies = []
    main = read(RESULTS / 'analysis.json')
    frozen = read(RESULTS / 'freeze.json')
    for arm in ARMS:
        expected6 = next(e for e in frozen['checkpoints'][arm] if e['generation'] == 6)
        output['positions'][arm] = {}
        output['matches'][arm] = {}
        for variant in 'ABCDE':
            output['positions'][arm][variant] = {}
            for suite in ('tactical', 'solved', 'deep'):
                directory = RESULTS / 'tasks' / f'{arm}-g6-{variant}-positions'
                assert read(directory / 'identity.json')['checkpoint'] == expected6
                evidence = read(directory / f'{suite}.json')
                assert evidence['complete']
                rows = evidence['rows']
                position_audit(rows, {r['id']: r for r in fixtures[suite]}, variant)
                tables[arm, variant, suite] = {r['id']: r for r in rows}
                baseline = tables[arm, 'A', suite]
                for row in rows:
                    a = baseline[row['id']]
                    if variant in 'ABE':
                        assert row['root_prior'] == a['root_prior']
                    assert row['root_value'] == a['root_value'] * {'A': 1, 'B': 0, 'C': 1, 'D': 0, 'E': .5}[variant]
                metrics = ('correct', 'avoidable', 'missed_win') if suite == 'tactical' else ('correct', 'mistake', 'win_draw', 'win_loss', 'draw_loss')
                entry = {}
                for metric in metrics:
                    entry[metric] = independent_interval(rows, [r[metric] for r in rows])
                    delta = independent_interval(rows, [r[metric] - baseline[r['id']][metric] for r in rows])
                    entry[metric]['delta_from_A'] = delta
                    discrepancies.extend((abs(entry[metric]['mean'] - main['positions'][arm][variant][suite][metric]['mean']),
                                          abs(delta['mean'] - main['positions'][arm][variant][suite]['delta_from_A'][metric]['mean'])))
                entry['fixed'] = sum(r['correct'] and not baseline[r['id']]['correct'] for r in rows)
                entry['introduced'] = sum(not r['correct'] and baseline[r['id']]['correct'] for r in rows)
                output['positions'][arm][variant][suite] = entry
            directory = RESULTS / 'tasks' / f'{arm}-g6-{variant}-matches'
            assert read(directory / 'identity.json')['checkpoint'] == expected6
            evidence = read(directory / 'matches.json')
            assert evidence['complete']
            records = evidence['records']
            assert len(records) == 80
            assert {(r['opening_id'], r['game_index']) for r in records} == {(opening, i) for opening in openings for i in (0, 1)}
            rows = []
            for record in records:
                opening = openings[record['opening_id']]
                assert record['family'] == opening['family'] and record['agent_color'] == record['game_index']
                assert record['moves'][:len(opening['moves'])] == opening['moves'] and not record['abandoned']
                board, _ = board_from_history(record['moves'], terminal=True)
                winner = winning_player(board)
                outcome = 'draw' if winner is None else ('win' if winner == record['agent_color'] else 'loss')
                assert record['winner'] == (-1 if winner is None else winner) and record['result'] == outcome
                assert record['score'] == {'win': 1., 'draw': .5, 'loss': 0.}[outcome]
                assert [d['move'] for d in record['decisions']] == record['moves'][len(opening['moves']):]
                assert all(d['color'] == (len(opening['moves']) + i) % 2 for i, d in enumerate(record['decisions']))
                rows.append(dict(id=f'{record["opening_id"]}:{record["game_index"]}', cluster=record['family'], score=record['score']))
            tables[arm, variant, 'matches'] = {r['id']: r for r in rows}
            baseline = tables[arm, 'A', 'matches']
            score = independent_interval(rows, [r['score'] for r in rows])
            delta = independent_interval(rows, [r['score'] - baseline[r['id']]['score'] for r in rows])
            output['matches'][arm][variant] = dict(wdl=dict(Counter(r['result'] for r in records)), score=score, delta_from_A=delta)
            discrepancies.extend((abs(score['mean'] - main['matches'][arm][variant]['score']['mean']),
                                  abs(delta['mean'] - main['matches'][arm][variant]['delta_from_A']['mean'])))
        for suite in ('tactical', 'solved', 'deep'):
            directory = RESULTS / 'tasks' / f'{arm}-g0-A-positions'
            expected0 = next(e for e in frozen['checkpoints'][arm] if e['generation'] == 0)
            assert read(directory / 'identity.json')['checkpoint'] == expected0
            position_audit(read(directory / f'{suite}.json')['rows'], {r['id']: r for r in fixtures[suite]}, 'A')
        evidence = read(RESULTS / 'tasks' / f'{arm}-values' / 'values.json')
        assert evidence['complete'] and evidence['identities']['g6'] == expected6
        assert evidence['identities']['g5'] == next(e for e in frozen['checkpoints'][arm] if e['generation'] == 5)
        assert evidence['prequential_g6_on_g7'] is None and evidence['label_audit']['errors'] == 0
        output['values'][arm] = {}
        datasets = {k: evidence[k] for k in ('training_g6', 'training_g5', 'prequential_g5_on_g6')}
        datasets.update({f'exact:{s}': rows for s, rows in evidence['exact'].items()})
        for kind, rows in datasets.items():
            assert len({r['id'] for r in rows}) == len(rows)
            assert all(math.isfinite(r['value']) and abs(r['value']) <= 1 and r['target'] in (-1, 0, 1) for r in rows)
            if kind.startswith('exact:'):
                labels = {r['id']: r for r in fixtures[kind.split(':')[1]]}
                assert len(rows) == len(labels) and {r['id'] for r in rows} == set(labels)
                assert all(r['target'] == labels[r['id']]['value'] and r['cluster'] == labels[r['id']]['cluster']
                           and r['ply'] == len(labels[r['id']]['moves']) and r['actor'] == r['ply'] % 2 for r in rows)
            summary = dict(mse=independent_interval(rows, [(r['value'] - r['target']) ** 2 for r in rows]),
                           zero_mse=independent_interval(rows, [r['target'] ** 2 for r in rows]))
            target_main = main['values'][arm]['exact'][kind.split(':')[1]] if kind.startswith('exact:') else main['values'][arm][kind]
            discrepancies.append(abs(summary['mse']['mean'] - target_main['mse']['mean']))
            if not kind.startswith('exact:'):
                labels = defaultdict(list)
                for row in rows:
                    labels[row['position_key']].append(row['target'])
                floor = sum(sum((z - sum(group) / len(group)) ** 2 for z in group) for group in labels.values()) / len(rows)
                summary['target_noise'] = dict(conflicting_repeated_positions=sum(len(set(zs)) > 1 for zs in labels.values()), empirical_identical_position_mse_floor=floor)
                discrepancies.append(abs(floor - target_main['target_noise']['empirical_identical_position_mse_floor']))
            output['values'][arm][kind] = summary
    for architecture in ('cnn', 'resnet3x64'):
        arms = [f'{seed}-{architecture}' for seed in (42, 314159)]
        output['pooled'][architecture] = {}
        for suite in ('tactical', 'solved', 'deep', 'matches'):
            metric = 'score' if suite == 'matches' else 'correct'
            rows = list(tables[arms[0], 'A', suite].values())
            assert all(set(tables[a, v, suite]) == {r['id'] for r in rows} for a in arms for v in 'ABCDE')
            contrasts = {}
            for name, coefficients in CONTRASTS.items():
                values = [sum(coefficients[v] * tables[a, v, suite][r['id']][metric] for a in arms for v in coefficients) / 2 for r in rows]
                contrasts[name] = independent_interval(rows, values)
                discrepancies.append(abs(contrasts[name]['mean'] - main['pooled'][architecture][suite]['contrasts'][name]['mean']))
            output['pooled'][architecture][suite] = contrasts
    assert max(discrepancies) < 1e-12
    output['comparison'] = dict(compared_means=len(discrepancies), max_mean_discrepancy=max(discrepancies),
                                independent_analysis_seed=20261012, resamples=10000,
                                exact_labels_limit='legal-vector/tactical consistency; inherited full-game WDL is not independently re-solved')
    save('independent-clean-recomputed.json', output)
    print(json.dumps(output['comparison']))


def replay_label_audit(runner, arm, check):
    """Reconstruct replay labels/keys independently; no network scoring or extra loads."""
    expected = []
    for game in runner.replay.iter_games():
        check()
        terminal, _ = board_from_history(game.moves, terminal=True)
        winner = winning_player(terminal)
        assert game.winner == (-1 if winner is None else winner)
        assert len(game.examples) == len(game.moves)
        board = [[-1] * 7 for _ in range(6)]
        prefix = tuple(game.moves[:8])
        cluster = repr(min(prefix, tuple(6 - c for c in prefix)))
        for ply, example in enumerate(game.examples):
            actor = ply % 2
            observation = tuple(tuple(0 if cell < 0 else 1 if cell == actor else -1 for cell in row)
                                for row in reversed(board))
            target = 0 if winner is None else (1 if winner == actor else -1)
            assert example.ply == ply and example.actor == actor and example.observation == observation
            assert example.outcome == target
            mirrored = tuple(row[::-1] for row in observation)
            family = actor, min(observation, mirrored)
            expected.append(dict(id=f'{game.generation}:{game.index}:{ply}', generation=game.generation,
                                 cluster=cluster, actor=actor, ply=ply, target=target, family=family,
                                 position_key=hashlib.sha256(repr((actor, observation)).encode()).hexdigest(),
                                 terminal_distance=len(game.moves) - ply))
            board = played(board, game.moves[ply], actor)
    raw = read(RESULTS / 'tasks' / f'{arm}-values' / 'values.json')
    training5_keys = {r['family'] for r in expected if r['generation'] <= 5}
    total = 0
    for kind, selected in (('training_g6', expected),
                           ('training_g5', [r for r in expected if r['generation'] <= 5]),
                           ('prequential_g5_on_g6', [r for r in expected if r['generation'] == 6])):
        check()
        records = raw[kind]
        assert len(records) == len(selected)
        repeats = Counter(r['family'] for r in selected)
        for actual, wanted in zip(records, selected):
            assert all(actual[field] == wanted[field] for field in ('id', 'cluster', 'actor', 'ply', 'target', 'position_key', 'terminal_distance'))
            assert actual['repetitions'] == repeats[wanted['family']]
            seen = wanted['family'] in training5_keys if kind == 'prequential_g5_on_g6' else None
            assert actual['seen_in_training'] == seen
        total += len(records)
    assert raw['label_audit'] == dict(errors=0, positions=total)
    return dict(arm=arm, independently_checked_labels=total, additional_model_inferences=0)


def reproduce(deadline=None):
    """All four frozen g6 arms x A/B/E x tactical200; at most 2400 roots/240s."""
    require_complete()
    from .common import load, enforce, runtime, sha
    from games.connect4.alphazero_v2.data import seeded_rng
    from games.connect4.alphazero_v2.network import V2Inference, V2Prediction
    from games.connect4.alphazero_v2.oracle import engine_position
    from games.connect4.alphazero_v2.search import EvaluationAgent
    runtime()
    frozen = enforce()
    fixtures = read(REPO / 'research/alphazero_resnet_selfplay/fixtures.json')['tactical']
    origin = time.monotonic()
    stop_epoch = min(time.time() + 240, deadline) if deadline is not None else time.time() + 240
    rows = []
    labels_checked = []

    def check():
        if time.time() >= stop_epoch or (RESULTS / 'CANCEL').exists():
            raise TimeoutError('Independent model reproduction bounded stop; no retries')

    class ScaledInference(V2Inference):
        # Reimplemented independently, without importing the phase Intervention.
        def __init__(self, base, factor):
            self.model, self.base, self.factor = base.model, base, factor

        def predict(self, state):
            p = self.base.predict(state)
            return V2Prediction(p.logits, p.policy, p.value * self.factor)

        def predict_batch(self, states):
            return tuple(V2Prediction(p.logits, p.policy, p.value * self.factor)
                         for p in self.base.predict_batch(states))

    try:
        for arm in ARMS:
            check()
            runner, model, base, entry = load(arm, 6, frozen)
            weights = runner.factory.hash(model)
            labels_checked.append(replay_label_audit(runner, arm, check))
            for variant, factor in (('A', 1.), ('B', 0.), ('E', .5)):
                saved = {r['id']: r for r in read(RESULTS / 'tasks' / f'{arm}-g6-{variant}-positions' / 'tactical.json')['rows']}
                for fixture in fixtures:
                    check()
                    rng = seeded_rng(f'phase4d4h-c-evaluation:tactical:{fixture["id"]}', 4404901)
                    result, selection = EvaluationAgent(ScaledInference(base, factor), 256, rng=rng, backend='bitboard').search(engine_position(fixture['moves']))
                    expected = saved[fixture['id']]
                    assert selection.move == expected['action'] and list(result.visits) == expected['visits']
                    assert list(result.root_prior) == expected['root_prior'] and result.root_value == expected['root_value']
                    assert result.root_noise is None and selection.temperature == 0 and sum(result.visits) == 256
                    rows.append(dict(arm=arm, variant=variant, id=fixture['id'], action=selection.move,
                                     visits=list(result.visits), matched=True, checkpoint_sha256=entry['sha256']))
                    check()
                assert runner.factory.hash(model) == weights and sha(entry['path']) == entry['sha256']
            save('independent-clean-model-rerun.json', dict(complete=False, roots=len(rows), rows=rows,
                                                          replay_label_audits=labels_checked, wall_seconds=time.monotonic() - origin))
        assert len(rows) == 2400
        check()
    except BaseException as error:
        save('independent-clean-model-rerun.json', dict(complete=False, roots=len(rows), rows=rows,
                                                      replay_label_audits=labels_checked, wall_seconds=time.monotonic() - origin, reason=repr(error)))
        raise
    save('independent-clean-model-rerun.json', dict(complete=True, roots=len(rows), rows=rows,
                                                  replay_label_audits=labels_checked,
                                                  wall_seconds=time.monotonic() - origin, ceiling_seconds=240,
                                                  same_host=True, variants=['A', 'B', 'E']))
    print('Clean independent model reproduction: 2400 exact action/visit matches')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('arithmetic', 'model-rerun'), default='arithmetic')
    parser.add_argument('--deadline', type=float)
    arguments = parser.parse_args()
    if arguments.mode == 'arithmetic':
        arithmetic()
    else:
        reproduce(arguments.deadline)


if __name__ == '__main__':
    main()
