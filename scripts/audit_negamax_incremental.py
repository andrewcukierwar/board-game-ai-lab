"""Audit complete incremental evidence and export all variant performance rows."""
import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import statistics

from scripts import benchmark_negamax_incremental as bench
from games.connect4.agents.negamax_agent import SearchState
from games.connect4.connect4 import Connect4
from tests.test_connect4_incremental_evaluation import array_evaluation, assert_state, snapshot
from tests.test_connect4_negamax import DRAW


def full_draw_path():
    """Check all 42 legal prefixes and all 42 undos with the array oracle."""
    game = Connect4()
    state = SearchState(game)
    initial = snapshot(state)
    games, saved, stack = [game], [], []
    for col in DRAW:
        saved.append(snapshot(state))
        child = Connect4(game.board, game.current_player)
        assert child.make_move(col)
        stack.append(array_evaluation(game.board, 0))
        state.play(col)
        assert_state(state, child.board, child.current_player, stack)
        games.append(child)
        game = child
    assert game.is_board_full() and game.check_winner() == -1
    for col in reversed(DRAW):
        state.undo(col)
        stack.pop()
        games.pop()
        game = games[-1]
        assert_state(state, game.board, game.current_player, stack)
        assert snapshot(state) == saved.pop()
    assert snapshot(state) == initial
    return 84


def audit(directory):
    config = json.loads((directory / 'manifest.json').read_text())
    bench.check_source(config, directory)
    rows = [json.loads(line) for line in (directory / 'results.jsonl').read_text().splitlines()]
    expected = {(f['id'], d) for f in config['positions'] for d in config['depths']}
    identities = [(r['position'], r['depth']) for r in rows]
    assert len(identities) == len(set(identities)) and set(identities) == expected
    assert json.loads((directory / 'analysis.json').read_text()) == bench.analyze(rows)
    source_trees = [ast.parse(source) for source in (bench.git_source(bench.AGENT_PATH),
                                                    Path(bench.AGENT_PATH).read_bytes())]
    unchanged = ['has_four', 'winning_squares', 'SearchTable', 'negamax', 'NegamaxAgent']
    for name in unchanged:
        nodes = [next(n for n in tree.body if getattr(n, 'name', None) == name) for tree in source_trees]
        assert ast.dump(nodes[0]) == ast.dump(nodes[1]), name
    for name in ('WIN_SCORE', 'CENTER_ORDER', 'EXACT', 'WEIGHTS', 'BOARD_MASK', 'BOTTOM_MASK', 'WINDOWS'):
        nodes = [next(n for n in tree.body if isinstance(n, ast.Assign) and any(
            any(isinstance(part, ast.Name) and part.id == name for part in ast.walk(target))
            for target in n.targets)) for tree in source_trees]
        assert ast.dump(nodes[0]) == ast.dump(nodes[1]), name
        unchanged.append(name)
    for name in ['legal', 'ordered_moves', 'terminal_value']:
        classes = [next(n for n in tree.body if getattr(n, 'name', None) == 'SearchState') for tree in source_trees]
        nodes = [next(n for n in cls.body if getattr(n, 'name', None) == name) for cls in classes]
        assert ast.dump(nodes[0]) == ast.dump(nodes[1]), name
        unchanged.append('SearchState.' + name)
    prior = {}
    for path in ('profiling.json', 'profiling-supplement/profiling.json'):
        for row in json.loads(bench.git_source(bench.EVIDENCE + path))['rows']:
            prior[row['position'], row['depth']] = row
    decisions, prior_matches = 0, 0
    tail_matches = 0
    tail_prior = json.loads(bench.git_source(bench.EVIDENCE + 'tail-diagnostic.json'))['rows'][0]
    csv_rows = []
    for row in rows:
        base = row['variants']['baseline']
        reference = base['samples'][0]
        if row['group'] == 'phase3a':
            old = prior[row['position'], row['depth']]
            normal = old['samples'][0]
            normal['scores'] = [[int(k), v] for k, v in normal['scores'].items()]
            bench.assert_same(reference, normal)
            assert base['counted']['leaf_evaluations'] == old['profile']['leaf_evaluations']
            prior_matches += 1
        if row['group'] == 'post-hoc-diagnostic' and row['depth'] == 10:
            normal = tail_prior['normal']
            normal['scores'] = [[int(k), v] for k, v in normal['scores'].items()]
            bench.assert_same(reference, normal)
            assert base['counted']['leaf_evaluations'] == tail_prior['profile']['leaf_evaluations']
            tail_matches += 1
        for name, variant in row['variants'].items():
            assert len(variant['samples']) == config['samples']
            assert len(variant['memory']) == config['memory_repetitions']
            assert variant['median_wall_seconds'] == statistics.median(s['wall_seconds'] for s in variant['samples'])
            assert variant['median_cpu_seconds'] == statistics.median(s['cpu_seconds'] for s in variant['samples'])
            assert variant['counted']['leaf_evaluations'] == base['counted']['leaf_evaluations']
            for run in variant['samples'] + variant['memory'] + [variant['counted']]:
                bench.assert_same(reference, run)
                assert run['wall_seconds'] > 0 and run['cpu_seconds'] > 0
                decisions += 1
            assert all(m['tt_bytes'] == base['memory'][0]['tt_bytes'] for m in variant['memory'])
            assert all(m['table_attributes_bytes'] == base['memory'][0]['table_attributes_bytes']
                       for m in variant['memory'])
            csv_rows.append(dict(position=row['position'], group=row['group'], depth=row['depth'], variant=name,
                median_wall_ms=variant['median_wall_seconds'] * 1000,
                median_cpu_ms=variant['median_cpu_seconds'] * 1000,
                wall_speedup=base['median_wall_seconds'] / variant['median_wall_seconds'],
                nodes=reference['stats']['nodes'], entries=reference['stats']['entries'],
                hits=reference['stats']['hits'], cutoffs=reference['stats']['cutoffs'],
                leaves=variant['counted']['leaf_evaluations'], tt_bytes=variant['memory'][0]['tt_bytes'],
                max_peak_bytes=max(m['peak_bytes'] for m in variant['memory']),
                move=reference['move'], scores=json.dumps(reference['scores'])))
    micro = json.loads((directory / 'microbenchmark.json').read_text())['rows']
    assert len(micro) == len(config['positions'])
    assert {r['position'] for r in micro} == {f['id'] for f in config['positions']}
    for row in micro:
        for samples in row['variants'].values():
            assert len(samples) == config['samples']
        for kind in ('calls', 'cycles'):
            assert len({s[kind]['checksum'] for samples in row['variants'].values() for s in samples}) == 1
    with (directory / 'performance.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(csv_rows)
    result = dict(conditions=len(rows), timed_decisions=len(rows) * len(config['variants']) * config['samples'],
                  warmups=len(rows) * len(config['variants']), verified_saved_decisions=decisions,
                  prior_phase3a_score_counter_leaf_matches=prior_matches,
                  prior_post_hoc_depth10_tail_matches=tail_matches,
                  unchanged_search_ast=unchanged, retained_tt_size_parity_all_repetitions=True,
                  independently_checked_full_draw_play_undo_transitions=full_draw_path(),
                  source_hashes_verified=True,
                  auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  evidence_sha256={name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                                   for name in ('manifest.json', 'results.jsonl', 'microbenchmark.json',
                                                'analysis.json', 'runtime-end.json', 'performance.csv')})
    archived = directory / 'pre-calibration'
    if archived.exists():
        original = json.loads((archived / 'manifest.json').read_text())
        assert hashlib.sha256((archived / 'harness.py').read_bytes()).hexdigest() == original['source_hashes'][
            'scripts/benchmark_negamax_incremental.py']
        assert hashlib.sha256((archived / 'DESIGN.md').read_bytes()).hexdigest() == original['design_sha256']
        assert original['positions'] == config['positions']
        assert original['source_hashes'][bench.AGENT_PATH] == config['source_hashes'][bench.AGENT_PATH]
        result['uncalibrated_run_preserved_with_matching_harness_design_candidate_and_fixtures'] = True
    bench.write_new(directory / 'audit.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    print(json.dumps(audit(parser.parse_args().directory), indent=2))


if __name__ == '__main__':
    main()
