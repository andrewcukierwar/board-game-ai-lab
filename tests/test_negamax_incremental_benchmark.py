"""Evidence harness must refuse drift, overwrite and result/counter mismatch."""
from copy import deepcopy
from pathlib import Path

import pytest

from scripts import benchmark_negamax_incremental as bench


def test_declaration_freezes_all_prior_fixtures_seeded_histories_and_tail(tmp_path):
    design = Path('docs/search-negamax-v2/incremental-evaluation/DESIGN.md')
    (tmp_path / 'DESIGN.md').write_bytes(design.read_bytes())
    config = bench.declare(tmp_path)
    bench.check_source(config, tmp_path)
    assert len(config['positions']) == 20
    assert sum(f['group'] == 'phase3a' for f in config['positions']) == 11
    assert [len(f['history']) for f in config['positions'] if f['group'] == 'seeded'] == [8, 12, 16, 20] * 2
    tail = config['positions'][-1]
    assert tail['group'] == 'post-hoc-diagnostic' and tail['history'] == [1, 4, 6, 0, 6]
    with pytest.raises(FileExistsError):
        bench.declare(tmp_path)
    (tmp_path / 'DESIGN.md').write_text('changed declaration')
    with pytest.raises(AssertionError):
        bench.check_source(config, tmp_path)


@pytest.mark.parametrize('key', ['move', 'scores', 'stats'])
def test_parity_check_rejects_any_result_or_counter_difference(key):
    run = dict(move=3, scores=[[3, 2], [2, 1]], stats=dict(nodes=3, entries=1, hits=0, cutoffs=1))
    other = deepcopy(run)
    other[key] = None
    with pytest.raises(AssertionError):
        bench.assert_same(run, other)


def test_parity_includes_root_insertion_order():
    run = dict(move=3, scores=[[3, 2], [2, 2]], stats={})
    other = deepcopy(run)
    other['scores'].reverse()
    with pytest.raises(AssertionError):
        bench.assert_same(run, other)


def test_analysis_excludes_tail_and_rejects_terminal_heavy_regression():
    def row(group, depth, candidate_time):
        old = dict(median_wall_seconds=.01,
                   counted=dict(stats=dict(nodes=5000), leaf_evaluations=3000),
                   memory=[dict(peak_bytes=100000, tt_bytes=90000)])
        new = deepcopy(old)
        new['median_wall_seconds'] = candidate_time
        return dict(group=group, depth=depth, variants={'baseline': old, **{
            variant: deepcopy(new) for variant in bench.VARIANTS}})

    rows = [row('phase3a', d, .005) for d in (4, 6, 8, 10)]
    rows.append(row('post-hoc-diagnostic', 10, 50))
    result = bench.analyze(rows)
    assert result['selected'] is not None
    assert all(c['eligible'] for c in result['comparisons'].values())
    rows.append(row('phase3a', 4, .1))
    result = bench.analyze(rows)
    assert result['selected'] is None
    assert all(not c['checks']['per_condition_overhead'] for c in result['comparisons'].values())
