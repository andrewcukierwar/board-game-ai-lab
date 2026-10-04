"""Final frozen data labels and four-run protocol; no candidate training/inference."""
import json
from collections import Counter
from copy import deepcopy

import pytest
pytest.importorskip('torch')

from games.connect4.dqn import replication as r
from games.connect4.dqn import validation_holdout as h
from games.connect4.dqn.experiment import parser
from test_dqn_validation import test_independent_labels as independent_labels

ROWS = json.loads(r.FINAL_PATH.read_text())['positions']


@pytest.mark.parametrize('row', ROWS, ids=lambda row: row['name'])
def test_final_independent_engine_labels(row):
    independent_labels(deepcopy(row))


def test_final_frozen_disjoint_balanced_families():
    assert h.digest(r.FINAL_PATH) == '8e63fd8be0419ddb4daf83ae226362f26999dced321617900938d61a60efb7da'
    assert h.validate(ROWS, exclude_paths=(h.OLD_PATH, h.PATH)) == 48
    assert len(ROWS) == 96
    assert Counter((x['category'],x['direction'],x['player']) for x in ROWS) == {
        (k,d,p):8 for k in ('win','block') for d in ('horizontal','vertical','diagonal') for p in (0,1)}
    exact = lambda g:(g.current_player,tuple(tuple(row) for row in g.board))
    assert len({exact(h.replay(x['moves'])) for x in ROWS}) == 96
    assert [sum(x['expected_action']==c for x in ROWS) for c in range(7)] == [14,14,14,12,14,14,14]
    for path in (h.OLD_PATH,h.PATH):
        old = {h.key(h.replay(x['moves'])) for x in json.loads(path.read_text())['positions']}
        assert not old & {h.key(h.replay(x['moves'])) for x in ROWS}
    with pytest.raises(ValueError, match='Duplicate'):
        h.validate(ROWS, exclude_paths=(r.FINAL_PATH,))


def test_exact_paired_experiment_configuration():
    assert r.SEEDS == (73,314)
    for seed in r.SEEDS:
        a = vars(parser().parse_args(r.training_args(seed,0.0,'unused-baseline')))
        b = vars(parser().parse_args(r.training_args(seed,0.5,'unused-augmented')))
        assert a.pop('horizontal_symmetry_probability') == 0.0
        assert b.pop('horizontal_symmetry_probability') == 0.5
        a.pop('output'); b.pop('output')
        assert a == b
        assert a['max_updates']==90000 and a['max_plies']==100000 and a['max_games']==7000 and a['max_seconds']==300
        assert a['batch_size']==64 and a['replay_capacity']==10000 and a['target_sync_interval']==100
        assert a['seed']==seed and a['epsilon_decay']==0.99997 and a['epsilon_min']==0.10
        assert a['threads']==a['interop_threads']==1
        assert a['archived_checkpoint'] is None and a['evaluation_games']==0
