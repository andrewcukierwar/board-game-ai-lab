"""Conditional symmetry never changes exact roots or the asymmetric tree."""
from copy import deepcopy
import json
from pathlib import Path
from math import inf
import pytest
from scripts import benchmark_negamax_v3 as bench
from scripts.negamax_v3_root_mirror import source_for
from scripts.negamax_v3_variants import load_source
from tests.test_connect4_negamax import position,root_oracle,oracle,DRAW

BASE=Path('docs/search-negamax-v3/tuning/terminal-proof-source.py').read_text()
MANIFEST=json.loads(Path('docs/search-negamax-v3/root-mirror/manifest.json').read_text())


@pytest.mark.parametrize('fixture',MANIFEST['positions'])
def test_legal_root_oracle_and_asymmetric_payload(fixture):
    m=load_source(source_for(BASE))
    ref=load_source(BASE)
    game=position(fixture['history'])
    caller=deepcopy(vars(game))
    agent=m.NegamaxAgent(3)
    with bench.capture(m) as held: scores=agent.score_moves(game)
    assert scores==root_oracle(game,3)
    assert agent.choose_move(game)==max(scores,key=scores.get)
    with bench.capture(ref) as baseline: refagent=ref.NegamaxAgent(3); refagent.score_moves(game)
    state=m.SearchState(game)
    symmetric=all(m.reflect(b)==b for b in state.pieces)
    if not symmetric:
        assert agent.last_stats==refagent.last_stats
        assert held[0].entries==baseline[0].entries
    assert held[0].mirror_enabled==symmetric
    assert vars(game)==caller


@pytest.mark.parametrize('method',['heuristic','terminal_value','ordered_moves'])
def test_nested_root_exception_restoration(method,monkeypatch):
    m=load_source(source_for(BASE))
    game=position([])
    state=m.SearchState(game)
    before=deepcopy({slot:getattr(state,slot) for slot in state.__slots__})
    original=getattr(m.SearchState,method)
    def fail(self,*args,**kwargs):
        if self.count>=3:raise RuntimeError('injected')
        return original(self,*args,**kwargs)
    monkeypatch.setattr(m.SearchState,method,fail)
    t=m.SearchTable();t.mirror_enabled=True
    with pytest.raises(RuntimeError):m.negamax(state,4,table=t)
    assert {slot:getattr(state,slot) for slot in state.__slots__}==before
    caller=deepcopy(vars(game))
    with pytest.raises(RuntimeError):m.NegamaxAgent(4).choose_move(game)
    assert vars(game)==caller


def test_all_retained_symmetric_bounds_against_array_truth():
    m=load_source(source_for(BASE))
    game=position([0,1,6,5])
    table=m.SearchTable();table.mirror_enabled=True
    for alpha,beta in [(-10,-9),(20,21),(-inf,inf)]:
        m.negamax(m.SearchState(game),3,alpha,beta,table)
        from games.connect4.agents.negamax_tt import unpack_key,unpack_entry
        for key,entry in table.entries.items():
            x,o,mover,depth=unpack_key(key)
            flag,value,hint=unpack_entry(entry)
            board=[['X' if x & (1 << (7*c+5-r)) else 'O' if o & (1 << (7*c+5-r)) else ' '
                    for c in range(7)] for r in range(6)]
            child=type(game)(board,mover)
            exact=oracle(child,depth)
            assert value==exact if flag=='exact' else value<=exact if flag=='lower' else value>=exact
            assert hint in child.get_valid_moves()
