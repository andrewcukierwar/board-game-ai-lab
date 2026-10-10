"""Exact provisional fast paths; no timing measurements here."""
from copy import deepcopy
from math import inf
import pytest
from scripts.negamax_v3_variants import variant_source,load_source
from scripts.negamax_v3_tuning import terminal_parent_proof,trusted_tt
from games.connect4.agents.negamax_tt import unpack_key,unpack_entry
from tests.test_connect4_negamax import position,root_oracle,oracle,DRAW


@pytest.fixture(params=[terminal_parent_proof,trusted_tt])
def module(request):
    return load_source(request.param(variant_source('direct')),'tuning')


@pytest.mark.parametrize('history,depth',[([],3),([5,4,3,6,2,4],3),([0,1,0,1,0,2],4),
                                         ([1,0,3,0,5,0,1,6,3,6,5,6],3),(DRAW[:36],6)])
def test_exact_roots(module,history,depth):
    game=position(history)
    before=deepcopy(vars(game))
    agent=module.NegamaxAgent(depth)
    assert agent.score_moves(game)==root_oracle(game,depth)
    baseline=load_source(variant_source('direct')).NegamaxAgent(depth)
    assert agent.choose_move(game)==baseline.choose_move(game)
    assert agent.last_stats==baseline.last_stats
    assert vars(game)==before


def test_terminals_initial_and_child_validity(module):
    for history in (DRAW,[0,1,0,1,0,1,0]):
        game=position(history)
        for mover in (0,1):
            game.current_player=mover
            for depth in (0,4,10**80):
                assert module.negamax(module.SearchState(game),depth,table=module.SearchTable())==oracle(game,depth)
    # Proven nonterminal parent transition: only the previous mover added a stone.
    game=position([0,1,0,1,0,2])
    state=module.SearchState(game)
    assert state.terminal_value(3) is None
    state.play(0)
    if 'previous_only' in module.SearchState.terminal_value.__code__.co_varnames:
        assert state.terminal_value(2,previous_only=True)==state.terminal_value(2)==-1000002
    state.undo(0)


def test_retained_bounds_and_rollback(module,monkeypatch):
    game=position([5,4,3,6,2,4])
    state,table=module.SearchState(game),module.SearchTable()
    before=deepcopy({slot:getattr(state,slot) for slot in state.__slots__})
    for a,b in [(-10,-9),(20,21),(-inf,inf)]:
        module.negamax(state,3,a,b,table)
        for key,entry in table.entries.items():
            x,o,mover,depth=unpack_key(key)
            flag,value,hint=unpack_entry(entry)
            board=[['X' if x & (1 << (7*c+5-r)) else 'O' if o & (1 << (7*c+5-r)) else ' '
                    for c in range(7)] for r in range(6)]
            child=type(game)(board,mover)
            exact=oracle(child,depth)
            assert value==exact if flag=='exact' else value<=exact if flag=='lower' else value>=exact
            assert hint in child.get_valid_moves()
    def fail(self): raise RuntimeError('injected')
    monkeypatch.setattr(module.SearchState,'heuristic',fail)
    with pytest.raises(RuntimeError):
        module.negamax(state,3,table=module.SearchTable())
    assert {slot:getattr(state,slot) for slot in state.__slots__}==before
