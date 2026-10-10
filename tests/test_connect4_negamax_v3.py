"""Independent exact-search, mirror identity, bounds and rollback checks."""
from copy import deepcopy
from math import inf
from random import Random

import pytest

from games.connect4.agents.negamax_tt import pack_key, unpack_key, pack_entry, unpack_entry
from scripts.negamax_v3_variants import load_variant
from tests.test_connect4_negamax import position, oracle, root_oracle, DRAW

VARIANTS = ('direct', 'mirror', 'mirror-selective', 'pvs')


@pytest.fixture(params=VARIANTS)
def module(request):
    return load_variant(request.param)


@pytest.mark.parametrize('history,depth', [([], 3), ([0,1,0,1,0,2],4),
    ([0,1,0,1,0],3), ([5,4,3,6,2,4],3), ([6,4,4,2,2,2,6,3],3),
    ([1,0,3,0,5,0,1,6,3,6,5,6],3), (DRAW[:36],6), (DRAW[:41],12)])
def test_root_oracle_and_caller(module, history, depth):
    for h in (history, [6-c for c in history]):
        game = position(h)
        before = deepcopy(vars(game))
        agent = module.NegamaxAgent(depth)
        scores = agent.score_moves(game)
        assert scores == root_oracle(game, depth)
        assert list(scores) == game.get_valid_moves()
        assert agent.choose_move(game) == max(scores, key=scores.get)
        assert vars(game) == before


@pytest.mark.parametrize('depth', [1,2,3,4])
def test_bounds_reuse_and_depth(module, depth):
    game = position([5,4,3,6,2,4])
    state = module.SearchState(game)
    expected = oracle(game, depth)
    for a, b, flag in [(expected-2,expected-1,'lower'), (expected+1,expected+2,'upper')]:
        table = module.SearchTable()
        value = module.negamax(state, depth, a, b, table)
        assert value >= b if flag == 'lower' else value <= a
        key = module.identity(state,depth,3 if module.__name__ == 'mirror-selective' else 1)[0] if hasattr(module,'identity') else pack_key(*state.pieces,state.mover,depth)
        assert unpack_entry(table.entries[key])[0] == flag
        assert module.negamax(state,depth,table=table) == expected
        assert unpack_entry(table.entries[key])[:2] == ('exact',expected)
        for alpha,beta in [(-10,-9),(0,1),(20,21),(-5,60)]:
            value = module.negamax(state,depth,alpha,beta,table)
            assert value <= alpha if expected <= alpha else value >= beta if expected >= beta else value == expected
        assert module.negamax(state,depth,table=table) == expected
    for mover in (0,1):
        game.current_player = mover
        assert module.negamax(module.SearchState(game),depth) == oracle(game,depth)


def test_reflection_involution_geometry_identity():
    m = load_variant('mirror')
    rng = Random(702)
    for cell in range(49):
        assert m.reflect(1 << cell) == 1 << ((6-cell//7)*7+cell%7)
    for _ in range(500):
        x, o = rng.getrandbits(49), rng.getrandbits(49)
        assert m.reflect(m.reflect(x)) == x
        for mover in (0,1):
            state = m.SearchState(position([]))
            state.pieces, state.mover = [x,o], mover
            for depth in (1,3,12,10**40):
                k, mirrored = m.identity(state,depth)
                xx, oo, mm, dd = unpack_key(k)
                assert (mm,dd) == (mover,depth)
                assert (xx,oo) == ((m.reflect(x),m.reflect(o)) if mirrored else (x,o))
                state.pieces = [m.reflect(x),m.reflect(o)]
                assert m.identity(state,depth)[0] == k
                state.pieces = [x,o]
    for hint in (None,3,0,6,7,14):
        assert m.reflect_hint(m.reflect_hint(hint)) == hint
    state = m.SearchState(position([3,3]))
    assert m.identity(state,3)[1] is False


@pytest.mark.parametrize('flag', ['exact','lower','upper'])
def test_mirrored_cache_hints_bounds(flag):
    m = load_variant('mirror')
    s = m.SearchState(position([0,1,0,2]))
    t = m.SearchTable()
    key, reflected = m.identity(s,3)
    for score in (-10**100,0,10**100):
        assert unpack_entry(pack_entry(flag,score,3)) == (flag,score,3)
    # A bound that cannot close this window only suggests the mapped column.
    value = oracle(position([0,1,0,2]),3)
    bound = value if flag == 'exact' else -10**20 if flag == 'lower' else 10**20
    move = 0
    t.entries[key] = pack_entry(flag,bound,m.reflect_hint(move) if reflected else move)
    assert m.negamax(s,3,table=t) == value
    mirror = m.SearchState(position([6,5,6,4]))
    assert m.negamax(mirror,3,table=t) == value
    assert t.hits > 0


@pytest.mark.parametrize('failure', ['heuristic','terminal_value','ordered_moves'])
def test_recursive_and_root_exception_restoration(module, monkeypatch, failure):
    game = position([3,2,4,3])
    state = module.SearchState(game)
    before = deepcopy({slot:getattr(state,slot) for slot in state.__slots__})
    original = getattr(module.SearchState,failure)
    def fail(self,*args,**kwargs):
        if self.count >= 7:
            raise RuntimeError('injected')
        return original(self,*args,**kwargs)
    monkeypatch.setattr(module.SearchState,failure,fail)
    with pytest.raises(RuntimeError,match='injected'):
        module.negamax(state,4,table=module.SearchTable())
    assert {slot:getattr(state,slot) for slot in state.__slots__} == before
    caller = deepcopy(vars(game))
    with pytest.raises(RuntimeError,match='injected'):
        module.NegamaxAgent(5).choose_move(game)
    assert vars(game) == caller


def test_seeded_exact_vectors(module):
    rng = Random(913)
    for _ in range(16):
        game = position([])
        for _ in range(rng.randrange(4,30)):
            game.make_move(rng.choice(game.get_valid_moves()))
            if game.is_game_over():
                break
        if not game.is_game_over():
            assert module.NegamaxAgent(3).score_moves(game) == root_oracle(game,3)
