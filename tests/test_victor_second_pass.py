"""Independent exhaustive suffix, interruption, ABI and hard-split regressions."""
import json
from pathlib import Path
from random import Random

import pytest

from games.connect4.victor import Position
from games.connect4.victor.native import NativeBudget, prove_moves, build
from victor_validation.exact_oracle import replay, solve
from test_victor_performance import random_history
from test_victor_native_search import check


@pytest.fixture(scope='module', autouse=True)
def compiled():
    import shutil
    if not shutil.which('cc'): pytest.skip('requires a build-time compiler')
    build()


def test_exhaustive_suffix_graphs_with_collisions_and_interruptions():
    rng=Random(209104)
    seen=set(); tested=0
    for _ in range(32):
        h,_=random_history(rng,32)
        pending=[replay(h)]
        while pending:
            p=pending.pop()
            if p in seen or p.terminal:continue
            seen.add(p)
            truth=solve(p,max_remaining=10,position_budget=1_000_000)
            assert truth.status=='exact'
            runtime=Position.from_board(p.board,p.turn)
            # Sweep many cancellation points and pathological replacement tables.
            for entries in (1,2,3,17):
                count=(tested*13)%67
                proof=prove_moves(runtime,NativeBudget(nodes=count,seconds=None,
                    table_entries=entries,claimeven=bool(tested%2)))
                assert proof.nodes<=count
                check(proof,dict(truth.move_values))
            if tested%19==0:
                full=prove_moves(runtime,NativeBudget(nodes=1_000_000,seconds=None,
                    table_entries=17,serial=True))
                assert full.status=='all_moves'
                assert dict(truth.move_values)=={c:lo for c,lo,hi in full.intervals}
            tested+=1
            pending.extend(p.drop(c) for c in p.legal_columns)
    print(f"Exhaustive suffix states: {tested}")
    assert tested>1000


@pytest.mark.parametrize('order',[(0,)*7,tuple(range(32)),(False,1,2,3,4,5,6),(-1,0,1,2,3,4,5)])
def test_invalid_order_never_enters_native(order):
    p=replay([3,2,3,4])
    with pytest.raises(ValueError):prove_moves(Position.from_board(p.board,p.turn),order=order)


def test_allocation_disabled_and_terminal_inputs():
    p=replay([3,2,3,4])
    assert prove_moves(Position.from_board(p.board,p.turn),NativeBudget(table_entries=0)).status=='unavailable'
    p=replay([0,1,0,1,0,1,0])
    q=prove_moves(Position.from_board(p.board,p.turn))
    assert q.status=='terminal' and not q.intervals and q.nodes==0


def test_fresh_split_disjoint_from_every_prior_artifact_and_book():
    from victor_validation.astra_experiment import histories
    from victor_validation.performance_benchmark import canonical
    root=Path('docs/victor-second-pass')
    rows=json.loads((root/'split.json').read_text())['positions']
    keys=[canonical(r['history']) for r in rows]
    assert len(set(keys))==len(keys)
    prior={canonical(h) for folder in ('victor-astra','victor-performance','victor-opening')
           for path in Path('docs',folder).glob('*.json')
           for h in histories(json.loads(path.read_text()))}
    book=json.loads(Path('games/connect4/victor/data/opening_book.json').read_text())
    prior.update(canonical([int(c)-1 for c in row[4]]) for row in book['entries'])
    assert not set(keys)&prior
    assert {r['mover'] for r in rows}=={0,1}
    assert {r['split'] for r in rows}=={'dev','heldout'}


def raw_groups():
    return tuple(tuple((r+i*dr,c+i*dc) for i in range(4))
        for r in range(6) for c in range(7) for dr,dc in ((0,1),(1,0),(1,1),(1,-1))
        if 0<=r+3*dr<6 and 0<=c+3*dc<7)


def pairing_facts(board):
    """Cell-based coverage and forced ownership, independent of native bit tricks."""
    groups=raw_groups()
    uppers={(r,c) for r in (0,2,4) for c in range(7)
            if board[r][c]==board[r+1][c]==' '}
    covered=all(any(board[r][c]=='O' or (r,c) in uppers for r,c in g) for g in groups)
    win=any(all(board[r][c]=='O' or (r,c) in uppers for r,c in g) for g in groups)
    return uppers,covered,win


def replay_pairing_policy(board, uppers):
    """Exhaust ALL White choices against an explicit legal Black response policy."""
    from functools import lru_cache
    pairs={}
    for r,c in uppers:pairs[r,c]=(r+1,c);pairs[r+1,c]=(r,c)
    singles=[(r,c) for r in range(6) for c in range(7)
             if board[r][c]==' ' and (r,c) not in pairs]
    assert len(singles)%2==0
    for a,b in zip(singles[::2],singles[1::2]):pairs[a]=b;pairs[b]=a
    groups=raw_groups()
    def winning(b,token):return any(all(b[r][c]==token for r,c in g) for g in groups)
    def drop(b,s,token):
        r,c=s
        assert b[r][c]==' ' and (r==5 or b[r+1][c]!=' ')
        return tuple(tuple(token if (i,j)==s else cell for j,cell in enumerate(row)) for i,row in enumerate(b))
    @lru_cache(None)
    def visit(b):
        legal=[(r,c) for r in range(6) for c in range(7)
               if b[r][c]==' ' and (r==5 or b[r+1][c]!=' ')]
        assert legal  # reaching a full drawn board contradicts the claimed win
        for square in legal:
            child=drop(b,square,'X')
            assert not winning(child,'X')
            child=drop(child,pairs[square],'O')
            if not winning(child,'O'):visit(child)
    visit(board)
    return visit.cache_info().currsize


def test_claimeven_win_has_executable_pairing_and_independent_exact_value():
    import ctypes
    from games.connect4.victor.native import _library
    from games.connect4.victor.exact import Bits
    lib=_library();f=lib.victor_claimeven_win
    f.argtypes=[ctypes.c_uint64,ctypes.c_uint64];f.restype=ctypes.c_int
    rng=Random(902310);hits=states=0;uncovered=0
    for _ in range(800):
        h,g=random_history(rng,rng.choice((32,34,36,38)))
        p=Position.from_board(g.board,0);b=Bits.from_position(p)
        uppers,covered,win=pairing_facts(p.board)
        assert bool(f(b.pieces[0],b.pieces[0]|b.pieces[1])) == (covered and win)
        truth=solve(replay(h),max_remaining=10,position_budget=1_000_000)
        assert truth.status=='exact'
        if covered and win:
            assert truth.mover_value==-1
            states+=replay_pairing_policy(p.board,uppers)
            hits+=1
        elif win and truth.mover_value==1:
            uncovered+=1  # Guaranteed Black squares alone do not prevent White winning first.
    print(f"CL winning positions: {hits}; policy states: {states}; unsafe without coverage: {uncovered}")
    assert hits>=10 and states>=20 and uncovered>=5
