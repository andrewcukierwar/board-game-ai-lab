"""Independent bounds, collision/interruption, and integration checks for move proofs."""
from dataclasses import replace
import ctypes
import json
from pathlib import Path
from random import Random
from concurrent.futures import ThreadPoolExecutor

import pytest

from games.connect4.victor import Position, SolverBudget, analyze_position
from games.connect4.victor.native import NativeBudget, build, prove_moves, _library
from games.connect4.victor.exact import Bits
from games.connect4.victor.coverage import search_covering_set
from games.connect4.victor.certificate_producer import certificate_from_witness
from games.connect4.victor.certificate import verify_certificate, CertificateStatus
from victor_validation.exact_oracle import replay, solve
from victor_validation.performance_benchmark import game_from
from test_victor_performance import random_history


@pytest.fixture(scope='module', autouse=True)
def compiled():
    import shutil
    if shutil.which('cc') is None:
        pytest.skip('native correctness tests require build-time C compiler')
    build()


def position(history):
    g=game_from(history)
    return Position.from_board(g.board,g.current_player)


def check(proof, truth):
    assert proof.intervals
    assert {c for c,_,_ in proof.intervals}==set(truth)
    assert all(-1<=lo<=truth[c]<=hi<=1 for c,lo,hi in proof.intervals)
    if proof.optimal_moves:
        assert all(truth[c]==max(truth.values()) for c in proof.optimal_moves)
    assert proof.admissible_moves
    if proof.lower>=0:
        assert all(truth[c]>=0 for c in proof.admissible_moves)


@pytest.mark.parametrize('nodes,entries', [(0,1),(1,1),(128,1),(2048,17),(200_000,8192)])
def test_every_partial_bound_against_frozen_independent_oracle(nodes,entries):
    rows=json.loads(Path('docs/victor-performance/suite.json').read_text())['positions']
    for row in rows[::5]:
        p=position(row['history'])
        proof=prove_moves(p,NativeBudget(nodes=nodes,seconds=None,table_entries=entries,
                                        claimeven=True))
        assert proof.nodes<=nodes
        check(proof,{int(c):v for c,v in row['move_values'].items()})


def test_native_vs_independent_exhaustive_python_oracle():
    rng=Random(71041)
    for remaining in (4,6,8,9,10)*8:
        history,game=random_history(rng,42-remaining)
        truth=solve(replay(history),max_remaining=10,position_budget=1_000_000)
        assert truth.status=='exact'
        p=Position.from_board(game.board,game.current_player)
        for strategic in (False,True):
            r=prove_moves(p,NativeBudget(nodes=1_000_000,seconds=None,serial=True,
                                         claimeven=strategic))
            assert r.status=='all_moves'
            check(r,dict(truth.move_values))
            assert {c:lo for c,lo,hi in r.intervals}==dict(truth.move_values)


def test_fast_claimeven_cutoff_has_independently_verified_certificate():
    lib=_library()
    f=lib.victor_claimeven
    f.argtypes=[ctypes.c_uint64,ctypes.c_uint64]
    f.restype=ctypes.c_int
    rng=Random(925)
    hits=0
    for _ in range(400):
        _,g=random_history(rng,rng.choice((16,20,24,28,32,34,36)))
        p=Position.from_board(g.board,g.current_player)
        b=Bits.from_position(p)
        if f(b.pieces[0],b.pieces[0]|b.pieces[1]):
            # General CL/BI/VE producer followed by INDEPENDENT raw-cell verifier.
            witness=search_covering_set(p).witness
            assert witness is not None
            assert verify_certificate(certificate_from_witness(witness)).status is CertificateStatus.HYPOTHESES_VERIFIED
            hits+=1
    assert hits>=10


def test_mirrors_determinism_and_concurrent_isolation():
    rows=json.loads(Path('docs/victor-astra/dev.json').read_text())['positions'][:12]
    budget=NativeBudget(nodes=100_000,seconds=None,table_entries=4096,claimeven=True)
    def run(row):
        a=prove_moves(position(row['history']),budget)
        b=prove_moves(position([6-c for c in row['history']]),budget,
                      order=tuple(6-c for c in (3,2,4,1,5,0,6)))
        truth={int(c):v for c,v in row['oracle']['move_values'].items()}
        check(a,truth)
        check(b,{6-c:v for c,v in truth.items()})
        # Internal centre tie ordering need not mirror under truncation.
        return a.intervals,a.nodes
    expected=[run(r) for r in rows]
    with ThreadPoolExecutor(4) as pool:
        assert list(pool.map(run,rows))==expected


def test_zero_time_and_missing_library_fail_closed(monkeypatch):
    p=position([3,2,3,4,0,6,1,2,6])
    r=prove_moves(p,NativeBudget(seconds=0))
    assert r.nodes==0
    monkeypatch.setattr('games.connect4.victor.native._library',lambda:None)
    assert prove_moves(p).status=='unavailable'
    old=SolverBudget(opening_book=False,cover_nodes=0)
    a=analyze_position(p.board,p.player_to_move,old)
    b=analyze_position(p.board,p.player_to_move,replace(old,native=NativeBudget()))
    assert (a.move,a.move_kind)==(b.move,b.move_kind)


def test_early_proved_move_and_partial_nonloss_are_distinct():
    rows=json.loads(Path('docs/victor-astra/dev.json').read_text())['positions']
    optimal=partial=0
    for row in rows:
        p=position(row['history'])
        result=analyze_position(p.board,p.player_to_move,SolverBudget(
            opening_book=False,cover_nodes=0,native=NativeBudget(nodes=20_000,seconds=None)))
        proof=result.move_proof
        assert result.move in game_from(row['history']).get_valid_moves()
        if proof.optimal_moves:
            assert result.move_kind=='exact' and result.exact_value==row['oracle']['value']
            assert result.justified_move
            optimal+=1
        elif proof.lower==0:
            assert result.move_kind=='search_nonloss' and result.exact_value is None
            assert result.justified_move and result.bound
            partial+=1
    assert optimal>50 and partial>0


@pytest.mark.parametrize('kwargs',[{'nodes':True},{'seconds':float('nan')},
    {'seconds':-1},{'table_entries':4_194_305},{'serial':1},{'claimeven':'yes'}])
def test_native_budget_validation(kwargs):
    with pytest.raises(ValueError): NativeBudget(**kwargs)


def test_known_white_refutation_failure_is_fixed_by_a_proof():
    rows = json.loads(Path('docs/victor-performance/suite.json').read_text())['positions']
    row = next(r for r in rows if r['id'] == 'e0-006')
    p = position(row['history'])
    result = analyze_position(p.board, 0, SolverBudget(
        opening_book=False, native=NativeBudget(seconds=None)))
    assert result.move == 1 and result.exact_value == 0
    assert result.move_proof.optimal_moves and result.justified_move


def test_native_split_has_no_canonical_overlap_or_book_leakage():
    from victor_validation.performance_benchmark import canonical
    path = Path('docs/victor-astra')
    dev = {canonical(r['history']) for r in json.loads((path / 'dev.json').read_text())['positions']}
    heldout = {canonical(r['history']) for r in json.loads((path / 'heldout.json').read_text())['positions']}
    book = json.loads(Path('games/connect4/victor/data/opening_book.json').read_text())
    keys = {canonical([int(c) - 1 for c in e[4]]) for e in book['entries']}
    assert len(dev) == len(heldout) == 300
    assert not dev & heldout and not (dev | heldout) & keys


def test_missing_native_library_preserves_retained_session(monkeypatch):
    from games.connect4.victor import VictorSolver
    monkeypatch.setattr('games.connect4.victor.native._library', lambda: None)
    old = VictorSolver(SolverBudget())
    new = VictorSolver(SolverBudget(native=NativeBudget()))
    # This Black move establishes a policy which subsequent calls must retain.
    game = game_from([3, 2, 3, 4, 0, 6, 1, 2, 6])
    for _ in range(8):
        a = old.choose_move(game)
        b = new.choose_move(game)
        assert (a, old.last_result.move_kind) == (b, new.last_result.move_kind)
        game.make_move(a)
        if game.is_game_over():
            break
        game.make_move(game.get_valid_moves()[0])
        if game.is_game_over():
            break
    with pytest.raises(ValueError):
        NativeBudget(nodes=2**64)
