"""Exact opening book: integrity, symmetry, fail-closed lookup and solver use.

Every stored value must come from a completed independent oracle search; the
runtime lookup must never act on absent, unresolved, edited or stale data.
"""
import copy
import hashlib
import json
from random import Random

import pytest

import games.connect4.victor.opening_book as book_module
from games.connect4.connect4 import Connect4
from games.connect4.victor import Position, SolverBudget, VictorSolver, analyze_position
from games.connect4.victor.exact import Bits
from games.connect4.victor.opening_book import (
    BOOK_PATH, BookIntegrityError, OpeningBook, canonical_key, entries_digest, preferred_move,
)
from victor_validation.native_oracle import SOURCE, OracleUnavailable, solve_histories
from victor_validation.opening_book_builder import TIE_BREAK_DEPTH

DATA = json.loads(BOOK_PATH.read_text())
BOOK = OpeningBook(DATA)
VALUES = {'+': 1, '=': 0, '-': -1}


def board_of(history):
    g = Connect4()
    for c in history:
        assert g.make_move(c)
    return g


def position(history):
    g = board_of(history)
    return Position.from_board(g.board, g.current_player)


def with_entries(data, entries, unresolved=None):
    out = copy.deepcopy(data)
    out['entries'] = entries
    out['unresolved'] = out['unresolved'] if unresolved is None else unresolved
    out['digest'] = entries_digest(out['entries'], out['unresolved'])
    return out


@pytest.fixture
def use_book(monkeypatch):
    def install(book):
        monkeypatch.setattr(book_module, 'default_book', lambda: book)
    return install


# ---------------------------------------------------------------- integrity

def test_book_file_is_exact_documented_and_current():
    assert DATA['schema'] == 'victor-opening-book-v1'
    assert DATA['digest'] == entries_digest(DATA['entries'], DATA['unresolved'])
    # Stale-data guard: regenerate the book whenever the oracle source changes.
    assert DATA['provenance']['oracle_sha256'] == hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    assert DATA['completion']['exact'] == len(DATA['entries']) == len(BOOK)
    assert DATA['completion']['unresolved'] == len(DATA['unresolved'])
    assert 'NOT a complete opening book' in DATA['description']
    assert max(len(e[4]) for e in DATA['entries']) <= DATA['generation']['max_plies']
    for key, player, value, moves, history, sources in DATA['entries']:
        legal = [m for m in moves if m != 'x']
        assert value == max(legal, key=VALUES.get) and sources
    # The packaged runtime path loads without the C compiler or tests package.
    assert book_module.default_book() is not None


def test_known_theorem_for_the_empty_board_and_first_replies():
    """Allen/Allis 1988: centre wins, columns c/e draw, the rest lose."""
    hit = BOOK.lookup(position(()))
    assert hit.status == 'exact' and hit.value == 1 and hit.optimal_moves == (3,)
    assert dict(hit.move_values) == {0: -1, 1: -1, 2: 0, 3: 1, 4: 0, 5: -1, 6: -1}
    # After the centre opening every Black reply loses.
    hit = BOOK.lookup(position((3,)))
    assert hit.value == -1 and set(dict(hit.move_values).values()) == {-1}


def test_every_entry_replays_and_its_mirror_maps_columns():
    for key, player, value, moves, history, _ in DATA['entries']:
        canonical = [int(c) - 1 for c in history]
        hit = BOOK.lookup(position(canonical))
        assert hit.status == 'exact' and hit.value == VALUES[value]
        assert dict(hit.move_values) == {c: VALUES[m] for c, m in enumerate(moves) if m != 'x'}
        mirrored = BOOK.lookup(position([6 - c for c in canonical]))
        assert mirrored.status == 'exact' and mirrored.value == hit.value
        assert dict(mirrored.move_values) == {6 - c: v for c, v in hit.move_values}


def test_parent_child_values_are_minimax_consistent():
    rows = {(int(k, 16), p): (v, m, h) for k, p, v, m, h, _ in DATA['entries']}
    linked = 0
    for (key, player), (value, moves, history) in rows.items():
        bits = Bits.from_position(position([int(c) - 1 for c in history]))
        for c, m in enumerate(moves):
            if m == 'x' or (child := bits.drop(c)).value is not None:
                continue
            other = rows.get((canonical_key(child)[0], child.turn))
            if other is not None:
                linked += 1
                assert VALUES[other[0]] == -VALUES[m]
    assert linked > 50


def test_digest_schema_and_malformed_rows_fail_closed(tmp_path, use_book):
    edited = copy.deepcopy(DATA)
    edited['entries'][0][2] = '='  # value edited without re-signing
    with pytest.raises(BookIntegrityError, match='digest'):
        OpeningBook(edited)
    with pytest.raises(BookIntegrityError, match='schema'):
        OpeningBook(dict(DATA, schema='victor-opening-book-v0'))
    with pytest.raises(BookIntegrityError, match='malformed'):
        OpeningBook(with_entries(DATA, [['zz', 0]]))
    path = tmp_path / 'book.json'
    path.write_text('{not json')
    with pytest.raises(BookIntegrityError):
        OpeningBook.load(path)
    # A disabled book never influences the solver.
    use_book(None)
    assert book_module.lookup(position(())).status == 'disabled'
    assert analyze_position(position(()).board, 0).move_kind != 'opening_book'


@pytest.mark.parametrize('tamper', ['history', 'legal', 'value'])
def test_structurally_inconsistent_entries_are_never_used(tamper, use_book):
    """Re-signed (digest-valid) but stale/incorrect rows are rejected per hit."""
    entries = copy.deepcopy(DATA['entries'])
    row = next(e for e in entries if e[4] == '4')  # after White's centre opening
    if tamper == 'history':
        row[4] = '3'
    elif tamper == 'legal':
        row[3] = 'x' + row[3][1:]
    else:
        row[2] = '+'  # claims a win the move values do not contain
    use_book(OpeningBook(with_entries(DATA, entries)))
    hit = book_module.lookup(position((3,)))
    assert hit.status == 'invalid'
    result = analyze_position(position((3,)).board, 1)
    assert result.move_kind != 'opening_book' and result.move in range(7)


def test_wrong_but_well_formed_values_are_caught_by_the_oracle_cross_check(oracle):
    """Runtime cannot detect a consistent lie; the offline oracle cross-check does."""
    deep = [e for e in DATA['entries'] if len(e[4]) >= 7][:3]
    flipped = []
    for key, player, value, moves, history, sources in deep:
        swap = {'+': '-', '-': '+', '=': '=', 'x': 'x'}
        flipped.append([key, player, swap[value] if value != '=' else value,
                        ''.join(swap[m] for m in moves), history, sources])
    results = oracle([[int(c) - 1 for c in e[4]] for e in flipped])
    mismatches = sum(''.join({1: '+', 0: '=', -1: '-'}[r.move_values[c]] if c in r.move_values
                             else 'x' for c in range(7)) != e[3]
                     for e, r in zip(flipped, results))
    assert mismatches == sum(any(m in '+-' for m in e[3]) for e in flipped) > 0


def test_unresolved_and_absent_positions_keep_the_existing_policy(use_book):
    history = (3, 3, 2)
    bits = Bits.from_position(position(history))
    key, _ = canonical_key(bits)
    entries = [e for e in DATA['entries'] if (int(e[0], 16), e[1]) != (key, bits.turn)]
    use_book(OpeningBook(with_entries(DATA, entries, [[format(key, 'x'), bits.turn, '443',
                                                       'oracle_unknown_after_9_nodes']])))
    hit = book_module.lookup(position(history))
    assert hit.status == 'unresolved' and 'unknown' in hit.detail
    expected = analyze_position(position(history).board, 1, SolverBudget(opening_book=False))
    result = analyze_position(position(history).board, 1)
    assert result.move_kind != 'opening_book' and result.move == expected.move
    absent = (0, 0, 6, 6, 0, 6, 1, 5, 1)  # nine stones: beyond the book by construction
    assert book_module.lookup(position(absent)).status == 'absent'


def test_terminal_positions_are_never_book_hits():
    assert BOOK.lookup(position((0, 1, 0, 1, 0, 1, 0))).status == 'absent'


# ------------------------------------------------------------ cross-checks

@pytest.fixture(scope='module')
def oracle():
    try:
        solve_histories([(3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2)], node_limit=1_000_000)
    except OracleUnavailable as exc:
        pytest.skip(str(exc))
    return solve_histories


def test_fresh_oracle_resolves_sampled_entries_identically(oracle):
    sample = Random(7).sample([e for e in DATA['entries'] if len(e[4]) >= 6], 24)
    results = oracle([[int(c) - 1 for c in e[4]] for e in sample], node_limit=2_000_000_000)
    for (key, player, value, moves, history, _), r in zip(sample, results):
        assert r.status == 'exact'
        assert {1: '+', 0: '=', -1: '-'}[r.value] == value
        assert ''.join({1: '+', 0: '=', -1: '-'}[r.move_values[c]] if c in r.move_values else 'x'
                       for c in range(7)) == moves


# --------------------------------------------------------------- solver use

def test_book_moves_are_optimal_legal_and_labelled_exact():
    rng = Random(3)
    for key, player, value, moves, history, _ in rng.sample(DATA['entries'], 60):
        p = position([int(c) - 1 for c in history])
        result = analyze_position(p.board, player)
        assert result.move_kind == 'opening_book' and result.justified_move
        assert result.exact_value == VALUES[value] and result.bound is None
        assert moves[result.move] == value  # an optimal legal column
        assert result.exact.status == 'opening_book' and result.exact.nodes == 0


def test_closure_covers_every_victor_decision_through_the_book_horizon():
    """Victor follows the book; every opponent reply stays inside it."""
    horizon = DATA['generation']['max_plies']
    assert DATA['completion']['complete_closure'] is True
    for start in ([()], [(c,) for c in range(7)]):
        frontier, seen = list(start), set()
        while frontier:
            history = frontier.pop()
            g = board_of(history)
            if history in seen or g.is_game_over():
                continue
            seen.add(history)
            hit = BOOK.lookup(Position.from_board(g.board, g.current_player))
            assert hit.status == 'exact', history
            move = preferred_move(g.board, g.current_player, hit.move_values, hit.value,
                                  TIE_BREAK_DEPTH)
            assert move in hit.optimal_moves
            if len(history) + 2 > horizon:
                continue
            g.make_move(move)
            if g.is_game_over():
                continue
            for reply in g.get_valid_moves():
                frontier.append(history + (move, reply))


def test_book_precedes_retained_plans_and_session_replays_cleanly():
    solver, g, rng = VictorSolver(SolverBudget(deadline=0.5)), Connect4(), Random(11)
    history = []
    while not g.is_game_over():
        if g.current_player == 1:
            column = solver.choose_move(Connect4(g.board, 1))
            if len(history) <= 7:
                assert solver.last_result.move_kind == 'opening_book'
        else:
            column = rng.choice(g.get_valid_moves())
        assert g.make_move(column)
        history.append(column)
    assert board_of(history).board == g.board and board_of(history).check_winner() == g.check_winner()


def test_book_exclusion_filters_benchmark_only_entries():
    only_benchmark = [e for e in DATA['entries'] if e[5] == 4]  # bit 2 = 'benchmark'
    assert only_benchmark, 'benchmark tier should add in-sample entries'
    history = [int(c) - 1 for c in only_benchmark[0][4]]
    p = position(history)
    assert analyze_position(p.board, p.player_to_move).move_kind == 'opening_book'
    held_out = SolverBudget(book_exclude=('benchmark',))
    assert analyze_position(p.board, p.player_to_move, held_out).move_kind != 'opening_book'
    with pytest.raises(ValueError):
        SolverBudget(book_exclude=('suite',))
    with pytest.raises(ValueError):
        SolverBudget(opening_book=1)
