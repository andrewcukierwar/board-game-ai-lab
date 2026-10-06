"""Phase 4D.3B torch-free evaluation infrastructure: oracle, reference Negamax, packages,
statistics and the paired arena. No model, training or learned checkpoint is involved."""
import contextlib
import io
import json
import math
from pathlib import Path
import random
import subprocess
import sys

import pytest

from games.connect4.connect4 import Connect4
from games.connect4.agents.negamax_agent import NegamaxAgent
from games.connect4.alphazero_v2 import packages as P
from games.connect4.alphazero_v2 import statistics as S
from games.connect4.alphazero_v2.arena import (ArenaCorrectnessError, GuardedUCTOpponent, NegamaxOpponent,
                                               RandomOpponent, StopEvaluation, run_paired_arena, scored)
from games.connect4.alphazero_v2.oracle import (BitboardSolver, BitboardState, SolverBudgetExceeded, board_key,
                                                engine_position, exhaustive_action_values, exhaustive_value,
                                                family_key, legal_moves_scan, mirror_moves, safe_moves_scan,
                                                tactical_label, winning_moves_scan)
from games.connect4.alphazero_v2.reference_negamax import (WIN_SCORE, ReferenceNegamaxAgent, State, Table,
                                                           minimax, negamax, root_scores)

ROOT = Path(__file__).resolve().parents[1]
DRAW = [2, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
        3, 3, 3, 3, 3, 3, 6, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5,
        6, 6, 6, 6, 6]


def random_history(rng, plies, *, allow_immediate_win=True):
    """A legal nonterminal history of exactly ``plies`` moves (retrying until one exists)."""
    while True:
        moves = []
        for _ in range(plies):
            game = engine_position(moves)
            if game.is_game_over():
                break
            moves.append(rng.choice(sorted(game.get_valid_moves())))
        if len(moves) == plies and not engine_position(moves).is_game_over():
            if allow_immediate_win or not winning_moves_scan(moves):
                return moves


def engine_winning_moves(moves):
    game = engine_position(moves)
    result = []
    for move in sorted(game.get_valid_moves()):
        child = Connect4(game.board, game.current_player)
        child.make_move(move)
        if child.check_winner() == game.current_player:
            result.append(move)
    return result


# Oracle -------------------------------------------------------------------------------------

def test_immediate_win_scans_agree_with_engine_and_bitboard():
    rng = random.Random(1)
    for _ in range(300):
        moves = random_history(rng, rng.randrange(0, 40))
        state = BitboardState.from_history(moves)
        bitboard = [c for c in range(7) if state.can_play(c) and state.is_winning_move(c)]
        assert winning_moves_scan(moves) == engine_winning_moves(moves) == bitboard
        assert legal_moves_scan(moves) == sorted(engine_position(moves).get_valid_moves())


@pytest.mark.parametrize('seed', range(4))
def test_independent_solvers_agree_on_late_positions(seed):
    rng = random.Random(100 + seed)
    for _ in range(6):
        moves = random_history(rng, rng.randrange(28, 38), allow_immediate_win=False)
        a = BitboardSolver().action_values(moves)
        assert a == exhaustive_action_values(moves)
        assert BitboardSolver().value(moves) == exhaustive_value(moves) == max(a.values())


def test_solver_values_respect_terminal_rules_and_mirrors():
    assert BitboardSolver().action_values(DRAW[:41]) == {6: 0}
    assert BitboardSolver().value([0, 1, 0, 1, 0, 1]) == 1          # X wins at once
    rng = random.Random(7)
    for _ in range(10):
        moves = random_history(rng, 30, allow_immediate_win=False)
        values = BitboardSolver().action_values(moves)
        mirrored = BitboardSolver().action_values(mirror_moves(moves))
        assert mirrored == {6 - a: v for a, v in values.items()}
    with pytest.raises(ValueError):
        BitboardSolver().value([0, 1, 0, 1, 0, 1, 0])


def test_solver_node_budget_is_enforced():
    with pytest.raises(SolverBudgetExceeded):
        BitboardSolver(max_nodes=50).value([3, 3])


def test_tactical_labels_follow_one_reply_rules():
    assert tactical_label([0, 1, 0, 1, 0, 2]) == ('unique_win', 0)
    assert tactical_label([0, 1, 0, 1, 2, 1]) == ('unique_safe', 1)
    assert tactical_label([]) == (None, None)
    assert safe_moves_scan([0, 1, 0, 1, 2, 1]) == [1]
    rng = random.Random(3)
    for _ in range(200):
        moves = random_history(rng, rng.randrange(6, 36))
        category, action = tactical_label(moves)
        if category is not None:
            row = P.tactical_row(moves, category, action)  # engine and scan labelers must agree
            assert row['expected_action'] == action


def test_family_keys_merge_reflections_and_transpositions():
    assert board_key([0, 1, 2, 3]) == board_key([2, 3, 0, 1])          # transposition
    assert board_key([0, 1, 2, 3]) == board_key([0, 3, 2, 1])          # another transposition
    assert board_key([0, 1, 2, 3]) != board_key([1, 0, 3, 2])          # colors swapped
    assert family_key([0, 1]) == family_key([6, 5])                    # reflection
    assert family_key([0]) != family_key([0, 1])                       # actor differs


# Reference Negamax ----------------------------------------------------------------------------

def legacy_scores(moves, depth, alpha, beta, agent):
    game = engine_position(moves)
    with contextlib.redirect_stdout(io.StringIO()):
        return agent.negamax(Connect4(game.board, game.current_player), depth, alpha, beta,
                             1 - 2 * game.current_player)


def test_review_cache_reproduction_legacy_wrong_reference_correct():
    moves = [5, 4, 3, 6, 2, 4]
    legacy = NegamaxAgent(3)
    assert legacy_scores(moves, 3, 0, 1, legacy) == (3, 5)
    assert legacy_scores(moves, 3, -math.inf, math.inf, legacy) == (4, 9)       # unsafe reuse
    assert legacy_scores(moves, 3, -math.inf, math.inf, NegamaxAgent(3)) == (3, 10)
    state, table = State.from_moves(moves), Table()
    assert negamax(state, 3, 0, 1, table) >= 1                                   # fail-high bound
    assert negamax(state, 3, -math.inf, math.inf, table) == minimax(state, 3) == 10


@pytest.mark.parametrize('depth', [1, 2, 3, 4])
def test_bound_typed_table_is_sound_under_arbitrary_window_sequences(depth):
    rng = random.Random(depth)
    for _ in range(12 if depth < 4 else 4):
        moves = random_history(rng, rng.randrange(0, 30))
        state, table = State.from_moves(moves), Table()
        exact = minimax(state, depth)
        for _ in range(6):
            a, b = sorted(rng.uniform(-60, 60) for _ in range(2))
            value = negamax(state, depth, a, b, table)
            if a < exact < b:
                assert value == exact
            elif exact <= a:
                assert value <= a
            else:
                assert value >= b
        assert negamax(state, depth, -math.inf, math.inf, table) == exact


def test_root_scores_are_exact_per_move_and_terminal_scores_dominate():
    rng = random.Random(11)
    for _ in range(10):
        moves = random_history(rng, rng.randrange(0, 25))
        state = State.from_moves(moves)
        for col, score in root_scores(state, 3).items():
            assert score == -minimax_after(state, col, 2)
    scores = root_scores(State.from_moves([0, 1, 0, 1, 0, 2]), 2)
    assert scores[0] == WIN_SCORE + 1 and all(abs(v) < WIN_SCORE for c, v in scores.items() if c != 0)
    assert root_scores(State.from_moves(DRAW[:41]), 3) == {6: 0}


def minimax_after(state, col, depth):
    state.play(col)
    try:
        return minimax(state, depth, col)
    finally:
        state.undo(col)


def test_reference_agent_prefers_faster_wins_blocks_and_never_mutates():
    # X can win now at column 0; a depth-4 search also sees slower wins but must take the fastest.
    game = engine_position([0, 1, 0, 1, 0, 2, 4, 2])
    before = [row[:] for row in game.board]
    assert ReferenceNegamaxAgent(4).choose_move(game) == 0
    assert [row[:] for row in game.board] == before
    # O must block X's vertical threat at column 0 at depth 2.
    assert ReferenceNegamaxAgent(2).choose_move(engine_position([0, 1, 0, 1, 0])) == 0
    moves = [3, 3, 2, 4]
    scores = root_scores(State.from_moves(moves), 2)
    tied = {c for c, v in scores.items() if v == max(scores.values())}
    picks = {ReferenceNegamaxAgent(2, rng=random.Random(s)).choose_move(engine_position(moves)) for s in range(60)}
    assert picks == tied
    assert ReferenceNegamaxAgent(2).choose_move(engine_position(moves)) == next(
        c for c in (3, 2, 4, 1, 5, 0, 6) if c in tied)  # center-first without an RNG


def test_legacy_negamax_agent_is_preserved_unchanged():
    result = subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--', 'games/connect4/agents/negamax_agent.py'],
                            cwd=ROOT)
    assert result.returncode == 0


# Statistics -------------------------------------------------------------------------------------

def games_for(results, families=None):
    families = families or [f"f{i // 2}" for i in range(len(results))]
    return [dict(family=f, stratum='prefix', agent_color=i % 2, result=r) for i, (f, r) in enumerate(zip(families, results))]


def test_bootstrap_is_deterministic_and_degenerate_cases_are_exact():
    wins = S.arena_summary(games_for(['win'] * 20), resamples=500)
    assert wins['score'] == 1 and wins['interval95'] == [1.0, 1.0] and wins['clusters'] == 10
    mixed = games_for(['win', 'loss', 'draw', 'win'] * 10)
    assert S.arena_summary(mixed, resamples=500) == S.arena_summary(mixed, resamples=500)
    summary = S.arena_summary(mixed, resamples=2000)
    assert summary['wins'] == 20 and summary['draws'] == 10 and summary['losses'] == 10
    assert summary['score'] == pytest.approx(25 / 40)
    low, high = summary['interval95']
    assert low < 25 / 40 < high
    assert summary['by_color']['X']['games'] == summary['by_color']['O']['games'] == 20


def test_empty_board_pairs_share_one_cluster():
    games = [dict(family='empty-board', stratum='empty', agent_color=i % 2, result='win') for i in range(8)]
    games += games_for(['loss'] * 4)
    summary = S.arena_summary(games, resamples=300)
    assert summary['clusters'] == 3 and summary['by_stratum']['empty']['score'] == 1


def test_paired_difference_requires_identical_families():
    a, b = games_for(['win'] * 4), games_for(['loss'] * 4)
    assert S.paired_difference(a, b, resamples=200)['difference'] == 1
    with pytest.raises(ValueError):
        S.paired_difference(a, games_for(['win'] * 2), resamples=200)


def test_arena_gate_rules():
    summary = dict(complete=True, score=0.7, interval95=[0.56, 0.8], by_color={'X': dict(score=0.8), 'O': dict(score=0.6)})
    rule = S.ACCEPTANCE_THRESHOLDS['arena']['negamax2']
    assert S.arena_gate(summary, rule)['passed']
    assert not S.arena_gate(dict(summary, complete=False), rule)['passed']
    assert not S.arena_gate(dict(summary, by_color={'X': dict(score=.95), 'O': dict(score=.45)}), rule)['passed']
    assert not S.arena_gate(dict(summary, interval95=[0.55, 0.8]), rule)['passed']           # strict
    random_rule = S.ACCEPTANCE_THRESHOLDS['arena']['random']
    assert S.arena_gate(dict(summary, score=.95, interval95=[.90, 1]), random_rule)['passed']  # >= .90


def test_frozen_thresholds_match_the_review():
    t = S.ACCEPTANCE_THRESHOLDS
    assert t['tactical'] == dict(immediate_win=.99, safe_response=.95, per_actor_immediate_win=.97,
                                 per_actor_safe_response=.90)
    assert t['solved'] == dict(optimal_preserving=.80, avoidable_loss_max=.05)
    assert (t['value']['class_balanced_mse_max'], t['value']['decisive_sign_min'], t['value']['draw_mae_max'],
            t['value']['wrong_sign_saturated_max'], t['value']['saturation']) == (.60, .85, .35, .05, .95)
    assert {k: (v['score'], v['lower_bound']) for k, v in t['arena'].items()} == {
        'random': (.95, .90), 'negamax1': (.80, .70), 'negamax2': (.65, .55), 'initial_v2_512': (.60, .50),
        'phase4d2f_512': (.60, .50), 'nn_only_vs_random': (.85, .75)}
    assert t['arena']['negamax2']['min_color_score'] == .50
    assert S.CHAMPION_GATE == dict(score=.55, lower_bound=.50, max_tactical_regression=.02, schedule=(5, 10, 15, 20))
    assert S.BOOTSTRAP_RESAMPLES == 10_000


def tactical_rows():
    rows = []
    for family, category, actor in (('w0', 'unique_win', 0), ('w1', 'unique_win', 1),
                                    ('s0', 'unique_safe', 0), ('s1', 'unique_safe', 1)):
        for orientation in ('base', 'mirror'):
            rows.append(dict(id=f'{family}-{orientation}', family=family, category=category, actor=actor,
                             expected_action=3))
    return rows


def test_tactical_metrics_average_seeds_then_rows_then_families():
    rows = tactical_rows()
    choices = {r['id']: [3, 3, 3, 3] for r in rows}
    choices['w0-base'] = [3, 3, 0, 0]          # family w0: (0.5 + 1) / 2 = .75
    choices['s1-mirror'] = [0, 0, 0, 0]        # family s1: (1 + 0) / 2 = .5
    m = S.tactical_metrics(rows, choices, {r['id']: [(1,) * 7] for r in rows})
    assert m['immediate_win'] == pytest.approx((0.75 + 1) / 2) and m['immediate_win_x'] == .75
    assert m['safe_response'] == pytest.approx(.75) and m['safe_response_o'] == .5
    assert m['zero_visit_winning_actions'] == 0
    decision = S.champion_decision(dict(complete=True, score=.6, interval95=[.52, .7]), m,
                                   dict(immediate_win=m['immediate_win'] + .021, safe_response=.75), True)
    assert not decision['promote'] and decision['checks']['tactical'] is False
    decision = S.champion_decision(dict(complete=True, score=.6, interval95=[.52, .7]), m,
                                   dict(immediate_win=m['immediate_win'] + .02, safe_response=.75), True)
    assert decision['promote']
    for arena in (dict(complete=False, score=.9, interval95=[.8, 1]), dict(complete=True, score=.54, interval95=[.51, .6]),
                  dict(complete=True, score=.6, interval95=[.5, .7])):
        assert not S.champion_decision(arena, m, m, True)['promote']
    assert not S.champion_decision(dict(complete=True, score=.9, interval95=[.8, 1]), m, m, False)['promote']


def solved_rows():
    return [dict(id='w-base', family='w', value=1, actor=0, action_values={'0': 1, '1': -1}),
            dict(id='w-mirror', family='w', value=1, actor=0, action_values={'6': 1, '5': -1}),
            dict(id='d-base', family='d', value=0, actor=1, action_values={'2': 0, '3': -1}),
            dict(id='l-base', family='l', value=-1, actor=1, action_values={'4': -1})]


def test_solved_decision_and_value_metrics():
    rows = solved_rows()
    choices = {'w-base': [0, 1], 'w-mirror': [6, 6], 'd-base': [3, 3], 'l-base': [4, 4]}
    m = S.solved_decision_metrics(rows, choices)
    assert m['optimal_preserving_win'] == pytest.approx(.75) and m['optimal_preserving_draw'] == 0
    assert m['optimal_preserving_loss'] == 1 and m['optimal_preserving'] == pytest.approx((.75 + 0 + 1) / 3)
    assert m['avoidable_loss'] == pytest.approx((.25 + 1) / 2)
    values = {'w-base': .5, 'w-mirror': -.96, 'd-base': .2, 'l-base': -1.0}
    v = S.value_metrics(rows, values)
    assert v['by_class']['win']['mse'] == pytest.approx((.25 + 1.96 ** 2) / 2)
    assert v['by_class']['win']['correct_sign'] == .5 and v['by_class']['loss']['correct_sign'] == 1
    assert v['by_class']['draw']['mae'] == pytest.approx(.2)
    assert v['wrong_sign_saturated_fraction'] == pytest.approx(1 / 3)
    assert not S.value_gate(v)['passed']


def test_calibration_summary_bins_and_game_clusters():
    records = [dict(game=g, value=0.9 if g % 2 else -0.9, outcome=1 if g % 2 else -1) for g in range(40)]
    summary = S.calibration_summary(records, constant=0.0, resamples=300)
    assert summary['games'] == 40 and summary['mse'] == pytest.approx(.01) and summary['mse_zero'] == 1
    assert summary['improvement_interval95'][0] > 0
    occupied = [b for b in summary['bins'] if b['positions']]
    assert [b['games'] for b in occupied] == [20, 20]
    assert S.calibration_gate(summary)['checks']['any_established_bin'] is False  # 20 < 30 games per bin


# Packages ---------------------------------------------------------------------------------------

@pytest.fixture(scope='module')
def tiny_packages(tmp_path_factory):
    directory = tmp_path_factory.mktemp('packages') / 'build'
    saved = {name: getattr(P, name) for name in ('EXCLUSION_SOURCES', 'TACTICAL_QUOTAS', 'SOLVED_QUOTAS',
                                                 'SOLVED_STAGES', 'OPENING_PREFIX_BASES', 'EMPTY_BOARD_PAIRS',
                                                 'OVERSAMPLE')}
    P.EXCLUSION_SOURCES = P.EXCLUSION_SOURCES[:3]  # in-repository sources only
    P.TACTICAL_QUOTAS, P.SOLVED_QUOTAS = dict(sealed=1, development=1), dict(sealed=1, development=1)
    P.SOLVED_STAGES = (('late', 24, 41),)
    P.OPENING_PREFIX_BASES, P.EMPTY_BOARD_PAIRS, P.OVERSAMPLE = dict(sealed=2, development=2), 1, 1
    try:
        P.build_all(directory, seed=7, log=lambda message: None)
    finally:
        for name, value in saved.items():
            setattr(P, name, value)
    return directory


def test_tiny_package_build_verifies_with_resolve_and_disjoint_splits(tiny_packages):
    summary = P.verify_directory(tiny_packages, resolve=True, log=lambda message: None)
    assert summary['tactical-sealed'] == 8 and summary['solved-sealed'] == 12
    families = {}
    for split in ('sealed', 'development'):
        families[split] = {family_key(r['moves']) for kind in ('tactical', 'solved', 'openings')
                           for r in P.load_package(tiny_packages / f'{kind}-{split}.json')['rows'] if r['moves']}
    assert not families['sealed'] & families['development']
    excluded = set(json.loads((tiny_packages / 'exclusions.json').read_text())['family_keys'])
    assert not (families['sealed'] | families['development']) & excluded
    for row in P.load_package(tiny_packages / 'solved-sealed.json')['rows']:
        if row['ply'] >= P.EXHAUSTIVE_MIN_PIECES and row['id'].endswith('base'):
            assert row['labels']['method_b'] is not None


def test_package_tampering_is_detected(tiny_packages, tmp_path):
    copy = tmp_path / 'copy'
    copy.mkdir()
    for path in tiny_packages.iterdir():
        (copy / path.name).write_bytes(path.read_bytes())
    document = json.loads((copy / 'tactical-sealed.json').read_text())
    document['rows'][0]['expected_action'] = (document['rows'][0]['expected_action'] + 1) % 7
    (copy / 'tactical-sealed.json').write_text(json.dumps(document))
    with pytest.raises(ValueError, match='hash'):
        P.load_package(copy / 'tactical-sealed.json')
    document['rows_sha256'] = P.rows_sha256(document['rows'])
    (copy / 'tactical-sealed.json').write_text(json.dumps(document))
    manifest = json.loads((copy / 'manifest.json').read_text())
    manifest['tactical-sealed']['sha256'] = P.file_sha256(copy / 'tactical-sealed.json')
    (copy / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='Tactical label'):
        P.verify_directory(copy, log=lambda message: None)


def test_frozen_packages_verify_structurally():
    if not (P.FROZEN_DIRECTORY / 'manifest.json').exists():
        pytest.skip('frozen packages not present')
    summary = P.verify_directory(P.FROZEN_DIRECTORY, log=lambda message: None)
    assert summary == {'openings-development': 100, 'openings-sealed': 100, 'tactical-development': 200,
                       'tactical-sealed': 800, 'solved-development': 300, 'solved-sealed': 600}


# Arena ------------------------------------------------------------------------------------------

OPENINGS = [dict(id='e0', moves=[], stratum='empty', family='empty-board'),
            dict(id='p0-base', moves=[0, 1], stratum='prefix', family='p0'),
            dict(id='p0-mirror', moves=[6, 5], stratum='prefix', family='p0')]


def test_paired_arena_swaps_sides_and_is_order_independent():
    records, status = run_paired_arena(NegamaxOpponent(2), RandomOpponent(), OPENINGS, namespace='t', seed=1)
    assert status['complete'] and len(records) == 6
    for opening in OPENINGS:
        pair = [r for r in records if r['opening_id'] == opening['id']]
        assert [r['agent_color'] for r in pair] == [0, 1]
        for record in pair:
            assert record['moves'][:len(opening['moves'])] == opening['moves']
            game = engine_position(record['moves'])
            assert game.is_game_over() and game.check_winner() == record['winner']
            expected = 'draw' if record['winner'] == -1 else ('win' if record['winner'] == record['agent_color'] else 'loss')
            assert record['result'] == expected
            owners = [d['color'] for d in record['decisions']]
            assert owners == [(len(opening['moves']) + i) % 2 for i in range(len(owners))]
    again, _ = run_paired_arena(NegamaxOpponent(2), RandomOpponent(), OPENINGS[1:], namespace='t', seed=1)
    strip = lambda rs: [{k: v for k, v in r.items() if k not in ('decisions', 'seconds')} for r in rs]  # noqa: E731
    assert strip(again) == strip(records[2:])


class Illegal:
    name = 'illegal'

    def describe(self):
        return {}

    def choose(self, game, rng):
        return 7, {}


class Mutator:
    name = 'mutator'

    def describe(self):
        return {}

    def choose(self, game, rng):
        move = sorted(game.get_valid_moves())[0]
        game.make_move(move)  # tries to corrupt the arena's game
        return sorted(game.get_valid_moves())[0], {}


def test_arena_rejects_illegal_moves_and_isolates_game_state():
    with pytest.raises(ArenaCorrectnessError):
        run_paired_arena(Illegal(), RandomOpponent(), OPENINGS[:1], namespace='t', seed=0)
    records, status = run_paired_arena(Mutator(), RandomOpponent(), OPENINGS[:1], namespace='t', seed=0)
    assert status['complete'] and all(engine_position(r['moves']).is_game_over() for r in records)


def test_arena_cooperative_stop_between_games_and_mid_game():
    calls = {'n': 0}

    def stop_after(limit):
        def check():
            calls['n'] += 1
            if calls['n'] > limit:
                raise StopEvaluation('budget')
        return check
    records, status = run_paired_arena(RandomOpponent(), RandomOpponent(), OPENINGS, namespace='t', seed=0,
                                       check=stop_after(3))
    assert not status['complete'] and records and records[-1]['abandoned']
    assert 'result' not in records[-1] and len(scored(records)) == len(records) - 1


def test_model_free_opponents_play_legally():
    for opponent in (RandomOpponent(), NegamaxOpponent(1), NegamaxOpponent(4), GuardedUCTOpponent(20)):
        move, _ = opponent.choose(engine_position([3, 3, 2]), random.Random(0))
        assert move in engine_position([3, 3, 2]).get_valid_moves()


def test_evaluation_core_imports_without_torch():
    script = '''
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] == "torch":
            raise AssertionError("torch import attempted: " + name)
sys.meta_path.insert(0, Block())
import games.connect4.alphazero_v2.oracle, games.connect4.alphazero_v2.reference_negamax
import games.connect4.alphazero_v2.packages, games.connect4.alphazero_v2.statistics
import games.connect4.alphazero_v2.arena, games.connect4.alphazero_v2.launch_control
import games.connect4.alphazero_v2.official_evidence
print("ok")
'''
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, timeout=60, cwd=ROOT)
    assert result.returncode == 0 and result.stdout.strip() == 'ok', result.stdout + result.stderr


def test_bootstrap_resamples_whole_opening_families_not_games():
    # Two perfectly correlated families: per-game resampling would give a narrow interval
    # around .5; resampling families must allow every mixture from all-loss to all-win.
    games = games_for(['win'] * 10, ['a'] * 10) + games_for(['loss'] * 10, ['b'] * 10)
    summary = S.arena_summary(games, resamples=2000)
    assert summary['clusters'] == 2 and summary['interval95'] == [0.0, 1.0]


def test_reference_ties_are_uniform_on_a_symmetric_board():
    moves = [3, 3, 3, 3, 3, 3]  # full center column; the board is mirror-symmetric
    scores = root_scores(State.from_moves(moves), 2)
    tied = {c for c, v in scores.items() if v == max(scores.values())}
    assert len(tied) >= 2 and tied == {6 - c for c in tied}
    picks = [ReferenceNegamaxAgent(2, rng=random.Random(s)).choose_move(engine_position(moves)) for s in range(80)]
    assert set(picks) == tied


@pytest.mark.parametrize('seed', range(3))
def test_solver_table_is_sound_under_window_sequences_and_reuse(seed):
    rng = random.Random(500 + seed)
    solver = BitboardSolver()  # one table reused across positions and windows
    for _ in range(8):
        moves = random_history(rng, rng.randrange(24, 32), allow_immediate_win=False)
        exact = exhaustive_value(moves)
        state = BitboardState.from_history(moves)
        for alpha, beta in ((0, 1), (-1, 0), (0, 1), (-1, 1)):
            value = solver._negamax(state, alpha, beta)
            if alpha < exact < beta:
                assert value == exact
            elif exact <= alpha:
                assert value <= alpha
            else:
                assert value >= beta
        assert solver.action_values(moves) == exhaustive_action_values(moves)


def test_candidate_pool_excludes_and_deduplicates_families():
    first = P.Pool(21, set())
    seen = [family_key(p) for _, p in zip(range(12), first.candidates(3, lambda p: len(p) >= 2))]
    assert len(set(seen)) == len(seen)                      # no duplicate family within a pool
    excluded = set(seen[:6])
    again = P.Pool(21, excluded)                            # same seed: would otherwise yield them first
    produced = [family_key(p) for _, p in zip(range(12), again.candidates(3, lambda p: len(p) >= 2))]
    assert not excluded & set(produced)


def test_safe_moves_scan_counts_an_immediate_win_as_safe():
    # X to move: X wins at column 0 while O threatens column 6; only the win (and the block) is safe.
    moves = [0, 6, 0, 6, 0, 6]
    assert winning_moves_scan(moves) == [0]
    assert sorted(safe_moves_scan(moves)) == [0, 6]


def test_tactical_metrics_weight_families_equally_when_sizes_differ():
    rows = [dict(id='a1', family='a', category='unique_win', actor=0, expected_action=0),
            dict(id='a2', family='a', category='unique_win', actor=0, expected_action=0),
            dict(id='a3', family='a', category='unique_win', actor=0, expected_action=0),
            dict(id='b1', family='b', category='unique_win', actor=0, expected_action=0)]
    choices = {'a1': [0], 'a2': [0], 'a3': [0], 'b1': [1]}
    assert S.tactical_metrics(rows, choices)['immediate_win'] == .5   # not 3/4
