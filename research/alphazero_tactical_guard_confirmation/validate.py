"""Focused prelaunch correctness of the FROZEN 4D.4J-A guard (never modified here).

Reference: independently authored matrix rules (4D.4J-A independent_verify.ReferenceBoard).
Positions: rules-only random continuations of the new confirmation openings
(no model, no outcomes) plus designed edge cases. Writes results/validation.json.
"""
import random, time, ast
from copy import deepcopy
from collections import Counter
from games.connect4.connect4 import Connect4
from games.connect4.alphazero_v2.bitboard import BitboardState
from games.connect4.alphazero_v2.search import EvaluationAgent
from games.connect4.alphazero_v2.oracle import engine_position
from research.alphazero_tactical_guard.guard import guard
from research.alphazero_tactical_guard.evaluate import decision
from research.alphazero_tactical_guard.independent_verify import ReferenceBoard, reference_selection, self_check
from .common import runtime, load, state, write, read, RESULTS, GUARD, GUARD_SHA256, sha


def bitboard(ref): return BitboardState.from_game(Connect4([list(r) for r in ref.cells], ref.actor))
def mirror(ref): return ReferenceBoard(tuple(tuple(reversed(r)) for r in ref.cells), ref.actor)
def swap(ref): return ReferenceBoard(tuple(tuple({'X': 'O', 'O': 'X', ' ': ' '}[v] for v in r) for r in ref.cells), 1 - ref.actor)


def main():
    runtime(); start = time.time(); self_check(); checks, coverage = Counter(), Counter()
    assert sha(GUARD) == GUARD_SHA256
    imports = {n.module for n in ast.walk(ast.parse(GUARD.read_text())) if isinstance(n, ast.ImportFrom)}
    assert imports == {'numbers', 'time', 'games.connect4.alphazero_v2.bitboard'}, imports  # no model, RNG or search import
    checks['static_import_isolation'] += 1

    def verify(ref):
        root = bitboard(ref); before = deepcopy(root); legal = list(ref.legal()); c = ref.classify()
        coverage[f'actor_{ref.actor}'] += 1
        coverage['immediate_win'] += bool(c['immediate_winning_actions']); coverage['multiple_wins'] += len(c['immediate_winning_actions']) > 1
        coverage['all_unsafe'] += not c['safe_actions']; coverage['full_column'] += len(legal) < 7
        coverage['unsafe_present'] += bool(c['unsafe_actions'] and c['safe_actions'])
        for original in legal:
            for vector in ([256 if a == original else 0 for a in range(7)], [1 if a in legal else 0 for a in range(7)],
                           [(7 * a + 3) % 11 if a in legal else 0 for a in range(7)]):
                saved = list(vector); chosen, reason, cl = reference_selection(ref, vector, original)
                r = guard(root, vector, original)
                assert (r['final_action'], r['reason']) == (chosen, reason)
                assert r['immediate_wins'] == cl['immediate_winning_actions'] and r['safe_actions'] == cl['safe_actions']
                assert r['unsafe_actions'] == cl['unsafe_actions'] and r['opponent_winning_replies'] == cl['opponent_winning_replies']
                assert r['final_action'] in legal and vector == saved == r['original_visits'] and root == before
                if reason == 'UNCHANGED': assert r['final_action'] == original and original in cl['safe_actions']
                if reason == 'NO_SAFE_MOVE': assert r['final_action'] == original and not cl['safe_actions']
                if reason == 'IMMEDIATE_WIN': assert r['final_action'] in cl['immediate_winning_actions']
                if reason == 'AVOID_IMMEDIATE_LOSS': assert original in cl['unsafe_actions'] and r['final_action'] in cl['safe_actions']
                # Proof obligation A on every replacement: an unsafe non-winning original loses to the witnessed reply.
                if r['changed'] and reason == 'AVOID_IMMEDIATE_LOSS':
                    after = ref.play(original); reply = cl['opponent_winning_replies'][str(original)][0]
                    assert after.winner() == -1 and after.play(reply).winner() == 1 - ref.actor
                    checks['replacement_original_forced_loss'] += 1
                checks[f'reason_{reason}'] += 1; checks['reference_decisions'] += 1

    rng = random.Random(4420005); rows = read(RESULTS / 'openings.json')['confirmation']; sampled = 0
    for row in rows:  # rules-only continuations: two per opening at random depths
        for _ in range(2):
            ref = ReferenceBoard.from_moves(row['moves'])
            for _ in range(rng.randrange(0, 30)):
                if ref.terminal(): break
                ref = ref.play(rng.choice(ref.legal()))
            if ref.terminal():
                try: guard(bitboard(ref), [0] * 7, 0)
                except ValueError: checks['terminal_rejection'] += 1
                else: raise AssertionError('Terminal accepted')
                continue
            for variant in (ref, mirror(ref), swap(ref), swap(mirror(ref))): verify(variant)
            sampled += 1
    designed = [ReferenceBoard.from_moves([c, (c + 1) % 7, c, (c + 1) % 7, c, (c + 2) % 7]) for c in range(7)]
    cells = [[' '] * 7 for _ in range(6)]; cells[5] = list(' OOO XX')
    designed.append(ReferenceBoard(tuple(tuple(r) for r in cells), 0))  # all-unsafe double threat
    for ref in designed:
        for variant in (ref, mirror(ref), swap(ref)): verify(variant)
    draw = [2,0,0,0,0,0,0,1,1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,3,6,4,4,4,4,4,4,5,5,5,5,5,5,6,6,6,6,6]
    verify(ReferenceBoard.from_moves(draw[:-1]))  # last-cell draw successor counted safe
    for terminal in [ReferenceBoard.from_moves(draw)] + [ref.play(ref.classify()['immediate_winning_actions'][0]) for ref in designed[:7]]:
        try: guard(bitboard(terminal), [0] * 7, 0)
        except ValueError: checks['terminal_rejection'] += 1
        else: raise AssertionError('Terminal accepted')
    for bad in ([256, 0, 0, 0, 0, 0], [True] + [0] * 6, [-1] + [0] * 6):
        try: guard(bitboard(ReferenceBoard.empty()), bad, 0)
        except ValueError: checks['malformed_visits_rejected'] += 1
        else: raise AssertionError('Malformed visits accepted')
    assert all(coverage[k] > 0 for k in ('actor_0', 'actor_1', 'immediate_win', 'multiple_wins', 'all_unsafe', 'full_column', 'unsafe_present'))
    # Original search, neural outputs, RNG streams, model and learner state preserved; guard adds no RNG/inference.
    for seed in (42, 314159):
        runner, model, inference, entry = load(seed); before = state(runner); weights = runner.factory.hash(model)
        for row in rows[::32]:
            game = engine_position(row['moves']); board = deepcopy(game.board); prediction = inference.predict(game)
            r1, r2, r3 = [random.Random(4420006) for _ in range(3)]
            canonical, selection = EvaluationAgent(inference, 256, rng=r1, backend='bitboard').search(game)
            _, disabled = decision(inference, game, r2, False); _, enabled = decision(inference, game, r3, True)
            assert disabled['original_action'] == selection.move == enabled['original_action']
            assert list(canonical.visits) == disabled['visits'] == enabled['visits'] == enabled['guard']['original_visits']
            assert disabled['root_value'] == enabled['root_value'] == canonical.root_value
            assert disabled['root_prior'] == enabled['root_prior'] == list(canonical.root_prior)
            assert disabled['root_logits'] == enabled['root_logits'] == list(canonical.root_logits)
            assert r1.getstate() == r2.getstate() == r3.getstate()
            assert prediction == inference.predict(game) and game.board == board
            checks['puct_model_rng_preservation'] += 1
        assert before == state(runner) and weights == runner.factory.hash(model)
    write(RESULTS / 'validation.json', dict(disposition='PASS', guard_sha256=GUARD_SHA256, checks=checks, coverage=coverage,
        sampled_continuation_positions=sampled, seconds=time.time() - start,
        reference='4D.4J-A independently authored top-first matrix directional scan (no bitboard rules)'))
    print('Validation PASS', dict(checks), dict(coverage))


if __name__ == '__main__':
    main()
