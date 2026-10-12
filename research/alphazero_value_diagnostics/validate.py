"""Focused correctness checks, with no training and no measured endpoint selection."""
import math
import time
from games.connect4.alphazero_v2.bitboard import BitboardState
from games.connect4.alphazero_v2.oracle import engine_position, winning_moves_scan, safe_moves_scan
from games.connect4.agents.mcts_nn_agent import Node, backup, terminal_value
from .common import ARMS, VARIANTS, RESULTS, load, runtime, write
from .interventions import Intervention
from .evaluate import fixtures, search


def main():
    runtime()
    started=time.monotonic()
    checks = []
    for arm in ARMS:
        runner, model, base, entry = load(arm, 6)
        weight = runner.factory.hash(model)
        histories = ([], [0], [0, 0, 0, 0, 0, 0], [0, 1, 0, 1, 0, 2])
        states = [BitboardState.from_game(engine_position(m)) for m in histories]
        for variant in VARIANTS:
            wrapper = Intervention(base, variant)
            assert wrapper.predict_batch([]) == ()
            raw, modified = base.predict_batch(states), wrapper.predict_batch(states)
            for state, p, q in zip(states, raw, modified):
                assert q.value == p.value * wrapper.scale
                assert all(q.policy[a] == 0 for a in range(7) if a not in state.get_valid_moves())
                assert math.isclose(sum(q.policy), 1, abs_tol=1e-14)
                if wrapper.uniform:
                    assert len({q.policy[a] for a in state.get_valid_moves()}) == 1
                else:
                    assert q.policy == p.policy and q.logits == p.logits
                single = wrapper.predict(state)
                assert wrapper.predict_batch([state])[0] == single
                assert abs(single.value - q.value) < 2e-6
                assert max(abs(a-b) for a,b in zip(single.policy,q.policy)) < 2e-6
            terminal = BitboardState.from_game(engine_position([0, 1, 0, 1, 0, 1, 0]))
            for call in (lambda: wrapper.predict(terminal), lambda: wrapper.predict_batch([terminal])):
                try:
                    call()
                except ValueError:
                    pass
                else:
                    raise AssertionError('Terminal neural evaluation accepted')
            # A simple terminal win has value -1 for the now-to-move loser, and +1 at its parent.
            node = Node(terminal)
            assert terminal_value(terminal) == -1
            parent = Node(states[-1]); node.parent = parent
            backup(node, -1)
            assert node.value_sum == -1 and parent.value_sum == 1
            checks.append(f'{arm}:{variant}:single-batch-prior-value-terminal-sign')
        # All variants must take the unique terminal win even after changing nonterminal values.
        row = dict(id='correctness-only-terminal', moves=[0,1,0,1,0,1], cluster='test',
                   category='unique_win', expected_action=0, safe_actions=list(range(7)), immediate_wins=[0])
        for variant in VARIANTS:
            assert search(Intervention(base,variant), 'tactical', row, 4404901)['action'] == 0
        assert runner.factory.hash(model) == weight
    for suite in ('solved','deep','tactical'):
        rows = fixtures()[suite]
        for row in rows:
            game = engine_position(row['moves'])
            assert not game.is_game_over()
            if suite == 'tactical':
                assert winning_moves_scan(row['moves']) == row['immediate_wins']
                assert safe_moves_scan(row['moves']) == row['safe_actions']
            else:
                assert set(map(int,row['action_values'])) == set(game.get_valid_moves())
                assert row['value'] == max(row['action_values'].values())
                assert set(row['optimal_actions']) == {int(a) for a,v in row['action_values'].items() if v == row['value']}
        checks.append(f'{suite}:all-nonterminal-legal-labels')
    # ID-paired row-weighted bootstrap respects unequal cluster sizes and duplicated rows.
    from .analyze import estimate, pair
    a=[dict(id='a',cluster='one',m=1),dict(id='b',cluster='one',m=1),dict(id='c',cluster='two',m=0)]
    b=[dict(id=r['id'],cluster=r['cluster'],m=0) for r in reversed(a)]
    assert estimate(a,[1,1,0])['mean']==2/3 and pair(a,b,'m')['mean']==2/3
    assert pair(a,a,'m')['interval95']==[0.,0.]
    checks.append('paired-id-bootstrap-unequal-clusters')
    write(RESULTS / 'validation.json', dict(passed=True, checks=checks,seconds=time.monotonic()-started))
    print(f'{len(checks)} focused groups passed')

if __name__ == '__main__':
    main()
