"""Read-only independent arithmetic/board review; no implementation imports or evaluation."""
import json
import hashlib
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / 'results'
REPO = HERE.parents[1]
ARMS = ('42-cnn', '42-resnet3x64', '314159-cnn', '314159-resnet3x64')


def read(path):
    return json.loads(path.read_text())


def win(board):
    for r in range(6):
        for c in range(7):
            if board[r][c] < 0:
                continue
            for dr, dc in ((1,0),(0,1),(1,1),(1,-1)):
                if (0 <= r+3*dr < 6 and 0 <= c+3*dc < 7
                        and all(board[r+i*dr][c+i*dc] == board[r][c] for i in range(4))):
                    return board[r][c]
    return None


def legal(board):
    return [c for c in range(7) if board[5][c] < 0]


def drop(board, col, actor):
    assert col in legal(board)
    result = [r[:] for r in board]
    row = next(r for r in range(6) if result[r][col] < 0)
    result[row][col] = actor
    return result


def replay(moves, terminal=False):
    board = [[-1]*7 for _ in range(6)]
    for ply, col in enumerate(moves):
        assert win(board) is None and legal(board)
        board = drop(board, col, ply % 2)
    assert (win(board) is not None or not legal(board)) == terminal
    return board, len(moves) % 2


def fixtures_audit(fixtures):
    result = {}
    for suite in ('tactical','solved','deep'):
        for row in fixtures[suite]:
            b, actor = replay(row['moves'])
            moves = legal(b)
            if 'actor' in row:
                assert row['actor'] == actor
            if suite == 'tactical':
                wins = [c for c in moves if win(drop(b,c,actor)) == actor]
                threats = [c for c in moves if win(drop(b,c,1-actor)) == 1-actor]
                safe = []
                for c in moves:
                    after = drop(b,c,actor)
                    if win(after) == actor or not any(win(drop(after,d,1-actor)) == 1-actor for d in legal(after)):
                        safe.append(c)
                assert moves == row['legal'] and wins == row['immediate_wins']
                assert threats == row['opponent_threats'] and safe == row['safe_actions']
                assert row['expected_action'] == (wins[0] if wins else safe[0])
                assert row['category'] == ('unique_win' if len(wins) == 1 else 'unique_safe')
            else:
                vector = {int(c):v for c,v in row['action_values'].items()}
                assert sorted(vector) == moves and max(vector.values()) == row['value']
                assert set(row['optimal_actions']) == {c for c,v in vector.items() if v == row['value']}
                assert all(vector[c] == 1 for c in moves if win(drop(b,c,actor)) == actor)
        result[suite] = dict(rows=len(fixtures[suite]),clusters=len({r['cluster'] for r in fixtures[suite]}))
    return result


def interval(rows, values):
    groups = defaultdict(lambda:[0.,0])
    for row,v in zip(rows,values):
        groups[row['cluster']][0] += v
        groups[row['cluster']][1] += 1
    totals = np.array([groups[k] for k in sorted(groups)])
    rng = np.random.default_rng(20261012)
    counts = rng.multinomial(len(totals), np.ones(len(totals))/len(totals),size=10000)
    samples = (counts @ totals[:,0]) / (counts @ totals[:,1])
    return dict(mean=float(np.mean(values)),interval95=np.quantile(samples,[.025,.975]).tolist(),clusters=len(totals))


def main():
    assert read(RESULTS/'status.json')['status'] == 'COMPLETE'
    fixtures = read(REPO/'research/alphazero_resnet_selfplay/fixtures.json')
    result = dict(fixture_audit=fixtures_audit(fixtures),positions={},matches={},values={},pooled={})
    data = {}
    for arm in ARMS:
        result['positions'][arm] = {}
        result['matches'][arm] = {}
        for variant in 'ABCDE':
            result['positions'][arm][variant] = {}
            for suite in ('tactical','solved','deep'):
                rows = read(RESULTS/'tasks'/f'{arm}-g6-{variant}-positions'/f'{suite}.json')['rows']
                label = {r['id']:r for r in fixtures[suite]}
                assert len(rows) == len(label) and {r['id'] for r in rows} == set(label)
                for row in rows:
                    q = label[row['id']]
                    b,_ = replay(q['moves']);moves = legal(b);action = row['action']
                    assert action in moves and row['cluster'] == q['cluster']
                    assert len(row['visits']) == 7 and sum(row['visits']) == 256
                    assert row['visits'][action] == max(row['visits'])
                    assert all(row['visits'][c] == 0 and row['root_prior'][c] == 0 for c in range(7) if c not in moves)
                    assert abs(sum(row['root_prior'])-1) < 1e-12
                    if suite == 'tactical':
                        correct = int(action == q['expected_action'])
                        assert row['avoidable'] == int(action not in q['safe_actions'])
                        assert row['missed_win'] == int(bool(q['immediate_wins']) and action not in q['immediate_wins'])
                    else:
                        chosen = q['action_values'][str(action)];target=q['value']
                        correct=int(action in q['optimal_actions'])
                        assert row['mistake'] == int(chosen < target)
                        assert row['win_draw'] == int(target == 1 and chosen == 0)
                        assert row['win_loss'] == int(target == 1 and chosen == -1)
                        assert row['draw_loss'] == int(target == 0 and chosen == -1)
                    assert row['correct'] == correct
                    row['correct'] = correct
                data[arm,variant,suite] = {r['id']:r for r in rows}
                base=data[arm,'A',suite]
                for row in rows:
                    a=base[row['id']]
                    if variant in 'ABE':assert row['root_prior'] == a['root_prior']
                    else:assert all(abs(row['root_prior'][c]-1/len(legal(replay(label[row['id']]['moves'])[0]))) < 1e-12 for c in legal(replay(label[row['id']]['moves'])[0]))
                    if variant in 'AC':assert row['root_value'] == a['root_value']
                    elif variant in 'BD':assert row['root_value'] == 0
                    else:assert row['root_value'] == a['root_value']*.5
                result['positions'][arm][variant][suite]=dict(accuracy=interval(rows,[r['correct'] for r in rows]),
                    delta=interval(rows,[r['correct']-base[r['id']]['correct'] for r in rows]),
                    fixed=sum(r['correct'] and not base[r['id']]['correct'] for r in rows),
                    introduced=sum(not r['correct'] and base[r['id']]['correct'] for r in rows))
                if suite == 'tactical':
                    result['positions'][arm][variant][suite]['categories'] = {
                        category: dict(accuracy=interval([r for r in rows if r['category']==category],
                            [r['correct'] for r in rows if r['category']==category]),
                            delta=interval([r for r in rows if r['category']==category],
                            [r['correct']-base[r['id']]['correct'] for r in rows if r['category']==category]))
                        for category in ('unique_win','unique_safe')}
            records=read(RESULTS/'tasks'/f'{arm}-g6-{variant}-matches'/'matches.json')['records']
            matchrows=[]
            for r in records:
                b,_=replay(r['moves'],True);winner=win(b)
                expected='draw' if winner is None else ('win' if winner == r['agent_color'] else 'loss')
                assert r['result']==expected and not r['abandoned']
                assert r['score']=={'win':1.,'draw':.5,'loss':0.}[expected]
                matchrows.append(dict(id=f'{r["opening_id"]}:{r["game_index"]}',cluster=r['family'],score=r['score']))
            assert len(records)==80 and len({r['family'] for r in records})==40
            assert all(sorted(r['agent_color'] for r in records if r['family']==f)==[0,1] for f in {r['family'] for r in records})
            data[arm,variant,'matches']={r['id']:r for r in matchrows}
            base=data[arm,'A','matches']
            result['matches'][arm][variant]=dict(wdl=dict(Counter(r['result'] for r in records)),
                score=interval(matchrows,[r['score'] for r in matchrows]),
                delta=interval(matchrows,[r['score']-base[r['id']]['score'] for r in matchrows]))
        raw=read(RESULTS/'tasks'/f'{arm}-values'/'values.json')
        result['values'][arm]={}
        for kind in ('training_g6','training_g5','prequential_g5_on_g6'):
            rows=raw[kind]
            result['values'][arm][kind]=dict(n=len(rows),mse=interval(rows,[(r['value']-r['target'])**2 for r in rows]),
                zero_mse=float(np.mean([r['target']**2 for r in rows])))
        result['values'][arm]['exact']={suite:dict(mse=interval(rows,[(r['value']-r['target'])**2 for r in rows]),
            zero_mse=float(np.mean([r['target']**2 for r in rows])),mean_value=float(np.mean([r['value'] for r in rows]))) for suite,rows in raw['exact'].items()}
    formulas={'zero_value_learned_prior':{'B':1,'A':-1},'zero_value_uniform_prior':{'D':1,'C':-1},
        'uniform_prior_learned_value':{'C':1,'A':-1},'uniform_prior_zero_value':{'D':1,'B':-1},
        'interaction':{'A':1,'B':-1,'C':-1,'D':1},'half_value':{'E':1,'A':-1}}
    for arch in ('cnn','resnet3x64'):
        arms=[f'{seed}-{arch}' for seed in (42,314159)]
        result['pooled'][arch]={}
        for suite in ('tactical','solved','deep','matches'):
            metric='score' if suite=='matches' else 'correct'
            rows=list(data[arms[0],'A',suite].values())
            result['pooled'][arch][suite]={name:interval(rows,[sum(sum(c*data[a,v,suite][r['id']][metric] for v,c in formula.items()) for a in arms)/2 for r in rows]) for name,formula in formulas.items()}
    main=read(RESULTS/'analysis.json')
    diffs=[]
    for arch,suites in result['pooled'].items():
        for suite,contrasts in suites.items():
            for name,r in contrasts.items():diffs.append(abs(r['mean']-main['pooled'][arch][suite]['contrasts'][name]['mean']))
    result['comparison']=dict(headline_contrasts=len(diffs),max_mean_discrepancy=max(diffs),
        independent_rng=20261012,bootstrap='10,000 multinomial family multiplicities, paired row-weighted means')
    assert max(diffs)<1e-12
    (RESULTS/'independent-recomputed.json').write_text(json.dumps(result,indent=1,sort_keys=True)+'\n')
    print(json.dumps(result['comparison']))


if __name__=='__main__':
    main()
