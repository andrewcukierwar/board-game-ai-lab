"""Identical original search, inference, seeds, and opponent in both variants."""
from time import perf_counter
from games.connect4.alphazero_v2.bitboard import BitboardState
from games.connect4.alphazero_v2.search import EvaluationAgent
from games.connect4.alphazero_v2.data import seeded_rng
from games.connect4.alphazero_v2.oracle import engine_position
from games.connect4.alphazero_v2.arena import paired_game, NegamaxOpponent
from research.alphazero_resnet_selfplay.evaluate import fixtures, openings, SEARCH_NAMESPACE, MATCH_NAMESPACE
from .guard import guard
from .common import write


def decision(inference, game, rng, enabled):
    root=BitboardState.from_game(game)
    before=perf_counter()
    result, selection=EvaluationAgent(inference,256,rng=rng,backend='bitboard').search(root)
    search_seconds=perf_counter()-before
    assert sum(result.visits)==256 and result.root_noise is None and selection.temperature==0
    diagnostic=guard(root,result.visits,selection.move) if enabled else None
    final=diagnostic['final_action'] if diagnostic else selection.move
    return final,dict(contract='connect4-guarded-decision-v1', original_action=selection.move,
        final_action=final,visits=list(result.visits), root_value=result.root_value,
        root_prior=list(result.root_prior),root_logits=list(result.root_logits),
        search_seconds=search_seconds,guard=diagnostic,
        total_seconds=perf_counter()-before, occupancy=root.occupied.bit_count())


class Agent:
    def __init__(self,inference,enabled):self.inference,self.enabled=inference,enabled;self.name='guarded' if enabled else 'control'
    def choose(self,game,rng):return decision(self.inference,game,rng,self.enabled)
    def describe(self):return dict(name=self.name,simulations=256,guard=self.enabled)


def positions(inference,seed,check,output):
    for suite,rows in fixtures().items():
        if suite not in ('tactical','solved','deep'):continue
        records=[]
        for row in rows:
            check(); game=engine_position(row['moves']); pair={}
            for variant,enabled in [('control',False),('guarded',True)]:
                rng=seeded_rng(f'{SEARCH_NAMESPACE}:{suite}:{row["id"]}',4404901)
                action, info=decision(inference,game,rng,enabled)
                if suite=='tactical':
                    info.update(correct=int(action==row['expected_action']),
                        avoidable=int(bool(row['safe_actions']) and action not in row['safe_actions']),
                        missed_win=int(bool(row['immediate_wins']) and action not in row['immediate_wins']))
                else:
                    value,chosen=row['value'],row['action_values'][str(action)]
                    info.update(correct=int(action in row['optimal_actions']),exact=value,chosen_value=chosen,
                        mistake=int(chosen<value),win_draw=int(value==1 and chosen==0),
                        win_loss=int(value==1 and chosen==-1),draw_loss=int(value==0 and chosen==-1))
                pair[variant]=info
            for field in ('original_action','visits','root_value','root_prior','root_logits'):
                assert pair['control'][field]==pair['guarded'][field],field
            records.append(dict(id=row['id'],cluster=row['cluster'],category=row.get('category'),**pair))
            if len(records)%50==0:write(output/f'{suite}-progress.json',dict(completed=len(records),total=len(rows)))
        write(output/f'{suite}.json',dict(complete=True,seed=seed,rows=records))


def matches(inference,seed,check,output,variant,rows=None,progress=True):
    records=[];agent=Agent(inference,variant=='guarded')
    for row in openings() if rows is None else rows:
        for index in (0,1):
            check()
            record=paired_game(agent,NegamaxOpponent(3),row,index,namespace=MATCH_NAMESPACE,seed=4404902,
                               check=check,keep_decisions=True)
            assert not record['abandoned']
            record['score']={'win':1.,'draw':.5,'loss':0.}[record['result']]
            records.append(record)
            if progress and output is not None:write(output/'progress.json',dict(games=len(records),total=80,variant=variant))
    if output is not None:write(output/f'{variant}-matches.json',dict(complete=True,seed=seed,variant=variant,records=records))
    return records
