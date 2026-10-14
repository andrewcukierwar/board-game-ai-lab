"""Bounded cost only; never retain endpoint scores or actions."""
import time
from .common import runtime,load,write,RESULTS,state
from .evaluate import fixtures,openings,decision,matches
from games.connect4.alphazero_v2.data import seeded_rng
from games.connect4.alphazero_v2.oracle import engine_position

def main():
    runtime();start=time.time();rows=[]
    for seed in (42,314159):
        runner,model,inference,entry=load(seed);before=state(runner);weights=runner.factory.hash(model)
        clock=time.perf_counter()
        for suite in ('tactical','solved','deep'):
            for row in fixtures()[suite][::max(1,len(fixtures()[suite])//2)][:2]:
                decision(inference,engine_position(row['moves']),seeded_rng('guard-cost-only',4404901),True)
        root=(time.perf_counter()-clock)/6
        clock=time.perf_counter();matches(inference,seed,lambda:None,None,'guarded',rows=openings()[:1],progress=False)
        game=(time.perf_counter()-clock)/2
        assert state(runner)==before and runner.factory.hash(model)==weights
        rows.append(dict(seed=seed,seconds_per_root=root,seconds_per_game=game))
    projected=sum(r['seconds_per_root']*1256+r['seconds_per_game']*160 for r in rows)
    write(RESULTS/'pilot.json',dict(cost_only=True,seconds=time.time()-start,rows=rows,primary_projected_seconds=projected,
        contingency_seconds=2*projected+300,secondary='OMITTED: prioritize primary and independent review',
        roots=12,matches=4,no_outcomes_retained=True))
    print('Cost-only pilot complete')
if __name__=='__main__':main()
