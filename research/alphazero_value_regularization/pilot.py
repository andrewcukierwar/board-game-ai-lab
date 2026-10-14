"""Cost-only disposable work; no model quality is retained or used."""
import time
from games.connect4.alphazero_v2.experimental import restore_rng
from games.connect4.alphazero_v2.search import EvaluationAgent
from games.connect4.alphazero_v2.oracle import engine_position
from games.connect4.alphazero_v2.data import seeded_rng
from research.alphazero_resnet_selfplay.evaluate import fixtures, openings, RawAgent, MATCH_NAMESPACE
from research.alphazero_resnet_selfplay.measures import score_examples
from games.connect4.alphazero_v2.arena import paired_game, NegamaxOpponent
from .common import runtime, write, RESULTS
from .checkpoint import initial, sample
from .objective import step


def main():
    start = time.perf_counter()
    runtime()
    costs = {}
    for arm in ('control','treatment'):
        r,_ = initial(42)
        t = time.perf_counter()
        for _ in range(12):
            batch,_ = sample(r)
            step(r,batch,arm)
        costs[arm+'_update_seconds'] = (time.perf_counter()-t)/12
    r,_ = initial(42)
    inference = r.factory.inference(r.model)
    rows = fixtures()
    t=time.perf_counter()
    for suite in ('tactical','solved','deep'):
        for row in rows[suite][::max(1,len(rows[suite])//4)][:4]:
            EvaluationAgent(inference,256,rng=seeded_rng('phase4d4i-b-cost-only',7),backend='bitboard').search(engine_position(row['moves']))
    costs['search_seconds_per_root']=(time.perf_counter()-t)/12
    t=time.perf_counter()
    agent=RawAgent(inference,256,name='cost-only',weights_sha256='discarded')
    for color in (0,1):
        paired_game(agent,NegamaxOpponent(3),openings()[0],color,namespace=MATCH_NAMESPACE,seed=4404902,keep_decisions=False)
    costs['match_seconds_per_game']=(time.perf_counter()-t)/2
    t=time.perf_counter()
    examples=[e for g in r.replay.iter_games() for e in g.examples]
    score_examples(r.model,examples)
    costs['full_training_value_seconds']=time.perf_counter()-t
    costs['positions']=len(examples)
    # One sequential process: both seeds x both arms, initial shared once per seed,
    # two nonzero milestones. 10 evaluations, 628 roots + 80 matches each.
    evaluation=10*(628*costs['search_seconds_per_root']+80*costs['match_seconds_per_game']+1.5*costs['full_training_value_seconds'])
    update=2*(costs['control_update_seconds']+costs['treatment_update_seconds'])
    # Discrete cost-only choice from fixed candidates; target ~24 minutes with 15% contingency.
    choices=[512,1024,2048,3072,4096,5120]
    projections={n:dict(training_seconds=n*update,evaluation_seconds=evaluation,
        expected_seconds=n*update+evaluation+120,contingency_seconds=1.15*(n*update+evaluation)+240) for n in choices}
    feasible=[n for n in choices if projections[n]['contingency_seconds']<=2400]
    if not feasible:
        raise RuntimeError('No prospectively acceptable compute budget')
    selected=min(feasible,key=lambda n:abs(projections[n]['expected_seconds']-1440))
    write(RESULTS/'pilot.json',dict(cost_only=True,quality_retained=False,costs=costs,
        candidate_updates=choices,projections=projections,selected_updates=selected,
        wall_seconds=time.perf_counter()-start))
    print('Cost-only pilot complete; fixed candidate',selected,flush=True)

if __name__=='__main__': main()
