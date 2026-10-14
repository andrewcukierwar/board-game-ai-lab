"""Cost-only pilot plus blinded opponent calibration; the guard is never run.

Calibration uses only the 16 calibration families (never confirmation openings)
and only the CONTROL CNN, so it cannot depend on the guard comparison.
"""
import time, statistics
from games.connect4.alphazero_v2.arena import paired_game, NegamaxOpponent
from games.connect4.alphazero_v2.data import seeded_rng
from games.connect4.alphazero_v2.oracle import engine_position
from games.connect4.alphazero_v2.reference_negamax import ReferenceNegamaxAgent
from research.alphazero_tactical_guard.common import load, state, runtime
from research.alphazero_tactical_guard.evaluate import Agent
from research.alphazero_tactical_guard.independent_verify import ReferenceBoard
from .common import RESULTS, read, write

NAMESPACE = 'phase4d4j-b-calibration'
SEED = 4420002


def score(record, color_of_interest):
    w = record['winner']
    return .5 if w == -1 else float(w == color_of_interest)


def late_roots(n):
    """Random-playout roots at fixed occupancy bins (rules only)."""
    rng, roots = seeded_rng('phase4d4j-b-cost-roots', SEED), []
    for occupancy in (4, 10, 16, 22, 28, 34):
        made = 0
        while made < n:
            board, moves = ReferenceBoard.empty(), []
            while len(moves) < occupancy and not board.terminal():
                m = rng.choice(board.legal()); moves.append(m); board = board.play(m)
            if len(moves) == occupancy and not board.terminal():
                roots.append(moves); made += 1
    return roots


def main():
    runtime(); start = time.time()
    openings = read(RESULTS/'openings.json')['calibration']
    assert len(openings) == 16
    out = dict(cost_only_pilot={}, calibration={}, guard_used=False, confirmation_openings_used=False)
    # Opponent cost per decision by occupancy (cost only).
    for depth in (3, 4, 5):
        rows = []
        for moves in late_roots(4):
            game = engine_position(moves); t = time.perf_counter()
            ReferenceNegamaxAgent(depth, rng=seeded_rng('cost', SEED)).choose_move(game)
            rows.append(dict(occupancy=len(moves), seconds=time.perf_counter() - t))
        out['cost_only_pilot'][f'negamax{depth}'] = dict(mean=statistics.mean(r['seconds'] for r in rows),
            worst=max(r['seconds'] for r in rows), by_occupancy={o: statistics.mean(r['seconds'] for r in rows if r['occupancy'] == o)
                                                              for o in sorted({r['occupancy'] for r in rows})})
    # Candidate strength versus Negamax3 (model-free).
    for depth in (4, 5):
        records = [paired_game(NegamaxOpponent(depth), NegamaxOpponent(3), row, i, namespace=NAMESPACE + f':h2h{depth}', seed=SEED,
                               keep_decisions=False) for row in openings for i in (0, 1)]
        s = [score(r, r['agent_color']) for r in records]
        out['calibration'][f'negamax{depth}_vs_negamax3'] = dict(games=len(s), score=sum(s)/len(s),
            mean_plies=statistics.mean(len(r['moves']) for r in records))
    # Control CNN (guard disabled) versus candidates: saturation check plus cost.
    for seed in (42, 314159):
        runner, model, inference, entry = load(seed); before = state(runner); weights = runner.factory.hash(model)
        agent = Agent(inference, False)
        for depth in (3, 4, 5):
            t = time.perf_counter(); records = []
            for row in openings[::2]:
                for i in (0, 1):
                    records.append(paired_game(agent, NegamaxOpponent(depth), row, i, namespace=NAMESPACE + f':cnn{depth}', seed=SEED, keep_decisions=True))
            seconds = time.perf_counter() - t
            agent_t = [d['seconds'] for r in records for d in r['decisions'] if d['color'] == r['agent_color']]
            opp_t = [d['seconds'] for r in records for d in r['decisions'] if d['color'] != r['agent_color']]
            entry_ = out['calibration'].setdefault(f'control_vs_negamax{depth}', dict(games=0, score_sum=0., seconds=0., agent_decisions=0, opponent_decisions=0, agent_seconds=0., opponent_seconds=0.))
            entry_['games'] += len(records); entry_['score_sum'] += sum(score(r, r['agent_color']) for r in records)
            entry_['seconds'] += seconds; entry_['agent_decisions'] += len(agent_t); entry_['opponent_decisions'] += len(opp_t)
            entry_['agent_seconds'] += sum(agent_t); entry_['opponent_seconds'] += sum(opp_t)
        assert before == state(runner) and weights == runner.factory.hash(model)
    for depth in (3, 4, 5):
        e = out['calibration'][f'control_vs_negamax{depth}']
        e['score'] = e['score_sum']/e['games']; e['seconds_per_game'] = e['seconds']/e['games']
        # Primary design: 2 seeds x 2 arms x 96 families x 2 colors = 768 games per opponent; plus 2x1256 regression searches.
        e['projected_768_games_cpu_seconds'] = 768*e['seconds_per_game']
    a = out['calibration']['control_vs_negamax3']['projected_768_games_cpu_seconds']
    regression = 2*1256*out['calibration']['control_vs_negamax3']['agent_seconds']/out['calibration']['control_vs_negamax3']['agent_decisions']
    rule = read(RESULTS/'opponent-selection-rule.json'); choice = None; eligibility = {}
    for depth in rule['candidates_in_preference_order']:
        h2h = out['calibration'][f'negamax{depth}_vs_negamax3']['score']; c = out['calibration'][f'control_vs_negamax{depth}']
        total = a + c['projected_768_games_cpu_seconds'] + regression
        eligibility[depth] = dict(i=h2h >= .55, ii=.10 <= c['score'] <= .90, iii=total <= 1800, projected_total_cpu_seconds=total)
    for test in (lambda e: e['i'] and e['ii'] and e['iii'], lambda e: e['i'] and e['iii']):
        choice = choice or next((d for d in rule['candidates_in_preference_order'] if test(eligibility[d])), None)
    out.update(eligibility=eligibility, opponent_b_depth=choice or 4, regression_cpu_seconds=regression, seconds=time.time() - start,
               note='calibration outcomes are opponent-selection evidence only; no guard was run and no confirmation opening was played')
    write(RESULTS/'calibration.json', out)
    print(out['eligibility'], 'chosen', out['opponent_b_depth'], round(out['seconds'], 1))


if __name__ == '__main__':
    main()
