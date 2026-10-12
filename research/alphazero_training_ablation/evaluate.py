"""Raw development evaluations, independent of all training RNG streams."""
import math
import time
from .common import ROOT, read, write
from games.connect4.alphazero_v2.data import seeded_rng
from games.connect4.alphazero_v2.evaluation import nn_only_choice, V2SearchArenaAgent
from games.connect4.alphazero_v2.arena import paired_game
from games.connect4.alphazero_v2.search import EvaluationAgent
from games.connect4.alphazero_v2.oracle import engine_position
from games.connect4.alphazero_v2.network import V2Inference, weights_sha256

class RawBitboardAgent(V2SearchArenaAgent):
    def choose(self, game, rng):
        result, selection = EvaluationAgent(self.inference, self.simulations, rng=rng, backend='bitboard').search(game)
        if sum(result.visits) != 512: raise ValueError('Evaluation simulation budget differs')
        return selection.move, dict(visits=list(result.visits), raw_value=result.root_value)

def positions(model, check, output):
    fixtures = read(ROOT / 'fixtures.json')
    inference = V2Inference(model)
    records = {}
    before = weights_sha256(model)
    for suite in ('solved', 'tactical', 'deep'):
        rows = []
        for row in fixtures[suite]:
            check()
            prediction = inference.predict(engine_position(row['moves']))
            action = nn_only_choice(inference, row, namespace='phase4d4g-policy')
            record = dict(id=row['id'], cluster=row['cluster'], family=row['family'], policy=list(prediction.policy),
                          value=prediction.value, policy_action=action)
            if suite in ('solved', 'deep'):
                mass = sum(prediction.policy[a] for a in row['optimal_actions'])
                record.update(ce=-math.log(max(mass,1e-300)), optimal_mass=mass,
                    accuracy=int(action in row['optimal_actions']), mse=(prediction.value-row['value'])**2,
                    policy_deep_mistake=int(row['action_values'][str(action)] < row['value']))
            if suite == 'tactical':
                record.update(category=row['category'], policy_correct=int(action == row['expected_action']),
                    policy_avoidable=int(action not in row['safe_actions']),
                    policy_missed_win=int(bool(row['immediate_wins']) and action not in row['immediate_wins']))
            # A single fixed search seed per position, shared by seeds/arms/generations.
            rng = seeded_rng(f'phase4d4g-evaluation:{suite}:{row["id"]}', fixtures['search_seed'])
            result, selection = EvaluationAgent(inference,512,rng=rng,backend='bitboard').search(engine_position(row['moves']))
            if sum(result.visits) != 512: raise ValueError('Evaluation simulation budget differs')
            record.update(puct_action=selection.move, visits=list(result.visits))
            if suite == 'tactical':
                record.update(puct_correct=int(selection.move == row['expected_action']),
                              puct_avoidable=int(selection.move not in row['safe_actions']),
                              puct_missed_win=int(bool(row['immediate_wins']) and selection.move not in row['immediate_wins']))
            else:
                record['puct_deep_mistake'] = int(row['action_values'][str(selection.move)] < row['value'])
            rows.append(record)
            # Durable partial rows are explicitly incomplete and never passed to analysis.
            if len(rows) % 25 == 0: write(output / f'{suite}-partial.json', dict(complete=False,rows=rows))
        records[suite] = rows
        write(output / f'{suite}.json', dict(complete=True,rows=rows))
    if before != weights_sha256(model): raise ValueError('Evaluation mutated model')
    return records

def matches(model, parent, seed, check, output):
    fixtures = read(ROOT / 'fixtures.json')
    before = (weights_sha256(model), weights_sha256(parent))
    agent = RawBitboardAgent(model,512,name='candidate')
    opponent = RawBitboardAgent(parent,512,name='parent')
    records = []
    for row in fixtures['openings']:
        for game_index in (0,1):
            check()
            record = paired_game(agent, opponent, row, game_index, namespace=f'phase4d4g-strength-seed-{seed}',
                seed=fixtures['match_seed'],check=check,keep_decisions=True)
            if record['abandoned']: raise RuntimeError('Abandoned strength game')
            records.append(record)
            write(output / 'matches-partial.json', dict(complete=False,records=records))
    if before != (weights_sha256(model), weights_sha256(parent)): raise ValueError('Match mutated model')
    write(output / 'matches.json',dict(complete=True,records=records))
