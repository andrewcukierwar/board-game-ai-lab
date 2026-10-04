"""Predetermined inference-only diagnostics and a single shared evaluation deadline.

Frozen tactical labels annotate measurements only. Nothing here enters training.
Temperature-zero MCTS uses the foundation's seeded maximum-visit tie convention.
"""
from collections import Counter
from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
import math
from pathlib import Path
import random
import time

from .agents.negamax_agent import NegamaxAgent
from .agents.random_agent import RandomAgent
from .agents.mcts_nn_agent import MCTSNNAgent
from .neural_mcts import NeuralInference

SNAPSHOT_GAMES = (50, 100, 150)
FROZEN_SUITES = {
    'diagnostic': 'ec99e0ee080370ab8cfc678c00a49ff552466de73ffe0955778ac87ee340543a',
    'validation': '1c79db45a09e00a8e8806448362eb319b5e9c81e431de03c7b5ece8b2d04bb2f',
    'replication': '8e63fd8be0419ddb4daf83ae226362f26999dced321617900938d61a60efb7da',
}
# Explicit, legal varied prefixes, paired reflections, both even and odd length.
OPENINGS = ((3, 2), (3, 4), (0, 3, 2), (6, 3, 4),
            (2, 4, 3, 2), (4, 2, 3, 4))
OPPONENTS = ('random', 'negamax1', 'negamax2')
MODES = ('nn_only', 'raw_mcts')
EVALUATION_CONFIG = dict(max_seconds=180, openings=OPENINGS, sides=(0, 1),
    opponents=OPPONENTS, modes=MODES, models=('initial', 'final'),
    opponent_seed=420202, search_seed=420203, simulations=32, exploration=1.41,
    mcts_temperature=0, tactical_guard=False, root_noise=False, learning=False,
    planned_games=144, schedule='opponent, opening, mode, side, model',
    nn_ties='lowest legal column', mcts_ties='seeded maximum-visit ties',
    negamax='existing agent, fresh cache per game, existing tie order')


def greedy(inference, game):
    policy = inference.predict_legal(game).policy
    return max(game.get_valid_moves(), key=lambda a: (policy[a], -a))


class SeededRandom:
    """Existing RandomAgent with a per-game RNG, restoring process RNG each call."""
    def __init__(self, seed):
        self.rng = random.Random(seed)
        self.agent = RandomAgent()

    def choose_move(self, game):
        state = random.getstate()
        try:
            random.setstate(self.rng.getstate())
            action = self.agent.choose_move(game)
            self.rng.setstate(random.getstate())
            return action
        finally:
            random.setstate(state)


def frozen_fixtures():
    from .neural_self_play import require, position
    suites = {}
    for name, expected_hash in FROZEN_SUITES.items():
        path = Path(__file__).with_name('dqn') / f'{name}_positions.json'
        require(hashlib.sha256(path.read_bytes()).hexdigest() == expected_hash, 'Frozen suite changed')
        rows = json.loads(path.read_text())['positions']
        for row in rows:
            game = position(row['moves'])
            require(not game.is_game_over() and game.current_player == row['player'], 'Invalid frozen fixture')
        suites[name] = rows
    return suites


def summarize(rows):
    """Concentration, tactical action accuracy and exact mirror-pair behavior."""
    summary = {}
    for mode, action_key in (('nn_only', 'action'), ('mcts', 'modal_action'), ('mcts_sampled', 'action')):
        key = 'mcts' if mode == 'mcts_sampled' else mode
        tactical = [r for r in rows if r['tactical_required']]
        actions = [r[key][action_key] for r in rows]
        pairs = []
        by_name = {r['name']: r for r in rows}
        for row in rows:
            original_name = row.get('mirror_of') or (row['name'].removesuffix('_mirror')
                                                    if row['name'].endswith('_mirror') else None)
            if original_name in by_name:
                original = by_name[original_name]
                pairs.append(dict(name=row['name'], action_reflected=row[key][action_key] == 6-original[key][action_key],
                    policy_l1=sum(abs(a-b) for a, b in zip(row[key].get('legal_policy', row[key].get('policy')),
                                  reversed(original[key].get('legal_policy', original[key].get('policy'))))),
                    value_l1=sum(abs(a-b) for a,b in zip(row[key]['predicted_wdl'], original[key]['predicted_wdl']))))
        summary[mode] = dict(tactical_correct=sum(r[key][action_key] in r['tactical_required'] for r in tactical),
            tactical_total=len(tactical), actions=dict(Counter(actions)),
            action_concentration=max(Counter(actions).values())/len(rows),
            mean_policy_entropy=sum(r[key]['entropy'] for r in rows)/len(rows),
            mean_max_policy=sum(r[key]['max_probability'] for r in rows)/len(rows),
            mirror_pairs=pairs)
    wdl = [r['nn_only']['predicted_wdl'] for r in rows]
    wins = [r['nn_only']['predicted_wdl'] for r in rows if r['immediate_wins']]
    summary['value'] = dict(mean_wdl=[sum(p[i] for p in wdl)/len(wdl) for i in range(3)],
        min_wdl=[min(p[i] for p in wdl) for i in range(3)], max_wdl=[max(p[i] for p in wdl) for i in range(3)],
        mean_max_probability=sum(max(p) for p in wdl)/len(wdl),
        saturated_099=sum(bool(max(p) >= .99) for p in wdl),
        mean_entropy=sum(-sum(x*math.log(x) for x in p if x > 0) for p in wdl)/len(wdl),
        known_win_count=len(wins), known_win_mean_probability=sum(p[0] for p in wins)/len(wins) if wins else None,
        known_win_brier=sum((1-p[0])**2+p[1]**2+p[2]**2 for p in wins)/len(wins) if wins else None,
        known_win_nll=sum(-math.log(max(p[0], 1e-300)) for p in wins)/len(wins) if wins else None,
        calibration_scope='Only immediate wins have proven game-theoretic WDL labels; blocks do not')
    return summary


def snapshot(inference, report):
    from .neural_self_play import diagnostics, require
    fixed = diagnostics(inference)
    suites = {}
    for name, fixtures in frozen_fixtures().items():
        rows = diagnostics(inference, positions=tuple((r['name'], tuple(r['moves'])) for r in fixtures))
        for row, fixture in zip(rows, fixtures):
            row.update(category=fixture['category'], direction=fixture['direction'],
                       mirror_of=fixture['mirror_of'], expected_action=fixture['expected_action'])
            if fixture['expected_action'] is not None:
                require(row['tactical_required'] == [fixture['expected_action']], 'Frozen tactical annotation differs')
        suites[name] = dict(sha256=FROZEN_SUITES[name], previously_inspected=True,
                           rows=rows, summary=summarize(rows))
    recent = report['losses'][-10:]
    return dict(completed_games=report['completed_games'], updates=report['updates'],
        loss_last10={k: sum(r[k] for r in recent)/len(recent) if recent else None
                     for k in ('policy_loss', 'value_loss', 'combined_loss', 'policy_target_entropy')},
        fixed_rows=fixed, fixed_summary=summarize(fixed), frozen_suites=suites,
        training_feedback=False, search_measurement_temperature=1,
        modal_action='lowest-index maximum visits; distinct from temperature-1 sampled action')


def evaluation_schedule():
    for opponent in OPPONENTS:
        for opening_index, opening in enumerate(OPENINGS):
            for mode in MODES:
                for side in (0, 1):
                    for model in ('initial', 'final'):
                        yield dict(model=model, mode=mode, opponent=opponent, side=side,
                                   opening_index=opening_index, opening=opening,
                                   opponent_seed=420202+opening_index*2+side,
                                   search_seed=420203+opening_index*2+side)


def evaluate(models, *, max_seconds=180, clock=time.monotonic, chooser_factory=None, opponent_factory=None):
    """One shared cooperative deadline. Complete records and quarantined partials.

    Factories enable synthetic deadline/legal-history tests without learned games.
    """
    from .neural_self_play import require, position
    require(type(max_seconds) in (int, float) and math.isfinite(max_seconds) and 0 < max_seconds <= 180,
            'Evaluation budget must be positive and <= 180 seconds')
    require(set(models) == {'initial', 'final'}, 'Evaluation requires the predeclared model pair')
    start = clock()
    games, partial, timings = [], None, []
    for spec in evaluation_schedule():
        if clock()-start >= max_seconds:
            break
        inference = models[spec['model']]
        require(not inference.model.training, 'Evaluation model must be in eval mode')
        if chooser_factory:
            choose = chooser_factory(inference, spec)
        elif spec['mode'] == 'nn_only':
            choose = lambda game: greedy(inference, game)
        else:
            choose = MCTSNNAgent(inference, 32, temperature=0, exploration=1.41,
                                tactical_guard=False, rng=random.Random(spec['search_seed'])).choose_move
        opponent = (opponent_factory(spec) if opponent_factory else
                    SeededRandom(spec['opponent_seed']) if spec['opponent'] == 'random' else
                    NegamaxAgent(int(spec['opponent'][-1])))
        game = position(spec['opening'])
        require(not game.is_game_over(), 'Evaluation opening is terminal')
        moves = list(spec['opening'])
        record = dict(spec, moves=moves)
        while not game.is_game_over():
            if clock()-start >= max_seconds:
                break
            before = deepcopy(game.__dict__)
            turn_start = clock()
            with redirect_stdout(io.StringIO()):
                action = choose(game) if game.current_player == spec['side'] else opponent.choose_move(game)
            elapsed = clock()-turn_start
            require(game.__dict__ == before, 'Evaluation mutated caller state')
            require(type(action) is int and action in game.get_valid_moves(), 'Illegal evaluation move')
            if clock()-start >= max_seconds:
                break
            timings.append(dict(model=spec['model'], mode=spec['mode'], opponent=spec['opponent'],
                                model_turn=game.current_player == spec['side'], seconds=elapsed))
            require(game.make_move(action), 'Evaluation engine rejected legal action')
            moves.append(action)
            require(len(moves) <= 42, 'Evaluation exceeded legal ply count')
        if not game.is_game_over():
            partial = dict(record, disposition='deadline; excluded from W/L/D')
            break
        winner = game.check_winner()
        replay = position(moves)
        require(replay.__dict__ == game.__dict__ and replay.is_game_over(), 'Evaluation replay mismatch')
        games.append(dict(record, winner=winner, plies=len(moves),
                          result='draw' if winner == -1 else 'win' if winner == spec['side'] else 'loss'))
    counts = []
    for model in ('initial', 'final'):
        for mode in MODES:
            for opponent in OPPONENTS:
                for side in (0, 1):
                    subset = [g for g in games if (g['model'],g['mode'],g['opponent'],g['side']) == (model,mode,opponent,side)]
                    counts.append(dict(model=model, mode=mode, opponent=opponent, side=side,
                        completed=len(subset), **{r:sum(g['result']==r for g in subset) for r in ('win','loss','draw')}))
    return dict(config=dict(EVALUATION_CONFIG, max_seconds=max_seconds), games=games, counts=counts,
                completed_games=len(games), planned_games=144, partial_game=partial,
                status='complete' if len(games)==144 else 'deadline_limited',
                elapsed_seconds=clock()-start, move_timings=timings,
                budget_scope='combined initial/final, NN Only/raw MCTS, all opponents and sides')
