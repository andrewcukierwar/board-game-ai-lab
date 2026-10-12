"""Replay-distribution analysis of candidate datasets; never trains or resumes.

Everything here describes sampling exposure only. Duplicated positions still
carry separately searched targets and outcomes, so "distinct exposure" is a
diversity proxy, not a measure of gradient quality or playing strength.
"""
from collections import Counter
import math
import random
import statistics

import torch

from games.connect4.alphazero_v2.config import V2Config
from games.connect4.alphazero_v2.data import (GenerationReplay, V2Example, encode_board, finalize_game,
                                             pre_move_ply, seeded_rng)
from games.connect4.alphazero_v2.diagnostics import entropy, family
from games.connect4.alphazero_v2.generation import rebuild_game
from games.connect4.alphazero_v2.oracle import engine_position
from research.alphazero_baseline import audit
from . import analysis as a, schedules as s

SAMPLING_DOMAIN = 'phase4d4d-counterfactual-replay-sampling'
RAREFACTION_SEED = 4404450
ENTROPY_BINS = ((0, .1), (.1, .3), (.3, .6), (.6, 1.), (1., 1.5), (1.5, math.inf))
MULTIPLICITY_BINS = ((1, 1), (2, 2), (3, 4), (5, 9), (10, 19), (20, math.inf))


def expected_distinct(counts, draws):
    """E[distinct states] in ``draws`` independent uniform draws over the examples."""
    n = sum(counts)
    return sum(1 - (1 - c / n) ** draws for c in counts)


def rarefaction(games, sizes, *, permutations=200, seed=RAREFACTION_SEED):
    """Mean unique positions among the first M examples over random whole-game orders."""
    rng = random.Random(seed)
    sizes = sorted(sizes)
    sums = {size: [0, 0] for size in sizes}
    games = sorted(games, key=lambda g: g['id'])
    for _ in range(permutations):
        exact, canon, seen, pending = set(), set(), 0, list(sizes)
        for game in rng.sample(games, len(games)):
            for p in game['plies']:
                exact.add(p['key'])
                canon.add(p['canon'])
                seen += 1
                if pending and seen == pending[0]:
                    sums[pending.pop(0)][0] += len(exact)
                    sums[seen][1] += len(canon)
            if not pending:
                break
        if pending:
            raise ValueError('Rarefaction size exceeds the dataset')
    return {str(size): dict(unique_positions=exact / permutations, unique_mirror_canonical=canon / permutations,
                            unique_per_example=exact / permutations / size)
            for size, (exact, canon) in sums.items()}


def profile(games, *, draws, rarefaction_sizes):
    """Replay-relevant description of one dataset at an externally fixed sampling budget."""
    rows = list(a.examples(games))
    late = [p for p in rows if p['ply'] >= 8]
    exact, canon = Counter(p['key'] for p in rows), Counter(p['canon'] for p in rows)
    late_canon = Counter(p['canon'] for p in late)
    multiplicity = {}
    for low, high in MULTIPLICITY_BINS:
        label = str(low) if low == high else f'{low}+' if high == math.inf else f'{low}-{high}'
        members = [c for c in canon.values() if low <= c <= high]
        multiplicity[label] = dict(states=len(members), examples=sum(members), example_share=sum(members) / len(rows))
    ordered = sorted(canon.values(), reverse=True)
    orientations = Counter()
    for key in exact:
        orientations[a.canonical(key)] += 1
    symmetric = sum(a.mirror_bits(k[0]) == k[0] and a.mirror_bits(k[1]) == k[1] for k in exact)
    targets = [[v / s.SIMULATIONS for v in p['visits']] for p in rows]
    entropies = [entropy(t) for t in targets]
    counts = a.total(games, 'tactics')
    labels = Counter(p['z'] for p in rows)
    by_stage = {}
    for name, low, high in s.STAGES:
        stage = Counter(p['z'] for p in a.examples(games, low, high))
        n = sum(stage.values())
        by_stage[name] = {k: (stage[v] / n if n else None) for k, v in (('win', 1), ('draw', 0), ('loss', -1))}
    return dict(
        examples=len(rows), unique_positions=len(exact), unique_mirror_canonical=len(canon),
        duplicated_example_share=sum(c for c in canon.values() if c > 1) / len(rows),
        max_multiplicity=ordered[0], multiplicity=multiplicity,
        concentration=dict(all=a.concentration(canon), plies_8_plus=a.concentration(late_canon),
                           top10_states_share=sum(ordered[:10]) / len(rows),
                           top1_percent_states_share=sum(ordered[:max(1, len(ordered) // 100)]) / len(rows)),
        mirror=dict(both_orientations_present=sum(n == 2 for n in orientations.values()),
                    self_symmetric_positions=symmetric, canonical_reduction=1 - len(canon) / len(exact)),
        tactical=dict(win_opportunity_share=counts['win_opps'] / len(rows),
                      mandatory_block_share=counts['block_states'] / len(rows),
                      choice_state_share=counts['choice_states'] / len(rows),
                      tactical_state_share=counts['tactical_states'] / len(rows),
                      tactical_examples=counts['tactical_states']),
        target_entropy=dict(a.describe(entropies), near_one_hot_share=sum(max(t) >= .9 for t in targets) / len(rows),
                            histogram={f'{low:g}-{high:g}': sum(low <= h < high for h in entropies) / len(rows)
                                       for low, high in ENTROPY_BINS}),
        outcome_labels=dict(win=labels[1] / len(rows), draw=labels[0] / len(rows), loss=labels[-1] / len(rows),
                            mean_first_player=statistics.fmean(p['z'] for p in rows if p['ply'] % 2 == 0),
                            mean_second_player=statistics.fmean(p['z'] for p in rows if p['ply'] % 2 == 1),
                            by_stage=by_stage,
                            game_results=dict(x=sum(g['winner'] == 0 for g in games) / len(games),
                                              o=sum(g['winner'] == 1 for g in games) / len(games),
                                              draw=sum(g['winner'] == -1 for g in games) / len(games))),
        rarefaction=rarefaction(games, rarefaction_sizes),
        equal_budget=dict(draws=draws, expected_distinct_states=expected_distinct(canon.values(), draws),
                          expected_distinct_per_draw=expected_distinct(canon.values(), draws) / draws,
                          expected_distinct_states_plies_8_plus_given_late_draws=expected_distinct(
                              late_canon.values(), round(draws * len(late) / len(rows)))))


# Archived official storage (read-only; never a runner, load_boundary or resume) ---------------

def archived_payload(campaign, seed, generation=20):
    path = audit.checkpoint(campaign, seed, generation).with_name(f'generation-{generation:04d}.resume.pt')
    return torch.load(path, map_location='cpu', weights_only=True)


def archived_games(payload, generation, seed):
    """One archived generation in the analysis structure (original schedule, no priors)."""
    stored = next(g for g in payload['replay']['generations'] if g['generation'] == generation)
    return [a.build_game((seed, f'archived-g{generation}', 0, game['index']), s.ORIGINAL, game['moves'], game['winner'],
                         [dict(visits=v, action=m, temperature=t) for m, v, t in
                          zip(game['moves'], game['visits'], game['action_temperatures'])])
            for game in stored['games']]


def archived_window(payload, generations):
    """{generation: CompletedGame tuple} through the project's own storage validation."""
    config = V2Config.from_dict(payload['config'])
    return {stored['generation']: tuple(rebuild_game(
        stored['generation'], game['index'], game['moves'], game['visits'], game['action_temperatures'],
        game['winner'], config.self_play_simulations, config.exploratory_plies) for game in stored['games'])
        for stored in payload['replay']['generations'] if stored['generation'] in generations}


def completed_games(games, generation):
    """Candidate games as CompletedGame objects, reindexed 0..n-1 in identity order."""
    result = []
    for index, game in enumerate(sorted(games, key=lambda g: g['id'])):
        engine, pending = engine_position([]), []
        for p in game['plies']:
            pending.append(V2Example(encode_board(engine), engine.current_player, pre_move_ply(engine),
                                     p['visits'], p['action'], p['temperature']))
            engine.make_move(p['action'])
        result.append(finalize_game(generation, index, game['moves'], pending))
    return tuple(result)


def counterfactual(window, candidate, *, updates, seeds, new_generation=21):
    """Archived window plus a PARTIAL candidate generation under the existing replay rules.

    Uses the real GenerationReplay (window/max-games enforcement, uniform
    batches without replacement inside a batch). Counterfactual exposure only.
    """
    config = V2Config()
    replay = GenerationReplay(config.replay_generations, config.replay_max_games)
    for generation in sorted(window):
        replay.add_generation(generation, window[generation])
    evicted = replay.add_generation(new_generation, candidate)
    states, sources, late = [], [], []
    for position in range(len(replay)):
        generation, _, ply = replay.address(position)
        states.append(family(replay.example(position)))
        sources.append(generation)
        late.append(ply >= 8)
    archive = {state for state, source in zip(states, sources) if source != new_generation}
    new = [i for i, source in enumerate(sources) if source == new_generation]
    novel = [i for i in new if states[i] not in archive]
    late_new = [i for i in new if late[i]]
    late_novel = [i for i in novel if late[i]]
    runs = []
    for seed in seeds:
        rng = seeded_rng(SAMPLING_DOMAIN, seed)
        drawn = []
        for _ in range(updates):
            positions, _ = replay.sample(config.batch_size, rng)
            drawn.extend(positions)
        from_new = [i for i in drawn if sources[i] == new_generation]
        runs.append(dict(draws=len(drawn), distinct_states=len({states[i] for i in drawn}),
                         distinct_addresses=len(set(drawn)), new_generation_draws=len(from_new),
                         new_generation_distinct_states=len({states[i] for i in from_new}),
                         new_generation_novel_draws=sum(states[i] not in archive for i in from_new),
                         novel_states_drawn=len({states[i] for i in from_new if states[i] not in archive})))
    summary = {key: dict(mean=statistics.fmean(r[key] for r in runs),
                         stdev=statistics.stdev(r[key] for r in runs) if len(runs) > 1 else None)
               for key in runs[0]}
    return dict(
        counterfactual=True, partial_generation_games=len(candidate),
        declared_games_per_generation=config.games_per_generation,
        window_generations=list(replay.generations), evicted_generations=list(evicted),
        window_games=replay.games, window_positions=len(replay),
        window_unique_states=len(set(states)), archive_unique_states=len(archive),
        candidate_examples=len(new), candidate_unique_states=len({states[i] for i in new}),
        candidate_novel_examples=len(novel), candidate_novel_states=len({states[i] for i in novel}),
        candidate_novel_states_per_1000_examples=1000 * len({states[i] for i in novel}) / len(new),
        candidate_examples_plies_8_plus=len(late_new),
        candidate_novel_states_plies_8_plus=len({states[i] for i in late_novel}),
        candidate_novel_example_share_plies_8_plus=len(late_novel) / len(late_new) if late_new else None,
        updates=updates, batch_size=config.batch_size, sampling_seeds=list(seeds),
        existing_rule_updates_for_this_candidate=config.updates_for(len(new)),
        expected_distinct_states_analytic=expected_distinct(Counter(states).values(), updates * config.batch_size),
        sampling=summary)
