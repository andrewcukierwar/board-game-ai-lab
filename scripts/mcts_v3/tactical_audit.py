"""Depth-limited tactical audit of recorded study games.

    python -m scripts.mcts_v3.tactical_audit --study confirm_primary \
        --matchups f1-time-400 f1-time-2000 --pairs 128 --depth 8 \
        --output docs/search-mcts-v3/tactical-audit/audit.json

Every recorded decision is re-scored with exact depth-limited Negamax, which
proves wins and losses that complete within ``depth`` plies. A decision is a
proven blunder when some move was not proven lost and the move played is. This
is a diagnostic of what each agent still misses, not a strength estimate:
errors beyond the horizon and all positional errors are invisible to it.
"""
import argparse
import json
from pathlib import Path

from games.connect4.agents.negamax_agent import WIN_SCORE, NegamaxAgent
from scripts.evaluate_mcts_strength import atomic_json, load_rows
from scripts.mcts_v3.analysis import role_of
from scripts.mcts_v3.harness import ROOT, position, validate_rows

EXAMPLES_PER_KIND = 4


def classify(scores, move, depth):
    """Return (kind, plies) for one decision from exact depth-limited scores.

    A root score of -(WIN_SCORE + d) means the opponent completes four
    ``depth - 1 - d`` plies after the move; +(WIN_SCORE + d) means the mover
    does so ``depth - 1 - d`` plies after it (0 = the move itself wins).
    """
    best, chosen = max(scores.values()), scores[move]
    if best <= -WIN_SCORE:
        return 'already_lost', None
    if chosen <= -WIN_SCORE:
        return 'blunder', depth - 1 - (-chosen - WIN_SCORE)
    if best >= WIN_SCORE:
        if chosen >= WIN_SCORE:
            return 'win_kept', depth - 1 - (best - WIN_SCORE)
        return 'win_missed', depth - 1 - (best - WIN_SCORE)
    return 'unproven', None


def audit(study, matchups, pairs, depth):
    config = json.loads((ROOT / study / 'study.json').read_text())
    rows = load_rows(ROOT / study / 'results.jsonl')
    validate_rows(config, rows)
    openings = {o['id']: o for o in config['openings']}
    wanted = {o['id'] for o in config['openings'][:pairs]}
    agent = NegamaxAgent(depth)
    result = []
    for matchup in matchups:
        tallies = {role: dict(decisions=0, already_lost=0, unproven=0, win_kept=0,
                              win_missed={}, blunder={}) for role in ('challenger', 'opponent')}
        examples = []
        games = [r for r in rows if r['matchup'] == matchup and r['opening'] in wanted]
        for row in games:
            opening = openings[row['opening']]
            game = position(opening['history'])
            for index, move in enumerate(row['moves']):
                role = role_of(row, opening, index)
                tally = tallies[role]
                tally['decisions'] += 1
                scores = agent.score_moves(game)
                kind, plies = classify(scores, move, depth)
                if kind in ('blunder', 'win_missed'):
                    key = str(plies)
                    tally[kind][key] = tally[kind].get(key, 0) + 1
                    if sum(e['role'] == role and e['kind'] == kind and e['plies'] == plies
                           for e in examples) < EXAMPLES_PER_KIND:
                        examples.append(dict(
                            role=role, kind=kind, plies=plies, game=row['id'],
                            history=opening['history'] + row['moves'][:index], played=move,
                            best=max(scores, key=scores.get),
                            simulations=row['simulations'][index]))
                else:
                    tally[kind] += 1
                game.make_move(move)
        spec = next(m for m in config['matchups'] if m['id'] == matchup)
        result.append(dict(matchup=matchup, challenger=spec['challenger'], opponent=spec['opponent'],
                           games=len(games), tallies=tallies, examples=examples))
    return dict(study=study, depth=depth, pairs=pairs, matchups=result,
                note='Blunder plies = opponent plies until four is completed after the move. '
                     'Only outcomes forced within the search depth are visible.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', required=True)
    parser.add_argument('--matchups', nargs='+', required=True)
    parser.add_argument('--pairs', type=int, default=128)
    parser.add_argument('--depth', type=int, default=8)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.study, args.matchups, args.pairs, args.depth)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, result)
    for row in result['matchups']:
        for role, tally in row['tallies'].items():
            blunders = sum(tally['blunder'].values())
            missed = sum(tally['win_missed'].values())
            print(f'{row["matchup"]:14s} {role:10s} decisions {tally["decisions"]:5d} '
                  f'blunders {blunders:4d} ({100 * blunders / tally["decisions"]:.2f}%) '
                  f'by plies {dict(sorted(tally["blunder"].items()))} '
                  f'missed wins {missed} of {missed + tally["win_kept"]}')


if __name__ == '__main__':
    main()
