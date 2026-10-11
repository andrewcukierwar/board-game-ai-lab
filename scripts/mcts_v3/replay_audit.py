"""Replay recorded study games and require exact reproduction.

    python -m scripts.mcts_v3.replay_audit --pairs 2 --output docs/search-mcts-v3/validation/replay-audit.json

Fixed-budget games are deterministic given their declared seeds, so every
replayed game must match its recorded moves, winner, per-move simulation
counts and final RNG fingerprints. Timings are expected to differ.
"""
import argparse
import json
from pathlib import Path

from scripts.evaluate_mcts_strength import atomic_json, load_rows
from scripts.mcts_v3.harness import ROOT, play_game, schedule, source_hashes, validate_rows

STABLE = ('id', 'matchup', 'opening', 'challenger_color', 'moves', 'simulations',
          'winner', 'challenger_score', 'final_rng', 'status')


def audit(study, pairs):
    config = json.loads((ROOT / study / 'study.json').read_text())
    rows = load_rows(ROOT / study / 'results.jsonl')
    validate_rows(config, rows)
    recorded = {row['id']: row for row in rows}
    wanted = {o['id'] for o in config['openings'][:pairs]}
    replayed = mismatches = 0
    for m, opening, color in schedule(config):
        game_id = f'{m["id"]}/{opening["id"]}/{color}'
        if opening['id'] not in wanted or game_id not in recorded:
            continue
        fresh = play_game(config, m, opening, color)
        replayed += 1
        mismatches += any(fresh.get(key) != recorded[game_id].get(key) for key in STABLE)
    return dict(study=study, recorded_games=len(rows), replayed=replayed, mismatches=mismatches,
                pinned_sources_unchanged=config['source_hashes'] == source_hashes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pairs', type=int, default=2)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    studies = sorted(path.parent.name for path in ROOT.glob('*/results.jsonl'))
    result = dict(pairs=args.pairs, studies=[audit(name, args.pairs) for name in studies])
    atomic_json(args.output, result)
    for row in result['studies']:
        print(row)
    if any(row['mismatches'] for row in result['studies']):
        raise SystemExit('Replay mismatch')


if __name__ == '__main__':
    main()
