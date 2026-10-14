"""Research-side reference outputs for production equivalence (Phase 4D.4K-A).

Runs in the research environment against the clean research commit, with the
4D.4J-A freeze enforced, using the ORIGINAL generation-6 checkpoint through the
strict loader and the unmodified 4D.4J-A ``evaluate.decision`` (PUCT256, guard
on/off). Positions have no historical record of their own: mirrors of the
nonsealed development fixtures (both actors), the manifest reference boards and
their mirrors, boards with full columns, and terminal boards (must be rejected).
Only the nonsealed fixtures.json (sha-checked by the research loader) is read.

    env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\
        VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \\
        ~/.venvs/board-game-alphazero-m4/bin/python -B scripts/alphazero/research_reference.py \\
        --research-root <phase4d4j-b worktree> --manifest models/alphazero/az2-hc-g6-cnn-s42-v1.json \\
        --output benchmarks/alphazero/phase4d4k-a/research-reference.json
"""
import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import sys

RESEARCH_COMMIT = "99ceb370cf9c197627f27e86099976304597902e"
SEED = 42
DOMAIN, RNG_SEED = "phase4d4k-a-equivalence", 4410002
THREADS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "NUMEXPR_NUM_THREADS")
TIMING = ("classification_seconds", "guard_seconds")
TERMINAL = {"x-vertical": [3, 4, 3, 4, 3, 4, 3], "o-horizontal": [6, 0, 6, 1, 5, 2, 0, 3],
            "x-diagonal": [0, 1, 1, 2, 2, 3, 2, 3, 3, 6, 3]}


def git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=True).stdout.strip()


def mirror(moves):
    return [6 - m for m in moves]


def full_column_boards(BitboardState, count=8):
    """Seeded walks biased towards tall columns; keep nonterminal boards with a full column."""
    rows, attempt = [], 0
    while len(rows) < count:
        rng, state, moves = random.Random(4410100 + attempt), BitboardState((0, 0), 0, 0), []
        attempt += 1
        target = 18 + attempt % 17
        while len(moves) < target:
            options = [m for m in sorted(state.get_valid_moves()) if not state.drop(m).is_game_over()]
            if not options:
                break
            heights = {m: (state.occupied >> (7 * m) & 63).bit_count() for m in options}
            move = max(options, key=lambda m: (heights[m], -m)) if rng.random() < 0.6 else rng.choice(options)
            state, moves = state.drop(move), moves + [move]
        full = [c for c in range(7) if not state.is_valid_move(c)]
        if len(moves) == target and full and not state.is_game_over():
            rows.append(moves)
    return rows


def draw_board(BitboardState):
    for attempt in range(100000):
        rng, state, moves = random.Random(4410200 + attempt), BitboardState((0, 0), 0, 0), []
        while not state.is_game_over():
            options = [m for m in sorted(state.get_valid_moves()) if state.drop(m).winner == -1]
            if not options:
                break
            move = rng.choice(options)
            state, moves = state.drop(move), moves + [move]
        if len(moves) == 42 and state.winner == -1:
            return moves
    raise SystemExit("No draw board found")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--research-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    research = Path(args.research_root).resolve()
    if git(research, "rev-parse", "HEAD") != RESEARCH_COMMIT or git(research, "status", "--porcelain",
                                                                       "--untracked-files=no"):
        raise SystemExit("Research root must be the clean confirmed research commit")
    if any(os.environ.get(name) != "1" for name in THREADS) or os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise SystemExit("Set all five library thread variables and PYTHONDONTWRITEBYTECODE to 1")
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("Refusing to overwrite reference evidence")
    manifest = json.loads(Path(args.manifest).read_text())
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(research))

    from research.alphazero_tactical_guard import common as ja
    from research.alphazero_tactical_guard.evaluate import decision
    from research.alphazero_resnet_selfplay.evaluate import fixtures
    from games.connect4.alphazero_v2.bitboard import BitboardState
    from games.connect4.alphazero_v2.data import seeded_rng
    from games.connect4.alphazero_v2.network import encode_bitboard_current_player
    from games.connect4.alphazero_v2.oracle import engine_position
    from games.connect4.neural_mcts import encode_current_player

    runtime = ja.runtime()
    ja.enforce()
    runner, model, inference, entry = ja.load(SEED)
    if entry["sha256"] != manifest["source"]["checkpoint_sha256"]:
        raise SystemExit("Manifest does not describe the loaded research checkpoint")
    before, weights = ja.state(runner), runner.factory.hash(model)

    suites = fixtures()
    positions = []
    positions += [(f"tactical-mirror-{r['id']}", "tactical-mirror", mirror(r["moves"])) for r in suites["tactical"]]
    positions += [(f"solved-mirror-{r['id']}", "solved-mirror", mirror(r["moves"])) for r in suites["solved"][::5]]
    positions += [(f"deep-mirror-{r['id']}", "deep-mirror", mirror(r["moves"])) for r in suites["deep"][::4]]
    for i, row in enumerate(manifest["reference_predictions"]):
        positions += [(f"reference-{i:02d}", "reference", row["moves"]),
                      (f"reference-mirror-{i:02d}", "reference-mirror", mirror(row["moves"]))]
    for i, moves in enumerate(full_column_boards(BitboardState)):
        positions += [(f"full-column-{i:02d}", "full-column", moves),
                      (f"full-column-mirror-{i:02d}", "full-column-mirror", mirror(moves))]
    terminal = dict(TERMINAL, draw=draw_board(BitboardState))
    positions += [(f"terminal-{name}", "terminal", moves) for name, moves in terminal.items()]

    rows = []
    for row_id, category, moves in positions:
        game = engine_position(list(moves))
        state = BitboardState.from_game(game)
        row = dict(id=row_id, category=category, moves=list(moves), current_player=game.current_player,
                   legal=sorted(game.get_valid_moves()), terminal=game.is_game_over())
        if row["terminal"]:
            for enabled in (False, True):
                try:
                    decision(inference, game, seeded_rng(f"{DOMAIN}:{row_id}", RNG_SEED), enabled)
                except ValueError:
                    continue
                raise SystemExit(f"Research search accepted terminal board {row_id}")
            row["rejected"] = True
            rows.append(row)
            continue
        engine_input = encode_current_player(game).flatten().tolist()
        bitboard_input = encode_bitboard_current_player(state).flatten().tolist()
        prediction = inference.predict(game)
        row.update(encoding=engine_input, bitboard_encoding_identical=[repr(v) for v in engine_input] == [
                       repr(v) for v in bitboard_input],
                   prediction=dict(logits=list(prediction.logits), policy=list(prediction.policy),
                                   value=prediction.value))
        for variant, enabled in (("control", False), ("guarded", True)):
            action, info = decision(inference, game, seeded_rng(f"{DOMAIN}:{row_id}", RNG_SEED), enabled)
            guard = info["guard"]
            if guard is not None:
                guard = {k: v for k, v in guard.items() if k not in TIMING}
            row[variant] = dict(final_action=action, original_action=info["original_action"], visits=info["visits"],
                                root_value=info["root_value"], root_prior=info["root_prior"],
                                root_logits=info["root_logits"], guard=guard)
        rows.append(row)
    if ja.state(runner) != before or runner.factory.hash(model) != weights:
        raise SystemExit("Reference generation changed research state")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(dict(format="connect4-alphazero-research-reference", format_version=1,
                       research_commit=RESEARCH_COMMIT, checkpoint_sha256=entry["sha256"], weights_sha256=weights,
                       research_decision="research.alphazero_tactical_guard.evaluate.decision (4D.4J-A)",
                       rng=dict(domain=DOMAIN, seed=RNG_SEED, rule="fresh seeded_rng(domain:id, seed) per variant"),
                       runtime=runtime, rows=rows), stream, indent=None, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(dict(rows=len(rows), terminal=sum(r["terminal"] for r in rows),
               changed=sum(1 for r in rows if not r["terminal"] and r["guarded"]["final_action"] != r["control"]["final_action"])))


if __name__ == "__main__":
    main()
