"""Preflight measurements (no research training, no learned checkpoints).

``search``: whole-search wall time of deployment-mode PUCT at 256 and 512
simulations with the seed-42 *untrained* v2 network on model-blind varied
positions, plus memory growth over repeated self-play games.

``artifacts``: resume/inference artifact bytes and save/load seconds at full
replay scale (8 generations x 256 games) for typical generated game lengths and
for the 86,016-position worst case (every game a 42-ply draw). Replays are
synthetic, legal and schedule-conforming; the optimizer state comes from a
single discarded update on synthetic data so that AdamW moments exist.

Both write a JSON report into a new output directory and never overwrite.
"""
import argparse
import gc
import json
from pathlib import Path
import random
import sys
import time

from ..connect4 import Connect4
from .config import V2Config, action_temperature
from .data import V2Example, encode_board, finalize_game, pre_move_ply
from .diagnostics import peak_rss_mib, quantiles
from .generation import GenerationRunner, initial_model, load_resume_boundary
from .network import V2Inference, weights_sha256
from .oracle import engine_position
from .packages import sample_game
from .provenance import configure_deterministic_runtime, execution_source_identity, git_provenance
from .search import EvaluationAgent, SelfPlayer, V2RootNoise
from .selfplay import play_game

PROFILE_SEED = 4_303_011
DRAW_GAME = (2, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
             3, 3, 3, 3, 3, 3, 6, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6)


def varied_positions(count, seed=PROFILE_SEED):
    """Model-blind nonterminal positions from seeded generated games, spread over game stages."""
    rng, positions = random.Random(seed), []
    while len(positions) < count:
        moves = sample_game(rng)
        ply = rng.randrange(len(moves))
        if not engine_position(moves[:ply]).is_game_over():
            positions.append(moves[:ply])
    return positions


def profile_search(output, *, positions=500, budgets=(256, 512), warmup=20, selfplay_games=6, threads=1):
    runtime = configure_deterministic_runtime(threads)
    inference = V2Inference(initial_model(42))
    histories = varied_positions(positions)
    for moves in histories[:warmup]:
        EvaluationAgent(inference, budgets[0], rng=random.Random(0)).search(engine_position(moves))
    report = dict(runtime=runtime, model="seed-42 untrained AlphaZeroV2Net (never trained or saved)",
                  weights_sha256=weights_sha256(inference.model), positions=len(histories), warmup=warmup,
                  position_plies=quantiles([len(m) for m in histories], (0.0, 0.5, 1.0)), budgets={})
    for budget in budgets:
        seconds, depth = [], []
        for index, moves in enumerate(histories):
            game = engine_position(moves)
            started = time.perf_counter()
            result, _ = EvaluationAgent(inference, budget, rng=random.Random(index)).search(game)
            seconds.append(time.perf_counter() - started)
            depth.append(result.max_depth)
        ms = [s * 1000 for s in seconds]
        report["budgets"][str(budget)] = dict(mean_ms=sum(ms) / len(ms), ms=quantiles(ms, (0.5, 0.95, 0.99, 1.0)),
                                              max_depth=quantiles(depth, (0.5, 1.0)), peak_rss_mib=peak_rss_mib())
    config = V2Config(self_play_simulations=budgets[0])
    player = SelfPlayer(inference, config, search_rng=random.Random(1), action_rng=random.Random(2),
                        root_noise=V2RootNoise(seed=3))
    rss, game_seconds, plies = [], [], []
    for index in range(selfplay_games):
        started = time.perf_counter()
        completed = play_game(player, 1, index)
        game_seconds.append(time.perf_counter() - started)
        plies.append(len(completed.moves))
        gc.collect()
        rss.append(peak_rss_mib())
    report["self_play_256"] = dict(games=selfplay_games, plies=plies, seconds=game_seconds,
                                   seconds_per_ply=sum(game_seconds) / sum(plies), peak_rss_mib_after_each=rss)
    return report


def synthetic_game(generation, index, moves, simulations, rng, exploratory_plies):
    game, pending = Connect4(), []
    for move in moves:
        legal = sorted(game.get_valid_moves())
        others = [a for a in legal if a != move]
        visits = [0] * 7
        for action in others:
            visits[action] = rng.randrange(0, simulations // (2 * len(legal)) + 1)
        visits[move] = simulations - sum(visits)
        ply = pre_move_ply(game)
        pending.append(V2Example(encode_board(game), game.current_player, ply, tuple(visits), move,
                                 action_temperature(ply, exploratory_plies)))
        if not game.make_move(move):
            raise RuntimeError("Synthetic history is illegal")
    return finalize_game(generation, index, moves, pending)


def full_scale_runner(kind, config, seed=PROFILE_SEED):
    rng = random.Random(seed)
    runner = GenerationRunner(config)
    total_games = total_positions = 0
    for generation in range(1, config.replay_generations + 1):
        games = []
        for index in range(config.games_per_generation):
            moves = list(DRAW_GAME) if kind == "worst_case" else sample_game(rng)
            games.append(synthetic_game(generation, index, moves, config.self_play_simulations, rng,
                                        config.exploratory_plies))
        runner.replay.add_generation(generation, games)
        total_games += len(games)
        total_positions += sum(len(g.examples) for g in games)
    _, batch = runner.replay.sample(config.batch_size, runner.sampling_rng)
    runner.trainer.step(batch)  # one discarded synthetic update so AdamW moments exist
    runner.completed_generations = config.replay_generations
    runner.total_games, runner.total_positions = total_games, total_positions
    return runner


def profile_artifacts(output, *, kinds=("typical", "worst_case"), threads=1):
    runtime = configure_deterministic_runtime(threads)
    config = V2Config()
    report = dict(runtime=runtime, config=config.to_dict(), kinds={})
    for kind in kinds:
        started = time.perf_counter()
        runner = full_scale_runner(kind, config)
        built = time.perf_counter() - started
        directory = Path(output) / kind
        directory.mkdir()
        started = time.perf_counter()
        artifacts = runner.save_boundary(directory)
        saved = time.perf_counter() - started
        resume, inference = Path(artifacts["resume"]), Path(artifacts["inference"])
        del runner
        gc.collect()
        started = time.perf_counter()
        loaded = load_resume_boundary(resume, restore_global_rng=False, expected_sha256=artifacts["resume_sha256"])
        load_seconds = time.perf_counter() - started
        report["kinds"][kind] = dict(
            replay_games=loaded.replay.games, replay_positions=len(loaded.replay),
            resume_bytes=resume.stat().st_size, inference_bytes=inference.stat().st_size,
            build_seconds=built, save_seconds_including_validation_reload=saved, load_seconds=load_seconds,
            peak_rss_mib=peak_rss_mib())
        del loaded
        gc.collect()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("search", "artifacts"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--positions", type=int, default=500)
    args = parser.parse_args(argv)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    if args.command == "search":
        report = profile_search(output, positions=args.positions)
    else:
        report = profile_artifacts(output)
    report.update(command=args.command, wall_seconds=time.time() - started,
                  execution_sha256=execution_source_identity()["sha256"], git=git_provenance())
    (output / f"{args.command}-profile.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print(json.dumps(report, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
