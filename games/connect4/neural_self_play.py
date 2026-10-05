"""One explicitly launched, bounded CPU neural self-play experiment; inert on import.

No historical initialization, tactical guards, noise or resume support.
Bounds are cooperative: check before every search, move and optimizer update.
An in-flight operation may finish beyond the deadline; partial games are quarantined.
"""
import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import resource
import signal
import subprocess
import sys
import time

import numpy as np
import torch

from .agents.mcts_agent import MCTSAgent
from .agents.mcts_nn_agent import MCTSNNAgent, visit_policy
from .connect4 import Connect4
from .neural_mcts import (
    CHECKPOINT_CONTRACT, Connect4Net, NeuralInference, encode_current_player,
    load_checkpoint, probability_vector, save_checkpoint,
)
from .train_mcts_nn import (
    POLICY_TARGET, NeuralTrainer, TrainingExample, capture_example,
    finalize_examples, outcome_for_player,
    reflect_completed_example,
)


@dataclass(frozen=True)
class Bounds:
    max_games: int = 20
    max_plies: int = 840
    max_updates: int = 200
    max_seconds: float = 900.0
    profile: str = "pilot"

    def __post_init__(self):
        if self.profile not in ("pilot", "scaled"):
            raise ValueError("Unknown bounded profile")
        ceilings = (20, 840, 200) if self.profile == "pilot" else (200, 8400, 2000)
        for name, ceiling in zip(("max_games", "max_plies", "max_updates"), ceilings):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= ceiling:
                raise ValueError(f"{name} must be a positive integer <= {ceiling}")
        if (type(self.max_seconds) not in (int, float) or not math.isfinite(self.max_seconds)
                or not 0 < self.max_seconds <= 900):
            raise ValueError("max_seconds must be positive, finite and <= 900")


    @classmethod
    def scaled(cls, **reduced_limits):
        values = dict(max_games=200, max_plies=8400, max_updates=2000, max_seconds=900, profile="scaled")
        values.update(reduced_limits)
        return cls(**values)


CONFIG = dict(seed=42, python_seed=42, numpy_seed=42, torch_seed=42,
              search_seed=42, training_sampling_seed=42, diagnostic_seed=42,
              simulations=32, exploration=1.41, temperature=1.0,
              tactical_guard=False, root_dirichlet_noise=False,
              horizontal_augmentation=False, horizontal_symmetry_probability=0.0,
              batch_size=32, learning_rate=0.001,
              device="cpu", dtype="float32", intra_op_threads=1, inter_op_threads=1,
              deterministic_algorithms=True, smoke_games=2,
              updates_per_additional_game=10,
              sampling="uniform without replacement per batch; skip if fewer than 32 examples",
              schedule="no updates in games 1-2; up to 10 after each later completed game; all limits checked first")

# Fixed legal, nonterminal probes; never fed to the trainer. Both colors and mirrors.
_BASE_POSITIONS = [
    ("opening_x", []), ("opening_o", [3]),
    ("middle_x", [3, 2, 4, 3, 2, 4, 1, 5]),
    ("middle_o", [3, 2, 4, 3, 2, 4, 1, 5, 1]),
    ("win_x", [0, 1, 0, 1, 0, 2]), ("win_o", [0, 1, 0, 1, 2, 1, 2]),
    ("block_x", [0, 1, 0, 1, 2, 1]), ("block_o", [0, 1, 0, 1, 0]),
    ("full_column_x", [0, 0, 0, 0, 0, 0]),
]
POSITIONS = tuple((name, tuple(moves)) for name, moves in _BASE_POSITIONS) + tuple(
    (name + "_mirror", tuple(6 - m for m in moves)) for name, moves in _BASE_POSITIONS[4:])


def require(condition, message):
    if not condition:
        raise ValueError(message)


def entropy(policy):
    return -sum(p * math.log(p) for p in policy if p > 0)


def fingerprint(example):
    return hashlib.sha256(json.dumps(asdict(example), sort_keys=True, allow_nan=False).encode()).hexdigest()


def memory_peak_mib():
    # macOS reports bytes; Linux reports KiB. ru_maxrss is process peak, not current RSS.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2 if sys.platform == "darwin" else 1024)


def position(moves):
    game = Connect4()
    for move in moves:
        require(not game.is_game_over() and move in game.get_valid_moves(), "Illegal diagnostic/history prefix")
        require(game.make_move(move), "Engine rejected validated history move")
    return game


def validate_search(result, game, agent):
    require(type(result.move) is int and result.move in game.get_valid_moves(), "Illegal selected move")
    require(result.root.game_state.__dict__ == game.__dict__, "Search root is not the pre-move state")
    require(result.tactical_guard is False and result.guard_applied == "disabled", "Tactical guard used")
    require(result.temperature == 1.0, "Changed target temperature")
    require(len(result.visits) == 7 and all(type(v) is int and v >= 0 for v in result.visits), "Invalid visits")
    require(sum(result.visits) == result.root.visits == agent.simulation_limit, "Simulation accounting failed")
    require(result.visits == tuple(result.root.children[a].visits if a in result.root.children else 0
                                   for a in range(7)), "Root-child visits disagree")
    policy = probability_vector(result.policy)
    require(np.array_equal(policy, visit_policy(result.visits, result.temperature)), "Changed visit target")
    require(all(policy[a] == 0 for a in range(7) if a not in game.get_valid_moves()), "Illegal target support")
    require(policy[result.move] > 0, "Selected move has no target support")


def validate_episode(examples, records, completed_game):
    """Reconstruct the entire actual episode to prove pre-move alignment and labels."""
    game = Connect4()
    require(len(examples) == len(records) > 0, "Missing episode records")
    actors = set()
    for example, record in zip(examples, records):
        require(not game.is_game_over(), "Example follows termination")
        require(TrainingExample(**asdict(example)) == example, "Example failed validation")
        expected = tuple(tuple(int(c) for c in row) for row in encode_current_player(game)[0, 0].tolist())
        require(example.observation == expected and example.acting_player == game.current_player,
                "Pre-move observation or acting player mismatch")
        require(record["example_sha256"] == fingerprint(example), "Pending example changed after later moves")
        require(example.policy == tuple(record["policy"]) and example.tactical_guard is False,
                "Policy/guard mismatch")
        require(sum(record["visits"]) == 32, "Recorded simulation budget changed")
        require(record["actor"] == example.acting_player and record["action"] in game.get_valid_moves(),
                "Recorded actor/action mismatch")
        require(example.outcome is None, "Premature label")
        actors.add(example.acting_player)
        require(game.make_move(record["action"]), "History replay rejected move")
    require(game.is_game_over() and game.__dict__ == completed_game.__dict__, "Wrong completed game provenance")
    require(actors == {0, 1}, "Both actors require valid examples")
    labeled = finalize_examples(examples, completed_game)
    winner = completed_game.check_winner()
    for original, example in zip(examples, labeled):
        require(example.outcome == outcome_for_player(winner, example.acting_player), "Wrong actor-relative label")
        require(example.observation == original.observation and example.policy == original.policy,
                "Finalization changed training data")
        require(TrainingExample(**asdict(example)) == example, "Completed example failed validation")
    return labeled


def label_counts(examples):
    counts = Counter(str(e.outcome) for e in examples)
    return {str(label): counts[str(label)] for label in (-1, 0, 1)}


class SymmetryAugmenter:
    """Private domain-separated stream; draws only for enabled minibatch samples."""
    domain = "connect4-completed-horizontal-symmetry-v1"

    def __init__(self, probability=0.0, *, seed=42):
        require(type(probability) in (int, float) and math.isfinite(probability)
                and 0 <= probability <= 1, "Symmetry probability must be finite in [0,1]")
        require(type(seed) is int, "Augmentation seed must be an integer")
        self.probability = float(probability)
        self.seed_digest = hashlib.sha256(f"{self.domain}:{seed}".encode()).hexdigest()
        self._rng = random.Random(int(self.seed_digest, 16))
        self.transformed = self.untransformed = 0

    def batch(self, examples):
        require(bool(examples) and all(isinstance(e, TrainingExample) and e.outcome is not None
                                      for e in examples), "Augmentation requires completed examples")
        flags = [self.probability > 0 and self._rng.random() < self.probability for _ in examples]
        batch = [reflect_completed_example(e) if flag else e for e, flag in zip(examples, flags)]
        self.transformed += sum(flags)
        self.untransformed += len(flags) - sum(flags)
        return batch, flags

    def record(self):
        total = self.transformed + self.untransformed
        return dict(domain=self.domain, seed_sha256=self.seed_digest, probability=self.probability,
                    transformed=self.transformed, untransformed=self.untransformed,
                    observed_rate=self.transformed / total if total else None)

    def rng_state(self):
        return self._rng.getstate()


def run_training(trainer, agent, bounds=Bounds(), *, sampling_rng=None,
                 horizontal_symmetry_probability=0.0, augmentation=None,
                 clock=time.monotonic, should_stop=lambda: False, emit=lambda event: None,
                 on_snapshot=lambda report: None):
    """Shared model; completed-game collection then persistent-trainer updates only.

    Injectable time, signals and search permit synthetic bounded tests. No retry.
    A game that reaches any limit mid-episode is quarantined, never labeled/trained.
    The final completed game is recorded before stopping; no updates start at a limit.
    """
    sampling_rng = random.Random(42) if sampling_rng is None else sampling_rng
    augmentation = (SymmetryAugmenter(horizontal_symmetry_probability) if augmentation is None else augmentation)
    require(augmentation.probability == horizontal_symmetry_probability, "Augmentation configuration mismatch")
    require(trainer.steps == 0 and not trainer.optimizer.state, "Trainer must be fresh")
    require(agent.inference.model is trainer.model, "Both sides must share the trainer model")
    require((agent.simulation_limit, agent.exploration, agent.temperature, agent.tactical_guard)
            == (32, 1.41, 1.0, False), "Search configuration differs from declared experiment")
    require(all(group["lr"] == .001 for group in trainer.optimizer.param_groups), "Changed optimizer learning rate")
    start = clock()
    report = dict(status="running", completed_games=0, started_games=0, actual_plies=0,
                  collected_plies=0, updates=0, wins_x=0, wins_o=0, draws=0,
                  smoke_gate="pending", smoke_updates=0, games=[], losses=[], partial_game=None)
    collection, episode, pending, records = [], None, [], []
    optimizer = trainer.optimizer

    def stop_reason():
        if should_stop():
            return "interrupted"
        for reason, value, limit in (("max_games", report["completed_games"], bounds.max_games),
                                    ("max_plies", report["actual_plies"], bounds.max_plies),
                                    ("max_updates", trainer.steps, bounds.max_updates)):
            if value >= limit:
                return reason
        if clock() - start >= bounds.max_seconds:
            return "max_seconds"
        return None

    try:
        while True:
            reason = stop_reason()
            if reason:
                report["stop_reason"] = reason
                break
            game = Connect4()
            report["started_games"] += 1
            episode = dict(index=report["started_games"], moves=[], update_start=trainer.steps)
            pending, records = [], []
            while not game.is_game_over():
                reason = stop_reason()
                if reason:
                    break
                before = deepcopy(game.__dict__)
                actor = game.current_player
                expected = encode_current_player(game)[0, 0].tolist()
                prediction = agent.inference.predict(game)
                search_start = clock()
                result = agent.search(game)
                search_seconds = clock() - search_start
                require(game.__dict__ == before, "Search mutated caller state")
                validate_search(result, game, agent)
                example = capture_example(result)  # Explicitly BEFORE applying the move.
                require(example.observation == tuple(tuple(int(c) for c in row) for row in expected)
                        and example.acting_player == actor, "Capture lost pre-move state/player")
                require(example.outcome is None, "Captured example already labeled")
                record = dict(event="ply", game=episode["index"], game_ply=len(pending) + 1,
                              actor=actor, action=result.move, observation=example.observation,
                              policy=example.policy, visits=result.visits,
                              temperature=result.temperature, tactical_guard=result.tactical_guard,
                              guard_applied=result.guard_applied, example_sha256=fingerprint(example),
                              policy_target_entropy=entropy(example.policy),
                              raw_policy=prediction.policy, predicted_wdl=prediction.value_probabilities,
                              search_seconds=search_seconds)
                # Release diagnostic trees promptly; only immutable data enters collection.
                del result
                reason = stop_reason()
                if reason:
                    emit(dict(event="unapplied_search", **{k: v for k, v in record.items() if k != "event"}))
                    break
                require(record["action"] in game.get_valid_moves(), "Move became illegal")
                require(game.make_move(record["action"]), "Engine rejected validated legal move")
                report["actual_plies"] += 1
                episode["moves"].append(record["action"])
                pending.append(example)
                records.append(record)
                require(all(fingerprint(e) == r["example_sha256"] for e, r in zip(pending, records)),
                        "Example mutated after a subsequent move")
                require(trainer.steps == episode["update_start"] and not trainer.model.training,
                        "Optimization during an episode")
                emit(dict(record, actual_ply=report["actual_plies"], updates=trainer.steps,
                          elapsed_seconds=clock() - start, peak_memory_mib=memory_peak_mib()))
            if not game.is_game_over():
                report["stop_reason"] = reason
                break
            completed = validate_episode(pending, records, game)
            require(len(collection) + len(completed) <= bounds.max_plies, "Training collection overflow")
            collection.extend(completed)
            report["completed_games"] += 1
            report["collected_plies"] = len(collection)
            winner = game.check_winner()
            report["draws" if winner == -1 else "wins_x" if winner == 0 else "wins_o"] += 1
            episode.update(winner=winner, plies=len(completed), labels=label_counts(completed))
            report["games"].append(episode)
            emit(dict(event="completed_game", **episode,
                      examples=[asdict(e) for e in completed], labels_by_actor={
                          str(p): label_counts([e for e in completed if e.acting_player == p]) for p in (0, 1)}))
            if report["completed_games"] == 2:
                require(trainer.steps == 0 and not optimizer.state, "Smoke gate had optimizer updates")
                report["smoke_gate"] = "passed"
                report["smoke_collected_plies"] = len(collection)
                emit(dict(event="smoke_gate", result="passed", games=2, plies=len(collection), updates=0,
                          invariants=["replayed pre-move observations", "finite normalized legal policies",
                                      "explicit correct actors", "32 root-child visits", "actual winner labels",
                                      "both actors", "immutable snapshots", "guard disabled"]))
            # Update only AFTER an additional completed game, on the declared schedule.
            if report["completed_games"] > 2 and len(collection) >= 32:
                require(report["smoke_gate"] == "passed", "Optimization before smoke gate")
                for _ in range(10):
                    if stop_reason():
                        break
                    indices = sampling_rng.sample(range(len(collection)), 32)
                    batch, reflected = augmentation.batch([collection[i] for i in indices])
                    trainer.step(batch)
                    require(trainer.optimizer is optimizer, "Optimizer was replaced")
                    require(trainer.last_metrics is not None and all(
                        math.isfinite(v) for v in trainer.last_metrics.values()), "Nonfinite/missing trainer metrics")
                    metrics = dict(trainer.last_metrics, update=trainer.steps, after_game=report["completed_games"])
                    report["losses"].append(metrics)
                    emit(dict(event="update", **metrics, sampled_indices=indices,
                              horizontal_reflected=reflected, transformed_samples=sum(reflected),
                              untransformed_samples=len(reflected)-sum(reflected),
                              elapsed_seconds=clock() - start, peak_memory_mib=memory_peak_mib()))
            episode["update_end"] = trainer.steps
            if bounds.profile == "scaled" and report["completed_games"] in (50, 100, 150):
                report["updates"] = trainer.steps
                on_snapshot(report)
            episode, pending, records = None, [], []
        report["status"] = "interrupted" if report["stop_reason"] == "interrupted" else "bounded_stop"
    except Exception as exc:
        report.update(status="invariant_failure", stop_reason="error", error=f"{type(exc).__name__}: {exc}")
        if report["smoke_gate"] == "pending":
            report["smoke_gate"] = "failed"
    finally:
        report.update(updates=trainer.steps, elapsed_seconds=clock() - start,
                      augmentation=augmentation.record(),
                      labels=label_counts(collection), labels_by_actor={
                          str(p): label_counts([e for e in collection if e.acting_player == p]) for p in (0, 1)},
                      peak_memory_mib=memory_peak_mib())
        if episode is not None and pending:
            report["partial_game"] = dict(episode, pending_plies=len(pending),
                                          disposition="quarantined; excluded from collection and optimization")
    return report


def diagnostics(inference, *, positions=POSITIONS):
    results = []
    helper = MCTSAgent(1)
    for name, moves in positions:
        game = position(moves)
        require(not game.is_game_over(), "Diagnostic must be nonterminal")
        prediction = inference.predict(game)
        legal = inference.predict_legal(game)
        # NN Only uses deterministic lowest-index legal maximum; MCTS remains tau=1.
        direct = max(game.get_valid_moves(), key=lambda a: (legal.policy[a], -a))
        agent = MCTSNNAgent(inference, 32, temperature=1, exploration=1.41,
                            tactical_guard=False, rng=random.Random(42))
        start = time.monotonic()
        result = agent.search(game)
        seconds = time.monotonic() - start
        validate_search(result, game, agent)
        wins, safe = helper._winning_moves(game), helper._safe_moves(game)
        required = wins if wins else safe if len(safe) < len(game.get_valid_moves()) else []
        modal = max(range(7), key=lambda a: (result.visits[a], -a))
        results.append(dict(name=name, moves=moves, actor=game.current_player,
                            legal_moves=game.get_valid_moves(), immediate_wins=wins, safe_moves=safe,
                            tactical_required=required,
                            nn_only=dict(action=direct, raw_policy=prediction.policy, legal_policy=legal.policy,
                                         max_probability=max(legal.policy), entropy=entropy(legal.policy),
                                         predicted_wdl=legal.value_probabilities,
                                         tactical_success=direct in required if required else None),
                            mcts=dict(action=result.move, modal_action=modal, policy=result.policy,
                                      visits=result.visits, max_probability=max(result.policy),
                                      entropy=entropy(result.policy), root_value=result.root.q_value,
                                      predicted_wdl=legal.value_probabilities, search_seconds=seconds,
                                      tactical_success=result.move in required if required else None,
                                      modal_tactical_success=modal in required if required else None,
                                      tactical_guard=False, guard_applied=result.guard_applied)))
    return results


def rng_states(search_rng, sampling_rng, augmentation=None):
    return dict(python=random.getstate(), numpy={
        "algorithm": np.random.get_state()[0], "keys": np.random.get_state()[1].tolist(),
        "position": np.random.get_state()[2], "has_gauss": np.random.get_state()[3],
        "cached_gaussian": np.random.get_state()[4]}, torch=torch.get_rng_state().tolist(),
        search=search_rng.getstate(), training_sampling=sampling_rng.getstate(),
        **({"augmentation": augmentation.rng_state()} if augmentation is not None else {}))


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_identity(output):
    root = Path(__file__).resolve().parents[2]
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args])
    patch = git("diff", "--binary", "HEAD")
    (output / "source.patch").write_bytes(patch)
    sources = {}
    for raw in git("ls-files", "--others", "--exclude-standard", "-z").split(b"\0"):
        if not raw:
            continue
        name = raw.decode()
        source = root / name
        require(not source.is_symlink(), "Cannot archive source symlinks")
        target = output / "untracked-source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        sources[name] = sha256(target)
    return dict(commit=git("rev-parse", "HEAD").decode().strip(),
                branch=git("branch", "--show-current").decode().strip(),
                status=git("status", "--porcelain=v1").decode(),
                patch_sha256=sha256(output / "source.patch"), untracked_sha256=sources)


def verify_checkpoint(path, model):
    start = time.monotonic()
    with torch.random.fork_rng(devices=[]):
        loaded = load_checkpoint(path)
    load_seconds = time.monotonic() - start
    require(loaded.model.representation_version == CHECKPOINT_CONTRACT["encoding"], "Reload representation mismatch")
    require(all(torch.equal(v, loaded.model.state_dict()[k]) for k, v in model.state_dict().items()),
            "Reload weight tensors differ")
    original = NeuralInference(model)
    for _, moves in POSITIONS:
        game = position(moves)
        require(original.predict(game) == loaded.predict(game), "Reload predictions differ")
        with torch.inference_mode():
            require(all(torch.equal(a, b) for a, b in zip(
                model(encode_current_player(game)), loaded.model(encode_current_player(game)))),
                "Reload logits differ")
    return dict(exact_weight_equality=True, exact_prediction_equality=True, exact_logit_equality=True,
                finite_predictions=True, fixed_positions=len(POSITIONS), contract=deepcopy(CHECKPOINT_CONTRACT),
                loading_seconds=load_seconds)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="New directory under ignored experiment-output/")
    parser.add_argument("--profile", choices=("pilot", "scaled"), default="pilot")
    parser.add_argument("--horizontal-symmetry-probability", type=float, default=0.0)
    parser.add_argument("--evaluation-baseline", type=Path,
                        help="Explicit canonical checkpoint for evaluation only; never initializes training")
    args = parser.parse_args(argv)
    bounds = Bounds.scaled() if args.profile == "scaled" else Bounds()
    augmentation = SymmetryAugmenter(args.horizontal_symmetry_probability)
    require(args.evaluation_baseline is None or args.profile == "scaled", "Baseline evaluation requires scaled profile")
    root = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    allowed = (root / "experiment-output").resolve()
    require(allowed in output.parents, "Output must be a new directory inside experiment-output/")
    require(platform.machine() == "arm64", "This experiment requires native ARM64 CPU execution")
    output.mkdir(parents=True, exist_ok=False)  # A launch marker; never reuse/restart a run directory.
    end_to_end = time.monotonic()
    identity = source_identity(output)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    search_rng, sampling_rng = random.Random(42), random.Random(42)
    initialization_rng = rng_states(search_rng, sampling_rng, augmentation)
    start = time.monotonic()
    model = Connect4Net().cpu().float().eval()  # Sole initialization; no checkpoint reader here.
    initialization_seconds = time.monotonic() - start
    trainer = NeuralTrainer(model, learning_rate=.001)
    agent = MCTSNNAgent(model, 32, temperature=1, exploration=1.41, rng=search_rng, tactical_guard=False)
    if args.profile == "scaled":
        from .neural_evaluation import EVALUATION_CONFIG, SNAPSHOT_GAMES, snapshot, evaluate
        if args.evaluation_baseline is None:
            initial_inference = NeuralInference(deepcopy(model).eval())
        else:
            with torch.random.fork_rng(devices=[]):
                initial_inference = load_checkpoint(args.evaluation_baseline)
    write_json(output / "config.json", dict(config=dict(CONFIG,
               horizontal_augmentation=augmentation.probability > 0,
               horizontal_symmetry_probability=augmentation.probability), bounds=asdict(bounds),
               augmentation=augmentation.record(),
               evaluation_models=dict(initial=(dict(path=str(args.evaluation_baseline),
                   sha256=sha256(args.evaluation_baseline)) if args.evaluation_baseline else "fresh untrained"),
                   final="new candidate"),
               measurement_protocol=(dict(snapshot_games=SNAPSHOT_GAMES, evaluation=EVALUATION_CONFIG,
                   frozen_labels="previously inspected; measurements only") if args.profile == "scaled" else None),
               checkpoint_contract=CHECKPOINT_CONTRACT, policy_target=POLICY_TARGET,
               source=identity, started_utc=datetime.now(timezone.utc).isoformat(),
               environment=dict(python=sys.version, platform=platform.platform(), machine=platform.machine(),
                                torch=torch.__version__, numpy=np.__version__, device="cpu",
                                intra_op_threads=torch.get_num_threads(), inter_op_threads=torch.get_num_interop_threads(),
                                deterministic_algorithms=torch.are_deterministic_algorithms_enabled()),
               initialization_seconds=initialization_seconds, checkpoint_initialization=None,
               initialization_peak_memory_mib=memory_peak_mib(),
               initial_weights_sha256=hashlib.sha256(b"".join(
                   t.detach().numpy().tobytes() for t in model.state_dict().values())).hexdigest()))
    write_json(output / "rng-initial.json", dict(before_initialization=initialization_rng,
                                                before_collection=rng_states(search_rng, sampling_rng, augmentation)))
    initial = diagnostics(agent.inference)
    write_json(output / "diagnostics-initial.json", initial)
    if args.profile == "scaled":
        write_json(output / "snapshot-initial.json", snapshot(agent.inference,
                   dict(completed_games=0, updates=0, losses=[])))
    stop = [False]
    def request_stop(signum, frame):
        stop[0] = True
    previous = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        with (output / "events.jsonl").open("x") as stream:
            def emit(event):
                stream.write(json.dumps(event, allow_nan=False) + "\n")
                stream.flush()
                if event["event"] in ("completed_game", "smoke_gate"):
                    print(json.dumps({k: v for k, v in event.items() if k not in ("examples", "moves")}), flush=True)
            def on_snapshot(current):
                name = f"snapshot-game{current['completed_games']}.json"
                write_json(output / name, snapshot(agent.inference, current))
                print(json.dumps(dict(event="snapshot", games=current['completed_games'],
                                      updates=current['updates'])), flush=True)
            report = run_training(trainer, agent, bounds, sampling_rng=sampling_rng,
                                  horizontal_symmetry_probability=augmentation.probability, augmentation=augmentation,
                                  should_stop=lambda: stop[0], emit=emit, on_snapshot=on_snapshot)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    write_json(output / "rng-final.json", rng_states(search_rng, sampling_rng, augmentation))
    # Failed gate/invariants never lead to more optimization or a candidate.
    if report["status"] == "invariant_failure" or report["smoke_gate"] != "passed":
        report["end_to_end_seconds"] = time.monotonic() - end_to_end
        write_json(output / "report.json", report)
        print(json.dumps(report), flush=True)
        return 1
    final = diagnostics(agent.inference)
    write_json(output / "diagnostics-final.json", final)
    if args.profile == "scaled":
        write_json(output / "snapshot-final.json", snapshot(agent.inference, report))
    try:
        start = time.monotonic()
        save_checkpoint(output / "candidate.pt", model)
        report["checkpoint_writing_seconds"] = time.monotonic() - start
        report["checkpoint_reload"] = verify_checkpoint(output / "candidate.pt", model)
    except Exception as exc:
        report.update(status="invariant_failure", error=f"Checkpoint: {type(exc).__name__}: {exc}")
    if args.profile == "scaled" and report["status"] != "invariant_failure":
        evaluation = evaluate(dict(initial=initial_inference, final=agent.inference))
        write_json(output / "opponent-evaluation.json", evaluation)
        report["evaluation"] = {k: evaluation[k] for k in
                                ("status", "completed_games", "planned_games", "elapsed_seconds")}
    report.update(end_to_end_seconds=time.monotonic() - end_to_end,
                  peak_memory_mib=memory_peak_mib(), exact_resumability=False,
                  initialization_seconds=initialization_seconds)
    write_json(output / "report.json", report)
    files = {str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob("*")) if p.is_file()}
    write_json(output / "manifest.json", dict(format_version=1, files_sha256=files,
                                             inference_artifact="candidate.pt", exact_resumability=False))
    print(json.dumps({k: v for k, v in report.items() if k not in ("games", "losses")}), flush=True)
    return 1 if report["status"] == "invariant_failure" else 0


if __name__ == "__main__":
    raise SystemExit(main())
