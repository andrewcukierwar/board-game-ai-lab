"""Serial frozen-generation lifecycle with atomic, resumable generation boundaries.

One generation: (A) freeze the learner into an eval snapshot, (B) collect
exactly ``games_per_generation`` complete self-play games with that snapshot
for both sides, (C) finalize them, (D) add them to replay (expiring whole old
generations), (E) run ``ceil(4 * new_positions / batch)`` persistent-AdamW
updates on uniform replay samples with random reflection, (F) optionally save
a boundary, (G) the trained learner collects the next generation.

Exact continuation is promised only from a completed boundary, under an
identical runtime identity (every field of ``provenance.runtime_identity``) and
identical execution-source content (``provenance.execution_source_identity``).
Both are enforced on load by default; a non-strict load is recorded in the
boundary lineage as not claiming exact continuation. A runner also refuses to
continue if its runtime or execution sources change while it is alive.
Mid-game or mid-generation state is never saved: an interrupted generation is
discarded and rerun from the last boundary.
Inert on import; no CLI. Running a research campaign needs separate approval.
"""
from copy import deepcopy
import hashlib
from pathlib import Path
import random
import time

import numpy as np
import torch

from ..connect4 import Connect4
from .artifacts import atomic_torch_save
from . import diagnostics
from .artifacts import file_sha256
from .config import RESUME_CONTRACT, V2Config, action_temperature
from .data import (GenerationReplay, ReflectionAugmenter, V2Example, encode_board, finalize_game,
                   pre_move_ply, seeded_rng)
from .provenance import (execution_source_identity, runtime_differences, runtime_identity,
                         source_differences, source_identity)
from .network import (AlphaZeroV2Net, V2Inference, frozen_copy, model_from_state, model_state,
                      save_inference_checkpoint, weights_sha256)
from .search import SelfPlayer, V2RootNoise
from .selfplay import collect_generation
from .training import V2Trainer

SEARCH_DOMAIN = "connect4-alphazero-v2-search-ties"
ACTION_DOMAIN = "connect4-alphazero-v2-action-selection"
SAMPLING_DOMAIN = "connect4-alphazero-v2-replay-sampling"
INITIALIZATION_DOMAIN = "connect4-alphazero-v2-model-initialization"
PAYLOAD_KEYS = {"contract", "config", "counters", "model_state_dict", "optimizer_state_dict",
                "replay", "rng", "augmentation_counts", "history", "learner_weights_sha256",
                "state_sha256", "runtime", "source", "lineage"}
# Execution sources as imported by this process; disk is re-checked against it.
PROCESS_EXECUTION_SOURCE = execution_source_identity()


def check_process_sources():
    """Raise if any execution source changed on disk since this process imported it."""
    changed = source_differences(PROCESS_EXECUTION_SOURCE, execution_source_identity())
    if changed:
        raise RuntimeError(f"Execution sources changed on disk since import: {changed}")


def _python_rng_state(rng_state):
    version, internal, gauss = rng_state
    return dict(version=version, internal=list(internal), gauss=gauss)


def _python_rng_tuple(state):
    return state["version"], tuple(state["internal"]), state["gauss"]


def _pcg64_to_storage(state):
    bit = state["bit_generator"]
    return dict(draws=state["draws"], bit_generator=dict(
        bit_generator=bit["bit_generator"], state={k: str(v) for k, v in bit["state"].items()},
        has_uint32=int(bit["has_uint32"]), uinteger=int(bit["uinteger"])))


def _pcg64_from_storage(state):
    bit = state["bit_generator"]
    return dict(draws=state["draws"], bit_generator=dict(
        bit_generator=bit["bit_generator"], state={k: int(v) for k, v in bit["state"].items()},
        has_uint32=bit["has_uint32"], uinteger=bit["uinteger"]))


def _update_digest(digest, value):
    """Canonical content hash over nested state (tensors by dtype/shape/bytes)."""
    if isinstance(value, torch.Tensor):
        digest.update(f"T{value.dtype}{tuple(value.shape)}".encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    elif isinstance(value, dict):
        digest.update(b"{")
        for key in sorted(value, key=repr):
            digest.update(repr(key).encode())
            _update_digest(digest, value[key])
        digest.update(b"}")
    elif isinstance(value, (list, tuple)):
        digest.update(b"[")
        for item in value:
            _update_digest(digest, item)
        digest.update(b"]")
    else:
        digest.update(repr(value).encode())


def rebuild_game(generation, index, moves, visits, temperatures, winner, simulations, exploratory_plies):
    """Reconstruct a stored game's examples by legal replay, then re-finalize it.

    Also checks the declared self-play execution: each stored action temperature
    follows the ply schedule and a tau=0 action is a maximum-visit action.
    """
    if not len(moves) == len(visits) == len(temperatures):
        raise ValueError("Stored game arrays differ in length")
    if any(sum(counts) != simulations for counts in visits):
        raise ValueError("Stored visits disagree with the self-play simulation budget")
    game, pending = Connect4(), []
    for move, counts, temperature in zip(moves, visits, temperatures):
        if game.is_game_over():
            raise ValueError("Stored history continues after termination")
        if float(temperature) != action_temperature(pre_move_ply(game), exploratory_plies):
            raise ValueError("Stored action temperature disagrees with the self-play schedule")
        if float(temperature) == 0 and counts[int(move)] != max(counts):
            raise ValueError("Stored tau=0 action is not a maximum-visit action")
        pending.append(V2Example(encode_board(game), game.current_player, pre_move_ply(game),
                                 tuple(int(v) for v in counts), int(move), float(temperature)))
        if not game.make_move(int(move)):
            raise ValueError("Stored history contains an illegal move")
    completed = finalize_game(generation, index, [int(m) for m in moves], pending)
    if completed.winner != winner:
        raise ValueError("Stored winner disagrees with replayed history")
    return completed


def initial_model(seed):
    """Fresh, seed-determined initialization (never v1 weights); global RNG untouched."""
    init_seed = int(hashlib.sha256(f"{INITIALIZATION_DOMAIN}:{seed}".encode()).hexdigest(), 16) % 2 ** 63
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(init_seed)
        return AlphaZeroV2Net().eval()


class GenerationRunner:
    """Owns the learner, persistent trainer, replay window and every RNG stream."""

    def __init__(self, config=V2Config()):
        if not isinstance(config, V2Config):
            raise ValueError("GenerationRunner requires a V2Config")
        self.config = config
        self._install(initial_model(config.seed))

    def _install(self, model):
        config = self.config
        check_process_sources()
        # Pinned for the runner's lifetime; every generation and boundary must match.
        self.runtime = runtime_identity()
        self.lineage = []
        self.model = model
        self.trainer = V2Trainer(model, config)
        self._optimizer = self.trainer.optimizer  # Persistent across every generation.
        self.replay = GenerationReplay(config.replay_generations, config.replay_max_games)
        self.search_rng = seeded_rng(SEARCH_DOMAIN, config.seed)
        self.action_rng = seeded_rng(ACTION_DOMAIN, config.seed)
        self.sampling_rng = seeded_rng(SAMPLING_DOMAIN, config.seed)
        self.root_noise = V2RootNoise(config.root_noise_epsilon, config.root_dirichlet_alpha, seed=config.seed)
        self.augmenter = ReflectionAugmenter(config.reflection_probability, seed=config.seed)
        self.completed_generations = self.total_games = self.total_positions = 0
        self.history = []
        self.phase = "idle"
        self.failed = False

    # Lifecycle -------------------------------------------------------------

    def run_generation(self, boundary_directory=None, *, check=None):
        """Run one complete generation; ``check()`` runs before every search and update.

        ``check`` may raise (budget exhausted, stop requested): the generation is
        then discarded exactly like any other interruption. The returned summary
        is the deterministic history entry plus non-deterministic ``resources``.
        """
        if self.failed:
            raise RuntimeError("Runner failed mid-generation; resume from the last valid boundary")
        if self.completed_generations >= self.config.max_generations:
            raise RuntimeError("Configured max_generations already completed")
        self.check_runtime_and_sources()
        config, generation = self.config, self.completed_generations + 1
        try:
            self.phase = "collecting"
            started = time.perf_counter()
            learner_hash, steps_before = weights_sha256(self.model), self.trainer.steps
            snapshot = frozen_copy(self.model)
            player = SelfPlayer(V2Inference(snapshot), config, search_rng=self.search_rng,
                                action_rng=self.action_rng, root_noise=self.root_noise)
            observer = diagnostics.CollectionObserver()
            games = collect_generation(player, generation, config.games_per_generation, check=check,
                                       observer=observer)
            collected = time.perf_counter()
            if (len(games) != config.games_per_generation or self.trainer.steps != steps_before
                    or weights_sha256(self.model) != learner_hash or weights_sha256(snapshot) != learner_hash):
                raise RuntimeError("Learner or snapshot changed during collection")
            new_positions = sum(len(game.examples) for game in games)
            evicted = self.replay.add_generation(generation, games)

            self.phase = "training"
            reflected_before = self.augmenter.reflected
            metrics, addresses = [], []
            for _ in range(config.updates_for(new_positions)):
                if check is not None:
                    check()
                positions, batch = self.replay.sample(config.batch_size, self.sampling_rng)
                addresses.extend(self.replay.address(p) for p in positions)
                batch, _ = self.augmenter.batch(batch)
                metrics.append(self.trainer.step(batch))
                if self.trainer.optimizer is not self._optimizer:
                    raise RuntimeError("Optimizer was replaced")
            self.completed_generations = generation
            self.total_games += len(games)
            self.total_positions += new_positions
            summary = dict(
                generation=generation, collection_weights_sha256=learner_hash,
                games=len(games), new_positions=new_positions,
                x_wins=sum(g.winner == 0 for g in games), o_wins=sum(g.winner == 1 for g in games),
                draws=sum(g.winner == -1 for g in games), evicted_generations=list(evicted),
                replay_generations=list(self.replay.generations), replay_games=self.replay.games,
                replay_positions=len(self.replay),
                replay_counts={str(g): c for g, c in self.replay.counts().items()},
                updates=len(metrics), trainer_steps=self.trainer.steps,
                reflected_samples=self.augmenter.reflected - reflected_before,
                sampled_positions=len(metrics) * config.batch_size,
                root_noise_draws=self.root_noise.draws,
                mean_policy_loss=_mean(m["policy_loss"] for m in metrics),
                mean_value_mse=_mean(m["value_mse"] for m in metrics),
                mean_combined_loss=_mean(m["combined_loss"] for m in metrics),
                max_gradient_norm=max(m["gradient_norm"] for m in metrics),
                clipped_updates=sum(m["clipped"] for m in metrics),
                learner_weights_sha256=weights_sha256(self.model),
                diagnostics=dict(search=observer.deterministic(), games=diagnostics.game_diagnostics(games),
                                 replay=diagnostics.replay_diagnostics(self.replay),
                                 sampling=diagnostics.sampling_diagnostics(addresses, generation),
                                 gradients=diagnostics.gradient_diagnostics(metrics)))
            self.history.append(summary)
            self.phase = "idle"
        except BaseException:
            self.failed, self.phase = True, "failed"
            raise
        finished = time.perf_counter()
        result = deepcopy(summary)
        result["resources"] = dict(observer.resources(), collection_seconds=collected - started,
                                   training_seconds=finished - collected, peak_rss_mib=diagnostics.peak_rss_mib())
        if boundary_directory is not None:
            result["artifacts"] = self.save_boundary(boundary_directory)
        return result

    def check_runtime_and_sources(self):
        """Refuse to continue after any change of runtime identity or execution sources."""
        changed = runtime_differences(self.runtime, runtime_identity())
        if changed:
            raise RuntimeError(f"Runtime identity changed while the runner was alive: {changed}")
        check_process_sources()

    # Boundary state ----------------------------------------------------------

    def _rng_payload(self):
        numpy_state = np.random.get_state()
        return dict(
            python=_python_rng_state(random.getstate()),
            numpy=dict(algorithm=numpy_state[0], keys=[int(k) for k in numpy_state[1]],
                       position=int(numpy_state[2]), has_gauss=int(numpy_state[3]),
                       cached_gaussian=float(numpy_state[4])),
            torch=torch.get_rng_state(),
            search=_python_rng_state(self.search_rng.getstate()),
            action=_python_rng_state(self.action_rng.getstate()),
            sampling=_python_rng_state(self.sampling_rng.getstate()),
            augmentation=_python_rng_state(self.augmenter.rng.getstate()),
            root_noise=_pcg64_to_storage(self.root_noise.get_state()))

    def _replay_payload(self):
        generations = []
        for generation in self.replay.generations:
            games = [g for g in self.replay.iter_games() if g.generation == generation]
            generations.append(dict(generation=generation, games=[dict(
                index=g.index, winner=g.winner, moves=list(g.moves),
                visits=[list(e.visits) for e in g.examples],
                action_temperatures=[e.action_temperature for e in g.examples]) for g in games]))
        return dict(generations=generations, digest=self.replay.digest())

    def state_sha256(self):
        """Identity of everything that determines continuation (excluding global RNGs)."""
        digest = hashlib.sha256()
        rng = self._rng_payload()
        for key in ("python", "numpy", "torch"):
            rng.pop(key)
        _update_digest(digest, dict(
            config=self.config.to_dict(), counters=self._counters(), model=self.model.state_dict(),
            optimizer=self.trainer.optimizer.state_dict(), replay=self.replay.digest(), rng=rng,
            augmentation=self._augmentation_counts(), history=self.history))
        return digest.hexdigest()

    def _counters(self):
        return dict(completed_generations=self.completed_generations, total_games=self.total_games,
                    total_positions=self.total_positions, trainer_steps=self.trainer.steps)

    def _augmentation_counts(self):
        return dict(reflected=self.augmenter.reflected, unreflected=self.augmenter.unreflected)

    def boundary_payload(self):
        if self.failed or self.phase != "idle":
            raise RuntimeError("Boundaries exist only between completed generations")
        self.check_runtime_and_sources()
        return dict(contract=deepcopy(RESUME_CONTRACT), config=self.config.to_dict(),
                    counters=self._counters(), model_state_dict=model_state(self.model),
                    optimizer_state_dict=deepcopy(self.trainer.optimizer.state_dict()),
                    replay=self._replay_payload(), rng=self._rng_payload(),
                    augmentation_counts=self._augmentation_counts(), history=deepcopy(self.history),
                    learner_weights_sha256=weights_sha256(self.model), state_sha256=self.state_sha256(),
                    runtime=deepcopy(self.runtime), source=source_identity(PROCESS_EXECUTION_SOURCE),
                    lineage=deepcopy(self.lineage))

    def save_resume_boundary(self, path):
        payload = self.boundary_payload()

        def validate(temporary):
            loaded = load_resume_boundary(temporary, restore_global_rng=False)
            if loaded.state_sha256() != payload["state_sha256"]:
                raise ValueError("Reloaded boundary state differs")
        return atomic_torch_save(path, payload, validate)

    def save_boundary(self, directory):
        """Write generation-NNNN.inference.pt and generation-NNNN.resume.pt (never overwrite)."""
        directory = Path(directory)
        stem = f"generation-{self.completed_generations:04d}"
        inference = directory / f"{stem}.inference.pt"
        resume = directory / f"{stem}.resume.pt"
        return dict(inference=str(inference), inference_sha256=save_inference_checkpoint(inference, self.model),
                    resume=str(resume), resume_sha256=self.save_resume_boundary(resume))


def _mean(values):
    values = list(values)
    return sum(values) / len(values)


def load_resume_boundary(path, *, strict_runtime=True, strict_source=True, restore_global_rng=True,
                         expected_sha256=None):
    """Restore a runner exactly at a completed generation boundary.

    Weights-only CPU load. Rejects inference/v1 artifacts, changed contracts,
    replay or state digests, an unexpected file hash and, by default, any
    runtime-identity or execution-source difference. A non-strict load records
    the differences in ``runner.lineage`` and does not claim exact continuation.
    Restores process-global Python/NumPy/torch RNGs unless told not to.
    """
    loaded_sha256 = file_sha256(path)
    if expected_sha256 is not None and loaded_sha256 != expected_sha256:
        raise ValueError("Resume boundary file hash differs from the expected manifest hash")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) != PAYLOAD_KEYS or payload["contract"] != RESUME_CONTRACT:
        raise ValueError("Expected an AlphaZero v2 resume boundary")
    runtime_changes = runtime_differences(payload["runtime"], runtime_identity())
    if strict_runtime and runtime_changes:
        raise ValueError(f"Runtime identity differs; exact continuation is not promised: {runtime_changes}")
    check_process_sources()
    source_changes = source_differences(payload["source"]["execution"], PROCESS_EXECUTION_SOURCE)
    if payload["source"]["execution"].get("sha256") != PROCESS_EXECUTION_SOURCE["sha256"] and not source_changes:
        source_changes = ["<combined digest>"]
    if strict_source and source_changes:
        raise ValueError(f"Execution source identity differs; exact continuation is not promised: {source_changes}")
    config = V2Config.from_dict(payload["config"])
    runner = GenerationRunner.__new__(GenerationRunner)
    runner.config = config
    runner._install(model_from_state(payload["model_state_dict"]))
    runner.trainer.optimizer.load_state_dict(payload["optimizer_state_dict"])
    for group, decay in zip(runner.trainer.optimizer.param_groups, (config.weight_decay, 0.0)):
        if (group["lr"], tuple(group["betas"]), group["eps"], group["weight_decay"]) != (
                config.learning_rate, config.adam_betas, config.adam_eps, decay):
            raise ValueError("Optimizer hyperparameters disagree with config")
    for stored in payload["replay"]["generations"]:
        games = tuple(rebuild_game(stored["generation"], g["index"], g["moves"], g["visits"],
                                   g["action_temperatures"], g["winner"], config.self_play_simulations,
                                   config.exploratory_plies)
                      for g in stored["games"])
        runner.replay.add_generation(stored["generation"], games)
    if runner.replay.digest() != payload["replay"]["digest"]:
        raise ValueError("Replay content digest mismatch")
    counters = payload["counters"]
    runner.completed_generations = counters["completed_generations"]
    runner.total_games, runner.total_positions = counters["total_games"], counters["total_positions"]
    runner.trainer.steps = counters["trainer_steps"]
    optimizer_steps = {int(state["step"]) for state in runner.trainer.optimizer.state.values()}
    if optimizer_steps != ({runner.trainer.steps} if runner.trainer.steps else set()):
        raise ValueError("Optimizer step state disagrees with the trainer step counter")
    runner.history = deepcopy(payload["history"])
    rng = payload["rng"]
    runner.search_rng.setstate(_python_rng_tuple(rng["search"]))
    runner.action_rng.setstate(_python_rng_tuple(rng["action"]))
    runner.sampling_rng.setstate(_python_rng_tuple(rng["sampling"]))
    runner.augmenter.rng.setstate(_python_rng_tuple(rng["augmentation"]))
    runner.augmenter.reflected = payload["augmentation_counts"]["reflected"]
    runner.augmenter.unreflected = payload["augmentation_counts"]["unreflected"]
    runner.root_noise.set_state(_pcg64_from_storage(rng["root_noise"]))
    if weights_sha256(runner.model) != payload["learner_weights_sha256"]:
        raise ValueError("Learner weight digest mismatch")
    if runner.state_sha256() != payload["state_sha256"]:
        raise ValueError("Boundary state digest mismatch")
    runner.lineage = deepcopy(payload["lineage"]) + [dict(
        loaded_file_sha256=loaded_sha256, loaded_state_sha256=payload["state_sha256"],
        completed_generations=runner.completed_generations, strict_runtime=strict_runtime,
        strict_source=strict_source, runtime_differences=runtime_changes, source_differences=source_changes,
        exact_continuation_claimed=not runtime_changes and not source_changes)]
    if restore_global_rng:
        random.setstate(_python_rng_tuple(rng["python"]))
        numpy_state = rng["numpy"]
        np.random.set_state((numpy_state["algorithm"], np.asarray(numpy_state["keys"], dtype=np.uint32),
                             numpy_state["position"], numpy_state["has_gauss"], numpy_state["cached_gaussian"]))
        torch.set_rng_state(rng["torch"])
    return runner
