"""Serial frozen-generation lifecycle with atomic, resumable generation boundaries.

One generation: (A) freeze the learner into an eval snapshot, (B) collect
exactly ``games_per_generation`` complete self-play games with that snapshot
for both sides, (C) finalize them, (D) add them to replay (expiring whole old
generations), (E) run ``ceil(4 * new_positions / batch)`` persistent-AdamW
updates on uniform replay samples with random reflection, (F) optionally save
a boundary, (G) the trained learner collects the next generation.

Exact continuation is promised only from a completed boundary, in the same
runtime (versions/threads). Mid-game or mid-generation state is never saved:
an interrupted generation is discarded and rerun from the last boundary.
Inert on import; no CLI. Running a research campaign needs separate approval.
"""
from copy import deepcopy
import hashlib
from pathlib import Path
import platform
import random
import subprocess

import numpy as np
import torch

from ..connect4 import Connect4
from .artifacts import atomic_torch_save
from .config import RESUME_CONTRACT, V2Config
from .data import (GenerationReplay, ReflectionAugmenter, V2Example, encode_board, finalize_game,
                   pre_move_ply, seeded_rng)
from .network import (AlphaZeroV2Net, V2Inference, frozen_copy, model_from_state, model_state,
                      save_inference_checkpoint, weights_sha256)
from .search import SelfPlayer, V2RootNoise
from .selfplay import collect_generation
from .training import V2Trainer

SEARCH_DOMAIN = "connect4-alphazero-v2-search-ties"
ACTION_DOMAIN = "connect4-alphazero-v2-action-selection"
SAMPLING_DOMAIN = "connect4-alphazero-v2-replay-sampling"
INITIALIZATION_DOMAIN = "connect4-alphazero-v2-model-initialization"
STRICT_RUNTIME_KEYS = ("python", "torch", "numpy", "machine", "intra_op_threads")
PAYLOAD_KEYS = {"contract", "config", "counters", "model_state_dict", "optimizer_state_dict",
                "replay", "rng", "augmentation_counts", "history", "learner_weights_sha256",
                "state_sha256", "runtime", "source"}


def runtime_identity():
    return dict(python=platform.python_version(), torch=str(torch.__version__), numpy=str(np.__version__),
                platform=platform.platform(), machine=platform.machine(),
                intra_op_threads=torch.get_num_threads(), inter_op_threads=torch.get_num_interop_threads(),
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled())


def source_identity():
    """Best-effort code identity; None fields when git is unavailable."""
    root = Path(__file__).resolve().parents[3]

    def git(*args):
        try:
            return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True,
                                  timeout=10, check=True).stdout
        except (OSError, subprocess.SubprocessError):
            return None
    commit, status = git("rev-parse", "HEAD"), git("status", "--porcelain=v1")
    return dict(commit=commit.strip() if commit else None,
                dirty=None if status is None else bool(status.strip()))


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


def rebuild_game(generation, index, moves, visits, temperatures, winner, simulations):
    """Reconstruct a stored game's examples by legal replay, then re-finalize it."""
    if not len(moves) == len(visits) == len(temperatures):
        raise ValueError("Stored game arrays differ in length")
    if any(sum(counts) != simulations for counts in visits):
        raise ValueError("Stored visits disagree with the self-play simulation budget")
    game, pending = Connect4(), []
    for move, counts, temperature in zip(moves, visits, temperatures):
        if game.is_game_over():
            raise ValueError("Stored history continues after termination")
        pending.append(V2Example(encode_board(game), game.current_player, pre_move_ply(game),
                                 tuple(int(v) for v in counts), int(move), float(temperature)))
        if not game.make_move(int(move)):
            raise ValueError("Stored history contains an illegal move")
    completed = finalize_game(generation, index, [int(m) for m in moves], pending)
    if completed.winner != winner:
        raise ValueError("Stored winner disagrees with replayed history")
    return completed


class GenerationRunner:
    """Owns the learner, persistent trainer, replay window and every RNG stream."""

    def __init__(self, config=V2Config()):
        if not isinstance(config, V2Config):
            raise ValueError("GenerationRunner requires a V2Config")
        self.config = config
        init_seed = int(hashlib.sha256(f"{INITIALIZATION_DOMAIN}:{config.seed}".encode()).hexdigest(), 16) % 2 ** 63
        with torch.random.fork_rng(devices=[]):  # Fresh initialization only; never v1 weights.
            torch.manual_seed(init_seed)
            model = AlphaZeroV2Net().eval()
        self._install(model)

    def _install(self, model):
        config = self.config
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

    def run_generation(self, boundary_directory=None):
        if self.failed:
            raise RuntimeError("Runner failed mid-generation; resume from the last valid boundary")
        if self.completed_generations >= self.config.max_generations:
            raise RuntimeError("Configured max_generations already completed")
        config, generation = self.config, self.completed_generations + 1
        try:
            self.phase = "collecting"
            learner_hash, steps_before = weights_sha256(self.model), self.trainer.steps
            snapshot = frozen_copy(self.model)
            player = SelfPlayer(V2Inference(snapshot), config, search_rng=self.search_rng,
                                action_rng=self.action_rng, root_noise=self.root_noise)
            games = collect_generation(player, generation, config.games_per_generation)
            if (len(games) != config.games_per_generation or self.trainer.steps != steps_before
                    or weights_sha256(self.model) != learner_hash or weights_sha256(snapshot) != learner_hash):
                raise RuntimeError("Learner or snapshot changed during collection")
            new_positions = sum(len(game.examples) for game in games)
            evicted = self.replay.add_generation(generation, games)

            self.phase = "training"
            reflected_before = self.augmenter.reflected
            metrics = []
            for _ in range(config.updates_for(new_positions)):
                _, batch = self.replay.sample(config.batch_size, self.sampling_rng)
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
                learner_weights_sha256=weights_sha256(self.model))
            self.history.append(summary)
            self.phase = "idle"
        except BaseException:
            self.failed, self.phase = True, "failed"
            raise
        result = deepcopy(summary)
        if boundary_directory is not None:
            result["artifacts"] = self.save_boundary(boundary_directory)
        return result

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
        return dict(contract=deepcopy(RESUME_CONTRACT), config=self.config.to_dict(),
                    counters=self._counters(), model_state_dict=model_state(self.model),
                    optimizer_state_dict=deepcopy(self.trainer.optimizer.state_dict()),
                    replay=self._replay_payload(), rng=self._rng_payload(),
                    augmentation_counts=self._augmentation_counts(), history=deepcopy(self.history),
                    learner_weights_sha256=weights_sha256(self.model), state_sha256=self.state_sha256(),
                    runtime=runtime_identity(), source=source_identity())

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


def load_resume_boundary(path, *, strict_runtime=True, restore_global_rng=True):
    """Restore a runner exactly at a completed generation boundary.

    Weights-only CPU load. Rejects inference/v1 artifacts, changed contracts,
    replay or state digests and (by default) a different runtime identity.
    Restores process-global Python/NumPy/torch RNGs unless told not to.
    """
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) != PAYLOAD_KEYS or payload["contract"] != RESUME_CONTRACT:
        raise ValueError("Expected an AlphaZero v2 resume boundary")
    runtime = runtime_identity()
    if strict_runtime and any(payload["runtime"][k] != runtime[k] for k in STRICT_RUNTIME_KEYS):
        raise ValueError("Runtime identity differs; exact continuation is not promised")
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
                                   g["action_temperatures"], g["winner"], config.self_play_simulations)
                      for g in stored["games"])
        runner.replay.add_generation(stored["generation"], games)
    if runner.replay.digest() != payload["replay"]["digest"]:
        raise ValueError("Replay content digest mismatch")
    counters = payload["counters"]
    runner.completed_generations = counters["completed_generations"]
    runner.total_games, runner.total_positions = counters["total_games"], counters["total_positions"]
    runner.trainer.steps = counters["trainer_steps"]
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
    if restore_global_rng:
        random.setstate(_python_rng_tuple(rng["python"]))
        numpy_state = rng["numpy"]
        np.random.set_state((numpy_state["algorithm"], np.asarray(numpy_state["keys"], dtype=np.uint32),
                             numpy_state["position"], numpy_state["has_gauss"], numpy_state["cached_gaussian"]))
        torch.set_rng_state(rng["torch"])
    return runner
