"""Production AlphaZero inference artifact contract and fixed decision adapter (Phase 4D.4K-A).

A production artifact is the existing minimal v2 inference checkpoint
(``connect4-alphazero-v2-inference`` v2: contract plus model tensors only) plus a
JSON manifest that binds it to its research source. Loading is strict and
offline: bounded manifest, approved model identity, exact size and SHA-256,
weights-only tensor load, tensor-content hash, parameter count and reference
predictions under a tolerance declared before cross-runtime measurement.

Decisions reuse the research search unchanged: ``EvaluationAgent`` PUCT with
256 simulations, c=1.41, no root noise and greedy maximum-visit selection on
the bitboard backend, optionally followed by the verified one-ply root guard.
Each decision builds its own tree and its own deterministic tie-break RNG,
derived only from the position, so no mutable search state is shared.
"""
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import threading
import time

import torch

from .bitboard import BitboardState
from .config import ARCHITECTURE, INFERENCE_CONTRACT, PUCT_EXPLORATION
from .data import seeded_rng
from .network import load_inference_checkpoint, validate_batch_state, weights_sha256
from .search import EvaluationAgent
from .tactical_guard import CONTRACT as GUARD_CONTRACT, guard

MANIFEST_FORMAT = "connect4-alphazero-production-artifact"
MANIFEST_VERSION = 1
DECISION_CONTRACT = "connect4-alphazero-production-decision-v1"
TIE_BREAK_DOMAIN = "connect4-alphazero-production-tie-break-v1"
SIMULATIONS = 256
PARAMETERS = 325_896
MAX_MANIFEST_BYTES = 256 * 1024
MAX_ARTIFACT_BYTES = 8 * 1024 * 1024
TORCH_THREADS = 1
# Declared in benchmarks/alphazero/phase4d4k-a/tolerance-declaration.json before
# any cross-runtime measurement. Visits and actions must still match exactly.
REFERENCE_TOLERANCE = 1e-4

# Changing the served model requires a reviewed code change, never a file swap.
APPROVED_ARTIFACTS = {
    "az2-hc-g6-cnn-s42-v1": "4c3cf26aa58495f43094436f231486969658d70b94b246a912a5bb54756dc528",
}

MANIFEST_KEYS = {"format", "format_version", "model_id", "artifact", "inference_contract", "architecture",
                 "search", "tactical_guard", "source", "export", "reference_predictions"}
SEARCH_SETTINGS = {"algorithm": "connect4-alphazero-v2-puct", "backend": "bitboard", "simulations": SIMULATIONS,
                   "exploration": PUCT_EXPLORATION, "root_noise": None, "temperature": 0.0,
                   "final_selection": "maximum-visit, seeded tie-break", "tie_break_domain": TIE_BREAK_DOMAIN}


class ArtifactError(Exception):
    """Controlled production-artifact failure; ``code`` is safe to report, the message is for logs."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha_text(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def read_manifest(path):
    """Bounded, strictly shaped manifest; the artifact must be a plain sibling file name."""
    path = Path(path)
    try:
        size = path.stat().st_size
    except OSError as error:
        raise ArtifactError("artifact_missing", "AlphaZero manifest is not readable") from error
    if not 0 < size <= MAX_MANIFEST_BYTES:
        raise ArtifactError("artifact_contract", "AlphaZero manifest size is outside its bound")
    try:
        manifest = json.loads(path.read_bytes().decode("utf-8"),
                              parse_constant=lambda name: (_ for _ in ()).throw(ValueError(name)))
    except (UnicodeDecodeError, ValueError) as error:
        raise ArtifactError("artifact_contract", "AlphaZero manifest is not strict JSON") from error
    if not isinstance(manifest, dict) or set(manifest) != MANIFEST_KEYS:
        raise ArtifactError("artifact_contract", "AlphaZero manifest fields do not match the contract")
    if manifest["format"] != MANIFEST_FORMAT or manifest["format_version"] != MANIFEST_VERSION:
        raise ArtifactError("artifact_contract", "Unsupported AlphaZero manifest format")
    artifact = manifest["artifact"]
    if (not isinstance(artifact, dict) or set(artifact) != {"file", "sha256", "size_bytes"}
            or not isinstance(artifact["file"], str) or Path(artifact["file"]).name != artifact["file"]
            or artifact["file"] in ("", ".", "..") or not _sha_text(artifact["sha256"])
            or type(artifact["size_bytes"]) is not int or not 0 < artifact["size_bytes"] <= MAX_ARTIFACT_BYTES):
        raise ArtifactError("artifact_contract", "AlphaZero manifest artifact entry is malformed")
    if manifest["inference_contract"] != INFERENCE_CONTRACT:
        raise ArtifactError("unsupported_architecture", "AlphaZero inference contract is not supported")
    architecture = manifest["architecture"]
    if (not isinstance(architecture, dict) or architecture.get("identity") != ARCHITECTURE
            or architecture.get("parameters") != PARAMETERS or not _sha_text(architecture.get("weights_sha256"))):
        raise ArtifactError("unsupported_architecture", "AlphaZero architecture is not supported")
    if manifest["search"] != SEARCH_SETTINGS:
        raise ArtifactError("artifact_contract", "AlphaZero search settings differ from the validated settings")
    guard_entry = manifest["tactical_guard"]
    if not isinstance(guard_entry, dict) or guard_entry.get("contract") != GUARD_CONTRACT:
        raise ArtifactError("artifact_contract", "AlphaZero tactical guard contract is not supported")
    references = manifest["reference_predictions"]
    if not isinstance(references, list) or not references:
        raise ArtifactError("artifact_contract", "AlphaZero manifest lacks reference predictions")
    return manifest


def configure_inference_threads():
    """Bound torch to one CPU thread for this process; nothing else in the API uses torch."""
    torch.set_num_threads(TORCH_THREADS)
    if torch.get_num_interop_threads() != TORCH_THREADS:
        try:
            torch.set_num_interop_threads(TORCH_THREADS)
        except RuntimeError:
            pass  # Already fixed by earlier torch work in this process; intra-op remains bounded.
    if torch.get_num_threads() != TORCH_THREADS:
        raise ArtifactError("runtime_unavailable", "Could not bound torch inference threads")


def position_rng(state):
    """Fresh tie-break stream that depends only on the exact position."""
    x, o = state.pieces
    return seeded_rng(f"{TIE_BREAK_DOMAIN}:{x}:{o}:{state.current_player}", 0)


def _moves_state(moves):
    state = BitboardState((0, 0), 0, 0)
    for move in moves:
        state = state.drop(move)
    return state


@dataclass(frozen=True)
class AlphaZeroDecision:
    final_action: int
    original_action: int
    visits: tuple
    root_value: float
    root_prior: tuple
    root_logits: tuple
    guard: dict | None
    search_seconds: float
    total_seconds: float

    @property
    def guard_reason(self):
        return None if self.guard is None else self.guard["reason"]


class ProductionModel:
    """One immutable, eval-mode model per process. Safe for concurrent decisions."""

    def __init__(self, manifest, inference, artifact_sha256):
        self.manifest = manifest
        self.inference = inference
        self.model_id = manifest["model_id"]
        self.artifact_sha256 = artifact_sha256

    def decide(self, game, *, tactical_guard, rng=None):
        """Mirror of the 4D.4J-A decision: original PUCT, then the optional guard."""
        if type(tactical_guard) is not bool:
            raise TypeError("tactical_guard must be a boolean")
        started = time.perf_counter()
        root = BitboardState.from_game(game)
        validate_batch_state(root)  # turn counts, gravity, no winner, nonterminal
        rng = position_rng(root) if rng is None else rng
        before = time.perf_counter()
        result, selection = EvaluationAgent(self.inference, SIMULATIONS, rng=rng, backend="bitboard").search(root)
        search_seconds = time.perf_counter() - before
        if sum(result.visits) != SIMULATIONS or result.root_noise is not None or selection.temperature != 0:
            raise RuntimeError("Search did not run with the validated settings")
        diagnostic = guard(root, result.visits, selection.move) if tactical_guard else None
        final = diagnostic["final_action"] if diagnostic else selection.move
        if not root.is_valid_move(final):
            raise RuntimeError("AlphaZero selected an illegal action")
        return AlphaZeroDecision(int(final), int(selection.move), tuple(result.visits), result.root_value,
                                 tuple(result.root_prior), tuple(result.root_logits), diagnostic,
                                 search_seconds, time.perf_counter() - started)


def check_reference_predictions(inference, references, tolerance=REFERENCE_TOLERANCE):
    """Operational numerics check against predictions recorded in the research runtime."""
    worst = 0.0
    for row in references:
        if not isinstance(row, dict) or set(row) != {"moves", "logits", "value"}:
            raise ArtifactError("artifact_contract", "Malformed reference prediction")
        try:
            state = _moves_state(row["moves"])
        except (TypeError, ValueError) as error:
            raise ArtifactError("artifact_contract", "Reference position is not legal") from error
        prediction = inference.predict(state)
        expected = list(row["logits"]) + [row["value"]]
        actual = list(prediction.logits) + [prediction.value]
        if len(expected) != 8 or any(type(v) is not float or not math.isfinite(v) for v in expected):
            raise ArtifactError("artifact_contract", "Malformed reference prediction values")
        worst = max(worst, *(abs(a - e) for a, e in zip(actual, expected)))
    if worst > tolerance:
        raise ArtifactError("self_check_failed", "AlphaZero predictions differ from the research reference")
    return worst


def load_production_model(manifest_path, *, approved=APPROVED_ARTIFACTS):
    """Strict offline load; raises ArtifactError with a safe code on any defect."""
    manifest_path = Path(manifest_path)
    manifest = read_manifest(manifest_path)
    expected = approved.get(manifest["model_id"])
    entry = manifest["artifact"]
    if expected is None or entry["sha256"] != expected:
        raise ArtifactError("artifact_integrity", "AlphaZero model is not an approved artifact")
    artifact = manifest_path.parent / entry["file"]
    try:
        if not artifact.is_file() or artifact.is_symlink():
            raise ArtifactError("artifact_missing", "AlphaZero artifact file is missing")
        if artifact.stat().st_size != entry["size_bytes"]:
            raise ArtifactError("artifact_integrity", "AlphaZero artifact size differs from its manifest")
        # Hash and load one in-memory copy, so the verified bytes are the loaded bytes.
        payload = artifact.read_bytes()
    except OSError as error:
        raise ArtifactError("artifact_missing", "AlphaZero artifact file is not readable") from error
    if len(payload) != entry["size_bytes"] or hashlib.sha256(payload).hexdigest() != expected:
        raise ArtifactError("artifact_integrity", "AlphaZero artifact SHA-256 differs from the approved hash")
    configure_inference_threads()
    try:
        inference = load_inference_checkpoint(io.BytesIO(payload))  # torch.load(weights_only=True), strict keys
    except Exception as error:
        raise ArtifactError("unsupported_architecture", "AlphaZero artifact failed strict inference loading") from error
    model = inference.model
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if sum(p.numel() for p in model.parameters()) != PARAMETERS:
        raise ArtifactError("unsupported_architecture", "AlphaZero parameter count differs")
    if weights_sha256(model) != manifest["architecture"]["weights_sha256"]:
        raise ArtifactError("artifact_integrity", "AlphaZero tensor content differs from its manifest")
    try:
        check_reference_predictions(inference, manifest["reference_predictions"])
    except ArtifactError:
        raise
    except Exception as error:  # e.g. a malformed reference that the engine rejects
        raise ArtifactError("self_check_failed", "AlphaZero reference predictions could not be evaluated") from error
    return ProductionModel(manifest, inference, expected)


_cache = {}
_cache_lock = threading.Lock()


def load_once(manifest_path, *, approved=APPROVED_ARTIFACTS):
    """Process-wide single load per (manifest, approved hash); failures are not cached."""
    key = (os.path.realpath(manifest_path), tuple(sorted(approved.items())))
    with _cache_lock:
        model = _cache.get(key)
        if model is None:
            model = load_production_model(manifest_path, approved=approved)
            _cache[key] = model
        return model
