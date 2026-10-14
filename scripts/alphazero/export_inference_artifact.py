"""Export one approved research CNN as an inference-only production artifact (Phase 4D.4K-A).

Runs ONLY in the compatible research environment, against a clean checkout of
the confirmed research commit, with the historical gates intact:

    env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\
        VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \\
        ~/.venvs/board-game-alphazero-m4/bin/python -B scripts/alphazero/export_inference_artifact.py \\
        --research-root <phase4d4j-b worktree at 99ceb37> --output-dir <new empty directory>

Every import of ``games``/``research`` resolves to the research root, never to
this checkout. The original checkpoint is loaded through the 4D.4J-A strict
loader (read-only), after the 4D.4J-A source/runtime freeze and the 4D.4J-B
guard/checkpoint freeze are enforced. Only model tensors and the existing v2
inference contract are written; optimizer, replay, RNG and training state are
not. Nothing is overwritten. The production loader verifies the result
separately (scripts/alphazero/verify_inference_artifact.py).
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

RESEARCH_COMMIT = "99ceb370cf9c197627f27e86099976304597902e"
SOURCE = {
    42: dict(sha256="ba2bde2ec71b3ac7486046d1cc26ab591bb59d6b764193ece08c6d6a6f24f58b", size_bytes=4903328),
}
GUARD_SHA256 = "d2ee54ba1fe9ddbcc4641538e2f5dd89a2435234ff194d6909bed72096632d04"
# Added by the independent results reviewer after the 4D.4J-B freeze; ROOT.glob('*.py') sees it.
JB_POST_FREEZE = {"research/alphazero_tactical_guard_confirmation/independent_verify.py"}
THREADS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "NUMEXPR_NUM_THREADS")
# Synthetic legal nonterminal positions (no fixture package is read): a seeded walk
# that never enters a terminal state, at plies 0..38 so both actors and opening,
# middle, late and near-terminal boards are covered. Stored moves are authoritative.
REFERENCE_PLIES = (0, 1, 2, 3, 5, 8, 12, 17, 23, 29, 34, 38)
REFERENCE_SEED = 4410001


def reference_moves(BitboardState):
    import random
    rows = []
    for ply in REFERENCE_PLIES:
        for attempt in range(1000):
            rng, state, moves = random.Random(REFERENCE_SEED * 1000 + ply * 10 + attempt), BitboardState((0, 0), 0, 0), []
            while len(moves) < ply:
                options = [m for m in sorted(state.get_valid_moves()) if not state.drop(m).is_game_over()]
                if not options:
                    break
                move = rng.choice(options)
                state, moves = state.drop(move), moves + [move]
            if len(moves) == ply:
                rows.append(moves)
                break
        else:
            raise SystemExit("Could not generate a reference position")
    return rows


def git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=True).stdout.strip()


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--research-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, choices=sorted(SOURCE), default=42)
    args = parser.parse_args()

    tool = Path(__file__).resolve()
    integration = tool.parents[2]
    tool_rel = tool.relative_to(integration).as_posix()
    if git(integration, "status", "--porcelain", "--", tool_rel):
        raise SystemExit("Commit the export tool before exporting; its committed identity is recorded")
    integration_commit = git(integration, "rev-parse", "HEAD")

    research = Path(args.research_root).resolve()
    if git(research, "rev-parse", "HEAD") != RESEARCH_COMMIT:
        raise SystemExit("Research root is not the confirmed research commit")
    if git(research, "status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("Research root has tracked modifications")
    if any(os.environ.get(name) != "1" for name in THREADS) or os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise SystemExit("Set all five library thread variables and PYTHONDONTWRITEBYTECODE to 1")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(research))

    output = Path(args.output_dir).resolve()
    model_id = f"az2-hc-g6-cnn-s{args.seed}-v1"
    artifact_path, manifest_path = output / f"{model_id}.pt", output / f"{model_id}.json"
    if artifact_path.exists() or manifest_path.exists():
        raise FileExistsError("Refusing to overwrite an existing artifact or manifest")
    output.mkdir(parents=True, exist_ok=True)

    from research.alphazero_tactical_guard import common as ja
    from research.alphazero_tactical_guard_confirmation import common as jb
    from games.connect4.alphazero_v2 import network
    from games.connect4.alphazero_v2.bitboard import BitboardState
    from games.connect4.alphazero_v2.config import ARCHITECTURE, INFERENCE_CONTRACT, PUCT_EXPLORATION
    from games.connect4.alphazero_v2.experimental import ORIGINAL
    from games.connect4.alphazero_v2.provenance import runtime_identity
    import torch

    if Path(ja.__file__).resolve().parents[2] != research or Path(network.__file__).resolve().parents[3] != research:
        raise SystemExit("Research modules did not resolve to the research root")

    runtime = jb.runtime()  # one-thread deterministic runtime, enforced
    ja.enforce()            # 4D.4J-A frozen execution sources + runtime + checkpoint bytes
    jb_freeze = jb.read(jb.RESULTS / "freeze.json")
    current = jb.source_map()
    differ = {k for k in set(current) | set(jb_freeze["files"]) if current.get(k) != jb_freeze["files"].get(k)}
    if differ != JB_POST_FREEZE or any(k in jb_freeze["files"] for k in JB_POST_FREEZE):
        raise SystemExit(f"4D.4J-B frozen sources differ beyond the documented post-freeze verifier: {sorted(differ)}")
    for rel in JB_POST_FREEZE:
        if git(research, "hash-object", rel) != git(research, "rev-parse", f"{RESEARCH_COMMIT}:{rel}"):
            raise SystemExit("Post-freeze verifier differs from the research commit")
    if runtime_identity() != jb_freeze["runtime"] or jb.sha(jb.GUARD) != GUARD_SHA256:
        raise SystemExit("4D.4J-B runtime or guard identity differs")

    expected = SOURCE[args.seed]
    runner, model, inference, entry = ja.load(args.seed)  # strict original-schedule generation-6 loader
    checkpoint = Path(entry["path"])
    if (entry["sha256"] != expected["sha256"] or sha256(checkpoint) != expected["sha256"]
            or checkpoint.stat().st_size != expected["size_bytes"]
            or {"path": entry["path"], "sha256": entry["sha256"]} not in [
                {"path": e["path"], "sha256": e["sha256"]} for e in jb_freeze["checkpoints"]]):
        raise SystemExit("Checkpoint identity differs from the validated research checkpoint")
    if (runner.completed_generations != 6 or runner.schedule != ORIGINAL or runner.factory.architecture != ARCHITECTURE
            or type(model) is not network.AlphaZeroV2Net or sum(p.numel() for p in model.parameters()) != 325_896):
        raise SystemExit("Checkpoint is not the original generation-6 CNN")
    before, weights = ja.state(runner), runner.factory.hash(model)

    references = reference_moves(BitboardState)
    states = []
    for moves in references:
        state = BitboardState((0, 0), 0, 0)
        for move in moves:
            state = state.drop(move)
        network.validate_batch_state(state)
        states.append(state)
    original = [inference.predict(s) for s in states]

    artifact_sha256 = network.save_inference_checkpoint(artifact_path, model)  # atomic, never overwrites
    reloaded = network.load_inference_checkpoint(artifact_path)
    payload = torch.load(artifact_path, map_location="cpu", weights_only=True)
    if (set(payload) != {"contract", "model_state_dict"} or payload["contract"] != INFERENCE_CONTRACT
            or network.weights_sha256(reloaded.model) != weights
            or any(not torch.equal(reloaded.model.state_dict()[k], v) for k, v in model.state_dict().items())):
        raise SystemExit("Reloaded artifact differs from the original model")
    if [reloaded.predict(s) for s in states] != original:
        raise SystemExit("Reloaded predictions are not bit-identical to the original model")
    if ja.state(runner) != before or runner.factory.hash(model) != weights or sha256(checkpoint) != expected["sha256"]:
        raise SystemExit("Export changed the research model, learner state or checkpoint")

    manifest = {
        "format": "connect4-alphazero-production-artifact",
        "format_version": 1,
        "model_id": model_id,
        "artifact": {"file": artifact_path.name, "sha256": artifact_sha256, "size_bytes": artifact_path.stat().st_size},
        "inference_contract": INFERENCE_CONTRACT,
        "architecture": {"identity": ARCHITECTURE, "parameters": 325_896, "weights_sha256": weights},
        "search": {"algorithm": "connect4-alphazero-v2-puct", "backend": "bitboard", "simulations": 256,
                   "exploration": PUCT_EXPLORATION, "root_noise": None, "temperature": 0.0,
                   "final_selection": "maximum-visit, seeded tie-break",
                   "tie_break_domain": "connect4-alphazero-production-tie-break-v1"},
        "tactical_guard": {"contract": "connect4-tactical-root-guard-v1", "source_sha256": GUARD_SHA256,
                           "source": "research/alphazero_tactical_guard/guard.py", "default_enabled": False},
        "source": {
            "description": f"Phase 4D.4H-C generation-6 CNN, seed {args.seed}, original schedule",
            "phase": "4D.4H-C", "seed": args.seed, "generation": 6, "arm": "cnn",
            "checkpoint_sha256": expected["sha256"], "checkpoint_size_bytes": expected["size_bytes"],
            "checkpoint": f"experiment-output/phase4d4h-c/run/tasks/{args.seed}-cnn-g6-train/generation-0006.pt",
            "checkpoint_worktree": "phase4d4h-c-worktree",
            "ledger": f"research/alphazero_resnet_selfplay/results/run-evidence/tasks/{args.seed}-cnn-boundaries.json",
            "research_commit": RESEARCH_COMMIT,
            "strict_loader": "research.alphazero_tactical_guard.common.load (4D.4J-A)",
            "validated_in": {"4D.4J-A": "ffb8009b014ab6c84b9eb00ccab986fcb6bc4b14",
                             "4D.4J-B": "ae4715ff30d6796b010f115185ad90514e346141"},
            "learner_state_unchanged": True,
        },
        "export": {
            "tool": tool_rel, "tool_sha256": sha256(tool), "integration_commit": integration_commit,
            "gates": {"4D.4J-A_enforce": "PASS",
                      "4D.4J-B_freeze": "PASS except documented post-freeze file " + ", ".join(sorted(JB_POST_FREEZE)),
                      "guard_sha256": "PASS", "checkpoint_sha256": "PASS", "reload_bit_identical": "PASS"},
            "runtime": runtime,
            "runtime_sha256": hashlib.sha256(json.dumps(runtime, sort_keys=True).encode()).hexdigest(),
        },
        "reference_predictions": [{"moves": list(m), "logits": list(p.logits), "value": p.value}
                                  for m, p in zip(references, original)],
    }
    with manifest_path.open("x") as stream:
        json.dump(manifest, stream, indent=1, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"model_id": model_id, "artifact_sha256": artifact_sha256,
                      "size_bytes": manifest["artifact"]["size_bytes"], "weights_sha256": weights,
                      "manifest_sha256": sha256(manifest_path)}, indent=1))


if __name__ == "__main__":
    main()
