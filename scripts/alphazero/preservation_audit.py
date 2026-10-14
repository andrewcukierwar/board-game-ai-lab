"""Hash-only historical preservation audit for Phase 4D.4K-A (read-only).

Runs in the research environment. Reuses the research packages' own
preservation checks (4D.4J-A and 4D.4J-B ``preservation()``: hash-only
inventories of G/H/I/J outputs, results, docs, canonical tracked files and the
150 official campaign files; sealed payloads are hashed, never parsed or
printed), then adds checkpoint hashes, research-worktree tracked cleanliness
(HEADs are observations: other sessions may commit), this branch's diff scope
and the live backend's container/checkout identity, without contacting it.

    PYTHONDONTWRITEBYTECODE=1 ~/.venvs/board-game-alphazero-m4/bin/python -B \\
        scripts/alphazero/preservation_audit.py --research-root <phase4d4j-b worktree> \\
        --live-baseline <file with container id, image, start, restarts, services HEAD> --output audit.json
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

CHECKPOINTS = {
    "phase4d4h-c-worktree/experiment-output/phase4d4h-c/run/tasks/42-cnn-g6-train/generation-0006.pt":
        "ba2bde2ec71b3ac7486046d1cc26ab591bb59d6b764193ece08c6d6a6f24f58b",
    "phase4d4h-c-worktree/experiment-output/phase4d4h-c/run/tasks/314159-cnn-g6-train/generation-0006.pt":
        "2646d7cdee7d7b069412688bf95fd9794bea30402b4952120c904dbcb399401d",
}
PROTECTED_PATHS = ("benchmarks/connect4", "models/connect4", "ui", "docs/mac-mini-deployment.md",
                   "scripts/deploy-mac-mini-api.sh", ".github", "docker-compose.yml")


def run(*args, cwd=None):
    done = subprocess.run(list(args), cwd=cwd, capture_output=True, text=True)
    return done.stdout.strip() if done.returncode == 0 else None


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--research-root", required=True)
    parser.add_argument("--live-baseline", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    research = Path(args.research_root).resolve()
    worktrees = research.parent
    integration = Path(__file__).resolve().parents[2]
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(research))
    from research.alphazero_tactical_guard import common as ja
    from research.alphazero_tactical_guard_confirmation import common as jb

    report = dict(format="connect4-alphazero-preservation-audit", format_version=1)
    for name, module in (("4D.4J-A", ja), ("4D.4J-B", jb)):
        result = module.preservation()
        changed = {k: v["changed"] for k, v in result.get("scopes", {}).items() if v["changed"]}
        report[name] = dict(intact=result["intact"], scopes=len(result.get("scopes", {})),
                            files=sum(v["files"] for v in result.get("scopes", {}).values()), changed=changed)
    official = ja.hc.preserve()  # raises unless all 150 official files are byte-identical
    report["official_campaign"] = dict(files=len(official), intact=True)
    report["checkpoints"] = {p: sha256(worktrees / p) == h for p, h in CHECKPOINTS.items()}
    observed = {}
    for tree in sorted(worktrees.glob("phase4d4*-worktree")) + [worktrees / "phase4d4a-main-reference"]:
        if tree.resolve() == integration:
            continue
        observed[tree.name] = dict(head=run("git", "-C", str(tree), "rev-parse", "HEAD"),
                                   tracked_clean=run("git", "-C", str(tree), "status", "--porcelain",
                                                     "--untracked-files=no") == "")
    historical = worktrees.parent
    observed["historical-checkout"] = dict(head=run("git", "-C", str(historical), "rev-parse", "HEAD"),
                                           tracked_clean=run("git", "-C", str(historical), "status", "--porcelain",
                                                             "--untracked-files=no") == "")
    report["worktrees"] = observed
    diff = run("git", "-C", str(integration), "diff", "--name-only", "origin/main", "--", *PROTECTED_PATHS)
    report["integration_branch_protected_paths_changed"] = diff.splitlines() if diff else []
    baseline = Path(args.live_baseline).read_text().split()
    live = run("docker", "inspect", "bgai-api", "--format", "{{.Id}} {{.Image}} {{.State.StartedAt}} {{.RestartCount}}")
    services = run("git", "-C", str(Path.home() / "Services/board-game-ai-lab-api"), "rev-parse", "HEAD")
    report["live_backend"] = dict(container_unchanged=live is not None and live.split() == baseline[:4],
                                  services_checkout_unchanged=services == baseline[4], services_head=services)
    report["intact"] = (all(report[k]["intact"] for k in ("4D.4J-A", "4D.4J-B", "official_campaign"))
                        and all(report["checkpoints"].values())
                        and all(v["tracked_clean"] for v in observed.values())
                        and not report["integration_branch_protected_paths_changed"]
                        and report["live_backend"]["container_unchanged"]
                        and report["live_backend"]["services_checkout_unchanged"])
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Refusing to overwrite the audit")
    output.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print(json.dumps(report, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
