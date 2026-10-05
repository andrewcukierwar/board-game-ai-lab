"""Extra precision probes on unmodified HEAD (B2 registration duplicate, B2 per-generation diagnostics, S1 status)."""
import json, os, sys, tempfile
from pathlib import Path
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(__file__))
import torch
torch.set_num_threads(1)
import reproduce_findings as R
from games.connect4.alphazero_v2 import campaign as C, provenance

out = {}
with tempfile.TemporaryDirectory() as root:
    packages = R.tiny_packages(root)
    declaration, token = R.tiny_declaration(packages, name="x.json")
    C.Campaign.require_runtime = lambda self: provenance.runtime_identity()
    # B2(c): crash between inference and resume registration; bypass the archive collision to expose the duplicate.
    directory = Path(root) / "dup"
    real_publish, real_archive = C.Campaign._publish, C.Campaign._archive_games
    def failing_publish(self, seed, attempt, kind, generation, path):
        if kind == "resume" and generation == 1:
            raise RuntimeError("injected crash between inference and resume registration")
        return real_publish(self, seed, attempt, kind, generation, path)
    C.Campaign._publish = failing_publish
    try:
        C.Campaign(directory, declaration, token).run_seed(42)
    except RuntimeError:
        pass
    C.Campaign._publish = real_publish
    def tolerant_archive(self, seed, runner, generation):
        if (self.run_dir(seed) / "games" / f"generation-{generation:04d}.jsonl").exists():
            return
        return real_archive(self, seed, runner, generation)
    C.Campaign._archive_games = tolerant_archive
    try:
        C.Campaign(directory, declaration, token).run_seed(42)
        out["B2_retry_after_partial_registration_with_archive_bypassed"] = "completed"
    except RuntimeError as error:
        out["B2_retry_after_partial_registration_with_archive_bypassed"] = f"RuntimeError: {error}"
    records = [json.loads(l) for l in (directory / "runs/seed-42/artifacts.jsonl").read_text().splitlines()]
    out["B2_inference_records_for_generation_1"] = sum(r["kind"] == "inference" and r["generation"] == 1 for r in records)
    C.Campaign._archive_games = real_archive
    # B2(b): which generations' diagnostics were computed on resume after a diagnostics crash in generation 1.
    directory = Path(root) / "diag"
    real_metrics = C.Campaign._raw_value_metrics
    state = dict(armed=True, calls=[])
    def metrics(self, attempt_dir, generation, rows):
        state["calls"].append(generation)
        if state["armed"]:
            state["armed"] = False
            raise RuntimeError("injected crash in generation-1 diagnostics")
        return real_metrics(self, attempt_dir, generation, rows)
    C.Campaign._raw_value_metrics = metrics
    try:
        C.Campaign(directory, declaration, token).run_seed(42)
    except RuntimeError:
        pass
    state["calls"] = []
    status = C.Campaign(directory, declaration, token).run_seed(42)
    C.Campaign._raw_value_metrics = real_metrics
    out["B2_diagnostics_on_resume"] = dict(status=status, generations_with_diagnostic_calls=state["calls"],
                                           summary_files=sorted(p.name for p in directory.rglob("*.summary.json")))
    # S1: unexpected exception inside final evaluation leaves attempt_end status "running".
    real_final_seed = C.Campaign._final_seed
    C.Campaign._final_seed = lambda self, *a: (_ for _ in ()).throw(KeyError("injected"))
    try:
        C.Campaign(directory, declaration, token).final_evaluate()
    except KeyError:
        pass
    C.Campaign._final_seed = real_final_seed
    ledger = [json.loads(l) for l in (directory / "ledger.jsonl").read_text().splitlines()]
    out["S1_final_attempt_end_status_after_exception"] = [r["status"] for r in ledger
                                                          if r["event"] == "attempt_end" and r["seed"] == "final"]
    # B4: two concurrent Campaign objects on one directory both construct and read the same prior totals.
    a, b = C.Campaign(directory, declaration, token), C.Campaign(directory, declaration, token)
    out["B4_two_concurrent_invocations_accepted"] = a.directory == b.directory
print(json.dumps(out, indent=1, sort_keys=True))
