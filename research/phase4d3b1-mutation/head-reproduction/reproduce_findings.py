"""Phase 4D.3B.1 step 1: reproduce every launch-readiness finding against UNMODIFIED HEAD.

Run from the repository root with the torch venv. Uses temporary directories only;
no repository file is modified (source edits are virtual, via monkeypatching).
"""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, os.getcwd())
import torch  # noqa: E402

torch.set_num_threads(1)

from games.connect4.alphazero_v2 import campaign as C  # noqa: E402
from games.connect4.alphazero_v2 import packages as P  # noqa: E402
from games.connect4.alphazero_v2 import provenance  # noqa: E402
from games.connect4.alphazero_v2.config import V2Config  # noqa: E402

RESULTS = {}
FROZEN = Path("games/connect4/alphazero_v2/frozen/campaign-declaration.json")
OLD_TOKEN = "2741314399741c1b20510b8a2beb42e0938b0aeafc9dd0c487201e57208867d8"


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def tiny_packages(root):
    directory = Path(root) / "packages"
    saved = {name: getattr(P, name) for name in ('EXCLUSION_SOURCES', 'TACTICAL_QUOTAS', 'SOLVED_QUOTAS',
                                                 'SOLVED_STAGES', 'OPENING_PREFIX_BASES', 'EMPTY_BOARD_PAIRS',
                                                 'OVERSAMPLE')}
    P.EXCLUSION_SOURCES = P.EXCLUSION_SOURCES[:3]
    P.TACTICAL_QUOTAS, P.SOLVED_QUOTAS = dict(sealed=1, development=1), dict(sealed=1, development=1)
    P.SOLVED_STAGES = (('late', 24, 41),)
    P.OPENING_PREFIX_BASES, P.EMPTY_BOARD_PAIRS, P.OVERSAMPLE = dict(sealed=2, development=2), 1, 1
    try:
        P.build_all(directory, seed=7, log=lambda message: None)
    finally:
        for name, value in saved.items():
            setattr(P, name, value)
    return directory


def tiny_declaration(package_dir, name="declaration.json", **budget_overrides):
    manifest = json.loads((Path(package_dir) / 'manifest.json').read_text())
    config = V2Config(self_play_simulations=2, games_per_generation=2, replay_generations=2,
                      replay_max_games=4, batch_size=8, max_generations=2).to_dict()
    declaration = C.build_declaration(
        {k: v for k, v in manifest.items() if k != 'exclusions'}, name='tiny-repro', kind='test', config=config,
        seeds=(42,), primary_seed=42,
        budgets=dict(dict(per_run_training_seconds=3600, campaign_seconds=7200, evaluation_games_ceiling=200,
                          planned_evaluation_games=0), **budget_overrides),
        evaluation=dict(simulations=2, tactical_seeds=[0], root_noise=False, tactical_guard=False, temperature=0,
                        ties='seeded uniform'),
        champion=dict(schedule=[1, 2], arena_openings=2, baseline_opponents=['random'],
                      baseline_rows=C.baseline_rows(empty_pairs=1, prefix_families=0),
                      gate={k: v for k, v in C.statistics.CHAMPION_GATE.items() if k != 'schedule'},
                      tactical_package='tactical-development'),
        final=dict(ladder=['random'], ladder_openings=1, calibration_games=1, calibration_constant=None,
                   one_time=True))
    path = Path(package_dir) / name
    path.write_text(json.dumps(declaration, indent=1, sort_keys=True))
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def b1_source_binding():
    """Virtual one-byte edit of search.py: digest changes, preflight and Campaign still accept."""
    before = provenance.execution_source_identity()["sha256"]
    original = Path.read_bytes
    target = (provenance.REPO_ROOT / "games/connect4/alphazero_v2/search.py").resolve()

    def edited(self):
        data = original(self)
        return data + b"#" if Path(self).resolve() == target else data
    Path.read_bytes = edited
    try:
        after = provenance.execution_source_identity()["sha256"]
        report = C.preflight(FROZEN)
        with tempfile.TemporaryDirectory() as tmp:
            campaign = C.Campaign(Path(tmp) / "c", FROZEN, OLD_TOKEN)
            accepted = campaign.declaration_sha256
    finally:
        Path.read_bytes = original
    declaration = json.loads(FROZEN.read_text())
    RESULTS["B1_source_not_bound"] = dict(
        digest_before=before, digest_after_virtual_edit=after,
        preflight_token_after_edit=report["declaration_sha256"],
        campaign_accepted_old_token_after_edit=accepted == OLD_TOKEN,
        declaration_has_source_or_runtime_identity=any(k in declaration for k in (
            "execution_source", "runtime_identity", "source")),
        final_evaluate_references_identity="execution_source_identity" in
        Path(C.__file__).read_text().split("def final_evaluate")[1].split("def _final_seed")[0])


def b2_generation_publication(package_dir):
    """Crash after archival strands the archive; crash in diagnostics silently skips them."""
    declaration, token = tiny_declaration(package_dir, name="b2.json")
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp) / "c1"
        C.Campaign.require_runtime = lambda self: provenance.runtime_identity()
        real_save = C.Campaign._save_boundary
        calls = dict(n=0)

        def failing_save(self, seed, attempt, attempt_dir, runner, events):
            calls["n"] += 1
            if calls["n"] == 2:  # generation 1's boundary, after generation 1's games were archived
                raise RuntimeError("injected crash after archival")
            return real_save(self, seed, attempt, attempt_dir, runner, events)
        C.Campaign._save_boundary = failing_save
        try:
            C.Campaign(directory, declaration, token).run_seed(42)
        except RuntimeError as error:
            out["first_attempt"] = str(error)
        C.Campaign._save_boundary = real_save
        try:
            C.Campaign(directory, declaration, token).run_seed(42)
            out["resume_after_archive_crash"] = "completed"
        except FileExistsError as error:
            out["resume_after_archive_crash"] = f"FileExistsError: {Path(str(error).split()[-1]).name}"

        directory = Path(tmp) / "c2"
        real_metrics = C.Campaign._raw_value_metrics
        diag = dict(n=0, armed=True)

        def failing_metrics(self, *args):
            diag["n"] += 1
            if diag["armed"]:
                diag["armed"] = False
                raise RuntimeError("injected crash in diagnostics")
            return real_metrics(self, *args)
        C.Campaign._raw_value_metrics = failing_metrics
        try:
            C.Campaign(directory, declaration, token).run_seed(42)
        except RuntimeError:
            pass
        diag["n"] = 0
        status = C.Campaign(directory, declaration, token).run_seed(42)
        C.Campaign._raw_value_metrics = real_metrics
        summaries = sorted(p.name for p in (directory / "runs/seed-42").rglob("generation-0001.summary.json"))
        out["resume_after_diagnostics_crash"] = dict(status=status, generation1_summary_files=summaries,
                                                     diagnostic_calls_on_resume=diag["n"])

        directory = Path(tmp) / "c3"
        real_publish = C.Campaign._publish

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
        try:
            C.Campaign(directory, declaration, token).run_seed(42)
            out["resume_after_partial_registration"] = "completed"
        except (RuntimeError, FileExistsError) as error:
            out["resume_after_partial_registration"] = f"{type(error).__name__}: {error}"
    RESULTS["B2_generation_publication"] = out


def b3_deadlines():
    out = {}
    # (a) Final calibration receives no check callback and completion is not rechecked.
    source = Path(C.__file__).read_text()
    call = source.split("calibration_games(\n")[1].split("))")[0] if "calibration_games(\n" in source else \
        source.split("calibration.extend(")[1].split("budget.count_evaluation_game")[0]
    out["calibration_call_passes_check"] = "check=" in call
    # Deterministic run of the final orchestration with fakes and a fake clock.
    clock = Clock()
    import games.connect4.alphazero_v2.evaluation as E

    class FakeAgent:
        name = "fake"
        inference = None

        def describe(self):
            return {}
    seen = {}

    def fake_calibration(model, config, *, games, seed, check=None):
        seen["check_supplied"] = check is not None
        clock.now = 2.0  # last calibration operation overruns the 1-second cap
        campaign._budget.request_stop("stop requested during calibration")
        return [dict(game=0, ply=0, actor=0, value=0.0, outcome=1)]
    patches = dict(load_v2_agent=lambda *a, **k: FakeAgent(), search_rows=lambda *a, **k: ({}, {}),
                   nn_only_rows=lambda *a, **k: {}, raw_values=lambda *a, **k: {},
                   calibration_games=fake_calibration, untrained_model=lambda seed: None,
                   V2NNOnlyAgent=lambda *a, **k: FakeAgent())
    saved = {k: getattr(E, k) for k in patches}
    for k, v in patches.items():
        setattr(E, k, v)
    saved_c = dict(package_rows=C.package_rows, training_overlap=C.training_overlap,
                   run_paired_arena=C.run_paired_arena, statistics=C.statistics)
    C.package_rows = lambda d, b, name: []
    C.training_overlap = lambda run, h: set()
    C.run_paired_arena = lambda *a, **k: ([], dict(complete=True, stop_reason=None))

    class Stats:
        def __getattr__(self, name):
            return lambda *a, **k: {}
    C.statistics = Stats()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            campaign = C.Campaign.__new__(C.Campaign)
            campaign.declaration = dict(seeds=[42], budgets=dict(campaign_seconds=1, per_run_training_seconds=1,
                                                                 evaluation_games_ceiling=10),
                                        final=dict(ladder=[], ladder_openings=0, calibration_games=1,
                                                   calibration_constant=None),
                                        evaluation=dict(simulations=2, tactical_seeds=[0]),
                                        config=V2Config().to_dict(), thresholds=dict(arena={}))
            campaign.declaration["config"].pop("seed")
            campaign.declaration_sha256, campaign.base, campaign.directory = "x", directory, directory
            campaign.clock, campaign.ledger = clock, C.JsonLog(directory / "ledger.jsonl")
            campaign.require_runtime = lambda: {}
            (directory / "runs/seed-42").mkdir(parents=True)
            (directory / "runs/seed-42/selection.json").write_text(json.dumps(dict(inference=dict(path="x", sha256="y"))))
            status = campaign.final_evaluate()
            out["final_with_overrun_and_stop"] = dict(status=status, check_supplied=seen["check_supplied"],
                                                      seed_result_written=(directory / "final/seed-42.json").exists())
    finally:
        for k, v in saved.items():
            setattr(E, k, v)
        for k, v in saved_c.items():
            setattr(C, k, v)
    # (b) Training: the check precedes updates; an overrunning last update is never rechecked.
    with tempfile.TemporaryDirectory() as tmp:
        clock = Clock()
        budget = C.Budget(C.JsonLog(Path(tmp) / "l.jsonl"), dict(budgets=dict(
            per_run_training_seconds=100, campaign_seconds=1000, evaluation_games_ceiling=3)), 42, 1, clock=clock)
        budget.begin_training()
        clock.now = 99
        budget.check("training")  # permits the last update
        clock.now = 101           # the update overruns
        budget.end_training()
        budget.check("evaluation")  # evaluation does not enforce the training cap
        out["training_overrun_then_evaluation_allowed"] = budget.training_seconds()
        # (c) Exactly reaching the game ceiling blocks later NON-game evaluation work.
        for _ in range(3):
            budget.check("evaluation")
            budget.count_evaluation_game({})
        try:
            budget.check("evaluation")  # e.g. the tactical pass after the last allowed arena game
            out["position_work_after_exact_ceiling"] = "allowed"
        except C.CampaignStop as stop:
            out["position_work_after_exact_ceiling"] = f"refused: {stop}"
    RESULTS["B3_deadlines"] = out


def b4_accounting():
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        ledger, clock = C.JsonLog(Path(tmp) / "ledger.jsonl"), Clock()
        budget = C.Budget(ledger, dict(budgets=dict(per_run_training_seconds=100, campaign_seconds=1000,
                                                    evaluation_games_ceiling=6400)), "final", 1, clock=clock)
        budget.heartbeat(force=True)
        for _ in range(5):
            budget.check("evaluation")
            budget.count_evaluation_game({})
        # hard crash here: no attempt_end
        out["in_memory_games"] = budget.evaluation_games
        out["recovered_games_after_crash"] = C.consumed(ledger.records())["evaluation_games"]
        with open(ledger.path, "a") as stream:
            stream.write('{"event": "heartbeat", "attempt": 2, "se')  # torn trailing record
        try:
            ledger.records()
            out["torn_record"] = "read"
        except json.JSONDecodeError as error:
            out["torn_record"] = f"JSONDecodeError: {error.msg}"
        out["training_games_counts_only_full_generations"] = \
            "budget.training_games += config.games_per_generation" in Path(C.__file__).read_text()
        out["campaign_lock_present"] = any(w in Path(C.__file__).read_text() for w in ("flock", "lockf", "O_EXCL"))
    RESULTS["B4_accounting"] = out


def s_findings(package_dir):
    from games.connect4.alphazero_v2.oracle import BitboardSolver, exhaustive_action_values
    out = {}
    try:
        out["S3_action_values_on_won_history"] = BitboardSolver().action_values([0, 1, 0, 1, 0, 1, 0])
    except Exception as error:  # noqa: BLE001
        out["S3_action_values_on_won_history"] = f"refused: {error}"
    try:
        out["S3_exhaustive_action_values_on_won_history"] = exhaustive_action_values([0, 1, 0, 1, 0, 1, 0])
    except Exception as error:  # noqa: BLE001
        out["S3_exhaustive_action_values_on_won_history"] = f"refused: {error}"
    source = Path(C.__file__).read_text()
    final_eval = source.split("def final_evaluate")[1].split("def _final_seed")[0]
    out["S1_final_evaluate_catches_unexpected_exceptions"] = "except BaseException" in final_eval
    final_seed = source.split("def _final_seed")[1].split("OPPONENT_NAMES")[0]
    out["S1_seed_result_written_only_after_complete_seed"] = "write_json_once(final / f\"seed-{seed}.json\", result)" in final_eval
    out["S2_final_result_keeps_row_choices"] = "choices=" in final_seed or "\"choices\"" in final_seed
    out["S2_learned_readiness_measurement_in_final"] = "p95" in final_seed or "readiness" in final_seed
    report = C.preflight(FROZEN)
    out["S4_preflight_inter_op_threads"] = report["runtime"]["inter_op_threads"]
    out["S4_preflight_deterministic"] = report["runtime"]["deterministic_algorithms"]
    out["S4_preflight_checks_phase4d2f_checkpoint"] = "phase4d2f" in source.split("def preflight")[1].split("def main")[0]
    out["S4_cpu_model_none_compares_equal"] = provenance.runtime_differences(
        dict(cpu_model=None), dict(cpu_model=None)) == []
    RESULTS["S_findings"] = out


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as root:
        packages = tiny_packages(root)
        b1_source_binding()
        b2_generation_publication(packages)
        b3_deadlines()
        b4_accounting()
        s_findings(packages)
    print(json.dumps(RESULTS, indent=1, sort_keys=True, default=str))
