"""Bounded, explicitly authorized AlphaZero v2 campaign launcher (Milestone 3 tool).

Nothing runs on import. Every command that does campaign work (``run``,
``final-evaluate``) fails closed unless all of the following hold:

* ``--authorize`` equals the SHA-256 of the exact declaration file, and that
  token is not rejected. The Phase 4D.3B token ``2741314399...`` is rejected.
* The declaration (format 2) validates, and every input it binds is identical
  in this process:
  - execution-source content, in its training and evaluation/launch groups;
  - frozen package and frozen-record bytes;
  - the retained Phase 4D.2f checkpoint;
  - campaign configuration and launch-control semantics;
  - the complete runtime identity, after the deterministic runtime is
    configured: Python, torch build, NumPy, platform/OS, machine, CPU model,
    threads, deterministic flags and the library thread environment.
  A runtime that cannot identify one of those fields is refused.
* No other invocation holds the campaign directory (``flock``).

Identity is re-verified at every phase boundary and before every publication.
Git commit and dirty state are recorded as provenance only.

Layout of a campaign directory (created once; nothing is overwritten):

    declaration.json         verbatim copy; its hash is the authorization token
    journal.jsonl            the single authority (launch_control): checksummed, fsynced
                             transitions, time leases, unit reservations and evidence
    campaign.lock            single-writer lock
    attempts/attempt-NNN/    source snapshot of each invocation (provenance)
    runs/seed-S/generations/generation-NNNN/   committed inference, resume and archived games
    runs/seed-S/staging/     generation outputs awaiting a commit record (never authoritative)
    runs/seed-S/*.json, final/*.json, outcome.json
                             views derived from the journal, written once and verified on
                             every recovery; status.json is a replaceable summary

Crash recovery, deadlines and budgets: see ``launch_control`` and
docs/phase4d3b1-launch-control-fixes.md.
"""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from . import statistics
from .arena import (PAIRED_GAMES, GuardedUCTOpponent, NegamaxOpponent, RandomOpponent, paired_game, scored)
from .config import V2Config
from .launch_control import (GAME_KINDS, JOURNAL_FORMAT, LAUNCH_CONTROL_VERSION, LEASE_SECONDS,
                             REJECTED_DECLARATION_TOKENS, RENEW_BELOW_SECONDS, Budget, BudgetExhausted,
                             CampaignLock, CampaignLocked, CampaignState, CampaignStop, InconsistentCampaign, Journal,
                             close_open_work, fsync_directory, publish_view, read_state, recover_open_attempts,
                             replace_view, sha256_bytes, sha256_file, view_bytes, write_once)
from .oracle import family_key
from .packages import load_package
from .provenance import (REPO_ROOT, SOURCE_SCHEME, configure_deterministic_runtime, execution_source_files,
                         execution_source_identity, git_provenance, required_thread_environment, runtime_differences,
                         runtime_identity, source_differences, source_group, source_groups,
                         unavailable_runtime_fields)

DECLARATION_FORMAT = "connect4-alphazero-v2-campaign-declaration"
DECLARATION_VERSION = 2
UNTRACKED_LIMIT_BYTES = 100 * 1024 * 1024
DECLARATION_KEYS = {"format", "format_version", "name", "kind", "seeds", "primary_seed", "config", "generations",
                    "budgets", "runtime", "evaluation", "champion", "final", "packages", "thresholds",
                    "retention", "phase4d2f_checkpoint", "notes",
                    # Phase 4D.3B.1 bindings:
                    "execution_source", "runtime_identity", "frozen_records", "launch_control"}
FROZEN_RECORD_NAMES = ("build-provenance.json", "exclusions.json", "manifest.json")
RUNTIME_IDENTITY_KEYS = tuple(sorted(runtime_identity()))
PHASE4D2F_CHECKPOINT = dict(path="experiment-output/phase4d2f-neural-value-anchoring-seed42-20261005/candidate.pt",
                            sha256="78276912bdef5f1b1b53ac6b2ec3ede2eed20a8ebb5796a9bdcbb57e4cb28c51")


class LaunchRefused(PermissionError):
    """The declaration does not authorize this invocation (token, source, runtime or frozen inputs)."""


def write_json_once(path, value):
    """Write JSON to a new file only (temporary file, fsync, hard-link publish)."""
    return write_once(path, (json.dumps(value, indent=1, sort_keys=True, allow_nan=False) + "\n").encode())


def json_normalized(value):
    return json.loads(json.dumps(value))


# Declaration ------------------------------------------------------------------------------

def launch_control_declaration():
    """Launch-control semantics bound by the declaration (the executable meaning is bound by source)."""
    return dict(
        version=LAUNCH_CONTROL_VERSION, journal=JOURNAL_FORMAT,
        single_writer="flock on campaign.lock; a concurrent invocation is refused",
        lease_seconds=LEASE_SECONDS, renew_below_seconds=RENEW_BELOW_SECONDS,
        time_accounting=("Active invocation seconds. Downtime between invocations is not charged. An attempt that "
                         "ends normally is charged its measured time; a hard interruption is charged through its "
                         "last durable lease."),
        evaluation_games=("Charged when the game's durable reservation precedes its first move. Abandoned and "
                          "unclosed reservations stay charged and are never evidence. A game may start only while "
                          "charged games are below the ceiling."),
        deadlines=("A check precedes every search, move and optimizer update and refuses to start work at a limit. "
                   "An in-flight primitive may finish past a cooperative deadline, but a generation, decision, "
                   "selection or final result is published only if every charged limit still holds at its "
                   "completion. Otherwise the campaign is INCOMPLETE."),
        recovery=("Journal-first commits. Completed units are never rerun. Interrupted units are abandoned "
                  "(still charged) and rerun. Interrupted generations are discarded and rerun from the last "
                  "committed boundary. Inconsistent published outputs are refused."),
        final_evaluation=("Selections are fixed by final_started before any sealed inference. Each sealed unit "
                          "completes at most once. The evaluation resumes at unit granularity; it is never "
                          "restarted from scratch and never reselected."),
        rejected_declaration_tokens=list(REJECTED_DECLARATION_TOKENS))


def execution_source_declaration():
    identity = execution_source_identity()
    return dict(identity, groups=source_groups(identity["files"]))


def frozen_record_entries(directory, names=FROZEN_RECORD_NAMES):
    return {name: dict(path=name, sha256=sha256_file(Path(directory) / name)) for name in names}


def load_declaration(path):
    path = Path(path)
    data = path.read_bytes()
    digest = sha256_bytes(data)
    if digest in REJECTED_DECLARATION_TOKENS:
        raise LaunchRefused(f"Declaration token {digest} is REJECTED / NOT AUTHORIZED (Phase 4D.3B launch-readiness "
                            "review); it must never authorize a campaign")
    declaration = json.loads(data)
    validate_declaration(declaration, path.parent)
    return declaration, digest


def validate_declaration(declaration, base):
    if set(declaration) != DECLARATION_KEYS:
        raise ValueError(f"Declaration keys differ: {sorted(set(declaration) ^ DECLARATION_KEYS)}")
    if (declaration["format"], declaration["format_version"]) != (DECLARATION_FORMAT, DECLARATION_VERSION):
        raise ValueError("Not a format-2 v2 campaign declaration")
    seeds = declaration["seeds"]
    if not seeds or len(set(seeds)) != len(seeds) or declaration["primary_seed"] not in seeds:
        raise ValueError("Seeds must be distinct and include the primary seed")
    for seed in seeds:
        config = V2Config.from_dict(dict(declaration["config"], seed=seed))
        if config.max_generations != declaration["generations"]:
            raise ValueError("config.max_generations must equal the declared generations")
    budgets = declaration["budgets"]
    for name in ("per_run_training_seconds", "campaign_seconds", "evaluation_games_ceiling"):
        if not isinstance(budgets[name], (int, float)) or budgets[name] <= 0:
            raise ValueError(f"Budget {name} must be positive")
    if budgets["planned_evaluation_games"] > budgets["evaluation_games_ceiling"]:
        raise ValueError("Planned evaluation games exceed the ceiling")
    schedule = declaration["champion"]["schedule"]
    if sorted(set(schedule)) != schedule or not all(1 <= g <= declaration["generations"] for g in schedule):
        raise ValueError("Champion schedule must be increasing generations within the campaign")
    if planned_evaluation_games(declaration) != budgets["planned_evaluation_games"]:
        raise ValueError(f"Planned evaluation games must be {planned_evaluation_games(declaration)}")
    # The launcher executes statistics.CHAMPION_GATE and ACCEPTANCE_THRESHOLDS; a declaration may not differ.
    if declaration["champion"]["gate"] != {k: v for k, v in statistics.CHAMPION_GATE.items() if k != "schedule"}:
        raise ValueError("Declared champion gate differs from the executed gate")
    if declaration["thresholds"] != json_normalized(statistics.ACCEPTANCE_THRESHOLDS):
        raise ValueError("Declared thresholds differ from the executed thresholds")
    if declaration["launch_control"] != json_normalized(launch_control_declaration()):
        raise ValueError("Declared launch-control semantics differ from this launcher")
    source = declaration["execution_source"]
    if (set(source) != {"scheme", "sha256", "files", "groups"} or source["scheme"] != SOURCE_SCHEME
            or source["sha256"] != hashlib.sha256(json.dumps(source["files"], sort_keys=True).encode()).hexdigest()
            or source["groups"] != source_groups(source["files"])):
        raise ValueError("Declared execution source is internally inconsistent")
    runtime = declaration["runtime_identity"]
    if not isinstance(runtime, dict) or tuple(sorted(runtime)) != RUNTIME_IDENTITY_KEYS:
        raise ValueError("Declared runtime identity must contain exactly the enforced runtime fields")
    if unavailable_runtime_fields(runtime):
        raise ValueError(f"Declared runtime identity has unavailable fields: {unavailable_runtime_fields(runtime)}")
    threads = declaration["runtime"]["threads"]
    if ((runtime["intra_op_threads"], runtime["inter_op_threads"], runtime["deterministic_algorithms"])
            != (threads, threads, declaration["runtime"]["deterministic_algorithms"])
            or runtime["thread_environment"] != required_thread_environment(threads)):
        raise ValueError("Declared runtime identity disagrees with the declared runtime settings")
    records = declaration["frozen_records"]
    if declaration["kind"] == "research" and sorted(records) != sorted(FROZEN_RECORD_NAMES):
        raise ValueError(f"A research declaration must bind {list(FROZEN_RECORD_NAMES)}")
    for name, entry in records.items():
        if sha256_file(Path(base) / entry["path"]) != entry["sha256"]:
            raise ValueError(f"Frozen record {name} differs from the declaration")
    if set(declaration["phase4d2f_checkpoint"]) != {"path", "sha256"}:
        raise ValueError("phase4d2f_checkpoint needs path and sha256")
    for name, entry in declaration["packages"].items():
        load_package(Path(base) / entry["path"], expected_sha256=entry["sha256"])
    return True


def identity_problems(declaration, runtime):
    """Every way this process differs from the declared execution source and runtime."""
    problems = []
    declared, current = declaration["execution_source"], execution_source_identity()
    changed = source_differences(declared, current)
    if declared["sha256"] != current["sha256"] and not changed:
        changed = ["<combined digest>"]
    by_group = {}
    for path in changed:
        by_group.setdefault(source_group(path), []).append(path)
    for group, files in sorted(by_group.items()):
        problems.append(f"{group} source differs from the declaration: {files}")
    missing = unavailable_runtime_fields(runtime)
    if missing:
        problems.append(f"runtime identity fields unavailable (an exact launch cannot be claimed): {missing}")
    differing = runtime_differences(declaration["runtime_identity"], runtime)
    if differing:
        problems.append(f"runtime differs from the declaration: {differing}")
    return problems


def checkpoint_problems(declaration):
    entry = declaration["phase4d2f_checkpoint"]
    path = REPO_ROOT / entry["path"]
    if not path.is_file():
        return [f"retained Phase 4D.2f checkpoint missing: {entry['path']}"]
    if sha256_file(path) != entry["sha256"]:
        return [f"retained Phase 4D.2f checkpoint hash differs: {entry['path']}"]
    return []


def import_execution_closure():
    """Load every declared execution module now, so later lazy imports cannot pick up edited code."""
    for relative in execution_source_files():
        module = relative[:-len(".py")].replace("/", ".")
        importlib.import_module(module[:-len(".__init__")] if module.endswith(".__init__") else module)


def package_rows(declaration, base, name):
    entry = declaration["packages"][name]
    return load_package(Path(base) / entry["path"], expected_sha256=entry["sha256"])["rows"]


def planned_evaluation_games(declaration):
    champion, final = declaration["champion"], declaration["final"]
    per_check = 2 * champion["arena_openings"] + 2 * len(champion["baseline_rows"]) * len(champion["baseline_opponents"])
    per_seed = (len(champion["schedule"]) * per_check
                + 2 * final["ladder_openings"] * len(final["ladder"])
                + 2 * final["ladder_openings"]  # NN Only versus Random
                + final["calibration_games"])
    return per_seed * len(declaration["seeds"])


def baseline_rows(prefix="development", empty_pairs=4, prefix_families=8):
    return ([f"{prefix}-empty-{i:02d}" for i in range(empty_pairs)]
            + [f"{prefix}-prefix-{i:02d}-{o}" for i in range(prefix_families) for o in ("base", "mirror")])


def build_declaration(packages, *, name="phase4d3c-alphazero-v2-two-seed", kind="research", config=None,
                      seeds=(42, 314159), primary_seed=42, budgets=None, evaluation=None, champion=None, final=None,
                      runtime=None, notes=None, execution_source=None, runtime_identity=None, frozen_records=None,
                      phase4d2f_checkpoint=None):
    """Assemble the frozen declaration; ``packages`` maps package name -> {path, sha256}.

    ``runtime_identity`` must be the identity of the configured launch runtime
    (``freeze`` records it); ``execution_source`` defaults to the current closure.
    """
    config = dict(V2Config().to_dict() if config is None else config)
    config.pop("seed", None)
    champion = champion or dict(schedule=list(statistics.CHAMPION_GATE["schedule"]), arena_openings=100,
                                baseline_opponents=["random", "negamax1", "negamax2"], baseline_rows=baseline_rows(),
                                gate={k: v for k, v in statistics.CHAMPION_GATE.items() if k != "schedule"},
                                tactical_package="tactical-development")
    final = final or dict(ladder=["random", "negamax1", "negamax2", "negamax4", "guarded_uct_800", "initial_v2_512",
                                  "phase4d2f_512"], ladder_openings=100, calibration_games=256,
                          calibration_constant=None, one_time=True)
    declaration = dict(
        format=DECLARATION_FORMAT, format_version=DECLARATION_VERSION, name=name, kind=kind, seeds=list(seeds),
        primary_seed=primary_seed, config=config, generations=config["max_generations"],
        budgets=budgets or dict(per_run_training_seconds=8 * 3600, campaign_seconds=24 * 3600,
                                evaluation_games_ceiling=6400, planned_evaluation_games=0),
        runtime=runtime or dict(threads=1, deterministic_algorithms=True, strict_resume_runtime=True,
                                strict_resume_source=True),
        evaluation=evaluation or dict(simulations=512, tactical_seeds=[0, 1, 2, 3], root_noise=False,
                                      tactical_guard=False, temperature=0, ties="seeded uniform"),
        champion=champion, final=final, packages=packages,
        thresholds=statistics.ACCEPTANCE_THRESHOLDS,
        retention=dict(resume_boundaries=2, inference_snapshots="all", archived_games="all"),
        phase4d2f_checkpoint=phase4d2f_checkpoint or dict(PHASE4D2F_CHECKPOINT),
        notes=notes or [],
        execution_source=execution_source or execution_source_declaration(),
        runtime_identity=runtime_identity, frozen_records=frozen_records or {},
        launch_control=launch_control_declaration())
    declaration = json_normalized(declaration)  # tuples -> lists, as stored
    declaration["budgets"]["planned_evaluation_games"] = planned_evaluation_games(declaration)
    return declaration


# Source snapshot -------------------------------------------------------------------------

def snapshot_source(directory):
    """Commit, tracked patch, untracked (non-ignored) files and execution identity (provenance only)."""
    directory = Path(directory)
    directory.mkdir(parents=True)

    def git(*args, binary=False):
        result = subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True, check=True, timeout=60)
        return result.stdout if binary else result.stdout.decode()
    (directory / "tracked.patch").write_bytes(git("diff", "--no-ext-diff", "--binary", "HEAD", binary=True))
    untracked = [name for name in git("ls-files", "--others", "--exclude-standard", "-z").split("\0") if name]
    total = sum((REPO_ROOT / name).stat().st_size for name in untracked)
    if total > UNTRACKED_LIMIT_BYTES:
        raise RuntimeError(f"Untracked files total {total} bytes; refusing to archive (preflight issue)")
    files = {}
    for name in untracked:
        target = directory / "untracked" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO_ROOT / name, target)
        files[name] = sha256_file(target)
    identity = dict(git=git_provenance(), execution=execution_source_identity(), untracked_files=files,
                    tracked_patch_sha256=sha256_file(directory / "tracked.patch"))
    write_json_once(directory / "source.json", identity)
    return identity


# Campaign ---------------------------------------------------------------------------------

class Campaign:
    """One invocation against one campaign directory (holds the single-writer lock until ``close``).

    ``fault(point, campaign)`` is an optional test hook called at named transitions.
    """

    def __init__(self, directory, declaration_path, authorization, *, clock=time.monotonic, fault=None):
        if authorization in REJECTED_DECLARATION_TOKENS:
            raise LaunchRefused(f"Authorization token {authorization} is REJECTED / NOT AUTHORIZED")
        declaration, digest = load_declaration(declaration_path)
        if authorization != digest:
            raise PermissionError("Authorization token must equal the declaration SHA-256")
        self.declaration, self.declaration_sha256 = declaration, digest
        self.base = Path(declaration_path).resolve().parent
        self.directory, self.clock, self._fault = Path(directory), clock, fault
        self.runtime = self.configure_runtime()
        problems = identity_problems(declaration, self.runtime) + checkpoint_problems(declaration)
        if problems:
            raise LaunchRefused("Launch refused; this process does not match the frozen declaration: "
                                + "; ".join(problems))
        import_execution_closure()
        self.verify_identity("launch")
        self.attempt, self._budget, self._extra_check, self._context, self._last_stop = None, None, None, {}, None
        self._open_directory(Path(declaration_path))

    # Runtime and identity --------------------------------------------------------------------

    def configure_runtime(self):
        runtime = self.declaration["runtime"]
        expected = required_thread_environment(runtime["threads"])
        wrong = {k: os.environ.get(k) for k, v in expected.items() if os.environ.get(k) != v}
        if wrong:
            raise RuntimeError("Set the library thread environment before starting Python: "
                               + " ".join(f"{k}={v}" for k, v in expected.items()) + f" (found {wrong})")
        return configure_deterministic_runtime(runtime["threads"]) if runtime["deterministic_algorithms"] \
            else runtime_identity()

    def verify_identity(self, where):
        """Fail closed if runtime or execution sources drifted from the declaration mid-campaign."""
        problems = identity_problems(self.declaration, runtime_identity())
        if problems:
            raise LaunchRefused(f"Identity changed during the campaign ({where}): " + "; ".join(problems))

    def fault(self, point):
        if self._fault is not None:
            self._fault(point, self)

    # Directory, lock and recovery -------------------------------------------------------------

    def _open_directory(self, declaration_path):
        directory = self.directory
        if directory.exists():
            entries = {p.name for p in directory.iterdir()}
            if "declaration.json" not in entries and entries - {"campaign.lock"}:
                raise FileExistsError(f"{directory} exists and is not a campaign directory")
        directory.mkdir(parents=True, exist_ok=True)
        self.lock = CampaignLock(directory)
        try:
            copy = directory / "declaration.json"
            if copy.exists():
                if sha256_file(copy) != self.declaration_sha256:
                    raise FileExistsError("Campaign directory exists for a different declaration")
            else:
                if (directory / "journal.jsonl").exists():
                    raise InconsistentCampaign("Journal exists without its declaration copy")
                data = declaration_path.read_bytes()
                if sha256_bytes(data) != self.declaration_sha256:
                    raise LaunchRefused("Declaration changed while launching")
                write_once(copy, data)
            self.state = CampaignState(self.declaration["generations"], self.declaration["champion"]["schedule"])
            self.journal = Journal(directory / "journal.jsonl", self.state)
            self.journal.repair()
            if self.state.declaration_sha256 is None:
                self.journal.append(dict(event="campaign_created", declaration_sha256=self.declaration_sha256,
                                         launch_control=LAUNCH_CONTROL_VERSION))
            elif self.state.declaration_sha256 != self.declaration_sha256:
                raise InconsistentCampaign("Journal belongs to a different declaration")
            self.recovered_attempts = recover_open_attempts(self.journal, self.declaration["seeds"])
        except BaseException:
            self.lock.release()
            raise

    def close(self):
        self.lock.release()

    def _close_open_work(self, number, reason):
        close_open_work(self.journal, number, reason, self.declaration["seeds"])

    # Attempts -----------------------------------------------------------------------------------

    def _attempt(self, scope, body, extra_check):
        number = self.attempt = len(self.state.attempts) + 1
        self.journal.append(dict(event="attempt_start", attempt=number, scope=str(scope), pid=os.getpid(),
                                 declaration_sha256=self.declaration_sha256, runtime=self.runtime,
                                 execution_sha256=self.declaration["execution_source"]["sha256"]))
        budget = self._budget = Budget(self.journal, self.declaration["budgets"], scope=scope, attempt=number,
                                       clock=self.clock)
        self._extra_check, self._last_stop = extra_check, None
        status = "running"
        try:
            budget.lease(force=True)
            source = snapshot_source(self.directory / "attempts" / f"attempt-{number:03d}" / "source")
            self.journal.append(dict(event="attempt_source", attempt=number, git=source["git"],
                                     execution_sha256=source["execution"]["sha256"],
                                     tracked_patch_sha256=source["tracked_patch_sha256"],
                                     untracked_files=len(source["untracked_files"])))
            self.verify_identity("attempt start")
            self._refresh_views()
            body()
            status = "completed"
        except BudgetExhausted as stop:
            status = f"incomplete: {stop}"
            self._close_open_work(number, status)
            self._declare_incomplete(scope, str(stop))
        except CampaignStop as stop:
            status = f"stopped: {stop}"
        except BaseException as error:
            status = f"failed: {type(error).__name__}: {error}"
            raise
        finally:
            if not self.journal.broken:
                self._close_open_work(number, status)
                budget.settle(status)
                self._write_status()
            self._budget = None
        return status

    def _check(self, phase):
        try:
            if self._extra_check is not None:
                self._extra_check(phase, self._context)
            self._budget.check(phase)
        except CampaignStop as stop:
            self._last_stop = stop
            raise

    def _unit(self, seed, unit, kind, compute):
        """Run one evaluation unit durably; a completed unit returns its journaled evidence unchanged."""
        evidence = self.state.evidence(unit)
        if evidence is not None:
            return evidence
        self._check("evaluation")
        if kind in GAME_KINDS:
            self._budget.require_game_slot()
        self.journal.append(dict(event="unit_begin", unit=unit, kind=kind, seed=seed, attempt=self.attempt))
        self._context = dict(unit=unit, kind=kind, seed=seed)
        self.fault(f"after_unit_begin:{kind}")
        try:
            evidence = compute()
        except CampaignStop as stop:
            self.journal.append(dict(event="unit_abandoned", unit=unit, kind=kind, seed=seed, attempt=self.attempt,
                                     reason=str(stop)))
            raise
        if isinstance(evidence, dict) and evidence.get("abandoned"):
            self.journal.append(dict(event="unit_abandoned", unit=unit, kind=kind, seed=seed, attempt=self.attempt,
                                     reason=evidence["stop_reason"], partial=evidence))
            raise self._last_stop or CampaignStop(evidence["stop_reason"])
        self.fault(f"before_unit_complete:{kind}")
        self.journal.append(dict(event="unit_complete", unit=unit, kind=kind, seed=seed, attempt=self.attempt,
                                 evidence=evidence, within_budget=self._budget.completion_violation() is None))
        return self.state.evidence(unit)

    def _completion_problem(self, *, training=False):
        """Recheck before publishing: nothing is published after a stop request or past a limit."""
        if self._budget.stop_reason is not None:
            return CampaignStop(self._budget.stop_reason)
        violation = self._budget.completion_violation(training=training)
        return BudgetExhausted(violation) if violation else None

    def _require_completion(self, *, training=False):
        problem = self._completion_problem(training=training)
        if problem is not None:
            raise problem

    # Paths and views ----------------------------------------------------------------------------

    def run_dir(self, seed):
        return self.directory / "runs" / f"seed-{seed}"

    def config_for(self, seed):
        return V2Config.from_dict(dict(self.declaration["config"], seed=seed))

    def _committed_file(self, seed, generation, kind):
        info = self.state.seed(seed)["committed"][generation]["files"][kind]
        path = self.run_dir(seed) / info["path"]
        if sha256_file(path) != info["sha256"]:
            raise InconsistentCampaign(f"Committed artifact differs from the journal: {path}")
        return path, info

    def _generation_dir(self, seed, generation):
        return self.run_dir(seed) / "generations" / f"generation-{generation:04d}"

    def _refresh_views(self):
        """Publish every journal-derived view that is missing; verify those present."""
        for seed in self.declaration["seeds"]:
            seed_state = self.state.seed(seed)
            self._recover_seed_files(seed)
            for generation in sorted(seed_state["diagnostics"]):
                publish_view(self._generation_dir(seed, generation) / "summary.json",
                             self._generation_view(seed, generation))
            for generation in sorted(seed_state["decisions"]):
                publish_view(self.run_dir(seed) / f"champion-check-{generation:04d}.json",
                             self._check_view(seed, generation))
            if seed_state["selected"] is not None:
                publish_view(self.run_dir(seed) / "selection.json", self._selection_view(seed))
        if self.state.final["started"] is not None:
            publish_view(self.directory / "final" / "started.json", self.state.final["started"])
            for seed in self.declaration["seeds"]:
                if str(seed) in self.state.final["seeds"]:
                    self._publish_final_seed(seed)
        if self.state.outcome is not None:
            publish_view(self.directory / "outcome.json", self._outcome_view())
            if self.state.final["started"] is not None:
                publish_view(self.directory / "final" / "result.json", self._outcome_view())

    def _write_status(self):
        replace_view(self.directory / "status.json", campaign_status_from_state(
            self.state, self.declaration, note="non-authoritative summary; the journal is the authority"))

    # Generation commits (B2) ----------------------------------------------------------------------

    def _commit_generation(self, seed, runner, summary):
        """Stage every generation output, journal the commit, then install it (journal-first)."""
        from .network import weights_sha256
        generation = runner.completed_generations
        run = self.run_dir(seed)
        staging = run / "staging" / f"generation-{generation:04d}.attempt-{self.attempt:03d}"
        staging.mkdir(parents=True)
        saved = runner.save_boundary(staging)
        self.fault("after_boundary_files")
        final_rel = Path("generations") / f"generation-{generation:04d}"
        files = {}
        for kind in ("inference", "resume"):
            path = Path(saved[kind])
            files[kind] = dict(path=str(final_rel / path.name), sha256=saved[f"{kind}_sha256"],
                               bytes=path.stat().st_size)
        if generation:
            games = [g for g in runner.replay.iter_games() if g.generation == generation]
            data = "".join(json.dumps(dict(generation=generation, index=g.index, moves=list(g.moves),
                                           winner=g.winner)) + "\n" for g in games).encode()
            name = f"generation-{generation:04d}.games.jsonl"
            files["games"] = dict(path=str(final_rel / name), sha256=write_once(staging / name, data), bytes=len(data))
        fsync_directory(staging)
        self._require_completion()
        self.verify_identity(f"seed {seed} commit generation {generation}")
        self.fault("before_commit_record")
        resources = None if summary is None else summary["resources"]
        self.journal.append(dict(
            event="generation_committed", seed=seed, generation=generation, attempt=self.attempt,
            staging=str(staging.relative_to(run)), files=files, learner_weights_sha256=weights_sha256(runner.model),
            state_sha256=runner.state_sha256(),
            summary=None if summary is None else {k: v for k, v in summary.items() if k != "resources"},
            resources=resources))
        self.fault("after_commit_record")
        self._install_committed(seed, generation)
        self.fault("after_commit_install")

    def _install_committed(self, seed, generation):
        """Make a committed generation's directory authoritative, finishing an interrupted install."""
        record = self.state.seed(seed)["committed"][generation]
        pruned = self.state.seed(seed)["pruned"]
        final_dir, staging = self._generation_dir(seed, generation), self.run_dir(seed) / record["staging"]
        source = final_dir if final_dir.exists() else staging
        if not source.exists():
            raise InconsistentCampaign(f"Committed generation {generation} of seed {seed} has no outputs")
        for kind, info in record["files"].items():
            path = source / Path(info["path"]).name
            if info["path"] in pruned and not path.exists():
                continue
            if not path.is_file() or sha256_file(path) != info["sha256"]:
                raise InconsistentCampaign(f"Committed {kind} artifact of generation {generation} differs from the "
                                           "journal")
        if source is staging:
            final_dir.parent.mkdir(parents=True, exist_ok=True)
            os.rename(staging, final_dir)
            fsync_directory(final_dir.parent)
            fsync_directory(staging.parent)

    def _recover_seed_files(self, seed):
        seed_state = self.state.seed(seed)
        for generation in sorted(seed_state["committed"]):
            self._install_committed(seed, generation)
        for path_name, record in seed_state["pruned"].items():
            path = self.run_dir(seed) / path_name
            if path.exists():  # interrupted between the prune record and the unlink
                if sha256_file(path) != record["sha256"]:
                    raise InconsistentCampaign(f"Refusing to finish pruning a modified artifact: {path}")
                path.unlink()
                fsync_directory(path.parent)

    def _prune(self, seed):
        keep = self.declaration["retention"]["resume_boundaries"]
        seed_state = self.state.seed(seed)
        live = [g for g in sorted(seed_state["committed"])
                if seed_state["committed"][g]["files"]["resume"]["path"] not in seed_state["pruned"]]
        for generation in live[:-keep] if len(live) > keep else []:
            path, info = self._committed_file(seed, generation, "resume")
            self.journal.append(dict(event="artifact_pruned", seed=seed, generation=generation, attempt=self.attempt,
                                     path=info["path"], sha256=info["sha256"], bytes=info["bytes"]))
            self.fault("after_prune_record")
            path.unlink()
            fsync_directory(path.parent)

    def _generation_view(self, seed, generation):
        record = self.state.seed(seed)["committed"][generation]
        diagnostics = self.state.seed(seed)["diagnostics"][generation]
        return dict(seed=seed, generation=generation, committed_attempt=record["attempt"], artifacts=record["files"],
                    learner_weights_sha256=record["learner_weights_sha256"], state_sha256=record["state_sha256"],
                    summary=record["summary"], resources=record["resources"],
                    development_raw_value=diagnostics["development_raw_value"])

    # Per-seed run --------------------------------------------------------------------------------

    def run_seed(self, seed, *, extra_check=None):
        """Run or resume one seed until its generations finish or a stop/limit ends the attempt."""
        if seed not in self.declaration["seeds"]:
            raise ValueError("Seed is not declared")
        if self.state.outcome is not None:
            self._refresh_views()
            return self._outcome_status()
        if self.state.seed(seed)["selected"] is not None:
            raise RuntimeError("This seed already completed champion selection")
        if self.state.final["started"] is not None:
            raise RuntimeError("The sealed final evaluation has started; seed runs are closed")
        from . import generation as generation_module  # torch-dependent
        if generation_module.PROCESS_EXECUTION_SOURCE["sha256"] != self.declaration["execution_source"]["sha256"]:
            raise LaunchRefused("The loaded generation sources differ from the declaration")
        return self._attempt(seed, lambda: self._seed_body(seed), extra_check)

    def _seed_body(self, seed):
        from .generation import GenerationRunner, load_resume_boundary
        seed_state = self.state.seed(seed)
        config = self.config_for(seed)
        if seed_state["committed"]:
            latest = max(seed_state["committed"])
            record = seed_state["committed"][latest]
            path, info = self._committed_file(seed, latest, "resume")
            runtime = self.declaration["runtime"]
            runner = load_resume_boundary(path, expected_sha256=info["sha256"],
                                          strict_runtime=runtime["strict_resume_runtime"],
                                          strict_source=runtime["strict_resume_source"])
            if runner.completed_generations != latest or runner.state_sha256() != record["state_sha256"]:
                raise InconsistentCampaign(f"Resume boundary of generation {latest} disagrees with its commit record")
        else:
            self._check("setup")
            runner = GenerationRunner(config)
            self._commit_generation(seed, runner, None)
        rows = package_rows(self.declaration, self.base, "solved-development")
        self._finish_pending(seed, rows)
        while runner.completed_generations < config.max_generations:
            generation = runner.completed_generations + 1
            self.verify_identity(f"seed {seed} generation {generation}")
            self._context = dict(runner=runner, generation=generation, seed=seed)
            self._check("training")
            self.fault("before_generation_begin")
            self.journal.append(dict(event="generation_begin", seed=seed, generation=generation, attempt=self.attempt,
                                     base_state_sha256=seed_state["committed"][generation - 1]["state_sha256"]))
            self.fault("after_generation_begin")

            def progress(event, **info):
                if event == "game":
                    self.journal.append(dict(event="selfplay_game", seed=seed, generation=generation,
                                             attempt=self.attempt, index=info["index"], plies=info["plies"]))
                elif event == "update":
                    self.journal.append(dict(event="optimizer_step", seed=seed, generation=generation,
                                             attempt=self.attempt, step=info["step"]))
                self.fault(f"progress:{event}")
            self._budget.begin_training()
            try:
                summary = runner.run_generation(check=lambda: self._check("training"), progress=progress)
            except BaseException as error:
                self._budget.end_training()
                self.journal.append(dict(event="generation_discarded", seed=seed, generation=generation,
                                         attempt=self.attempt, reason=f"{type(error).__name__}: {error}"))
                raise
            self._budget.end_training()
            self.fault("after_training")
            problem = self._completion_problem(training=True)
            if problem is not None:
                self.journal.append(dict(event="generation_discarded", seed=seed, generation=generation,
                                         attempt=self.attempt, reason=str(problem)))
                raise problem
            self._commit_generation(seed, runner, summary)
            self._finish_pending(seed, rows)
        self._select(seed)

    def _finish_pending(self, seed, rows):
        """Post-commit work in generation order: diagnostics, pruning, then a scheduled champion check."""
        seed_state = self.state.seed(seed)
        for generation in sorted(g for g in seed_state["committed"] if g):
            if generation not in seed_state["diagnostics"]:
                self._diagnostics(seed, generation, rows)
            self._prune(seed)
            if generation in self.declaration["champion"]["schedule"] and generation not in seed_state["decisions"]:
                self._champion_check(seed, generation)

    def _diagnostics(self, seed, generation, rows):
        from .evaluation import raw_values
        from .network import load_inference_checkpoint
        self._check("evaluation")
        path, _ = self._committed_file(seed, generation, "inference")
        metrics = statistics.value_metrics(rows, raw_values(load_inference_checkpoint(path), rows))
        self._require_completion()
        self.fault("before_diagnostics_record")
        self.journal.append(dict(event="generation_diagnostics", seed=seed, generation=generation,
                                 attempt=self.attempt, development_raw_value=metrics))
        self.fault("after_diagnostics_record")
        publish_view(self._generation_dir(seed, generation) / "summary.json", self._generation_view(seed, generation))

    # Champion selection (development evidence only) ---------------------------------------------

    def _v2_agent(self, seed, generation, name):
        from .evaluation import load_v2_agent
        path, info = self._committed_file(seed, generation, "inference")
        agent = load_v2_agent(path, self.declaration["evaluation"]["simulations"], expected_sha256=info["sha256"],
                              name=name)
        agent.identity["path"] = info["path"]  # campaign-relative
        return agent

    def _arena_units(self, seed, prefix, kind, agent, opponent, rows, namespace):
        records = []
        for row in rows:
            for game_index, _ in PAIRED_GAMES:
                records.append(self._unit(seed, f"{prefix}/{row['id']}/{game_index}", kind, lambda: paired_game(
                    agent, opponent, row, game_index, namespace=namespace, seed=seed,
                    check=lambda: self._check("evaluation"), keep_decisions=False)))
        return records

    def _development_tactics(self, seed, generation, agent, rows):
        from .evaluation import search_row
        evaluation = self.declaration["evaluation"]
        evidence = {row["id"]: self._unit(
            seed, f"seed{seed}/development-tactics/g{generation:04d}/{row['id']}", "development_tactics_row",
            lambda: search_row(agent.inference, row, simulations=evaluation["simulations"],
                               seeds=evaluation["tactical_seeds"], namespace="development-tactics",
                               check=lambda: self._check("evaluation"))) for row in rows}
        return statistics.tactical_metrics(rows, {k: v["choices"] for k, v in evidence.items()},
                                           {k: v["visits"] for k, v in evidence.items()})

    def _champion_check(self, seed, generation):
        self.verify_identity(f"seed {seed} champion check {generation}")
        decl = self.declaration["champion"]
        champion_generation = self.state.champion_generation(seed)
        candidate = self._v2_agent(seed, generation, f"generation-{generation}")
        incumbent = self._v2_agent(seed, champion_generation, f"champion-generation-{champion_generation}")
        openings = package_rows(self.declaration, self.base, "openings-development")
        tactics = package_rows(self.declaration, self.base, decl["tactical_package"])
        prefix = f"seed{seed}/development-g{generation:04d}"
        arena_rows = openings[:decl["arena_openings"]]
        records = self._arena_units(seed, f"{prefix}/arena", "development_arena_game", candidate, incumbent,
                                    arena_rows, f"seed{seed}-champion-g{generation}")
        by_id = {row["id"]: row for row in openings}
        baselines = {}
        for name in decl["baseline_opponents"]:
            rows = [by_id[i] for i in decl["baseline_rows"]]
            base_records = self._arena_units(seed, f"{prefix}/baseline-{name}", "development_baseline_game",
                                             candidate, opponent_by_name(name), rows,
                                             f"seed{seed}-baseline-{name}-g{generation}")
            baselines[name] = statistics.arena_summary(scored(base_records), planned_games=2 * len(rows))
        candidate_tactics = self._development_tactics(seed, generation, candidate, tactics)
        champion_tactics = self._development_tactics(seed, champion_generation, incumbent, tactics)
        arena = statistics.arena_summary(scored(records), planned_games=2 * len(arena_rows))
        # Every unit completed and no correctness error was raised.
        decision = statistics.champion_decision(arena, candidate_tactics, champion_tactics, True)
        self._require_completion()
        self.verify_identity(f"seed {seed} champion decision {generation}")
        self.fault("before_decision_record")
        self.journal.append(dict(event="champion_decision", seed=seed, generation=generation, attempt=self.attempt,
                                 previous=champion_generation, promote=decision["promote"], decision=decision,
                                 arena=arena, baselines=baselines, candidate_tactics=candidate_tactics,
                                 champion_tactics=champion_tactics, candidate=candidate.describe(),
                                 champion=incumbent.describe()))
        self.fault("after_decision_record")
        publish_view(self.run_dir(seed) / f"champion-check-{generation:04d}.json", self._check_view(seed, generation))

    def _check_view(self, seed, generation):
        record = self.state.seed(seed)["decisions"][generation]
        prefixes = (f"seed{seed}/development-g{generation:04d}/",
                    f"seed{seed}/development-tactics/g{generation:04d}/",
                    f"seed{seed}/development-tactics/g{record['previous']:04d}/")
        units = sorted(u for u in self.state.units if u.startswith(prefixes))
        return dict(record, records={unit: self.state.evidence(unit) for unit in units})

    def _select(self, seed):
        seed_state = self.state.seed(seed)
        generations = self.declaration["generations"]
        missing = ([g for g in range(generations + 1) if g not in seed_state["committed"]]
                   + [g for g in range(1, generations + 1) if g not in seed_state["diagnostics"]]
                   + [g for g in self.declaration["champion"]["schedule"] if g not in seed_state["decisions"]])
        if missing:
            raise InconsistentCampaign(f"Selection attempted with missing work: {missing}")
        self._require_completion()
        champion = self.state.champion_generation(seed)
        self.fault("before_selection_record")
        self.journal.append(dict(event="seed_selected", seed=seed, attempt=self.attempt, generation=champion,
                                 inference=seed_state["committed"][champion]["files"]["inference"],
                                 decisions={str(g): d["promote"] for g, d in sorted(seed_state["decisions"].items())},
                                 rule="development evidence only; sealed packages untouched"))
        self.fault("after_selection_record")
        publish_view(self.run_dir(seed) / "selection.json", self._selection_view(seed))

    def _selection_view(self, seed):
        return dict(self.state.seed(seed)["selected"], declaration_sha256=self.declaration_sha256)

    # Sealed final evaluation ------------------------------------------------------------------------

    def final_evaluate(self, *, extra_check=None):
        if self.state.outcome is not None:
            self._refresh_views()
            return self._outcome_status()
        selections = {}
        for seed in self.declaration["seeds"]:
            selected = self.state.seed(seed)["selected"]
            if selected is None:
                raise RuntimeError(f"Seed {seed} has no development champion selection")
            selections[str(seed)] = selected
        # Every earlier attempt (training and development evaluation) ran under the declared identity.
        for number, info in sorted(self.state.attempts.items()):
            start = info["start"]
            if (runtime_differences(self.declaration["runtime_identity"], start["runtime"])
                    or start["execution_sha256"] != self.declaration["execution_source"]["sha256"]
                    or start["declaration_sha256"] != self.declaration_sha256):
                raise LaunchRefused(f"Attempt {number} ran under a different identity; final evaluation refused")
        return self._attempt("final", lambda: self._final_body(selections), extra_check)

    def _final_body(self, selections):
        started = self.state.final["started"]
        if started is None:
            self._check("evaluation")
            self.verify_identity("final start")
            descriptions = {str(seed): self._final_descriptions(seed, selections[str(seed)])
                            for seed in self.declaration["seeds"]}
            self.fault("before_final_started")
            self.journal.append(dict(event="final_started", attempt=self.attempt, selections=selections,
                                     descriptions=descriptions, runtime=self.runtime,
                                     execution_sha256=self.declaration["execution_source"]["sha256"],
                                     declaration_sha256=self.declaration_sha256))
            self.fault("after_final_started")
        elif started["selections"] != selections:
            raise InconsistentCampaign("Selections changed after the sealed evaluation started")
        publish_view(self.directory / "final" / "started.json", self.state.final["started"])
        for seed in self.declaration["seeds"]:
            if str(seed) not in self.state.final["seeds"]:
                self._final_seed_units(seed)
                result = self._final_seed_result(seed)
                self._require_completion()
                self.verify_identity(f"final seed {seed}")
                self.fault("before_final_seed_record")
                self.journal.append(dict(event="final_seed_complete", seed=seed, attempt=self.attempt,
                                         result_sha256=sha256_bytes(view_bytes(result))))
                self.fault("after_final_seed_record")
            self._publish_final_seed(seed)
        self._require_completion()
        self.fault("before_campaign_complete")
        self.journal.append(dict(event="campaign_complete", attempt=self.attempt, status="COMPLETE",
                                 final_results={k: v["result_sha256"] for k, v in self.state.final["seeds"].items()}))
        publish_view(self.directory / "final" / "result.json", self._outcome_view())
        publish_view(self.directory / "outcome.json", self._outcome_view())

    def _selected_agent(self, seed, selection):
        from .evaluation import load_v2_agent
        path = self.run_dir(seed) / selection["inference"]["path"]
        agent = load_v2_agent(path, self.declaration["evaluation"]["simulations"],
                              expected_sha256=selection["inference"]["sha256"], name=f"seed{seed}-champion")
        agent.identity["path"] = selection["inference"]["path"]
        return agent

    def _ladder_opponents(self, seed):
        from .evaluation import V1SearchArenaAgent, V2SearchArenaAgent, untrained_model
        final, simulations = self.declaration["final"], self.declaration["evaluation"]["simulations"]
        opponents = {name: opponent_by_name(name) for name in final["ladder"] if name in OPPONENT_NAMES}
        if "initial_v2_512" in final["ladder"]:
            opponents["initial_v2_512"] = V2SearchArenaAgent(untrained_model(seed), simulations, name="initial_v2_512")
        if "phase4d2f_512" in final["ladder"]:
            checkpoint = self.declaration["phase4d2f_checkpoint"]
            opponents["phase4d2f_512"] = V1SearchArenaAgent(REPO_ROOT / checkpoint["path"], checkpoint["sha256"],
                                                            simulations)
            opponents["phase4d2f_512"].identity["path"] = checkpoint["path"]
        return opponents

    def _final_descriptions(self, seed, selection):
        opponents = self._ladder_opponents(seed)
        return dict(agent=self._selected_agent(seed, selection).describe(),
                    opponents={name: opponents[name].describe() for name in self.declaration["final"]["ladder"]})

    def _final_seed_units(self, seed):
        from .evaluation import V2NNOnlyAgent, calibration_games, nn_only_choice, raw_value, search_row
        final, evaluation = self.declaration["final"], self.declaration["evaluation"]
        simulations, tie_seeds = evaluation["simulations"], evaluation["tactical_seeds"]
        agent = self._selected_agent(seed, self.state.final["started"]["selections"][str(seed)])
        tactics = package_rows(self.declaration, self.base, "tactical-sealed")
        solved = package_rows(self.declaration, self.base, "solved-sealed")
        openings = package_rows(self.declaration, self.base, "openings-sealed")[:final["ladder_openings"]]
        prefix = f"final/seed{seed}"

        def check():
            self._check("evaluation")
        for row in tactics:
            self._unit(seed, f"{prefix}/tactical/{row['id']}", "final_tactical_row", lambda: dict(
                search_row(agent.inference, row, simulations=simulations, seeds=tie_seeds, namespace="sealed-tactics",
                           check=check), nn_only_choice=nn_only_choice(agent.inference, row)))
        for row in solved:
            self._unit(seed, f"{prefix}/solved/{row['id']}", "final_solved_row", lambda: dict(
                search_row(agent.inference, row, simulations=simulations, seeds=tie_seeds, namespace="sealed-solved",
                           check=check), raw_value=raw_value(agent.inference, row)))
        opponents = self._ladder_opponents(seed)
        for name in final["ladder"]:
            self._arena_units(seed, f"{prefix}/ladder-{name}", "final_ladder_game", agent, opponents[name], openings,
                              f"seed{seed}-final-{name}")
        nn_only = V2NNOnlyAgent(agent.inference, name=f"seed{seed}-champion-nn-only")
        self._arena_units(seed, f"{prefix}/nn-only-vs-random", "final_nn_only_game", nn_only, RandomOpponent(),
                          openings, f"seed{seed}-final-nn-only")
        config = self.config_for(seed)
        for index in range(final["calibration_games"]):
            self._unit(seed, f"{prefix}/calibration/{index:04d}", "calibration_game", lambda: dict(records=[
                dict(r, game=index) for r in calibration_games(agent.inference, config, games=1,
                                                               seed=seed * 1_000_003 + index, check=check)]))

    def _final_seed_result(self, seed):
        """The sealed result for one seed, computed from journaled evidence only (deterministic)."""
        final, thresholds = self.declaration["final"], self.declaration["thresholds"]
        started = self.state.final["started"]
        tactics = package_rows(self.declaration, self.base, "tactical-sealed")
        solved = package_rows(self.declaration, self.base, "solved-sealed")
        openings = package_rows(self.declaration, self.base, "openings-sealed")[:final["ladder_openings"]]
        prefix = f"final/seed{seed}"

        def evidence(unit):
            value = self.state.evidence(unit)
            if value is None:
                raise InconsistentCampaign(f"Final evidence missing: {unit}")
            return value

        def arena(name, label):
            records = [evidence(f"{prefix}/{label}/{row['id']}/{g}") for row in openings for g, _ in PAIRED_GAMES]
            summary = statistics.arena_summary(scored(records), planned_games=2 * len(openings))
            rule = thresholds["arena"].get(name)
            return dict(summary=summary, records=records,
                        gate=statistics.arena_gate(summary, rule) if rule else None)
        tactical = {row["id"]: evidence(f"{prefix}/tactical/{row['id']}") for row in tactics}
        solved_ev = {row["id"]: evidence(f"{prefix}/solved/{row['id']}") for row in solved}
        overlap = training_overlap(self.run_dir(seed), [r["moves"] for r in tactics + solved])
        choices = {k: v["choices"] for k, v in tactical.items()}
        visits = {k: v["visits"] for k, v in tactical.items()}
        nn_choices = {k: [v["nn_only_choice"]] for k, v in tactical.items()}
        solved_choices = {k: v["choices"] for k, v in solved_ev.items()}
        values = {k: v["raw_value"] for k, v in solved_ev.items()}
        ladder = {}
        for name in final["ladder"]:
            opponent = started["descriptions"][str(seed)]["opponents"][name]
            ladder[name] = dict(arena(name, f"ladder-{name}"), opponent=opponent)
        calibration = [r for index in range(final["calibration_games"])
                       for r in evidence(f"{prefix}/calibration/{index:04d}")["records"]]
        return dict(
            seed=seed, completeness="complete", selection=started["selections"][str(seed)],
            agent=started["descriptions"][str(seed)]["agent"], overlap_families=sorted(overlap),
            tactical=with_overlap_sensitivity(tactics, overlap,
                                              lambda rows: statistics.tactical_metrics(rows, choices, visits)),
            tactical_nn_only=with_overlap_sensitivity(tactics, overlap,
                                                      lambda rows: statistics.tactical_metrics(rows, nn_choices)),
            solved=with_overlap_sensitivity(solved, overlap,
                                            lambda rows: statistics.solved_decision_metrics(rows, solved_choices)),
            value=with_overlap_sensitivity(solved, overlap, lambda rows: statistics.value_metrics(rows, values)),
            ladder=ladder, nn_only_vs_random=arena("nn_only_vs_random", "nn-only-vs-random"),
            calibration=statistics.calibration_summary(calibration, constant=final.get("calibration_constant")),
            evidence=dict(tactical=tactical, solved=solved_ev, calibration=calibration))

    def _publish_final_seed(self, seed):
        path = self.directory / "final" / f"seed-{seed}.json"
        expected = self.state.final["seeds"][str(seed)]["result_sha256"]
        if path.exists():
            if sha256_file(path) != expected:
                raise InconsistentCampaign(f"{path} disagrees with the journal")
            return
        result = self._final_seed_result(seed)
        if sha256_bytes(view_bytes(result)) != expected:
            raise InconsistentCampaign(f"Recomputed final result for seed {seed} disagrees with the journal")
        publish_view(path, result)

    # Outcomes --------------------------------------------------------------------------------------

    def _declare_incomplete(self, scope, reason):
        if self.state.outcome is not None:
            return
        self.journal.append(dict(event="campaign_incomplete", attempt=self.attempt, scope=str(scope), status="INCOMPLETE",
                                 reason=reason, progress=progress_summary(self.state, self.declaration)))
        publish_view(self.directory / "outcome.json", self._outcome_view())
        if self.state.final["started"] is not None:
            publish_view(self.directory / "final" / "result.json", self._outcome_view())

    def _outcome_view(self):
        return dict(self.state.outcome, declaration_sha256=self.declaration_sha256,
                    note=("COMPLETE means every declared evidence unit finished within the declared limits; "
                          "acceptance is judged separately against the declared thresholds. INCOMPLETE is final: "
                          "missing evidence is never filled by extending a limit."))

    def _outcome_status(self):
        outcome = self.state.outcome
        return "completed" if outcome["event"] == "campaign_complete" else f"incomplete: {outcome['reason']}"


OPPONENT_NAMES = ("random", "negamax1", "negamax2", "negamax4", "guarded_uct_800")


def opponent_by_name(name):
    if name == "random":
        return RandomOpponent()
    if name.startswith("negamax"):
        return NegamaxOpponent(int(name[len("negamax"):]))
    if name.startswith("guarded_uct_"):
        return GuardedUCTOpponent(int(name[len("guarded_uct_"):]))
    raise ValueError(f"Unknown opponent {name}")


def training_overlap(run_dir, histories):
    """Package families (board+actor and reflection) that appear in any archived training position."""
    wanted = {family_key(list(moves)) for moves in histories}
    found = set()
    for path in sorted((Path(run_dir) / "generations").glob("generation-*/generation-*.games.jsonl")):
        for line in path.read_text().splitlines():
            moves = json.loads(line)["moves"]
            for ply in range(len(moves)):
                key = family_key(moves[:ply])
                if key in wanted:
                    found.add(key)
    return found


def with_overlap_sensitivity(rows, overlap, metric):
    """Prespecified sensitivity: drop every package family seen in training (either orientation)."""
    kept = [r for r in rows if family_key(r["moves"]) not in overlap]
    return dict(complete_set=metric(rows), overlap_excluded=metric(kept) if kept else None,
                overlap_excluded_rows=len(rows) - len(kept))


# Status ---------------------------------------------------------------------------------------

def progress_summary(state, declaration):
    seeds = {}
    for seed in declaration["seeds"]:
        s = state.seed(seed)
        seeds[str(seed)] = dict(
            committed_generations=max(s["committed"]) if s["committed"] else None,
            required_generations=declaration["generations"], diagnostics=sorted(s["diagnostics"]),
            decisions={str(g): d["promote"] for g, d in sorted(s["decisions"].items())},
            pending_checks=[g for g in declaration["champion"]["schedule"] if g not in s["decisions"]],
            selected_generation=None if s["selected"] is None else s["selected"]["generation"])
    final_units = {}
    for unit, info in state.units.items():
        if unit.startswith("final/"):
            entry = final_units.setdefault(info["kind"], dict(completed=0, incomplete=0))
            entry["completed" if info["completed"] else "incomplete"] += 1
    return dict(seeds=seeds, final=dict(started=state.final["started"] is not None,
                                        seeds_complete=sorted(state.final["seeds"]), units=final_units))


def campaign_status_from_state(state, declaration, note=None):
    limits = declaration["budgets"]
    consumption = state.consumption()
    return dict(
        note=note, outcome=None if state.outcome is None else dict(event=state.outcome["event"],
                                                                   reason=state.outcome.get("reason")),
        open_attempts=state.open_attempts(), progress=progress_summary(state, declaration), consumption=consumption,
        limits=dict(limits, remaining_campaign_seconds=limits["campaign_seconds"] - consumption["charged_seconds"],
                    remaining_evaluation_games=limits["evaluation_games_ceiling"]
                    - consumption["evaluation_games_charged"]))


def campaign_status(directory):
    """Read-only status (no lock, no recovery): an open attempt is either running or was hard-interrupted."""
    directory = Path(directory)
    declaration = json.loads((directory / "declaration.json").read_text())
    state, journal = read_state(directory, generations=declaration["generations"],
                                schedule=declaration["champion"]["schedule"])
    status = campaign_status_from_state(state, declaration, note="read-only; open attempts are running or "
                                        "were hard-interrupted and will be charged through their last lease")
    status["journal"] = dict(records=len(journal.records), torn_tail_bytes=len(journal.torn_tail or b""))
    return status


# Preflight and freeze -------------------------------------------------------------------------

def preflight(declaration_path):
    """Run every launch gate without starting work; ``ok`` is true only if a launch would be accepted."""
    path = Path(declaration_path)
    gates, report = [], dict(declaration_sha256=sha256_file(path))

    def gate(name, check):
        try:
            detail = check()
            gates.append(dict(name=name, ok=True, detail=detail))
            return detail
        except Exception as error:  # noqa: BLE001 - every failure is a refused gate
            gates.append(dict(name=name, ok=False, detail=f"{type(error).__name__}: {error}"))
            return None
    loaded = gate("declaration", lambda: load_declaration(path)[0]["name"])
    declaration = json.loads(path.read_text()) if loaded else None

    def environment():
        threads = declaration["runtime"]["threads"]
        wrong = {k: os.environ.get(k) for k, v in required_thread_environment(threads).items() if os.environ.get(k) != v}
        if wrong:
            raise RuntimeError(f"library thread environment not pinned: {wrong}")
        return required_thread_environment(threads)

    def runtime():
        identity = configure_deterministic_runtime(declaration["runtime"]["threads"])
        report["runtime"] = identity
        missing = unavailable_runtime_fields(identity)
        if missing:
            raise RuntimeError(f"unavailable runtime identity fields: {missing}")
        changed = runtime_differences(declaration["runtime_identity"], identity)
        if changed:
            raise RuntimeError(f"runtime differs from the declaration: {changed}")
        return "configured runtime equals the declared runtime identity"

    def source():
        current = execution_source_identity()
        report["execution_sha256"] = current["sha256"]
        problems = [p for p in identity_problems(declaration, declaration["runtime_identity"]) if "source" in p]
        if problems:
            raise RuntimeError("; ".join(problems))
        return dict(sha256=current["sha256"], files=len(current["files"]),
                    groups={k: v["sha256"] for k, v in source_groups(current["files"]).items()})

    def checkpoint():
        problems = checkpoint_problems(declaration)
        if problems:
            raise RuntimeError("; ".join(problems))
        return declaration["phase4d2f_checkpoint"]["sha256"]
    if declaration is not None:
        gate("thread_environment", environment)
        gate("runtime_identity", runtime)
        gate("execution_source", source)
        gate("phase4d2f_checkpoint", checkpoint)
        report["planned_evaluation_games"] = planned_evaluation_games(declaration)
    report.update(ok=bool(gates) and all(g["ok"] for g in gates), gates=gates, git=git_provenance())
    return report


def freeze(output, *, name="phase4d3c-alphazero-v2-two-seed", notes=None):
    """Write a new format-2 declaration from the frozen packages beside ``output`` and this runtime."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    threads = 1
    wrong = {k: os.environ.get(k) for k, v in required_thread_environment(threads).items() if os.environ.get(k) != v}
    if wrong:
        raise RuntimeError(f"Pin the library thread environment before freezing: {wrong}")
    identity = configure_deterministic_runtime(threads)
    if unavailable_runtime_fields(identity):
        raise RuntimeError(f"Cannot freeze an unidentified runtime: {unavailable_runtime_fields(identity)}")
    manifest = json.loads((output.parent / "manifest.json").read_text())
    declaration = build_declaration({k: v for k, v in manifest.items() if k != "exclusions"}, name=name,
                                    runtime_identity=identity, frozen_records=frozen_record_entries(output.parent),
                                    notes=notes)
    validate_declaration(declaration, output.parent)
    problems = identity_problems(declaration, identity) + checkpoint_problems(declaration)
    if problems:
        raise RuntimeError("; ".join(problems))
    token = write_json_once(output, declaration)
    if token in REJECTED_DECLARATION_TOKENS:
        raise RuntimeError("A freeze reproduced a rejected token")
    return token


FREEZE_NOTES = [
    "Phase 4D.3B.1 frozen declaration (format 2) for the Milestone 3 two-seed campaign; running it needs "
    "separate authorization.",
    "The authorization token is the SHA-256 of this exact file. The Phase 4D.3B token "
    "2741314399741c1b20510b8a2beb42e0938b0aeafc9dd0c487201e57208867d8 is REJECTED / NOT AUTHORIZED.",
    "It binds execution-source content, the complete runtime identity, frozen packages and records, the retained "
    "4D.2f checkpoint and the launch-control semantics; any difference refuses the launch.",
    "Seed 42 is primary; seed 314159 is replication, not a second chance.",
    "Package and frozen-record paths are relative to this file; the Phase 4D.2f checkpoint path is relative to the "
    "repository root.",
]


# CLI --------------------------------------------------------------------------------------------

EXIT_COMPLETED, EXIT_REFUSED, EXIT_STOPPED, EXIT_INCOMPLETE = 0, 2, 3, 4


def main(argv=None):
    parser = argparse.ArgumentParser(description="AlphaZero v2 bounded campaign (requires authorization)")
    sub = parser.add_subparsers(dest="command", required=True)
    pre = sub.add_parser("preflight")
    pre.add_argument("--declaration", required=True)
    frz = sub.add_parser("freeze")
    frz.add_argument("--output", required=True)
    stat = sub.add_parser("status")
    stat.add_argument("--campaign-dir", required=True)
    for name in ("run", "final-evaluate"):
        command = sub.add_parser(name)
        command.add_argument("--declaration", required=True)
        command.add_argument("--campaign-dir", required=True)
        command.add_argument("--authorize", required=True)
        if name == "run":
            command.add_argument("--seed", type=int, required=True)
    args = parser.parse_args(argv)
    if args.command == "preflight":
        report = preflight(args.declaration)
        print(json.dumps(report, indent=1, sort_keys=True))
        return EXIT_COMPLETED if report["ok"] else EXIT_REFUSED
    if args.command == "freeze":
        print(freeze(args.output, notes=FREEZE_NOTES))
        return EXIT_COMPLETED
    if args.command == "status":
        print(json.dumps(campaign_status(args.campaign_dir), indent=1, sort_keys=True))
        return EXIT_COMPLETED
    try:
        campaign = Campaign(args.campaign_dir, args.declaration, args.authorize)
    except (PermissionError, FileExistsError, RuntimeError) as error:
        print(f"refused: {type(error).__name__}: {error}")
        return EXIT_REFUSED

    def stop(signum, frame):
        if campaign._budget is not None:
            campaign._budget.request_stop(f"signal {signum}")
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    status = campaign.run_seed(args.seed) if args.command == "run" else campaign.final_evaluate()
    print(status)
    if status == "completed":
        return EXIT_COMPLETED
    return EXIT_INCOMPLETE if status.startswith("incomplete") else EXIT_STOPPED


if __name__ == "__main__":
    sys.exit(main())
