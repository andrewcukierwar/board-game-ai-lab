"""Single-owner, fail-closed AlphaZero v2 official campaign launcher (Milestone 3 tool, Phase 4D.3B.4).

Nothing runs on import. ``launch`` fails closed unless all of the following hold:

* ``--authorize`` equals the SHA-256 of the exact declaration file, and neither
  is a rejected token (Phase 4D.3B ``2741314399...``, Phase 4D.3B.1
  ``8a8a52b100...``, Phase 4D.3B.2 ``ed641aea2f...`` and Phase 4D.3B.3
  ``34b4d89963...`` are rejected).
* The declaration (format 3) validates, and every input it binds is identical
  in this process: execution-source content (training and evaluation/launch
  groups), frozen package and frozen-record bytes, the retained Phase 4D.2f
  checkpoint, campaign configuration, launch-control semantics, and the complete
  runtime identity after the deterministic runtime is configured. A runtime
  that cannot identify one of those fields is refused.
* ``--campaign-dir`` does not exist. An official campaign always starts in a
  fresh directory.

The official campaign is NON-RESUMABLE. One process owns the directory (``flock``)
and runs seed by seed, the development selection, then the sealed final
evaluation, under one declaration. Evidence lives in this process and is
published as write-once files. The process never reads evidence back from disk
and never loads a generation resume boundary. Any stop, limit, identity drift or
error ends the campaign INCOMPLETE. If the owner dies, a later invocation records
INCOMPLETE and refuses: see ``launch_control``.

COMPLETE is a one-way terminal commit (``Campaign._complete``). Every fallible
step (evidence publication and validation, identity, counts) happens before one
completion barrier, whose single monotonic reading is the campaign's
authoritative endpoint. After it, only the atomic COMPLETE write remains; it
certifies work that is already complete, and no later signal or error can
change a visible COMPLETE.

``official_results`` accepts a COMPLETE record only if it passes a strict
schema (``complete_record_problems``): every required field present and in its
domain, every per-seed mapping covering exactly the declared seeds, and every
quantity stored in two places equal in both.

Layout of a campaign directory (created once; nothing is overwritten except state.json):

    declaration.json        verbatim copy; its hash is the authorization token
    state.json              the single authority (atomically replaced on each transition)
    campaign.lock           single-owner lock
    source/                 source snapshot (provenance)
    runs/seed-S/generations/generation-NNNN/   inference, resume (diagnostic only), games, summary
    runs/seed-S/champion-check-NNNN.json, runs/seed-S/selection.json
    final/started.json, final/seed-S.json
                            official only if state.json is COMPLETE and lists their hashes
"""
import argparse
import hashlib
import importlib
import json
import math
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
from .launch_control import (COMPLETE, INCOMPLETE, INTERRUPTED_MESSAGE, LAUNCH_CONTROL_VERSION,
                             REJECTED_DECLARATION_TOKENS, RUNNING_FINAL, SELECTION_COMPLETE, STATE_FORMAT, Budget,
                             CampaignLock, CampaignLocked, CampaignStateFile, CampaignStop,
                             InterruptedCampaign, StateCorrupt, TerminalCampaign, fsync_directory, publish_view,
                             running_seed, sha256_bytes, sha256_file, view_bytes, write_once)
from .oracle import family_key
from .packages import load_package
from .provenance import (REPO_ROOT, SOURCE_SCHEME, configure_deterministic_runtime, execution_source_files,
                         execution_source_identity, git_provenance, required_thread_environment, runtime_differences,
                         runtime_identity, source_differences, source_group, source_groups,
                         unavailable_runtime_fields)

DECLARATION_FORMAT = "connect4-alphazero-v2-campaign-declaration"
DECLARATION_VERSION = 3
UNTRACKED_LIMIT_BYTES = 100 * 1024 * 1024
DECLARATION_KEYS = {"format", "format_version", "name", "kind", "seeds", "primary_seed", "config", "generations",
                    "budgets", "runtime", "evaluation", "champion", "final", "packages", "thresholds",
                    "retention", "phase4d2f_checkpoint", "notes",
                    "execution_source", "runtime_identity", "frozen_records", "launch_control"}
FROZEN_RECORD_NAMES = ("build-provenance.json", "exclusions.json", "manifest.json")
RUNTIME_IDENTITY_KEYS = tuple(sorted(runtime_identity()))
PHASE4D2F_CHECKPOINT = dict(path="experiment-output/phase4d2f-neural-value-anchoring-seed42-20261005/candidate.pt",
                            sha256="78276912bdef5f1b1b53ac6b2ec3ede2eed20a8ebb5796a9bdcbb57e4cb28c51")
DEVELOPMENT_GAME_KINDS = ("development_arena_game", "development_baseline_game")
SEALED_GAME_KINDS = ("final_ladder_game", "final_nn_only_game")
CALIBRATION_GAME_KINDS = ("calibration_game",)
GAME_KINDS = DEVELOPMENT_GAME_KINDS + SEALED_GAME_KINDS + CALIBRATION_GAME_KINDS
ROW_KINDS = ("development_tactics_row", "final_tactical_row", "final_solved_row")


class LaunchRefused(PermissionError):
    """The declaration does not authorize this invocation (token, source, runtime or frozen inputs)."""


class IdentityDrift(LaunchRefused):
    """Bound source or runtime identity changed after launch: the campaign ends INCOMPLETE."""


class ArtifactMismatch(RuntimeError):
    """An artifact written by this campaign no longer has the hash recorded when it was written."""


class NotOfficialEvidence(RuntimeError):
    """Campaign artifacts that are not certified by a COMPLETE state (forensic only)."""


def write_json_once(path, value):
    """Write JSON to a new file only (temporary file, fsync, hard-link publish)."""
    return write_once(path, view_bytes(value))


def json_normalized(value):
    return json.loads(json.dumps(value))


# Declaration ------------------------------------------------------------------------------

def launch_control_declaration():
    """Launch-control semantics bound by the declaration (the executable meaning is bound by source)."""
    return dict(
        version=LAUNCH_CONTROL_VERSION, state_file=STATE_FORMAT,
        resume=("Forbidden. An official campaign never resumes after a crash, reboot or drift, never repairs state "
                "to continue, never retries interrupted work and never reuses partial evidence. Generation resume "
                "artifacts are diagnostic only; the official launcher never loads them."),
        single_owner=("One orchestrator process holds an exclusive flock on campaign.lock for the whole campaign and "
                      "runs every seed, the development selection and the sealed final evaluation sequentially. A "
                      "second process is refused. A campaign always starts in a new directory."),
        states=("CREATED -> RUNNING_SEED_<seed> for each declared seed in order -> DEVELOPMENT_SELECTION_COMPLETE -> "
                "RUNNING_FINAL_EVALUATION -> COMPLETE. INCOMPLETE may follow any non-terminal state. COMPLETE and "
                "INCOMPLETE are terminal. Each transition atomically replaces state.json."),
        interruption=("An owner that ends without a terminal state leaves the campaign INCOMPLETE. A later "
                      "invocation records INCOMPLETE, reports that the campaign cannot be resumed and exits nonzero."),
        identity=("Declared execution sources and runtime identity are revalidated at launch, around every "
                  "generation and evaluation unit, and before every transition. Any difference ends the campaign "
                  "INCOMPLETE."),
        deadlines=("Monotonic seconds of the owning process. Per seed, collection+optimization stays within "
                   "per_run_training_seconds; the whole campaign stays within campaign_seconds. A check refuses to "
                   "start any unit, search, move or update at a limit. Work that finishes past a limit is never "
                   "accepted. Reaching any limit ends the campaign INCOMPLETE. Limits are inclusive at completion: "
                   "COMPLETE needs elapsed seconds at the completion barrier <= campaign_seconds and each seed's "
                   "training seconds <= per_run_training_seconds."),
        terminal_commit=("Every result file is published, flushed and re-validated, identity is re-verified and every "
                         "count is checked before one completion barrier. The barrier latches one monotonic reading "
                         "before reading the stop flag; that reading is the campaign's authoritative endpoint. A stop "
                         "handled or a limit exceeded at the barrier ends the campaign INCOMPLETE. After it, only the "
                         "atomic COMPLETE write remains: the scientific campaign is complete at the barrier, and the "
                         "write certifies it. A later signal is recorded but cannot change the outcome. INCOMPLETE "
                         "is written only if the visible state is non-terminal, so no error path can replace a "
                         "visible COMPLETE. A write that fails before COMPLETE is visible ends INCOMPLETE."),
        acceptance=("Official results require a COMPLETE record that passes a strict schema. Every required field "
                    "must be present and in its domain: finite non-negative seconds, non-negative integer counts and "
                    "SHA-256 digests. Every per-seed mapping must cover exactly the declared seeds (the research "
                    "declaration's seeds are exactly 42 and 314159), with no missing, extra or non-canonical key. "
                    "Quantities stored twice (completion record and counters; history, started.json and result "
                    "selections) must agree, and totals must equal their per-kind units. Duplicate JSON keys and "
                    "NaN/Infinity are refused."),
        evaluation_games=("Reserved before a game's first move. No game starts once evaluation_games_ceiling games "
                          "have started."),
        final_evaluation=("Both selections are recorded durably in RUNNING_FINAL_EVALUATION before any sealed "
                          "inference. An interrupted sealed evaluation ends the campaign INCOMPLETE and is never "
                          "restarted under this declaration."),
        counters=("Live provenance snapshotted into state.json. After a crash they are lower bounds and authorize "
                  "nothing."),
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
        raise LaunchRefused(f"Declaration token {digest} is REJECTED / NOT AUTHORIZED (Phase 4D.3B / 4D.3B.1 / "
                            "4D.3B.2 / 4D.3B.3 launch reviews); it must never authorize a campaign")
    declaration = json.loads(data)
    validate_declaration(declaration, path.parent)
    return declaration, digest


def validate_declaration(declaration, base):
    if set(declaration) != DECLARATION_KEYS:
        raise ValueError(f"Declaration keys differ: {sorted(set(declaration) ^ DECLARATION_KEYS)}")
    if (declaration["format"], declaration["format_version"]) != (DECLARATION_FORMAT, DECLARATION_VERSION):
        raise ValueError(f"Not a format-{DECLARATION_VERSION} v2 campaign declaration")
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
    if set(declaration["runtime"]) != {"threads", "deterministic_algorithms"}:
        raise ValueError("Declared runtime settings must be exactly threads and deterministic_algorithms")
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
        runtime=runtime or dict(threads=1, deterministic_algorithms=True),
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
    """The single owner of one new official campaign directory; ``run`` executes it exactly once.

    ``fault(point, campaign)`` is an optional test hook called at named points.
    """

    def __init__(self, directory, declaration_path, authorization, *, clock=time.monotonic, fault=None):
        if authorization in REJECTED_DECLARATION_TOKENS:
            raise LaunchRefused(f"Authorization token {authorization} is REJECTED / NOT AUTHORIZED")
        declaration, digest = load_declaration(declaration_path)
        if authorization != digest:
            raise PermissionError("Authorization token must equal the declaration SHA-256")
        self.declaration, self.declaration_sha256 = declaration, digest
        self.seeds = list(declaration["seeds"])
        self.base = Path(declaration_path).resolve().parent
        self.directory, self.clock, self._fault = Path(directory), clock, fault
        self.runtime = self.configure_runtime()
        problems = identity_problems(declaration, self.runtime) + checkpoint_problems(declaration)
        if problems:
            raise LaunchRefused("Launch refused; this process does not match the frozen declaration: "
                                + "; ".join(problems))
        import_execution_closure()
        self.verify_identity("launch")
        self._extra_check, self._context, self._last_stop, self._ran = None, {}, None, False
        self.evidence, self.seed_records, self.final_results = {}, {}, {}
        self.post_commit_errors = []
        self.units = {kind: dict(started=0, completed=0) for kind in GAME_KINDS + ROW_KINDS}
        self.training = {str(seed): dict(generations_attempted=0, generations_completed=0,
                                         selfplay_games_attempted=0, selfplay_games_completed=0, plies=0,
                                         optimizer_steps=0) for seed in self.seeds}
        self._create_directory(Path(declaration_path))
        self.budget = Budget(declaration["budgets"], clock=clock)

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
        """Raise IdentityDrift if runtime or execution sources differ from the declaration."""
        problems = identity_problems(self.declaration, runtime_identity())
        if problems:
            raise IdentityDrift(f"Identity changed during the campaign ({where}): " + "; ".join(problems))

    def fault(self, point):
        if self._fault is not None:
            self._fault(point, self)

    # Directory and ownership -------------------------------------------------------------------

    def _create_directory(self, declaration_path):
        directory = self.directory
        if directory.exists():
            self._refuse_existing()
        directory.mkdir(parents=True)  # never exist_ok: a concurrent creator loses here
        self.lock = CampaignLock(directory)
        try:
            data = declaration_path.read_bytes()
            if sha256_bytes(data) != self.declaration_sha256:
                raise LaunchRefused("Declaration changed while launching")
            write_once(directory / "declaration.json", data)
            self.state = CampaignStateFile.create(directory, declaration_sha256=self.declaration_sha256,
                                                  seeds=self.seeds)
        except BaseException:
            self.lock.release()
            raise

    def _refuse_existing(self):
        """An existing directory is never continued. An interrupted official campaign is marked INCOMPLETE."""
        directory = self.directory
        if not (directory / CampaignStateFile.NAME).exists():
            raise FileExistsError(f"{directory} exists and is not a new directory; an official campaign always "
                                  "starts in a new campaign directory")
        lock = CampaignLock(directory)  # CampaignLocked if the owner is still alive
        try:
            try:
                state = CampaignStateFile.load(directory)
            except StateCorrupt as error:
                raise TerminalCampaign(f"{error}. The campaign is INCOMPLETE and cannot be resumed.") from error
            if state.document["declaration_sha256"] != self.declaration_sha256:
                raise FileExistsError(f"{directory} holds a campaign for a different declaration")
            if state.terminal:
                raise TerminalCampaign(f"Official campaign is already {state.state}; a terminal campaign never "
                                       "runs or transitions again")
            owner = state.document["owner"]
            state.mark_incomplete(detected_by_pid=os.getpid(), reason=(
                f"interrupted: owner pid {owner['pid']} ended while {state.state}; official campaigns are "
                "non-resumable"))
            raise InterruptedCampaign(INTERRUPTED_MESSAGE)
        finally:
            lock.release()

    def close(self):
        self.lock.release()

    # Lifecycle --------------------------------------------------------------------------------

    def run(self, *, extra_check=None):
        """Run the whole campaign once: every seed, selection, sealed evaluation. Returns a status string.

        The status reports the visible terminal state. An error raised after COMPLETE is visible
        changes nothing: it is listed in ``post_commit_errors`` and the status stays "completed".
        """
        if self._ran:
            raise RuntimeError("An official campaign runs once")
        self._ran, self._extra_check = True, extra_check
        status = None
        try:
            with self.budget.phase("setup"):
                self.source = snapshot_source(self.directory / "source")
            self.verify_identity("campaign start")
            selections = {}
            for seed in self.seeds:
                self._transition(running_seed(seed))
                selections[str(seed)] = self._run_seed(seed)
            self._transition(SELECTION_COMPLETE, selections=selections)
            self._final(selections)
        except BaseException as error:
            status = self._incomplete(error)  # conditional: never replaces a visible COMPLETE
            if self.state.state == COMPLETE:
                self.post_commit_errors.append(f"{type(error).__name__}: {error}")
            elif not isinstance(error, (CampaignStop, IdentityDrift)):
                raise
        finally:
            self.budget.end_training()
            self.close()
        return "completed" if self.state.state == COMPLETE else status

    def _transition(self, new_state, **info):
        self.verify_identity(f"before {new_state}")
        problem = self.budget.completion_violation(seeds=self.seeds)
        if problem is not None:
            raise problem
        self.state.transition(new_state, counters=self.counters(), **info)
        self.fault(f"after_transition:{new_state}")

    def _incomplete(self, reason):
        """Every error and stop path ends here: INCOMPLETE only if the visible state is still non-terminal."""
        text = f"{type(reason).__name__}: {reason}" if not isinstance(reason, CampaignStop) else str(reason)
        try:
            self.state.mark_incomplete(reason=text, counters=self.counters())
        except Exception:  # noqa: BLE001 - the state stays non-terminal; a later invocation marks it INCOMPLETE
            pass
        return f"incomplete: {text}"

    def _check(self, phase, seed=None):
        try:
            if self._extra_check is not None:
                self._extra_check(phase, self._context)
            self.budget.check(phase, seed)
        except CampaignStop as stop:
            self._last_stop = stop
            raise

    def _require_within_limits(self, seeds=()):
        problem = self.budget.completion_violation(seeds=seeds)
        if problem is not None:
            self._last_stop = problem
            raise problem

    def counters(self):
        games = lambda kinds, key: sum(self.units[k][key] for k in kinds)  # noqa: E731
        return dict(
            note="live provenance; lower bounds unless the state is terminal",
            time=self.budget.snapshot(self.seeds), training=json_normalized(self.training),
            evaluation=dict(
                by_kind=json_normalized(self.units),
                development_games=games(DEVELOPMENT_GAME_KINDS, "completed"),
                sealed_games=games(SEALED_GAME_KINDS, "completed"),
                calibration_games=games(CALIBRATION_GAME_KINDS, "completed"),
                total_games_started=games(GAME_KINDS, "started"), total_games_completed=games(GAME_KINDS, "completed"),
                ceiling=self.declaration["budgets"]["evaluation_games_ceiling"]))

    # Evidence units ------------------------------------------------------------------------------

    def _unit(self, seed, unit, kind, compute):
        """Compute one evidence unit exactly once, under the bound identity and within every limit."""
        if unit in self.evidence:
            raise RuntimeError(f"Evidence unit {unit} was already computed; units never run twice")
        self._check("evaluation")
        self.verify_identity(f"before {unit}")
        if kind in GAME_KINDS:
            self.budget.start_game()
        self.units[kind]["started"] += 1
        self._context = dict(unit=unit, kind=kind, seed=seed)
        self.fault(f"after_unit_begin:{kind}")
        evidence = compute()
        if isinstance(evidence, dict) and evidence.get("abandoned"):
            raise self._last_stop or CampaignStop(evidence["stop_reason"])
        self.fault(f"before_unit_complete:{kind}")
        self._require_within_limits()
        self.verify_identity(f"after {unit}")
        self.units[kind]["completed"] += 1
        self.evidence[unit] = evidence
        return evidence

    def _evidence(self, unit):
        if unit not in self.evidence:
            raise RuntimeError(f"Evidence missing: {unit}")
        return self.evidence[unit]

    # Paths and artifacts --------------------------------------------------------------------------

    def run_dir(self, seed):
        return self.directory / "runs" / f"seed-{seed}"

    def config_for(self, seed):
        return V2Config.from_dict(dict(self.declaration["config"], seed=seed))

    def _generation_dir(self, seed, generation):
        return self.run_dir(seed) / "generations" / f"generation-{generation:04d}"

    def _artifact(self, seed, generation, kind):
        """Path of an artifact this campaign wrote, re-verified against the hash recorded at write time."""
        info = self.seed_records[str(seed)]["generations"][generation]["files"][kind]
        path = self.run_dir(seed) / info["path"]
        if sha256_file(path) != info["sha256"]:
            raise ArtifactMismatch(f"{path} differs from the hash recorded when it was written")
        return path, info

    # Per-seed run --------------------------------------------------------------------------------

    def _run_seed(self, seed):
        from .generation import GenerationRunner
        record = self.seed_records[str(seed)] = dict(generations={}, diagnostics={}, decisions={})
        config = self.config_for(seed)
        with self.budget.phase(f"seed-{seed}:setup"):
            self._check("setup")
            runner = GenerationRunner(config)
            self._save_generation(seed, runner, None)
        rows = package_rows(self.declaration, self.base, "solved-development")
        schedule = self.declaration["champion"]["schedule"]
        while runner.completed_generations < config.max_generations:
            generation = runner.completed_generations + 1
            self._context = dict(runner=runner, generation=generation, seed=seed)
            with self.budget.phase(f"seed-{seed}:training"):
                summary = self._train_generation(seed, runner, generation)
            with self.budget.phase(f"seed-{seed}:development"):
                self._save_generation(seed, runner, summary)
                self._diagnostics(seed, generation, rows)
                self._prune(seed)
                if generation in schedule:
                    self._champion_check(seed, generation)
            self.state.record_counters(self.counters())
        with self.budget.phase(f"seed-{seed}:development"):
            return self._select(seed, record)

    def _train_generation(self, seed, runner, generation):
        counters = self.training[str(seed)]
        collecting = dict(active=True)

        def progress(event, **info):
            if event == "game":
                counters["selfplay_games_attempted"] += 1
                counters["selfplay_games_completed"] += 1
                counters["plies"] += info["plies"]
            elif event == "collected":
                collecting["active"] = False
            elif event == "update":
                counters["optimizer_steps"] += 1
            self.fault(f"progress:{event}")
        self.verify_identity(f"seed {seed} before generation {generation}")
        self._check("training", seed)
        counters["generations_attempted"] += 1
        self.budget.begin_training(seed)
        try:
            summary = runner.run_generation(check=lambda: self._check("training", seed), progress=progress)
        except BaseException:
            if collecting["active"]:
                counters["selfplay_games_attempted"] += 1  # the game in progress when collection stopped
            raise
        finally:
            self.budget.end_training()
        self.fault("after_training")
        self._require_within_limits(seeds=[seed])
        self.verify_identity(f"seed {seed} after generation {generation}")
        counters["generations_completed"] += 1
        return summary

    def _save_generation(self, seed, runner, summary):
        """Write a completed generation's inference, resume (diagnostic) and games once; record their hashes."""
        from .network import weights_sha256
        generation = runner.completed_generations
        directory = self._generation_dir(seed, generation)
        directory.mkdir(parents=True)
        saved = runner.save_boundary(directory)
        relative = Path("generations") / directory.name
        files = {kind: dict(path=str(relative / Path(saved[kind]).name), sha256=saved[f"{kind}_sha256"],
                            bytes=Path(saved[kind]).stat().st_size) for kind in ("inference", "resume")}
        if generation:
            games = [g for g in runner.replay.iter_games() if g.generation == generation]
            data = "".join(json.dumps(dict(generation=generation, index=g.index, moves=list(g.moves),
                                           winner=g.winner)) + "\n" for g in games).encode()
            name = f"generation-{generation:04d}.games.jsonl"
            files["games"] = dict(path=str(relative / name), sha256=write_once(directory / name, data), bytes=len(data))
        fsync_directory(directory)
        self.seed_records[str(seed)]["generations"][generation] = dict(
            seed=seed, generation=generation, files=files, learner_weights_sha256=weights_sha256(runner.model),
            state_sha256=runner.state_sha256(),
            summary=None if summary is None else {k: v for k, v in summary.items() if k != "resources"},
            resources=None if summary is None else summary["resources"], resume_pruned=False)
        self.fault("after_generation_saved")

    def _prune(self, seed):
        """Delete resume boundaries beyond the retention count (diagnostic artifacts; never loaded here)."""
        keep = self.declaration["retention"]["resume_boundaries"]
        generations = self.seed_records[str(seed)]["generations"]
        live = [g for g in sorted(generations) if not generations[g]["resume_pruned"]]
        for generation in live[:-keep] if len(live) > keep else []:
            path, _ = self._artifact(seed, generation, "resume")
            path.unlink()
            fsync_directory(path.parent)
            generations[generation]["resume_pruned"] = True

    def _diagnostics(self, seed, generation, rows):
        from .evaluation import raw_values
        from .network import load_inference_checkpoint
        self._check("evaluation")
        path, _ = self._artifact(seed, generation, "inference")
        metrics = statistics.value_metrics(rows, raw_values(load_inference_checkpoint(path), rows))
        self._require_within_limits()
        self.verify_identity(f"seed {seed} diagnostics {generation}")
        record = self.seed_records[str(seed)]
        record["diagnostics"][generation] = metrics
        publish_view(self._generation_dir(seed, generation) / "summary.json",
                     dict(record["generations"][generation], development_raw_value=metrics))

    # Champion selection (development evidence only) ---------------------------------------------

    def _v2_agent(self, seed, generation, name):
        from .evaluation import load_v2_agent
        path, info = self._artifact(seed, generation, "inference")
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
        evidence = {}
        for row in rows:
            unit = f"seed{seed}/development-tactics/g{generation:04d}/{row['id']}"
            # The incumbent's rows may already exist from the check that promoted it; they are reused in-process.
            evidence[row["id"]] = self.evidence[unit] if unit in self.evidence else self._unit(
                seed, unit, "development_tactics_row",
                lambda: search_row(agent.inference, row, simulations=evaluation["simulations"],
                                   seeds=evaluation["tactical_seeds"], namespace="development-tactics",
                                   check=lambda: self._check("evaluation")))
        return statistics.tactical_metrics(rows, {k: v["choices"] for k, v in evidence.items()},
                                           {k: v["visits"] for k, v in evidence.items()})

    def champion_generation(self, seed):
        champion = 0
        for generation, decision in sorted(self.seed_records[str(seed)]["decisions"].items()):
            if decision["promote"]:
                champion = generation
        return champion

    def _champion_check(self, seed, generation):
        self.verify_identity(f"seed {seed} champion check {generation}")
        decl = self.declaration["champion"]
        champion_generation = self.champion_generation(seed)
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
        self._require_within_limits()
        self.verify_identity(f"seed {seed} champion decision {generation}")
        prefixes = (f"{prefix}/", f"seed{seed}/development-tactics/g{generation:04d}/",
                    f"seed{seed}/development-tactics/g{champion_generation:04d}/")
        record = dict(seed=seed, generation=generation, previous=champion_generation, promote=decision["promote"],
                      decision=decision, arena=arena, baselines=baselines, candidate_tactics=candidate_tactics,
                      champion_tactics=champion_tactics, candidate=candidate.describe(), champion=incumbent.describe())
        self.seed_records[str(seed)]["decisions"][generation] = record
        publish_view(self.run_dir(seed) / f"champion-check-{generation:04d}.json", dict(
            record, records={u: e for u, e in sorted(self.evidence.items()) if u.startswith(prefixes)}))

    def _select(self, seed, record):
        generations = self.declaration["generations"]
        missing = ([g for g in range(generations + 1) if g not in record["generations"]]
                   + [g for g in range(1, generations + 1) if g not in record["diagnostics"]]
                   + [g for g in self.declaration["champion"]["schedule"] if g not in record["decisions"]])
        if missing:
            raise RuntimeError(f"Selection attempted with missing work: {missing}")
        self._require_within_limits()
        self.verify_identity(f"seed {seed} selection")
        champion = self.champion_generation(seed)
        _, inference = self._artifact(seed, champion, "inference")
        selection = dict(seed=seed, generation=champion, inference=inference,
                         decisions={str(g): d["promote"] for g, d in sorted(record["decisions"].items())},
                         rule="development evidence only; sealed packages untouched",
                         declaration_sha256=self.declaration_sha256)
        publish_view(self.run_dir(seed) / "selection.json", selection)
        return selection

    # Sealed final evaluation ------------------------------------------------------------------------

    def _final(self, selections):
        with self.budget.phase("final"):
            self._check("evaluation")
            self.verify_identity("final start")
            descriptions = {str(seed): self._final_descriptions(seed, selections[str(seed)]) for seed in self.seeds}
            self.final_started = dict(selections=selections, descriptions=descriptions, runtime=self.runtime,
                                      execution_sha256=self.declaration["execution_source"]["sha256"],
                                      declaration_sha256=self.declaration_sha256)
            self.started_sha256 = publish_view(self.directory / "final" / "started.json", self.final_started)
            # Selections are fixed durably here, before any sealed inference.
            self._transition(RUNNING_FINAL, selections=selections, started_sha256=self.started_sha256)
            for seed in self.seeds:
                self._final_seed_units(seed)
                result = self._final_seed_result(seed)
                self._require_within_limits()
                self.verify_identity(f"final seed {seed}")
                self.final_results[str(seed)] = publish_view(self.directory / "final" / f"seed-{seed}.json", result)
                self.state.record_counters(self.counters())
        self.fault("before_campaign_complete")
        self._complete()

    def _complete(self):
        """The completion barrier, then the one-way terminal commit.

        Before the barrier, every fallible step: all evidence published (write-once,
        fsynced) and re-validated exactly as ``official_results`` will validate it,
        source/runtime identity re-verified, and every count checked. The barrier
        latches one monotonic reading (the authoritative endpoint) before reading the
        stop flag, then checks every limit against that reading. The scientific
        campaign is complete when the barrier succeeds. After it there is no
        computation, selection, evaluation or evidence publication: only the atomic
        COMPLETE write, which certifies the already-completed work.
        """
        if self.state.state != RUNNING_FINAL:
            raise RuntimeError(f"Completion attempted from {self.state.state}")
        seeds = required_seed_keys(self.declaration)
        problems = (certified_evidence_problems(self.directory, self.declaration_sha256, seeds, self.final_results,
                                                self.started_sha256, self.final_started["selections"])
                    + counter_problems(self.declaration, self.counters(), seeds))
        if problems:
            raise RuntimeError("Campaign evidence is not complete: " + "; ".join(problems))
        self.verify_identity("campaign complete")
        self.fault("before_completion_barrier")
        time_account = self.budget.cross_completion_barrier(self.seeds)  # CampaignStop/BudgetExhausted if not
        # ---- The scientific campaign is complete. Only terminal certification follows. ----
        account = dict(self.counters(), time=time_account)
        outcome = dict(
            status=COMPLETE, final_results=dict(self.final_results), started_sha256=self.started_sha256,
            completion=completion_record(time_account, self.declaration["budgets"]),
            note=("COMPLETE means every declared evidence unit finished in one uninterrupted owning process "
                  "within the declared limits, as measured at the completion barrier; acceptance is judged "
                  "separately against the declared thresholds."))
        self.fault("after_completion_barrier")
        self._commit_complete(account, outcome)
        self.fault("after_complete")

    def _commit_complete(self, account, outcome):
        """Atomically publish COMPLETE. A failure before it is visible propagates (the campaign ends INCOMPLETE);
        a failure after it is visible never downgrades it."""
        try:
            self.state.terminate(COMPLETE, counters=account, outcome=outcome)
        except BaseException as error:
            if self.state.state != COMPLETE:  # ``terminate`` re-read the visible state on failure
                raise
            self.post_commit_errors.append(f"{type(error).__name__} after COMPLETE was visible: {error}")
            try:
                fsync_directory(self.directory)  # retry the directory flush that may have failed
            except OSError as retry:
                self.post_commit_errors.append(f"directory fsync retry failed ({retry}): COMPLETE is visible; if "
                                               "the rename is later lost, the earlier state can only become "
                                               "INCOMPLETE")

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
        agent = self._selected_agent(seed, self.final_started["selections"][str(seed)])
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
        """The sealed result for one seed, computed from this process's evidence only (deterministic)."""
        final, thresholds = self.declaration["final"], self.declaration["thresholds"]
        started = self.final_started
        tactics = package_rows(self.declaration, self.base, "tactical-sealed")
        solved = package_rows(self.declaration, self.base, "solved-sealed")
        openings = package_rows(self.declaration, self.base, "openings-sealed")[:final["ladder_openings"]]
        prefix = f"final/seed{seed}"
        evidence = self._evidence

        def arena(name, label):
            records = [evidence(f"{prefix}/{label}/{row['id']}/{g}") for row in openings for g, _ in PAIRED_GAMES]
            summary = statistics.arena_summary(scored(records), planned_games=2 * len(openings))
            rule = thresholds["arena"].get(name)
            return dict(summary=summary, records=records,
                        gate=statistics.arena_gate(summary, rule) if rule else None)
        tactical = {row["id"]: evidence(f"{prefix}/tactical/{row['id']}") for row in tactics}
        solved_ev = {row["id"]: evidence(f"{prefix}/solved/{row['id']}") for row in solved}
        overlap = training_overlap(self._archived_games(seed), [r["moves"] for r in tactics + solved])
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

    def _archived_games(self, seed):
        """Every archived training game of this seed, from files re-verified against their recorded hashes."""
        generations = self.seed_records[str(seed)]["generations"]
        return [self._artifact(seed, g, "games")[0] for g in sorted(generations) if g]


OPPONENT_NAMES = ("random", "negamax1", "negamax2", "negamax4", "guarded_uct_800")


def opponent_by_name(name):
    if name == "random":
        return RandomOpponent()
    if name.startswith("negamax"):
        return NegamaxOpponent(int(name[len("negamax"):]))
    if name.startswith("guarded_uct_"):
        return GuardedUCTOpponent(int(name[len("guarded_uct_"):]))
    raise ValueError(f"Unknown opponent {name}")


def training_overlap(game_files, histories):
    """Package families (board+actor and reflection) that appear in any archived training position."""
    wanted = {family_key(list(moves)) for moves in histories}
    found = set()
    for path in game_files:
        for line in Path(path).read_text().splitlines():
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


# Status and official results --------------------------------------------------------------------

def campaign_status(directory):
    """Read-only view of state.json (no lock, no writes)."""
    directory = Path(directory)
    try:
        state = CampaignStateFile.load(directory)
    except StateCorrupt as error:
        return dict(state=None, error=str(error), note="unreadable state: the campaign is not COMPLETE")
    document = state.document
    note = {COMPLETE: "terminal: COMPLETE (official results are certified by this state)",
            INCOMPLETE: "terminal: INCOMPLETE (artifacts are forensic only)"}.get(
        state.state, "non-terminal: running in its owning process, or interrupted; an interrupted campaign is "
                     "INCOMPLETE and the next launch invocation records that")
    return dict(document, note=note)


# Strict COMPLETE schema ----------------------------------------------------------------------------
# Acceptance validates the whole COMPLETE record against this schema. Every required field must be
# present, of its type and domain. Every per-seed mapping must cover exactly the declared seeds. Every
# quantity stored twice must agree in both places. Nothing is validated only "if present".

OFFICIAL_SEEDS = (42, 314159)  # the frozen research declaration's seeds
OUTCOME_KEYS = {"status", "final_results", "started_sha256", "completion", "note"}
COMPLETION_KEYS = {"endpoint", "elapsed_seconds", "training_seconds", "evaluation_games_started", "limits"}
COMPLETION_LIMITS = ("campaign_seconds", "per_run_training_seconds", "evaluation_games_ceiling")
COUNTER_KEYS = {"note", "time", "training", "evaluation"}
TIME_KEYS = {"elapsed_seconds", "phase_seconds", "training_seconds", "evaluation_games_started"}
TRAINING_KEYS = {"generations_attempted", "generations_completed", "selfplay_games_attempted",
                 "selfplay_games_completed", "plies", "optimizer_steps"}
EVALUATION_KEYS = {"by_kind", "development_games", "sealed_games", "calibration_games", "total_games_started",
                   "total_games_completed", "ceiling"}
STARTED_KEYS = {"selections", "descriptions", "runtime", "execution_sha256", "declaration_sha256"}


class MalformedRecord(ValueError):
    """A record acceptance relies on is not strictly well formed (duplicate keys, NaN/Infinity, bad seeds)."""


def strict_json(data):
    """Parse JSON for acceptance. Duplicate object keys and non-finite numbers are refused, never resolved."""
    def pairs(items):
        keys = [key for key, _ in items]
        if len(set(keys)) != len(keys):
            raise MalformedRecord(f"duplicate JSON object keys {sorted({k for k in keys if keys.count(k) > 1})}")
        return dict(items)

    def constant(name):
        raise MalformedRecord(f"non-finite JSON number {name}")
    return json.loads(data, object_pairs_hook=pairs, parse_constant=constant)


def is_count(value):
    return type(value) is int and value >= 0


def is_seconds(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def is_sha256(value):
    return isinstance(value, str) and len(value) == 64 and set(value) <= set("0123456789abcdef")


def required_seed_keys(declaration):
    """The exact key set of every per-seed mapping: the declared seeds as canonical decimal strings."""
    seeds = declaration["seeds"]
    if (not isinstance(seeds, list) or not seeds or any(type(s) is not int for s in seeds)
            or len(set(seeds)) != len(seeds)):
        raise MalformedRecord(f"declared seeds are malformed: {seeds!r}")
    if declaration["kind"] == "research" and seeds != list(OFFICIAL_SEEDS):
        raise MalformedRecord(f"a research declaration must declare exactly seeds {list(OFFICIAL_SEEDS)}")
    return {str(seed) for seed in seeds}


def keys_problems(name, value, keys):
    if not isinstance(value, dict):
        return [f"{name} is missing or not an object"]
    if set(value) != set(keys):
        return [f"{name} must have exactly keys {sorted(keys)}; differs by {sorted(set(value) ^ set(keys))}"]
    return []


def seed_mapping_problems(name, value, seeds, valid=None, domain="valid"):
    """A required per-seed mapping: exactly the declared seeds (no missing, extra or non-canonical key)."""
    if not isinstance(value, dict):
        return [f"{name} is missing or not a per-seed mapping"]
    if set(value) != seeds:
        return [f"{name} must cover exactly seeds {sorted(seeds)}; has {sorted(value)}"]
    if valid is None:
        return []
    return [f"{name}[{seed}] is not {domain}: {value[seed]!r}" for seed in sorted(value) if not valid(value[seed])]


def selection_problems(name, selections, seeds, declaration_sha256):
    def valid(seed_selection):
        return isinstance(seed_selection, dict) and seed_selection.get("declaration_sha256") == declaration_sha256
    problems = seed_mapping_problems(name, selections, seeds, valid, f"a selection under {declaration_sha256}")
    if not problems:
        problems = [f"{name}[{seed}] names seed {selections[seed].get('seed')!r}" for seed in sorted(seeds)
                    if type(selections[seed].get("seed")) is not int or str(selections[seed]["seed"]) != seed]
    return problems


def completion_record(time_account, budgets):
    """The COMPLETE record's authoritative endpoint: the account measured at the completion barrier."""
    return dict(endpoint="completion barrier: one monotonic reading in the owning process; limits inclusive",
                elapsed_seconds=time_account["elapsed_seconds"],
                training_seconds=dict(time_account["training_seconds"]),
                evaluation_games_started=time_account["evaluation_games_started"],
                limits={k: budgets[k] for k in COMPLETION_LIMITS})


def safe_sha256_file(path):
    try:
        return sha256_file(path)
    except OSError:
        return None


def certified_evidence_problems(directory, declaration_sha256, seeds, final_results, started_sha256, selections):
    """Every way the files a COMPLETE record certifies differ from what acceptance requires.

    ``seeds`` is the required key set; ``selections`` are the selections recorded in state.json.
    """
    directory, problems = Path(directory), []
    if not is_sha256(declaration_sha256) or declaration_sha256 in REJECTED_DECLARATION_TOKENS:
        problems.append(f"declaration {declaration_sha256!r} is malformed or REJECTED / NOT AUTHORIZED")
    if safe_sha256_file(directory / "declaration.json") != declaration_sha256:
        problems.append("declaration.json differs from the declaration the campaign ran under")
    problems += selection_problems("recorded selections", selections, seeds, declaration_sha256)
    started = None
    if not is_sha256(started_sha256) or safe_sha256_file(directory / "final" / "started.json") != started_sha256:
        problems.append("final/started.json differs from the hash recorded when the sealed phase started")
    else:
        started = strict_json((directory / "final" / "started.json").read_bytes())
        started_problems = keys_problems("final/started.json", started, STARTED_KEYS)
        if not started_problems:
            if started["declaration_sha256"] != declaration_sha256:
                started_problems.append("final/started.json names another declaration")
            started_problems += seed_mapping_problems("final/started.json selections", started["selections"], seeds)
            started_problems += seed_mapping_problems(
                "final/started.json descriptions", started["descriptions"], seeds,
                lambda value: isinstance(value, dict) and "agent" in value, "an agent description")
            if started["selections"] != json_normalized(selections):
                started_problems.append("final/started.json selections differ from the selections in state.json")
        problems += started_problems
    problems += seed_mapping_problems("outcome.final_results", final_results, seeds, is_sha256, "a SHA-256")
    if problems:
        return problems
    for seed in sorted(seeds):
        path = directory / "final" / f"seed-{seed}.json"
        if safe_sha256_file(path) != final_results[seed]:
            problems.append(f"{path} differs from the hash certified by COMPLETE")
            continue
        result = strict_json(path.read_bytes())
        if (not isinstance(result, dict) or type(result.get("seed")) is not int or str(result["seed"]) != seed
                or result.get("completeness") != "complete"):
            problems.append(f"{path} is not a complete sealed result for seed {seed}")
        elif (result.get("selection") != started["selections"][seed]
              or result.get("agent") != started["descriptions"][seed]["agent"]):
            problems.append(f"{path} selection or agent differs from final/started.json")
    return problems


def counter_problems(declaration, counters, seeds):
    """The counter account of a COMPLETE campaign: exact structure, seed coverage, domains, internal totals.
    Limits are checked against the completion record (``completion_problems``) and at the barrier."""
    problems = keys_problems("counters", counters, COUNTER_KEYS)
    if problems:
        return problems
    time_account, training, evaluation = counters["time"], counters["training"], counters["evaluation"]
    time_problems = keys_problems("counters.time", time_account, TIME_KEYS)
    if not time_problems:
        if not is_seconds(time_account["elapsed_seconds"]):
            time_problems.append("counters.time.elapsed_seconds is not finite non-negative seconds")
        time_problems += seed_mapping_problems("counters.time.training_seconds", time_account["training_seconds"],
                                               seeds, is_seconds, "finite non-negative seconds")
        phases = time_account["phase_seconds"]  # diagnostic: typed, but no required phase names
        if not isinstance(phases, dict) or not all(is_seconds(v) for v in phases.values()):
            time_problems.append("counters.time.phase_seconds is not a mapping of phase names to seconds")
        if not is_count(time_account["evaluation_games_started"]):
            time_problems.append("counters.time.evaluation_games_started is not a non-negative integer")
    problems += time_problems
    training_problems = seed_mapping_problems("counters.training", training, seeds)
    if not training_problems:
        generations = declaration["generations"]
        games = generations * declaration["config"]["games_per_generation"]
        for seed in sorted(seeds):
            name, entry = f"counters.training[{seed}]", training[seed]
            entry_problems = keys_problems(name, entry, TRAINING_KEYS)
            if not entry_problems and not all(is_count(v) for v in entry.values()):
                entry_problems.append(f"{name} has values that are not non-negative integers")
            if not entry_problems and not (entry["generations_attempted"] == entry["generations_completed"]
                                           == generations):
                entry_problems.append(f"seed {seed} completed {entry['generations_completed']} of {generations} "
                                      f"generations ({entry['generations_attempted']} attempted)")
            if not entry_problems and not (entry["selfplay_games_attempted"] == entry["selfplay_games_completed"]
                                           == games):
                entry_problems.append(f"seed {seed} completed {entry['selfplay_games_completed']} of {games} "
                                      f"self-play games ({entry['selfplay_games_attempted']} attempted)")
            training_problems += entry_problems
    problems += training_problems
    evaluation_problems = keys_problems("counters.evaluation", evaluation, EVALUATION_KEYS)
    if not evaluation_problems:
        by_kind = evaluation["by_kind"]
        evaluation_problems = keys_problems("counters.evaluation.by_kind", by_kind, GAME_KINDS + ROW_KINDS)
        if not evaluation_problems:
            for kind, units in sorted(by_kind.items()):
                if keys_problems(kind, units, ("started", "completed")) or not all(map(is_count, units.values())):
                    evaluation_problems.append(f"counters.evaluation.by_kind[{kind}] is malformed")
                elif units["started"] != units["completed"]:
                    evaluation_problems.append(f"{kind}: {units['started']} evidence units started, "
                                               f"{units['completed']} completed")
    if not evaluation_problems:
        def total(kinds, key):
            return sum(by_kind[kind][key] for kind in kinds)
        ceiling = declaration["budgets"]["evaluation_games_ceiling"]
        expected = dict(development_games=total(DEVELOPMENT_GAME_KINDS, "completed"),
                        sealed_games=total(SEALED_GAME_KINDS, "completed"),
                        calibration_games=total(CALIBRATION_GAME_KINDS, "completed"),
                        total_games_started=total(GAME_KINDS, "started"),
                        total_games_completed=total(GAME_KINDS, "completed"), ceiling=ceiling)
        for key, value in expected.items():
            if not is_count(evaluation[key]) or evaluation[key] != value:
                evaluation_problems.append(f"counters.evaluation.{key} is {evaluation[key]!r}; the units give {value}")
        if evaluation["total_games_started"] > ceiling:
            evaluation_problems.append("evaluation games started exceed the declared ceiling")
        if not time_problems and time_account["evaluation_games_started"] != evaluation["total_games_started"]:
            evaluation_problems.append("counters.time.evaluation_games_started differs from the games started")
    return problems + evaluation_problems


def completion_problems(completion, counters, budgets, seeds):
    """The completion record: exact structure and seed coverage; its endpoint, every seed's training seconds
    and the games started within the declared limits (inclusive); and equal to the account it certifies."""
    problems = keys_problems("outcome.completion", completion, COMPLETION_KEYS)
    if problems:
        return problems
    if not isinstance(completion["endpoint"], str):
        problems.append("outcome.completion.endpoint is not a description")
    if completion["limits"] != {k: budgets[k] for k in COMPLETION_LIMITS}:
        problems.append("completion record limits differ from the declaration")
    if not is_seconds(completion["elapsed_seconds"]):
        problems.append("outcome.completion.elapsed_seconds is not finite non-negative seconds")
    elif not completion["elapsed_seconds"] <= budgets["campaign_seconds"]:
        problems.append("completion endpoint is past the campaign wall-clock budget")
    training = completion["training_seconds"]
    training_problems = seed_mapping_problems("outcome.completion.training_seconds", training, seeds, is_seconds,
                                              "finite non-negative seconds")
    if not training_problems:
        training_problems = [f"seed {seed} training seconds exceed the per-seed budget at completion"
                             for seed in sorted(seeds) if not training[seed] <= budgets["per_run_training_seconds"]]
    problems += training_problems
    if not is_count(completion["evaluation_games_started"]):
        problems.append("outcome.completion.evaluation_games_started is not a non-negative integer")
    elif not completion["evaluation_games_started"] <= budgets["evaluation_games_ceiling"]:
        problems.append("evaluation games started exceed the ceiling at completion")
    # The same quantities are stored in the account the record certifies: both copies must agree exactly.
    time_account = counters.get("time") if isinstance(counters, dict) else None
    for key in ("elapsed_seconds", "training_seconds", "evaluation_games_started"):
        if not isinstance(time_account, dict) or time_account.get(key) != completion[key]:
            problems.append(f"outcome.completion.{key} differs from counters.time.{key}")
    return problems


def history_problems(document, seeds):
    """The selections and sealed-phase start recorded in the transition history (each state appears once)."""
    entries = {entry["state"]: entry for entry in document["history"]}
    token, problems = document["declaration_sha256"], []
    for state in (SELECTION_COMPLETE, RUNNING_FINAL):
        problems += selection_problems(f"history {state} selections", entries[state].get("selections"), seeds, token)
    if not problems and entries[SELECTION_COMPLETE]["selections"] != entries[RUNNING_FINAL]["selections"]:
        problems.append("selections changed between DEVELOPMENT_SELECTION_COMPLETE and RUNNING_FINAL_EVALUATION")
    if entries[RUNNING_FINAL].get("started_sha256") != document["outcome"]["started_sha256"]:
        problems.append("started.json hash differs from the one recorded on entering the sealed phase")
    return problems


def complete_record_problems(directory, document):
    """Every way a COMPLETE state document (strictly parsed) fails the frozen schema; empty means acceptable."""
    outcome = document["outcome"]
    problems = keys_problems("outcome", outcome, OUTCOME_KEYS)
    if problems:
        return problems
    if outcome["status"] != COMPLETE or not isinstance(outcome["note"], str):
        problems.append("outcome status or note is malformed")
    token = document["declaration_sha256"]
    if not is_sha256(token) or safe_sha256_file(directory / "declaration.json") != token:
        return problems + ["declaration.json differs from the declaration the campaign ran under"]
    declaration = strict_json((directory / "declaration.json").read_bytes())
    seeds = required_seed_keys(declaration)
    if document["seeds"] != declaration["seeds"]:
        problems.append("state seeds differ from the declaration")
    problems += history_problems(document, seeds)
    if problems:
        return problems
    selections = next(h["selections"] for h in document["history"] if h["state"] == RUNNING_FINAL)
    return (certified_evidence_problems(directory, token, seeds, outcome["final_results"], outcome["started_sha256"],
                                        selections)
            + counter_problems(declaration, document["counters"], seeds)
            + completion_problems(outcome["completion"], document["counters"], declaration["budgets"], seeds))


def official_results(directory):
    """The sealed results, only if state.json is a COMPLETE record that passes the strict frozen schema
    (``complete_record_problems``) and every file it certifies validates. A COMPLETE string alone, or a record
    missing any required field or seed, is never enough."""
    directory = Path(directory)
    state = CampaignStateFile.load(directory)
    if state.state != COMPLETE:
        raise NotOfficialEvidence(f"Campaign is {state.state}; its artifacts are forensic only and are never "
                                  "official evidence")
    try:
        document = strict_json(state.data)
        problems = complete_record_problems(directory, document)
        if not problems:
            results = {seed: strict_json((directory / "final" / f"seed-{seed}.json").read_bytes())
                       for seed in sorted(document["outcome"]["final_results"])}
    except (MalformedRecord, KeyError, TypeError, ValueError, AttributeError, OSError) as error:
        problems = [f"malformed COMPLETE record: {type(error).__name__}: {error}"]
    if problems:
        raise NotOfficialEvidence("COMPLETE is not accepted: " + "; ".join(problems))
    return dict(declaration_sha256=document["declaration_sha256"], results=results,
                completion=document["outcome"]["completion"])


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
    """Write a new declaration from the frozen packages beside ``output`` and this runtime."""
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
    "Phase 4D.3B.4 frozen declaration (format 3) for the Milestone 3 two-seed campaign; running it needs "
    "separate authorization.",
    "The authorization token is the SHA-256 of this exact file. The Phase 4D.3B token "
    "2741314399741c1b20510b8a2beb42e0938b0aeafc9dd0c487201e57208867d8, the Phase 4D.3B.1 token "
    "8a8a52b100c7e32aa5e3e5b05469578084c8e12f37750ba81fee071990ff5e39, the Phase 4D.3B.2 token "
    "ed641aea2fef86a29000df55d1e46d9a32bf48dc3ec8f5dbcc0f8449668d0bbb and the Phase 4D.3B.3 token "
    "34b4d899639d59d1bd4408ecc505b7eb20cb4d9c2fdc12095402c246887ac390 are REJECTED / NOT AUTHORIZED.",
    "The official campaign is NON-RESUMABLE: one owning process runs both seeds, the development selection and the "
    "sealed final evaluation; any interruption, limit or identity drift before the completion barrier makes it "
    "INCOMPLETE.",
    "COMPLETE is a one-way terminal commit: all evidence is published and validated before one completion barrier, "
    "whose monotonic reading is the authoritative endpoint; nothing after it can downgrade a visible COMPLETE.",
    "Official results require a COMPLETE record that passes the strict schema, with exact coverage of seeds 42 and "
    "314159 in every per-seed mapping.",
    "It binds execution-source content, the complete runtime identity, frozen packages and records, the retained "
    "4D.2f checkpoint and the launch-control semantics; any difference refuses the launch.",
    "Seed 42 is primary; seed 314159 is replication, not a second chance.",
    "Package and frozen-record paths are relative to this file; the Phase 4D.2f checkpoint path is relative to the "
    "repository root.",
]


# CLI --------------------------------------------------------------------------------------------

EXIT_COMPLETED, EXIT_REFUSED, EXIT_INCOMPLETE = 0, 2, 4
STOP_SIGNALS = (signal.SIGINT, signal.SIGTERM)


def install_stop_handlers(campaign):
    """SIGINT/SIGTERM request a cooperative stop. The handler only records: it never raises and never writes
    state, so it cannot interrupt the terminal commit or replace COMPLETE. Before the completion barrier the
    stop ends the campaign INCOMPLETE; after it, the request is recorded and cannot change the outcome."""
    def stop(signum, frame):
        campaign.budget.request_stop(f"signal {signum}")
    for signum in STOP_SIGNALS:
        signal.signal(signum, stop)


def main(argv=None):
    parser = argparse.ArgumentParser(description="AlphaZero v2 official campaign (requires authorization; "
                                                 "non-resumable)")
    sub = parser.add_subparsers(dest="command", required=True)
    pre = sub.add_parser("preflight")
    pre.add_argument("--declaration", required=True)
    frz = sub.add_parser("freeze")
    frz.add_argument("--output", required=True)
    stat = sub.add_parser("status")
    stat.add_argument("--campaign-dir", required=True)
    command = sub.add_parser("launch")
    command.add_argument("--declaration", required=True)
    command.add_argument("--campaign-dir", required=True, help="a new directory; existing ones are never continued")
    command.add_argument("--authorize", required=True)
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
    except InterruptedCampaign as error:
        print(str(error))
        return EXIT_INCOMPLETE
    except (PermissionError, FileExistsError, RuntimeError) as error:
        print(f"refused: {type(error).__name__}: {error}")
        return EXIT_REFUSED

    install_stop_handlers(campaign)
    status = campaign.run()
    print(status)
    for problem in campaign.post_commit_errors:
        print(f"warning: {problem}; the certified COMPLETE outcome is unchanged", file=sys.stderr)
    for reason in campaign.budget.late_stop_requests:
        print(f"note: {reason} arrived after the completion barrier and cannot change the outcome", file=sys.stderr)
    return EXIT_COMPLETED if status == "completed" else EXIT_INCOMPLETE


if __name__ == "__main__":
    sys.exit(main())
