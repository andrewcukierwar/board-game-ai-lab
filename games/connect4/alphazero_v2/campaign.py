"""Single-owner, fail-closed AlphaZero v2 official campaign launcher (Milestone 3 tool, Phase 4D.3B.5).

Nothing runs on import. ``launch`` fails closed unless all of the following hold:

* ``--authorize`` equals the SHA-256 of the exact declaration file, and neither
  is a rejected token (Phase 4D.3B ``2741314399...``, Phase 4D.3B.1
  ``8a8a52b100...``, Phase 4D.3B.2 ``ed641aea2f...``, Phase 4D.3B.3
  ``34b4d89963...`` and Phase 4D.3B.4 ``9f7259827e...`` are rejected).
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

Every artifact ``official_results`` relies on has one authoritative schema in
``official_evidence``, and the same validators run at three points: before
``final/started.json`` and each ``final/seed-S.json`` is published, before the
completion barrier (read back from disk), and in ``official_results``. A sealed
result has one producer, ``official_evidence.derive_seed_result``: acceptance
re-derives every summary from the result's raw evidence and requires the whole
result to equal it.

Layout of a campaign directory (created once; nothing is overwritten except state.json):

    declaration.json        verbatim copy; its hash is the authorization token
    packages/NAME.json      verbatim copy of every declared package (hashes bound by the declaration)
    state.json              the single authority (atomically replaced on each transition)
    campaign.lock           single-owner lock
    source/                 source snapshot (provenance)
    runs/seed-S/generations/generation-NNNN/   inference, resume (diagnostic only), games, summary
    runs/seed-S/champion-check-NNNN.json, runs/seed-S/selection.json
    final/started.json, final/seed-S.json
                            official only if state.json is COMPLETE and lists their hashes
"""
import argparse
import importlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from . import official_evidence, statistics
from .arena import (PAIRED_GAMES, GuardedUCTOpponent, NegamaxOpponent, RandomOpponent, paired_game, scored)
from .config import V2Config
from .launch_control import (COMPLETE, INCOMPLETE, INTERRUPTED_MESSAGE, REJECTED_DECLARATION_TOKENS,  # noqa: F401
                             RUNNING_FINAL, SELECTION_COMPLETE, Budget, CampaignLock, CampaignLocked,
                             CampaignStateFile, CampaignStop, InterruptedCampaign, StateCorrupt, TerminalCampaign,
                             fsync_directory, launch_control_declaration, publish_view, running_seed, sha256_bytes,
                             sha256_file, view_bytes, write_once)
from .official_evidence import (CALIBRATION_GAME_KINDS, DECLARATION_FORMAT, DECLARATION_KEYS,  # noqa: F401
                                DECLARATION_VERSION, DEVELOPMENT_GAME_KINDS, FROZEN_RECORD_NAMES, GAME_KINDS,
                                OFFICIAL_SEEDS, OPPONENT_NAMES, PACKAGE_DIRECTORY, ROW_KINDS, SEALED_GAME_KINDS,
                                SELECTION_RULE, MalformedRecord, champion_name, completion_record, nn_only_name,
                                planned_evaluation_games, required_seed_keys, strict_json)
from .official_evidence import normalized as json_normalized
from .oracle import family_key
from .packages import load_package
from .provenance import (REPO_ROOT, SOURCE_SCHEME, configure_deterministic_runtime, execution_source_files,
                         execution_source_identity, git_provenance, required_thread_environment, runtime_differences,
                         runtime_identity, source_differences, source_group, source_groups,
                         unavailable_runtime_fields)

UNTRACKED_LIMIT_BYTES = 100 * 1024 * 1024
RUNTIME_IDENTITY_KEYS = tuple(sorted(runtime_identity()))
PHASE4D2F_CHECKPOINT = dict(path="experiment-output/phase4d2f-neural-value-anchoring-seed42-20261005/candidate.pt",
                            sha256="78276912bdef5f1b1b53ac6b2ec3ede2eed20a8ebb5796a9bdcbb57e4cb28c51")


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


# Declaration ------------------------------------------------------------------------------

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
                            "4D.3B.2 / 4D.3B.3 / 4D.3B.4 launch reviews); it must never authorize a campaign")
    declaration = json.loads(data)
    validate_declaration(declaration, path.parent)
    return declaration, digest


def validate_declaration(declaration, base):
    # The shared evidence contract: every check that needs neither torch nor files, the same function acceptance
    # re-applies to a certified declaration.
    problems = official_evidence.declaration_problems(declaration)
    if problems:
        raise ValueError("Declaration violates the official evidence contract: " + "; ".join(problems))
    # Launch-only checks: the execution-source partition and the runtime field names come from this process.
    source = declaration["execution_source"]
    if source["scheme"] != SOURCE_SCHEME or source["groups"] != source_groups(source["files"]):
        raise ValueError("Declared execution source is internally inconsistent")
    runtime = declaration["runtime_identity"]
    if tuple(sorted(runtime)) != RUNTIME_IDENTITY_KEYS:
        raise ValueError("Declared runtime identity must contain exactly the enforced runtime fields")
    if runtime["thread_environment"] != required_thread_environment(declaration["runtime"]["threads"]):
        raise ValueError("Declared runtime identity disagrees with the declared runtime settings")
    for name, entry in declaration["frozen_records"].items():
        if sha256_file(Path(base) / entry["path"]) != entry["sha256"]:
            raise ValueError(f"Frozen record {name} differs from the declaration")
    documents = {name: load_package(Path(base) / entry["path"], expected_sha256=entry["sha256"])
                 for name, entry in declaration["packages"].items()}
    problems = official_evidence.package_problems(declaration, documents)
    if problems:
        raise ValueError("Declared packages violate the official evidence contract: " + "; ".join(problems))
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
            self._snapshot_packages()
            self.state = CampaignStateFile.create(directory, declaration_sha256=self.declaration_sha256,
                                                  seeds=self.seeds)
        except BaseException:
            self.lock.release()
            raise

    def _snapshot_packages(self):
        """Copy every declared package into the campaign directory, verified against the declared hash, so the
        directory alone holds every input official acceptance derives its expectations from."""
        target = self.directory / PACKAGE_DIRECTORY
        target.mkdir()
        for name, entry in sorted(self.declaration["packages"].items()):
            data = (self.base / entry["path"]).read_bytes()
            if sha256_bytes(data) != entry["sha256"]:
                raise LaunchRefused(f"Package {name} changed while launching")
            write_once(target / f"{name}.json", data)

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
                         rule=SELECTION_RULE,
                         declaration_sha256=self.declaration_sha256)
        publish_view(self.run_dir(seed) / "selection.json", selection)
        return selection

    # Sealed final evaluation ------------------------------------------------------------------------

    def _final(self, selections):
        with self.budget.phase("final"):
            self._check("evaluation")
            self.verify_identity("final start")
            self.context = self._official_context()
            descriptions = {str(seed): self._final_descriptions(seed, selections[str(seed)]) for seed in self.seeds}
            self.final_started = json_normalized(dict(
                selections=selections, descriptions=descriptions, runtime=self.runtime,
                execution_sha256=self.declaration["execution_source"]["sha256"],
                declaration_sha256=self.declaration_sha256))
            # Contract call site A (started.json): validated before it is published.
            self._require_contract("final/started.json", official_evidence.started_problems(
                self.context, self.final_started, self.final_started["selections"]))
            self.started_sha256 = publish_view(self.directory / "final" / "started.json", self.final_started)
            # Selections are fixed durably here, before any sealed inference.
            self._transition(RUNNING_FINAL, selections=selections, started_sha256=self.started_sha256)
            for seed in self.seeds:
                self._final_seed_units(seed)
                result = json_normalized(self._final_seed_result(seed))
                # Contract call site A (final/seed-S.json): the exact bytes about to be published are validated.
                self._require_contract(f"final/seed-{seed}.json", official_evidence.seed_result_problems(
                    self.context, self.final_started, str(seed), result))
                self._require_within_limits()
                self.verify_identity(f"final seed {seed}")
                self.final_results[str(seed)] = publish_view(self.directory / "final" / f"seed-{seed}.json", result)
                self.state.record_counters(self.counters())
        self.fault("before_campaign_complete")
        self._complete()

    def _official_context(self):
        context, problems = official_evidence.load_context(self.directory, self.declaration_sha256)
        self._require_contract("the campaign directory's declaration and packages", problems)
        return context

    @staticmethod
    def _require_contract(name, problems):
        if problems:
            raise RuntimeError(f"{name} violates the official evidence contract: " + "; ".join(problems))

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
        # Contract call site B: every certified file read back from disk, exactly as official_results reads it.
        context, problems = official_evidence.load_context(self.directory, self.declaration_sha256)
        if not problems:
            problems = (official_evidence.certified_evidence_problems(
                self.directory, context, self.final_results, self.started_sha256, self.final_started["selections"])
                + official_evidence.counter_problems(context, json_normalized(self.counters())))
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
                              expected_sha256=selection["inference"]["sha256"], name=champion_name(seed))
        agent.identity["path"] = selection["inference"]["path"]
        return agent

    def _final_descriptions(self, seed, selection):
        opponents = ladder_opponents(self.declaration, seed)
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
        opponents = ladder_opponents(self.declaration, seed)
        for name in final["ladder"]:
            self._arena_units(seed, f"{prefix}/ladder-{name}", "final_ladder_game", agent, opponents[name], openings,
                              f"seed{seed}-final-{name}")
        nn_only = V2NNOnlyAgent(agent.inference, name=nn_only_name(seed))
        self._arena_units(seed, f"{prefix}/nn-only-vs-random", "final_nn_only_game", nn_only, RandomOpponent(),
                          openings, f"seed{seed}-final-nn-only")
        config = self.config_for(seed)
        for index in range(final["calibration_games"]):
            self._unit(seed, f"{prefix}/calibration/{index:04d}", "calibration_game", lambda: dict(records=[
                dict(r, game=index) for r in calibration_games(agent.inference, config, games=1,
                                                               seed=seed * 1_000_003 + index, check=check)]))

    def _final_seed_result(self, seed):
        """The sealed result for one seed: this process's raw evidence, summarized by the one result producer."""
        final = self.declaration["final"]
        tactics = package_rows(self.declaration, self.base, "tactical-sealed")
        solved = package_rows(self.declaration, self.base, "solved-sealed")
        openings = package_rows(self.declaration, self.base, "openings-sealed")[:final["ladder_openings"]]
        prefix = f"final/seed{seed}"
        evidence = self._evidence

        def records(label):
            return [evidence(f"{prefix}/{label}/{row['id']}/{g}") for row in openings for g, _ in PAIRED_GAMES]
        raw = dict(
            overlap_families=sorted(training_overlap(self._archived_games(seed),
                                                     [r["moves"] for r in tactics + solved])),
            tactical={row["id"]: evidence(f"{prefix}/tactical/{row['id']}") for row in tactics},
            solved={row["id"]: evidence(f"{prefix}/solved/{row['id']}") for row in solved},
            calibration=[r for index in range(final["calibration_games"])
                         for r in evidence(f"{prefix}/calibration/{index:04d}")["records"]],
            ladder={name: records(f"ladder-{name}") for name in final["ladder"]},
            nn_only_vs_random=records("nn-only-vs-random"))
        return official_evidence.derive_seed_result(self.context, seed, self.final_started, raw)

    def _archived_games(self, seed):
        """Every archived training game of this seed, from files re-verified against their recorded hashes."""
        generations = self.seed_records[str(seed)]["generations"]
        return [self._artifact(seed, g, "games")[0] for g in sorted(generations) if g]


def opponent_by_name(name):
    if name == "random":
        return RandomOpponent()
    if name.startswith("negamax"):
        return NegamaxOpponent(int(name[len("negamax"):]))
    if name.startswith("guarded_uct_"):
        return GuardedUCTOpponent(int(name[len("guarded_uct_"):]))
    raise ValueError(f"Unknown opponent {name}")


def ladder_opponents(declaration, seed):
    """The declared final-ladder opponents for one seed (``declaration["final"]["ladder"]`` names)."""
    from .evaluation import V1SearchArenaAgent, V2SearchArenaAgent, untrained_model
    final, simulations = declaration["final"], declaration["evaluation"]["simulations"]
    opponents = {name: opponent_by_name(name) for name in final["ladder"] if name in OPPONENT_NAMES}
    if "initial_v2_512" in final["ladder"]:
        opponents["initial_v2_512"] = V2SearchArenaAgent(untrained_model(seed), simulations, name="initial_v2_512")
    if "phase4d2f_512" in final["ladder"]:
        checkpoint = declaration["phase4d2f_checkpoint"]
        opponents["phase4d2f_512"] = V1SearchArenaAgent(REPO_ROOT / checkpoint["path"], checkpoint["sha256"],
                                                        simulations)
        opponents["phase4d2f_512"].identity["path"] = checkpoint["path"]
    return opponents


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


# Official results ----------------------------------------------------------------------------------
# Acceptance is the official evidence contract (official_evidence.complete_record_problems): the same validators
# the owning process ran before publishing and before the completion barrier. Nothing is validated "if present".

def official_results(directory):
    """The sealed results, only if state.json is a COMPLETE record whose every certified artifact satisfies the
    official evidence contract. A COMPLETE string alone, a hash match alone, or a record or result missing any
    required field, seed, row, opponent, game or summary is never enough. Returns exactly the validated documents."""
    directory = Path(directory)
    state = CampaignStateFile.load(directory)
    if state.state != COMPLETE:
        raise NotOfficialEvidence(f"Campaign is {state.state}; its artifacts are forensic only and are never "
                                  "official evidence")
    results = {}
    try:
        document = strict_json(state.data)
        problems = official_evidence.complete_record_problems(directory, document, results)
    except (MalformedRecord, OSError) as error:
        problems = [f"malformed COMPLETE record: {type(error).__name__}: {error}"]
    except Exception as error:  # noqa: BLE001 - fail closed: a validator defect can only refuse, never accept
        problems = [f"internal validation error (refused): {type(error).__name__}: {error}"]
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
    "Phase 4D.3B.5 frozen declaration (format 3) for the Milestone 3 two-seed campaign; running it needs "
    "separate authorization.",
    "The authorization token is the SHA-256 of this exact file. The Phase 4D.3B token "
    "2741314399741c1b20510b8a2beb42e0938b0aeafc9dd0c487201e57208867d8, the Phase 4D.3B.1 token "
    "8a8a52b100c7e32aa5e3e5b05469578084c8e12f37750ba81fee071990ff5e39, the Phase 4D.3B.2 token "
    "ed641aea2fef86a29000df55d1e46d9a32bf48dc3ec8f5dbcc0f8449668d0bbb, the Phase 4D.3B.3 token "
    "34b4d899639d59d1bd4408ecc505b7eb20cb4d9c2fdc12095402c246887ac390 and the Phase 4D.3B.4 token "
    "9f7259827e8abee6db81113147e48c2a3420248296d09df559f2783c5d7d85f4 are REJECTED / NOT AUTHORIZED.",
    "The official campaign is NON-RESUMABLE: one owning process runs both seeds, the development selection and the "
    "sealed final evaluation; any interruption, limit or identity drift before the completion barrier makes it "
    "INCOMPLETE.",
    "COMPLETE is a one-way terminal commit: all evidence is published and validated before one completion barrier, "
    "whose monotonic reading is the authoritative endpoint; nothing after it can downgrade a visible COMPLETE.",
    "Official results require a COMPLETE record whose every certified artifact satisfies the one official evidence "
    "contract (official_evidence), with exact coverage of seeds 42 and 314159, the declared ladder, the sealed "
    "package rows, the declared openings and calibration games, and every summary re-derived from its evidence.",
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
