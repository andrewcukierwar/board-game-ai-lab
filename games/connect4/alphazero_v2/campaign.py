"""Bounded, explicitly authorized AlphaZero v2 campaign launcher (Milestone 3 tool).

Nothing runs on import. ``run`` and ``final-evaluate`` refuse to start unless
``--authorize`` equals the SHA-256 of the exact declaration file, the library
thread environment is pinned, and the deterministic runtime can be configured.

Layout of a campaign directory (created once, never reused or overwritten):

    declaration.json             verbatim copy; its hash is the authorization token
    ledger.jsonl                 append-only, fsynced: attempts, heartbeats, budgets, pruning
    runs/seed-S/
        artifacts.jsonl          every published artifact with SHA-256 and size
        champion.json            current development champion (replaced atomically)
        selection.json           final development-only champion selection (written once)
        games/generation-NNNN.jsonl   archived game records (outside the active replay)
        attempt-NNN/             source snapshot, events, boundaries, summaries, evaluations
    final/                       sealed evaluation (one-time; started.json marker first)

Budgets (from the declaration) are cumulative across attempts through the
ledger: per-run collection+optimization seconds, campaign wall-clock seconds
(including evaluation) and the evaluation-game ceiling. Checks run before
every self-play search, optimizer update, arena game and arena move. Abandoned
work counts. Time between the last heartbeat and a crash is not recoverable and
is disclosed rather than estimated.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from . import statistics
from .arena import (GuardedUCTOpponent, NegamaxOpponent, RandomOpponent, StopEvaluation, run_paired_arena,
                    scored)
from .config import V2Config
from .oracle import family_key
from .packages import load_package
from .provenance import (REPO_ROOT, configure_deterministic_runtime, execution_source_identity, git_provenance,
                         required_thread_environment, runtime_identity)

DECLARATION_FORMAT = "connect4-alphazero-v2-campaign-declaration"
DECLARATION_VERSION = 1
HEARTBEAT_SECONDS = 30.0
UNTRACKED_LIMIT_BYTES = 100 * 1024 * 1024
DECLARATION_KEYS = {"format", "format_version", "name", "kind", "seeds", "primary_seed", "config", "generations",
                    "budgets", "runtime", "evaluation", "champion", "final", "packages", "thresholds",
                    "retention", "phase4d2f_checkpoint", "notes"}


class CampaignStop(StopEvaluation):
    """Cooperative stop: budget exhausted or stop requested. Work in flight is discarded."""


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_once(path, value):
    """Write JSON to a new file only (temporary file, fsync, hard-link publish)."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(temporary, "w") as stream:
        json.dump(value, stream, indent=1, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        os.unlink(temporary)
    return sha256_file(path)


def replace_json(path, value):
    """Atomically replace a small mutable state file (champion pointer only)."""
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(temporary, "w") as stream:
        json.dump(value, stream, indent=1, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class JsonLog:
    """Append-only JSON lines with fsync after every record."""

    def __init__(self, path):
        self.path = Path(path)

    def append(self, record):
        with open(self.path, "a") as stream:
            stream.write(json.dumps(dict(record, utc=utc_now()), sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def records(self):
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]


# Declaration ------------------------------------------------------------------------------

def load_declaration(path):
    path = Path(path)
    declaration = json.loads(path.read_text())
    validate_declaration(declaration, path.parent)
    return declaration, sha256_file(path)


def validate_declaration(declaration, base):
    if set(declaration) != DECLARATION_KEYS:
        raise ValueError(f"Declaration keys differ: {sorted(set(declaration) ^ DECLARATION_KEYS)}")
    if (declaration["format"], declaration["format_version"]) != (DECLARATION_FORMAT, DECLARATION_VERSION):
        raise ValueError("Not a v2 campaign declaration")
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
    for name, entry in declaration["packages"].items():
        load_package(base / entry["path"], expected_sha256=entry["sha256"])
    return True


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
                      runtime=None, notes=None):
    """Assemble the frozen declaration; ``packages`` maps package name -> {path, sha256}."""
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
        phase4d2f_checkpoint=dict(path="experiment-output/phase4d2f-neural-value-anchoring-seed42-20261005/"
                                       "candidate.pt",
                                  sha256="78276912bdef5f1b1b53ac6b2ec3ede2eed20a8ebb5796a9bdcbb57e4cb28c51"),
        notes=notes or [])
    declaration = json.loads(json.dumps(declaration))  # tuples -> lists, as stored
    declaration["budgets"]["planned_evaluation_games"] = planned_evaluation_games(declaration)
    return declaration


# Budgets -----------------------------------------------------------------------------------

class Budget:
    """Cumulative budget accounting across attempts, enforced by ``check(phase)``."""

    def __init__(self, ledger, declaration, seed, attempt, clock=time.monotonic):
        self.ledger, self.declaration, self.seed, self.attempt, self.clock = ledger, declaration, seed, attempt, clock
        self.limits = declaration["budgets"]
        prior = consumed(ledger.records())
        self.prior_total = prior["total_seconds"]
        self.prior_training = prior["training_seconds"].get(str(seed), 0.0)
        self.prior_evaluation_games = prior["evaluation_games"]
        self.started = clock()
        self.training_accumulated = 0.0
        self.training_started = None
        self.evaluation_games = 0
        self.training_games = 0
        self.stop_reason = None
        self.last_heartbeat = -1e18

    def request_stop(self, reason="stop requested"):
        self.stop_reason = reason

    def training_seconds(self):
        running = self.clock() - self.training_started if self.training_started is not None else 0.0
        return self.prior_training + self.training_accumulated + running

    def total_seconds(self):
        return self.prior_total + self.clock() - self.started

    def begin_training(self):
        self.training_started = self.clock()

    def end_training(self):
        if self.training_started is not None:
            self.training_accumulated += self.clock() - self.training_started
            self.training_started = None
        self.heartbeat(force=True)

    def counters(self):
        return dict(attempt=self.attempt, seed=self.seed,
                    attempt_total_seconds=self.clock() - self.started,
                    attempt_training_seconds=self.training_seconds() - self.prior_training,
                    attempt_evaluation_games=self.evaluation_games, attempt_training_games=self.training_games)

    def heartbeat(self, force=False):
        now = self.clock()
        if force or now - self.last_heartbeat >= HEARTBEAT_SECONDS:
            self.last_heartbeat = now
            self.ledger.append(dict(event="heartbeat", **self.counters()))

    def check(self, phase):
        if self.stop_reason is not None:
            raise CampaignStop(self.stop_reason)
        if self.total_seconds() >= self.limits["campaign_seconds"]:
            raise CampaignStop("campaign wall-clock budget exhausted")
        if phase == "training" and self.training_seconds() >= self.limits["per_run_training_seconds"]:
            raise CampaignStop("per-run collection+optimization budget exhausted")
        if phase == "evaluation" and (self.prior_evaluation_games + self.evaluation_games
                                      >= self.limits["evaluation_games_ceiling"]):
            raise CampaignStop("evaluation-game ceiling reached")
        self.heartbeat()

    def count_evaluation_game(self, record):
        self.evaluation_games += 1
        self.heartbeat()


def consumed(records):
    """Budget consumed by all attempts so far: last heartbeat of each attempt (seed-scoped training)."""
    last = {}
    for record in records:
        if record.get("event") in ("heartbeat", "attempt_end"):
            last[(record["seed"], record["attempt"])] = record
    training = {}
    for (seed, _), record in last.items():
        training[str(seed)] = training.get(str(seed), 0.0) + record["attempt_training_seconds"]
    return dict(total_seconds=sum(r["attempt_total_seconds"] for r in last.values()),
                training_seconds=training,
                evaluation_games=sum(r["attempt_evaluation_games"] for r in last.values()),
                training_games=sum(r["attempt_training_games"] for r in last.values()))


# Source snapshot -------------------------------------------------------------------------

def snapshot_source(directory):
    """Commit, tracked patch, untracked (non-ignored) files and execution identity."""
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
    def __init__(self, directory, declaration_path, authorization, *, clock=time.monotonic,
                 strict_runtime=True, strict_source=True):
        declaration, digest = load_declaration(declaration_path)
        if authorization != digest:
            raise PermissionError("Authorization token must equal the declaration SHA-256")
        self.declaration, self.declaration_sha256 = declaration, digest
        self.base = Path(declaration_path).resolve().parent
        self.directory = Path(directory)
        self.clock = clock
        self.strict_runtime, self.strict_source = strict_runtime, strict_source
        if self.directory.exists():
            copy = self.directory / "declaration.json"
            if not copy.is_file() or sha256_file(copy) != digest:
                raise FileExistsError("Campaign directory exists for a different declaration")
        else:
            self.directory.mkdir(parents=True)
            shutil.copy2(declaration_path, self.directory / "declaration.json")
        self.ledger = JsonLog(self.directory / "ledger.jsonl")

    # Runtime -----------------------------------------------------------------------------

    def require_runtime(self):
        runtime = self.declaration["runtime"]
        expected = required_thread_environment(runtime["threads"])
        wrong = {k: os.environ.get(k) for k, v in expected.items() if os.environ.get(k) != v}
        if wrong:
            raise RuntimeError("Set the library thread environment before starting Python: "
                               + " ".join(f"{k}={v}" for k, v in expected.items()) + f" (found {wrong})")
        identity = configure_deterministic_runtime(runtime["threads"]) if runtime["deterministic_algorithms"] \
            else runtime_identity()
        return identity

    # Per-seed run ------------------------------------------------------------------------

    def run_dir(self, seed):
        return self.directory / "runs" / f"seed-{seed}"

    def config_for(self, seed):
        return V2Config.from_dict(dict(self.declaration["config"], seed=seed))

    def _new_attempt(self, seed):
        run = self.run_dir(seed)
        run.mkdir(parents=True, exist_ok=True)
        existing = sorted(p.name for p in run.glob("attempt-*"))
        number = len(existing) + 1
        attempt = run / f"attempt-{number:03d}"
        attempt.mkdir()
        return number, attempt

    def _artifacts(self, seed):
        return JsonLog(self.run_dir(seed) / "artifacts.jsonl")

    def _publish(self, seed, attempt, kind, generation, path):
        path = Path(path)
        record = dict(kind=kind, generation=generation, attempt=attempt,
                      path=str(path.relative_to(self.run_dir(seed))), sha256=sha256_file(path),
                      bytes=path.stat().st_size)
        self._artifacts(seed).append(record)
        return record

    def latest_boundary(self, seed):
        pruned = {r["path"] for r in self._artifacts(seed).records() if r["kind"] == "pruned"}
        boundaries = [r for r in self._artifacts(seed).records() if r["kind"] == "resume" and r["path"] not in pruned]
        return sorted(boundaries, key=lambda r: r["generation"])

    def prune(self, seed, attempt):
        keep = self.declaration["retention"]["resume_boundaries"]
        boundaries = self.latest_boundary(seed)
        for record in boundaries[:-keep] if len(boundaries) > keep else []:
            path = self.run_dir(seed) / record["path"]
            if sha256_file(path) != record["sha256"]:
                raise RuntimeError(f"Refusing to prune a modified artifact: {path}")
            path.unlink()
            self._artifacts(seed).append(dict(kind="pruned", generation=record["generation"], attempt=attempt,
                                              path=record["path"], sha256=record["sha256"], bytes=record["bytes"]))
            self.ledger.append(dict(event="pruned", seed=seed, attempt=attempt, path=record["path"],
                                    sha256=record["sha256"]))

    def run_seed(self, seed, *, extra_check=None):
        """Run or resume one seed until its generations finish or a budget/stop ends the attempt."""
        if seed not in self.declaration["seeds"]:
            raise ValueError("Seed is not declared")
        if (self.run_dir(seed) / "selection.json").exists():
            raise RuntimeError("This seed already completed champion selection")
        from .generation import GenerationRunner, load_resume_boundary  # torch-dependent
        runtime = self.require_runtime()
        number, attempt_dir = self._new_attempt(seed)
        events = JsonLog(attempt_dir / "events.jsonl")
        budget = self._budget = Budget(self.ledger, self.declaration, seed, number, clock=self.clock)
        self.ledger.append(dict(event="attempt_start", seed=seed, attempt=number, runtime=runtime,
                                declaration_sha256=self.declaration_sha256))
        status = "running"
        try:
            source = snapshot_source(attempt_dir / "source")
            events.append(dict(event="source", execution_sha256=source["execution"]["sha256"], git=source["git"]))
            boundaries = self.latest_boundary(seed)
            if boundaries:
                record = boundaries[-1]
                runner = load_resume_boundary(self.run_dir(seed) / record["path"], expected_sha256=record["sha256"],
                                              strict_runtime=self.strict_runtime, strict_source=self.strict_source)
                events.append(dict(event="resumed", boundary=record, lineage=runner.lineage[-1]))
            else:
                runner = GenerationRunner(self.config_for(seed))
                self._save_boundary(seed, number, attempt_dir, runner, events)
            self._initialize_champion(seed)
            config = runner.config
            dev_solved = package_rows(self.declaration, self.base, "solved-development")

            def check(phase):
                if extra_check is not None:
                    extra_check(phase, runner)
                budget.check(phase)
            # A scheduled champion check interrupted in an earlier attempt is rerun, never skipped.
            for generation in self._pending_checks(seed, runner.completed_generations):
                self._champion_check(seed, generation, attempt_dir, self._inference_record(seed, generation),
                                     budget, check, events)
            while runner.completed_generations < config.max_generations:
                check("training")
                generation = runner.completed_generations + 1
                budget.begin_training()
                try:
                    summary = runner.run_generation(check=lambda: check("training"))
                finally:
                    budget.end_training()
                budget.training_games += config.games_per_generation
                self._archive_games(seed, runner, generation)
                artifacts = self._save_boundary(seed, number, attempt_dir, runner, events)
                summary["development_raw_value"] = self._raw_value_metrics(attempt_dir, generation, dev_solved)
                write_json_once(attempt_dir / f"generation-{generation:04d}.summary.json",
                                dict(summary, artifacts=artifacts))
                events.append(dict(event="generation_complete", generation=generation,
                                   learner_weights_sha256=summary["learner_weights_sha256"],
                                   resources=summary["resources"]))
                self.prune(seed, number)
                if generation in self.declaration["champion"]["schedule"]:
                    self._champion_check(seed, generation, attempt_dir, artifacts["inference"], budget, check, events)
            self._select_champion(seed)
            status = "completed"
        except CampaignStop as stop:
            status = f"stopped: {stop}"
        except BaseException as error:
            status = f"failed: {type(error).__name__}: {error}"
            raise
        finally:
            self.ledger.append(dict(event="attempt_end", status=status, **budget.counters()))
        return status

    def _save_boundary(self, seed, attempt, attempt_dir, runner, events):
        generation = runner.completed_generations
        saved = runner.save_boundary(attempt_dir)
        records = dict(inference=self._publish(seed, attempt, "inference", generation, saved["inference"]),
                       resume=self._publish(seed, attempt, "resume", generation, saved["resume"]))
        events.append(dict(event="boundary", generation=generation, **records))
        return records

    def _archive_games(self, seed, runner, generation):
        directory = self.run_dir(seed) / "games"
        directory.mkdir(exist_ok=True)
        path = directory / f"generation-{generation:04d}.jsonl"
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
        games = [g for g in runner.replay.iter_games() if g.generation == generation]
        with open(path, "w") as stream:
            for game in games:
                stream.write(json.dumps(dict(generation=generation, index=game.index, moves=list(game.moves),
                                             winner=game.winner)) + "\n")
        self._artifacts(seed).append(dict(kind="games", generation=generation, attempt=None,
                                          path=str(path.relative_to(self.run_dir(seed))), sha256=sha256_file(path),
                                          bytes=path.stat().st_size))

    def _raw_value_metrics(self, attempt_dir, generation, rows):
        from .evaluation import raw_values
        from .network import load_inference_checkpoint
        inference = load_inference_checkpoint(attempt_dir / f"generation-{generation:04d}.inference.pt")
        return statistics.value_metrics(rows, raw_values(inference, rows))

    # Champion selection (development evidence only) ---------------------------------------

    def _inference_record(self, seed, generation):
        records = [r for r in self._artifacts(seed).records() if r["kind"] == "inference" and r["generation"] == generation]
        if len(records) != 1:
            raise RuntimeError(f"Expected exactly one inference artifact for generation {generation}")
        return records[0]

    def _initialize_champion(self, seed):
        path = self.run_dir(seed) / "champion.json"
        if not path.exists():
            replace_json(path, dict(generation=0, inference=self._inference_record(seed, 0), tactics=None,
                                    decisions=[], incomplete_checks=[]))

    def _pending_checks(self, seed, completed):
        champion = json.loads((self.run_dir(seed) / "champion.json").read_text())
        decided = {d["generation"] for d in champion["decisions"]}
        return [g for g in self.declaration["champion"]["schedule"] if g <= completed and g not in decided]

    def _champion_check(self, seed, generation, attempt_dir, inference_record, budget, check, events):
        from .evaluation import load_v2_agent, search_rows
        decl, champion_path = self.declaration["champion"], self.run_dir(seed) / "champion.json"
        champion = json.loads(champion_path.read_text())
        simulations = self.declaration["evaluation"]["simulations"]
        seeds = self.declaration["evaluation"]["tactical_seeds"]
        run = self.run_dir(seed)
        candidate = load_v2_agent(run / inference_record["path"], simulations, expected_sha256=inference_record["sha256"],
                                  name=f"generation-{generation}")
        incumbent = load_v2_agent(run / champion["inference"]["path"], simulations,
                                  expected_sha256=champion["inference"]["sha256"],
                                  name=f"champion-generation-{champion['generation']}")
        openings = package_rows(self.declaration, self.base, "openings-development")
        tactics = package_rows(self.declaration, self.base, "tactical-development")
        arena_rows = openings[:decl["arena_openings"]]
        records, status = run_paired_arena(candidate, incumbent, arena_rows, namespace=f"seed{seed}-champion-g{generation}",
                                           seed=seed, check=lambda: check("evaluation"),
                                           on_game=budget.count_evaluation_game, keep_decisions=False)
        if not status["complete"]:
            self._incomplete_check(seed, generation, attempt_dir, champion, status, records, events)
        arena = statistics.arena_summary(scored(records), planned_games=2 * len(arena_rows))
        baselines = {}
        by_id = {row["id"]: row for row in openings}
        for name in decl["baseline_opponents"]:
            opponent = opponent_by_name(name)
            rows = [by_id[i] for i in decl["baseline_rows"]]
            base_records, base_status = run_paired_arena(candidate, opponent, rows,
                                                         namespace=f"seed{seed}-baseline-{name}-g{generation}",
                                                         seed=seed, check=lambda: check("evaluation"),
                                                         on_game=budget.count_evaluation_game, keep_decisions=False)
            if not base_status["complete"]:
                self._incomplete_check(seed, generation, attempt_dir, champion, base_status, base_records, events)
            baselines[name] = dict(status=base_status, summary=statistics.arena_summary(
                scored(base_records), planned_games=2 * len(rows)))
        candidate_choices, candidate_visits = search_rows(candidate.inference, tactics, simulations=simulations,
                                                          seeds=seeds, namespace="development-tactics",
                                                          check=lambda: check("evaluation"))
        candidate_tactics = statistics.tactical_metrics(tactics, candidate_choices, candidate_visits)
        if champion["tactics"] is None:
            choices, visits = search_rows(incumbent.inference, tactics, simulations=simulations, seeds=seeds,
                                          namespace="development-tactics", check=lambda: check("evaluation"))
            champion["tactics"] = statistics.tactical_metrics(tactics, choices, visits)
        # Reaching this point means every arena completed and no correctness error was raised.
        decision = statistics.champion_decision(arena, candidate_tactics, champion["tactics"], True)
        evaluation = dict(generation=generation, candidate=candidate.describe(), champion=incumbent.describe(),
                          arena_status=status, arena=arena, baselines=baselines, candidate_tactics=candidate_tactics,
                          champion_tactics=champion["tactics"], decision=decision, records=records)
        write_json_once(attempt_dir / f"champion-check-{generation:04d}.json", evaluation)
        champion["decisions"].append(dict(generation=generation, decision=decision, previous=champion["generation"]))
        if decision["promote"]:
            champion.update(generation=generation, inference=inference_record, tactics=candidate_tactics)
        replace_json(champion_path, champion)
        events.append(dict(event="champion_check", generation=generation, promote=decision["promote"]))

    def _incomplete_check(self, seed, generation, attempt_dir, champion, status, records, events):
        """Record incomplete development evidence (never used for a decision) and stop; rerun on resume."""
        write_json_once(attempt_dir / f"champion-check-{generation:04d}.incomplete.json",
                        dict(generation=generation, status=status, records=records))
        champion["incomplete_checks"].append(dict(generation=generation, attempt=attempt_dir.name, status=status))
        replace_json(self.run_dir(seed) / "champion.json", champion)
        events.append(dict(event="champion_check_incomplete", generation=generation, status=status))
        raise CampaignStop(f"development evaluation incomplete: {status['stop_reason']}")

    def _select_champion(self, seed):
        champion = json.loads((self.run_dir(seed) / "champion.json").read_text())
        write_json_once(self.run_dir(seed) / "selection.json", dict(
            seed=seed, selected_generation=champion["generation"], inference=champion["inference"],
            decisions=champion["decisions"], rule="development evidence only; sealed packages untouched",
            declaration_sha256=self.declaration_sha256))

    # Sealed final evaluation (one time) ---------------------------------------------------

    def final_evaluate(self, *, extra_check=None):
        selections = {}
        for seed in self.declaration["seeds"]:
            path = self.run_dir(seed) / "selection.json"
            if not path.exists():
                raise RuntimeError(f"Seed {seed} has no development champion selection")
            selections[seed] = json.loads(path.read_text())
        final = self.directory / "final"
        final.mkdir(exist_ok=True)
        runtime = self.require_runtime()
        write_json_once(final / "started.json", dict(selections=selections, runtime=runtime,
                                                     declaration_sha256=self.declaration_sha256))
        number = 1
        budget = self._budget = Budget(self.ledger, self.declaration, "final", number, clock=self.clock)
        self.ledger.append(dict(event="attempt_start", seed="final", attempt=number, runtime=runtime))
        status = "running"
        try:
            for seed, selection in selections.items():
                def check(phase=None):
                    if extra_check is not None:
                        extra_check("evaluation", None)
                    budget.check("evaluation")
                result = self._final_seed(seed, selection, budget, check)
                write_json_once(final / f"seed-{seed}.json", result)
            status = "completed"
        except CampaignStop as stop:
            status = f"stopped: {stop}"
        finally:
            self.ledger.append(dict(event="attempt_end", status=status, **budget.counters()))
        return status

    def _final_seed(self, seed, selection, budget, check):
        from .evaluation import (PHASE4D2F_CHECKPOINT, V1SearchArenaAgent, V2NNOnlyAgent, V2SearchArenaAgent,
                                 calibration_games, load_v2_agent, nn_only_rows, raw_values, search_rows,
                                 untrained_model)
        final, evaluation = self.declaration["final"], self.declaration["evaluation"]
        simulations, seeds = evaluation["simulations"], evaluation["tactical_seeds"]
        run = self.run_dir(seed)
        agent = load_v2_agent(run / selection["inference"]["path"], simulations,
                              expected_sha256=selection["inference"]["sha256"], name=f"seed{seed}-champion")
        tactics = package_rows(self.declaration, self.base, "tactical-sealed")
        solved = package_rows(self.declaration, self.base, "solved-sealed")
        openings = package_rows(self.declaration, self.base, "openings-sealed")[:final["ladder_openings"]]
        overlap = training_overlap(run, [r["moves"] for r in tactics + solved])
        result = dict(seed=seed, selection=selection, agent=agent.describe(), overlap_families=sorted(overlap))
        choices, visits = search_rows(agent.inference, tactics, simulations=simulations, seeds=seeds,
                                      namespace="sealed-tactics", check=check)
        result["tactical"] = with_overlap_sensitivity(
            tactics, overlap, lambda rows: statistics.tactical_metrics(rows, choices, visits))
        result["tactical_nn_only"] = statistics.tactical_metrics(tactics, nn_only_rows(agent.inference, tactics))
        solved_choices, _ = search_rows(agent.inference, solved, simulations=simulations, seeds=seeds,
                                        namespace="sealed-solved", check=check)
        values = raw_values(agent.inference, solved)
        result["solved"] = with_overlap_sensitivity(
            solved, overlap, lambda rows: statistics.solved_decision_metrics(rows, solved_choices))
        result["value"] = with_overlap_sensitivity(solved, overlap, lambda rows: statistics.value_metrics(rows, values))
        opponents = {name: opponent_by_name(name) for name in final["ladder"] if name in OPPONENT_NAMES}
        if "initial_v2_512" in final["ladder"]:
            opponents["initial_v2_512"] = V2SearchArenaAgent(untrained_model(seed), simulations, name="initial_v2_512")
        if "phase4d2f_512" in final["ladder"]:
            checkpoint = self.declaration["phase4d2f_checkpoint"]
            opponents["phase4d2f_512"] = V1SearchArenaAgent(REPO_ROOT / checkpoint["path"], checkpoint["sha256"],
                                                            simulations)
        ladder = {}
        for name in final["ladder"]:
            records, status = run_paired_arena(agent, opponents[name], openings, namespace=f"seed{seed}-final-{name}",
                                               seed=seed, check=check, on_game=budget.count_evaluation_game,
                                               keep_decisions=False)
            summary = statistics.arena_summary(scored(records), planned_games=2 * len(openings)) \
                if scored(records) else None
            ladder[name] = dict(status=status, summary=summary, opponent=opponents[name].describe(), records=records,
                                gate=statistics.arena_gate(summary, self.declaration["thresholds"]["arena"][name])
                                if summary and name in self.declaration["thresholds"]["arena"] else None)
            if not status["complete"]:
                raise CampaignStop(f"final ladder incomplete: {name}")
        result["ladder"] = ladder
        nn_only = V2NNOnlyAgent(agent.inference, name=f"seed{seed}-champion-nn-only")
        records, status = run_paired_arena(nn_only, RandomOpponent(), openings, namespace=f"seed{seed}-final-nn-only",
                                           seed=seed, check=check, on_game=budget.count_evaluation_game,
                                           keep_decisions=False)
        summary = statistics.arena_summary(scored(records), planned_games=2 * len(openings)) if scored(records) else None
        result["nn_only_vs_random"] = dict(status=status, summary=summary, records=records)
        calibration = []
        for index in range(final["calibration_games"]):
            check()
            calibration.extend(dict(r, game=index) for r in calibration_games(
                agent.inference, self.config_for(seed), games=1, seed=seed * 1_000_003 + index))
            budget.count_evaluation_game(None)
        result["calibration"] = statistics.calibration_summary(calibration, constant=final.get("calibration_constant"))
        return result


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
    for path in sorted((Path(run_dir) / "games").glob("generation-*.jsonl")):
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


# CLI ----------------------------------------------------------------------------------------

def preflight(declaration_path):
    declaration, digest = load_declaration(declaration_path)
    return dict(declaration_sha256=digest, planned_evaluation_games=planned_evaluation_games(declaration),
                runtime=runtime_identity(), execution_sha256=execution_source_identity()["sha256"],
                git=git_provenance(), required_environment=required_thread_environment(
                    declaration["runtime"]["threads"]))


def main(argv=None):
    parser = argparse.ArgumentParser(description="AlphaZero v2 bounded campaign (requires authorization)")
    sub = parser.add_subparsers(dest="command", required=True)
    pre = sub.add_parser("preflight")
    pre.add_argument("--declaration", required=True)
    for name in ("run", "final-evaluate"):
        command = sub.add_parser(name)
        command.add_argument("--declaration", required=True)
        command.add_argument("--campaign-dir", required=True)
        command.add_argument("--authorize", required=True)
        if name == "run":
            command.add_argument("--seed", type=int, required=True)
    args = parser.parse_args(argv)
    if args.command == "preflight":
        print(json.dumps(preflight(args.declaration), indent=1, sort_keys=True))
        return 0
    campaign = Campaign(args.campaign_dir, args.declaration, args.authorize)

    def stop(signum, frame):
        if getattr(campaign, "_budget", None) is not None:
            campaign._budget.request_stop(f"signal {signum}")
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    status = campaign.run_seed(args.seed) if args.command == "run" else campaign.final_evaluate()
    print(status)
    return 0 if status == "completed" else 3


if __name__ == "__main__":
    sys.exit(main())
