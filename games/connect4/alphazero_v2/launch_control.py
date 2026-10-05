"""Durable launch control for the AlphaZero v2 campaign (torch-free, Phase 4D.3B.1).

The campaign journal is the single authority for what has happened:

* Every record is one line ``{"seq", "record", "sha256"}``. A record is written,
  flushed and fsynced *before* the action it authorizes and *after* the result
  it certifies. A trailing record torn by a crash fails its checksum. It is
  discarded on recovery and kept in a side file. Any other damage is refused.
* ``CampaignState`` folds the journal into seed, generation, champion and final
  progress, unit evidence and cumulative consumption. Every transition is
  validated before it is written, so an impossible sequence (a duplicate
  commit, decision, selection or unit completion, a skipped generation)
  cannot be recorded or replayed.
* ``Budget`` enforces the declared limits. Active time is charged through
  durable *leases*: before work continues past the current reservation, a new
  upper bound is journaled. A hard crash is charged up to its last lease,
  never less than the time it used. An evaluation game is charged when its
  ``unit_begin`` record is durable, before its first move.
* ``CampaignLock`` (``flock``) admits one writer per campaign directory.

Unit lifecycle: *attempted* = ``unit_begin`` (charged here if the unit is a
game); *completed* = ``unit_complete`` carrying its evidence (counted as
evidence); otherwise ``unit_abandoned``, cooperatively or on recovery (still
charged, never evidence, rerun on resume with a new charge). Completed units
are never rerun.
"""
from collections import Counter
from datetime import datetime, timezone
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path

from .arena import StopEvaluation

LAUNCH_CONTROL_VERSION = "connect4-alphazero-v2-launch-control-v1"
JOURNAL_FORMAT = "connect4-alphazero-v2-campaign-journal-v1"
LEASE_SECONDS = 300.0
RENEW_BELOW_SECONDS = 150.0
# Declaration tokens that must never authorize a campaign (Phase 4D.3B launch-readiness review).
REJECTED_DECLARATION_TOKENS = ("2741314399741c1b20510b8a2beb42e0938b0aeafc9dd0c487201e57208867d8",)
GAME_KINDS = ("development_arena_game", "development_baseline_game", "final_ladder_game", "final_nn_only_game",
              "calibration_game")
POSITION_KINDS = ("development_tactics_row", "final_tactical_row", "final_solved_row")
UNIT_KINDS = GAME_KINDS + POSITION_KINDS


class CampaignStop(StopEvaluation):
    """Cooperative stop (stop requested or limit reached): work in flight is abandoned, never scored."""


class BudgetExhausted(CampaignStop):
    """A declared limit is exhausted: the campaign can no longer complete its required evidence."""


class CampaignLocked(RuntimeError):
    pass


class JournalCorrupt(RuntimeError):
    pass


class InconsistentCampaign(RuntimeError):
    """Published outputs disagree with the journal; recovery refuses to guess."""


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def record_digest(seq, record):
    return hashlib.sha256(canonical_json(dict(seq=seq, record=record)).encode()).hexdigest()


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_once(path, data):
    """Publish bytes to a new path only (temporary file, fsync, hard link, directory fsync)."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(temporary, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        os.unlink(temporary)
    fsync_directory(path.parent)
    return sha256_bytes(data)


def view_bytes(value):
    return (json.dumps(value, indent=1, sort_keys=True, allow_nan=False) + "\n").encode()


def publish_view(path, value):
    """Write a journal-derived view once; an existing view must be byte-identical."""
    path, data = Path(path), view_bytes(value)
    if path.exists():
        if path.read_bytes() != data:
            raise InconsistentCampaign(f"{path} exists with content that disagrees with the journal")
        return sha256_bytes(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    return write_once(path, data)


def replace_view(path, value):
    """Atomically replace a non-authoritative summary (status.json only)."""
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(temporary, "wb") as stream:
        stream.write(view_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


# Single writer -----------------------------------------------------------------------------

class CampaignLock:
    """Exclusive ``flock`` on ``campaign.lock``; released by ``release`` or process death."""

    def __init__(self, directory):
        self.path = Path(directory) / "campaign.lock"
        descriptor = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            os.close(descriptor)
            if error.errno in (errno.EWOULDBLOCK, errno.EAGAIN, errno.EACCES):
                raise CampaignLocked("Another invocation holds this campaign directory; "
                                     "concurrent invocations are refused") from error
            raise
        self.descriptor = descriptor
        os.ftruncate(descriptor, 0)
        os.write(descriptor, f"{os.getpid()}\n".encode())

    def release(self):
        if self.descriptor is not None:
            fcntl.flock(self.descriptor, fcntl.LOCK_UN)
            os.close(self.descriptor)
            self.descriptor = None


# Journal ------------------------------------------------------------------------------------

class Journal:
    """Checksummed, fsynced, append-only journal that feeds a ``CampaignState``."""

    def __init__(self, path, state):
        self.path, self.state = Path(path), state
        self.records, self.torn_tail, self.broken = [], None, False
        data = self.path.read_bytes() if self.path.exists() else b""
        offset, valid_end, damaged_at = 0, 0, None
        while offset < len(data):
            newline = data.find(b"\n", offset)
            end = len(data) if newline < 0 else newline + 1
            parsed = self._parse(data[offset:end]) if newline >= 0 else None  # unterminated = never fsynced
            if parsed is None:
                damaged_at = offset if damaged_at is None else damaged_at
            else:
                if damaged_at is not None:
                    raise JournalCorrupt(f"Damaged journal bytes at {damaged_at} precede a valid record")
                seq, record = parsed
                if seq != len(self.records):
                    raise JournalCorrupt(f"Journal sequence {seq} found where {len(self.records)} was expected")
                state.apply(record)
                self.records.append(record)
                valid_end = end
            offset = end
        self.valid_end = valid_end
        if valid_end < len(data):
            self.torn_tail = data[valid_end:]

    @staticmethod
    def _parse(line):
        try:
            entry = json.loads(line)
            if set(entry) != {"seq", "record", "sha256"} or record_digest(entry["seq"], entry["record"]) != entry["sha256"]:
                return None
            return entry["seq"], entry["record"]
        except (ValueError, TypeError, UnicodeDecodeError):
            return None

    def repair(self):
        """Discard a torn trailing record (lock holder only); its bytes are preserved beside the journal."""
        if self.torn_tail is None:
            return None
        saved = self.path.with_name(f"{self.path.name}.torn-{self.valid_end}")
        write_once(saved, self.torn_tail)
        with open(self.path, "r+b") as stream:
            stream.truncate(self.valid_end)
            stream.flush()
            os.fsync(stream.fileno())
        tail, self.torn_tail = self.torn_tail, None
        return self.append(dict(event="torn_tail_discarded", bytes=len(tail), sha256=sha256_bytes(tail),
                                saved_as=saved.name))

    def append(self, record):
        if self.broken:
            raise RuntimeError("Journal write failed earlier in this process; refusing further records")
        if self.torn_tail is not None:
            raise RuntimeError("Repair the torn journal tail before appending")
        record = json.loads(canonical_json(dict(record, utc=utc_now())))
        self.state.apply(record)  # validates the transition before it becomes durable
        seq = len(self.records)
        line = canonical_json(dict(seq=seq, record=record, sha256=record_digest(seq, record))) + "\n"
        created = not self.path.exists()
        try:
            with open(self.path, "ab") as stream:
                stream.write(line.encode())
                stream.flush()
                os.fsync(stream.fileno())
            if created:
                fsync_directory(self.path.parent)
        except BaseException:
            self.broken = self.state.broken = True
            raise
        self.records.append(record)
        return record


# State fold ---------------------------------------------------------------------------------

class CampaignState:
    """Deterministic fold of the journal; ``apply`` refuses invalid transitions."""

    def __init__(self, generations=None, schedule=()):
        self.generations, self.schedule = generations, tuple(schedule)
        self.declaration_sha256 = None
        self.attempts, self.units, self.seeds = {}, {}, {}
        self.final = dict(started=None, seeds={})
        self.outcome = None
        self.games_charged = 0
        self.events = Counter()
        self.broken = False

    def seed(self, seed):
        return self.seeds.setdefault(str(seed), dict(
            committed={}, open_generation=None, begun=0, discarded=[], diagnostics={}, decisions={}, selected=None,
            pruned={}, selfplay_games=0, training_plies=0, optimizer_steps=0))

    def apply(self, record):
        handler = getattr(self, "_on_" + record.get("event", ""), None)
        if handler is None:
            raise JournalCorrupt(f"Unknown journal event {record.get('event')!r}")
        handler(record)
        self.events[record["event"]] += 1

    @staticmethod
    def _require(condition, message):
        if not condition:
            raise JournalCorrupt(message)

    # Campaign and attempts -----------------------------------------------------------------

    def _on_campaign_created(self, record):
        self._require(self.declaration_sha256 is None, "campaign_created twice")
        self.declaration_sha256 = record["declaration_sha256"]

    def _on_torn_tail_discarded(self, record):
        pass

    def _open_attempt(self, number):
        info = self.attempts.get(number)
        self._require(info is not None and info["end"] is None and info["recovered"] is None,
                      f"Attempt {number} is not open")
        return info

    def _on_attempt_start(self, record):
        self._require(self.declaration_sha256 is not None, "attempt before campaign_created")
        self._require(record["attempt"] == len(self.attempts) + 1, "attempt numbers must be consecutive")
        self._require(not self.open_attempts(), "an attempt started while another is open")
        self.attempts[record["attempt"]] = dict(scope=record["scope"], start=record, lease=None, end=None,
                                                recovered=None)

    def _on_attempt_source(self, record):
        self._open_attempt(record["attempt"])["source"] = record

    def _on_lease(self, record):
        self._open_attempt(record["attempt"])["lease"] = record

    def _on_attempt_end(self, record):
        self._require(not self._open_units(record["attempt"]), "attempt ended with open units")
        self._open_attempt(record["attempt"])["end"] = record

    def _on_attempt_recovered(self, record):
        self._require(not self._open_units(record["attempt"]), "attempt recovered with open units")
        self._open_attempt(record["attempt"])["recovered"] = record

    def open_attempts(self):
        return [n for n, info in self.attempts.items() if info["end"] is None and info["recovered"] is None]

    def attempt_charge(self, number):
        info = self.attempts[number]
        settled = info["end"] or info["recovered"] or info["lease"]
        return (0.0, 0.0) if settled is None else (settled["total_seconds"], settled["training_seconds"])

    def charged_seconds(self):
        return sum(self.attempt_charge(n)[0] for n in self.attempts)

    def charged_training_seconds(self, scope):
        return sum(self.attempt_charge(n)[1] for n, info in self.attempts.items() if info["scope"] == str(scope))

    # Units ----------------------------------------------------------------------------------

    def _open_units(self, attempt):
        return [unit for unit, info in self.units.items() if info["open"] == attempt]

    def _on_unit_begin(self, record):
        self._open_attempt(record["attempt"])
        self._require(record["kind"] in UNIT_KINDS, f"unknown unit kind {record['kind']}")
        info = self.units.setdefault(record["unit"], dict(kind=record["kind"], seed=record["seed"], begins=[],
                                                          abandoned=[], completed=None, open=None))
        self._require(info["kind"] == record["kind"], f"unit {record['unit']} changed kind")
        self._require(info["completed"] is None, f"unit {record['unit']} is already complete")
        self._require(info["open"] is None, f"unit {record['unit']} is already open")
        info["begins"].append(record["attempt"])
        info["open"] = record["attempt"]
        if record["kind"] in GAME_KINDS:
            self.games_charged += 1

    def _close_unit(self, record):
        info = self.units.get(record["unit"])
        self._require(info is not None and info["open"] == record["attempt"],
                      f"unit {record['unit']} is not open in attempt {record['attempt']}")
        info["open"] = None
        return info

    def _on_unit_complete(self, record):
        self._close_unit(record)["completed"] = record

    def _on_unit_abandoned(self, record):
        self._close_unit(record)["abandoned"].append(record)

    def evidence(self, unit):
        info = self.units.get(unit)
        return None if info is None or info["completed"] is None else info["completed"]["evidence"]

    # Training -------------------------------------------------------------------------------

    def _on_selfplay_game(self, record):
        self._open_attempt(record["attempt"])
        seed = self.seed(record["seed"])
        seed["selfplay_games"] += 1
        seed["training_plies"] += record["plies"]

    def _on_optimizer_step(self, record):
        self._open_attempt(record["attempt"])
        self.seed(record["seed"])["optimizer_steps"] += 1

    def latest_generation(self, seed):
        committed = self.seed(seed)["committed"]
        return max(committed) if committed else None

    def _on_generation_begin(self, record):
        self._open_attempt(record["attempt"])
        seed = self.seed(record["seed"])
        latest = self.latest_generation(record["seed"])
        self._require(latest is not None and record["generation"] == latest + 1,
                      f"generation {record['generation']} begun out of order")
        self._require(seed["open_generation"] is None, "a generation is already open")
        self._require(self.generations is None or record["generation"] <= self.generations,
                      "generation beyond the declared count")
        seed["open_generation"] = (record["attempt"], record["generation"])
        seed["begun"] += 1

    def _on_generation_discarded(self, record):
        seed = self.seed(record["seed"])
        self._require(seed["open_generation"] == (record["attempt"], record["generation"]),
                      "discarded generation is not open")
        seed["open_generation"] = None
        seed["discarded"].append(record)

    def _on_generation_committed(self, record):
        self._open_attempt(record["attempt"])
        seed, generation = self.seed(record["seed"]), record["generation"]
        latest = self.latest_generation(record["seed"])
        self._require(generation == (0 if latest is None else latest + 1),
                      f"generation {generation} committed out of order")
        if generation:
            self._require(seed["open_generation"] == (record["attempt"], generation), "committed generation not open")
        seed["open_generation"] = None
        seed["committed"][generation] = record

    def _on_generation_diagnostics(self, record):
        seed, generation = self.seed(record["seed"]), record["generation"]
        self._require(generation in seed["committed"] and generation not in seed["diagnostics"],
                      f"diagnostics for generation {generation} out of order")
        seed["diagnostics"][generation] = record

    def _on_artifact_pruned(self, record):
        seed = self.seed(record["seed"])
        self._require(record["path"] not in seed["pruned"], "artifact pruned twice")
        seed["pruned"][record["path"]] = record

    # Champion and selection ------------------------------------------------------------------

    def _on_champion_decision(self, record):
        seed, generation = self.seed(record["seed"]), record["generation"]
        self._require(generation in seed["committed"], "decision for an uncommitted generation")
        self._require(generation not in seed["decisions"], f"duplicate champion decision for generation {generation}")
        self._require(not self.schedule or generation in self.schedule, "decision outside the champion schedule")
        self._require(record["previous"] == self.champion_generation(record["seed"]), "decision against a stale champion")
        seed["decisions"][generation] = record

    def champion_generation(self, seed):
        champion = 0
        for generation, decision in sorted(self.seed(seed)["decisions"].items()):
            if decision["promote"]:
                champion = generation
        return champion

    def _on_seed_selected(self, record):
        seed = self.seed(record["seed"])
        self._require(seed["selected"] is None, "seed selected twice")
        self._require(record["generation"] == self.champion_generation(record["seed"]),
                      "selection differs from the development champion")
        seed["selected"] = record

    # Final and outcome ------------------------------------------------------------------------

    def _on_final_started(self, record):
        self._require(self.final["started"] is None, "final evaluation started twice")
        self.final["started"] = record

    def _on_final_seed_complete(self, record):
        self._require(self.final["started"] is not None, "final seed before final_started")
        self._require(str(record["seed"]) not in self.final["seeds"], "final seed completed twice")
        self.final["seeds"][str(record["seed"])] = record

    def _on_campaign_incomplete(self, record):
        self._require(self.outcome is None, "campaign outcome recorded twice")
        self.outcome = record

    def _on_campaign_complete(self, record):
        self._require(self.outcome is None, "campaign outcome recorded twice")
        self.outcome = record

    # Accounting -------------------------------------------------------------------------------

    def consumption(self):
        """Monotonic, durable consumption: attempted versus accepted work."""
        units = {}
        for kind in UNIT_KINDS:
            selected = [u for u in self.units.values() if u["kind"] == kind]
            units[kind] = dict(attempted=sum(len(u["begins"]) for u in selected),
                               completed=sum(u["completed"] is not None for u in selected),
                               abandoned=sum(len(u["abandoned"]) for u in selected),
                               restarted_units=sum(len(u["begins"]) > 1 for u in selected))
        seeds = {}
        for name, seed in sorted(self.seeds.items()):
            accepted = [r["summary"] for g, r in seed["committed"].items() if g and r["summary"]]
            seeds[name] = dict(
                generations_begun=seed["begun"], generations_committed=len([g for g in seed["committed"] if g]),
                generations_discarded=len(seed["discarded"]), selfplay_games_completed=seed["selfplay_games"],
                training_plies_completed=seed["training_plies"], optimizer_steps_attempted=seed["optimizer_steps"],
                accepted_games=sum(s["games"] for s in accepted), accepted_plies=sum(s["new_positions"] for s in accepted),
                accepted_optimizer_steps=sum(s["updates"] for s in accepted),
                charged_training_seconds=self.charged_training_seconds(name))
        return dict(charged_seconds=self.charged_seconds(), evaluation_games_charged=self.games_charged,
                    evaluation_games_completed=sum(units[k]["completed"] for k in GAME_KINDS),
                    units=units, seeds=seeds,
                    attempts=dict(total=len(self.attempts),
                                  ended=sum(i["end"] is not None for i in self.attempts.values()),
                                  recovered_after_hard_interruption=sum(i["recovered"] is not None
                                                                        for i in self.attempts.values()),
                                  open=len(self.open_attempts())))


# Budgets ------------------------------------------------------------------------------------

class Budget:
    """Cumulative limits for one attempt, charged through durable leases and game reservations.

    ``check(phase)`` precedes every search, move and optimizer update: it refuses
    to *start* work once a limit is reached (``>=``). ``require_game_slot`` refuses
    to start a game once charged games reach the ceiling; finishing exactly at the
    ceiling is allowed and non-game work is unaffected. ``completion_violation``
    is the recheck before a phase is published: a primitive may overrun a
    cooperative deadline, but a phase is never published past a limit (``>``).
    """

    def __init__(self, journal, limits, *, scope, attempt, clock):
        self.journal, self.state, self.limits = journal, journal.state, limits
        self.scope, self.attempt, self.clock = str(scope), attempt, clock
        self.prior_total = self.state.charged_seconds()
        self.prior_training = self.state.charged_training_seconds(self.scope)
        self.started = clock()
        self.training_accumulated, self.training_started = 0.0, None
        self.reserved_total = self.reserved_training = 0.0
        self.stop_reason, self.last_stop = None, None

    def request_stop(self, reason="stop requested"):
        self.stop_reason = reason

    def attempt_seconds(self):
        return self.clock() - self.started

    def attempt_training_seconds(self):
        running = self.clock() - self.training_started if self.training_started is not None else 0.0
        return self.training_accumulated + running

    def total_seconds(self):
        return self.prior_total + self.attempt_seconds()

    def training_seconds(self):
        return self.prior_training + self.attempt_training_seconds()

    def _write_lease(self):
        self.journal.append(dict(event="lease", attempt=self.attempt, scope=self.scope,
                                 total_seconds=self.reserved_total, training_seconds=self.reserved_training))

    def lease(self, *, force=False):
        """Extend the durable reservation before the current one can be outrun."""
        elapsed, training = self.attempt_seconds(), self.attempt_training_seconds()
        renew_total = force or self.reserved_total - elapsed < RENEW_BELOW_SECONDS
        renew_training = self.training_started is not None and (
            force or self.reserved_training - training < RENEW_BELOW_SECONDS)
        if renew_total or renew_training:
            if renew_total:
                self.reserved_total = elapsed + LEASE_SECONDS
            if renew_training:
                self.reserved_training = training + LEASE_SECONDS
            self._write_lease()

    def begin_training(self):
        self.training_started = self.clock()
        self.lease(force=True)

    def end_training(self):
        if self.training_started is None:
            return
        self.training_accumulated += self.clock() - self.training_started
        self.training_started = None
        self.reserved_training = self.training_accumulated  # settled: the exact training time used
        self._write_lease()

    def _stop(self, stop):
        self.last_stop = stop
        raise stop

    def check(self, phase):
        if self.stop_reason is not None:
            self._stop(CampaignStop(self.stop_reason))
        if self.total_seconds() >= self.limits["campaign_seconds"]:
            self._stop(BudgetExhausted("campaign wall-clock budget exhausted"))
        if phase == "training" and self.training_seconds() >= self.limits["per_run_training_seconds"]:
            self._stop(BudgetExhausted("per-run collection+optimization budget exhausted"))
        self.lease()

    def require_game_slot(self):
        if self.state.games_charged >= self.limits["evaluation_games_ceiling"]:
            self._stop(BudgetExhausted("evaluation-game ceiling reached"))

    def completion_violation(self, *, training=False):
        if self.total_seconds() > self.limits["campaign_seconds"]:
            return "campaign wall-clock budget exceeded before the phase completed"
        if training and self.training_seconds() > self.limits["per_run_training_seconds"]:
            return "per-run collection+optimization budget exceeded before the generation completed"
        return None

    def settle(self, status):
        self.end_training()
        self.journal.append(dict(event="attempt_end", attempt=self.attempt, scope=self.scope, status=status,
                                 total_seconds=self.attempt_seconds(), training_seconds=self.attempt_training_seconds()))


# Recovery ------------------------------------------------------------------------------------

def close_open_work(journal, attempt, reason, seeds):
    """Abandon every unit and discard every generation still open in ``attempt`` (charges are kept)."""
    state = journal.state
    for unit, info in list(state.units.items()):
        if info["open"] == attempt:
            journal.append(dict(event="unit_abandoned", unit=unit, kind=info["kind"], seed=info["seed"],
                                attempt=attempt, reason=reason))
    for seed in seeds:
        open_generation = state.seed(seed)["open_generation"]
        if open_generation is not None and open_generation[0] == attempt:
            journal.append(dict(event="generation_discarded", seed=seed, generation=open_generation[1],
                                attempt=attempt, reason=reason))


def recover_open_attempts(journal, seeds):
    """Close attempts that ended without ``attempt_end`` (hard interruption) with conservative charges.

    Only the lock holder may call this: an open attempt then cannot still be running.
    """
    state, recovered = journal.state, []
    for number in state.open_attempts():
        reason = "hard interruption; recovered by a later invocation"
        close_open_work(journal, number, reason, seeds)
        total, training = state.attempt_charge(number)
        journal.append(dict(event="attempt_recovered", attempt=number, scope=state.attempts[number]["scope"],
                            total_seconds=total, training_seconds=training,
                            reason=reason + "; charged through its last durable lease"))
        recovered.append(number)
    return recovered


def read_state(directory, *, generations=None, schedule=()):
    """Read-only fold for status reporting (no lock, no repair; a torn tail is reported, not removed)."""
    state = CampaignState(generations, schedule)
    journal = Journal(Path(directory) / "journal.jsonl", state)
    return state, journal
