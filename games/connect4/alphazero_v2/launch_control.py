"""Fail-closed, non-resumable control for the official AlphaZero v2 campaign (torch-free, Phase 4D.3B.2).

The official campaign is one owning process holding one exclusive lock on its
campaign directory. It runs every seed, the development selection and the
sealed final evaluation sequentially under one declaration. There is no resume:

* ``state.json`` is the single authority. It is replaced atomically (temporary
  file, fsync, rename, directory fsync) on every transition, so a crash leaves
  either the previous or the next complete state, never a torn one.
* States advance along one fixed line: CREATED, RUNNING_SEED_<seed> for each
  declared seed in order, DEVELOPMENT_SELECTION_COMPLETE,
  RUNNING_FINAL_EVALUATION, COMPLETE. INCOMPLETE may follow any non-terminal
  state. COMPLETE and INCOMPLETE are terminal and never transition again.
* A non-terminal state with no live owner means the owner died. A later
  invocation marks it INCOMPLETE and refuses; it never repairs and continues.
* ``Budget`` measures monotonic seconds inside the owning process only. Limits
  are checked before every substantial unit and every search, move and
  optimizer update. Nothing is reconstructed across processes, so no leases.
* Counters are live provenance. If the process dies before writing them, the
  campaign is INCOMPLETE and the last written counters are lower bounds.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket

from .arena import StopEvaluation

LAUNCH_CONTROL_VERSION = "connect4-alphazero-v2-launch-control-v2-fail-closed"
STATE_FORMAT = "connect4-alphazero-v2-official-campaign-state-v1"
# Declaration tokens that must never authorize a campaign:
# Phase 4D.3B (launch-readiness review) and Phase 4D.3B.1 (final launch review, NO GO).
REJECTED_DECLARATION_TOKENS = ("2741314399741c1b20510b8a2beb42e0938b0aeafc9dd0c487201e57208867d8",
                               "8a8a52b100c7e32aa5e3e5b05469578084c8e12f37750ba81fee071990ff5e39")
CREATED, SELECTION_COMPLETE = "CREATED", "DEVELOPMENT_SELECTION_COMPLETE"
RUNNING_FINAL, COMPLETE, INCOMPLETE = "RUNNING_FINAL_EVALUATION", "COMPLETE", "INCOMPLETE"
TERMINAL_STATES = (COMPLETE, INCOMPLETE)
INTERRUPTED_MESSAGE = "Existing interrupted official campaign is INCOMPLETE and cannot be resumed."


def running_seed(seed):
    return f"RUNNING_SEED_{seed}"


def state_sequence(seeds):
    """The only order in which non-INCOMPLETE states may be entered."""
    return (CREATED, *(running_seed(s) for s in seeds), SELECTION_COMPLETE, RUNNING_FINAL, COMPLETE)


class CampaignStop(StopEvaluation):
    """Cooperative stop (stop requested or limit reached). Official campaigns then end INCOMPLETE."""


class BudgetExhausted(CampaignStop):
    """A declared limit was reached: the required evidence cannot be completed within it."""


class CampaignLocked(RuntimeError):
    """Another live process owns this campaign directory."""


class TerminalCampaign(RuntimeError):
    """The campaign already reached COMPLETE or INCOMPLETE; nothing may run or transition again."""


class InterruptedCampaign(TerminalCampaign):
    """The owner ended without a terminal state; the campaign is now INCOMPLETE and cannot resume."""


class InvalidTransition(RuntimeError):
    pass


class StateCorrupt(RuntimeError):
    """``state.json`` cannot be read or validated; the campaign is not COMPLETE and cannot continue."""


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def full_fsync(descriptor):
    """fsync, plus F_FULLFSYNC where available (macOS: also flush the drive's write cache)."""
    os.fsync(descriptor)
    if hasattr(fcntl, "F_FULLFSYNC"):
        try:
            fcntl.fcntl(descriptor, fcntl.F_FULLFSYNC)
        except OSError:
            pass  # not supported by this filesystem; the plain fsync above still happened


def fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        full_fsync(descriptor)
    finally:
        os.close(descriptor)


def view_bytes(value):
    return (json.dumps(value, indent=1, sort_keys=True, allow_nan=False) + "\n").encode()


def write_once(path, data):
    """Publish bytes to a new path only (temporary file, fsync, hard link, directory fsync)."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(temporary, "wb") as stream:
        stream.write(data)
        stream.flush()
        full_fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        os.unlink(temporary)
    fsync_directory(path.parent)
    return sha256_bytes(data)


def publish_view(path, value):
    """Write an immutable JSON artifact once; returns its SHA-256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return write_once(path, view_bytes(value))


def atomic_replace(path, data):
    """Replace a file atomically: a reader sees the old or the new bytes, never a mixture."""
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(temporary, "wb") as stream:
        stream.write(data)
        stream.flush()
        full_fsync(stream.fileno())
    os.replace(temporary, path)
    fsync_directory(path.parent)


# Single owner -------------------------------------------------------------------------------

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
                raise CampaignLocked("Another live process owns this official campaign directory; "
                                     "a second process may never advance it") from error
            raise
        self.descriptor = descriptor
        os.ftruncate(descriptor, 0)
        os.write(descriptor, f"{os.getpid()}\n".encode())

    def release(self):
        if self.descriptor is not None:
            fcntl.flock(self.descriptor, fcntl.LOCK_UN)
            os.close(self.descriptor)
            self.descriptor = None


# Authoritative state ------------------------------------------------------------------------

class CampaignStateFile:
    """``state.json``: current state, transition history and the last counter snapshot."""

    NAME = "state.json"

    def __init__(self, path, document):
        self.path, self.document = Path(path), document

    @classmethod
    def create(cls, directory, *, declaration_sha256, seeds):
        document = dict(format=STATE_FORMAT, launch_control=LAUNCH_CONTROL_VERSION,
                        declaration_sha256=declaration_sha256, seeds=list(seeds), state=CREATED,
                        owner=dict(pid=os.getpid(), host=socket.gethostname(), started_utc=utc_now()),
                        history=[dict(state=CREATED, utc=utc_now())], counters=None, outcome=None)
        path = Path(directory) / cls.NAME
        write_once(path, view_bytes(document))
        return cls(path, document)

    @classmethod
    def load(cls, directory):
        path = Path(directory) / cls.NAME
        try:
            document = json.loads(path.read_bytes())
        except (OSError, ValueError) as error:
            raise StateCorrupt(f"{path} is missing or unreadable: {error}") from error
        cls.validate(document)
        return cls(path, document)

    @staticmethod
    def validate(document):
        required = {"format", "launch_control", "declaration_sha256", "seeds", "state", "owner", "history",
                    "counters", "outcome"}
        if not isinstance(document, dict) or set(document) != required or document["format"] != STATE_FORMAT:
            raise StateCorrupt("state.json is not an official campaign state document")
        sequence = state_sequence(document["seeds"])
        visited = [entry.get("state") for entry in document["history"]]
        if not visited or visited[-1] != document["state"]:
            raise StateCorrupt("state.json history does not end at its current state")
        ordinary = visited[:-1] if visited[-1] == INCOMPLETE else visited
        if INCOMPLETE in ordinary or tuple(ordinary) != sequence[:len(ordinary)]:
            raise StateCorrupt(f"state.json history is not a valid transition sequence: {visited}")
        if (document["outcome"] is not None) != (document["state"] in TERMINAL_STATES):
            raise StateCorrupt("state.json outcome disagrees with its state")

    @property
    def state(self):
        return self.document["state"]

    @property
    def terminal(self):
        return self.state in TERMINAL_STATES

    def _write(self, document):
        self.validate(document)
        atomic_replace(self.path, view_bytes(document))
        self.document = document

    def transition(self, new_state, *, counters=None, outcome=None, **info):
        """Durably enter ``new_state``; only the next state in sequence, or INCOMPLETE, is accepted."""
        if self.terminal:
            raise InvalidTransition(f"Campaign is {self.state}; terminal states never transition again")
        sequence = state_sequence(self.document["seeds"])
        expected = sequence[sequence.index(self.state) + 1]
        if new_state not in (expected, INCOMPLETE):
            raise InvalidTransition(f"{self.state} -> {new_state} is not allowed (next is {expected} or "
                                    f"{INCOMPLETE})")
        if (new_state in TERMINAL_STATES) != (outcome is not None):
            raise InvalidTransition("A terminal state needs an outcome; a running state has none")
        document = json.loads(json.dumps(self.document))
        document["state"] = new_state
        document["history"].append(dict(info, state=new_state, utc=utc_now()))
        if counters is not None:
            document["counters"] = counters
        document["outcome"] = outcome
        self._write(document)

    def record_counters(self, counters):
        """Snapshot live counters (lower bounds if the process later dies)."""
        if self.terminal:
            raise InvalidTransition(f"Campaign is {self.state}; terminal states are never rewritten")
        self._write(dict(json.loads(json.dumps(self.document)), counters=counters))


# Budgets ------------------------------------------------------------------------------------

class Budget:
    """Declared limits measured with the owning process's monotonic clock.

    ``check(phase, seed)`` precedes every unit, search, move and optimizer update
    and refuses to *start* work once a limit is reached (``>=``).
    ``completion_violation`` is the recheck before anything is accepted or
    published (``>``): an in-flight primitive may overrun a cooperative check,
    but its result is then never accepted and the campaign ends INCOMPLETE.
    """

    def __init__(self, limits, *, clock):
        self.limits, self.clock = limits, clock
        self.started = clock()
        self.training_accumulated, self.training_seed, self.training_started = {}, None, None
        self.phase_seconds = {}
        self.games_started = 0
        self.stop_reason = None

    def request_stop(self, reason="stop requested"):
        self.stop_reason = reason

    def elapsed(self):
        return self.clock() - self.started

    def training_seconds(self, seed):
        seconds = self.training_accumulated.get(str(seed), 0.0)
        if self.training_seed == str(seed):
            seconds += self.clock() - self.training_started
        return seconds

    def begin_training(self, seed):
        if self.training_seed is not None:
            raise RuntimeError("Training time is already being measured")
        self.training_seed, self.training_started = str(seed), self.clock()

    def end_training(self):
        if self.training_seed is None:
            return
        seed = self.training_seed
        self.training_accumulated[seed] = self.training_seconds(seed)
        self.training_seed = self.training_started = None

    @contextmanager
    def phase(self, name):
        started = self.clock()
        try:
            yield
        finally:
            self.phase_seconds[name] = self.phase_seconds.get(name, 0.0) + self.clock() - started

    def check(self, phase, seed=None):
        if self.stop_reason is not None:
            raise CampaignStop(self.stop_reason)
        if self.elapsed() >= self.limits["campaign_seconds"]:
            raise BudgetExhausted("campaign wall-clock budget exhausted")
        if phase == "training" and self.training_seconds(seed) >= self.limits["per_run_training_seconds"]:
            raise BudgetExhausted(f"seed {seed} collection+optimization budget exhausted")

    def start_game(self):
        """Reserve one evaluation game before its first move; refuse at the ceiling."""
        if self.games_started >= self.limits["evaluation_games_ceiling"]:
            raise BudgetExhausted("evaluation-game ceiling reached")
        self.games_started += 1

    def completion_violation(self, *, seeds=()):
        if self.stop_reason is not None:
            return CampaignStop(self.stop_reason)
        if self.elapsed() > self.limits["campaign_seconds"]:
            return BudgetExhausted("campaign wall-clock budget exceeded before the work was accepted")
        for seed in seeds:
            if self.training_seconds(seed) > self.limits["per_run_training_seconds"]:
                return BudgetExhausted(f"seed {seed} collection+optimization budget exceeded before the "
                                       "generation was accepted")
        return None

    def snapshot(self, seeds):
        return dict(elapsed_seconds=self.elapsed(), phase_seconds=dict(sorted(self.phase_seconds.items())),
                    training_seconds={str(s): self.training_seconds(s) for s in seeds},
                    evaluation_games_started=self.games_started)
