"""Fail-closed, non-resumable control for the official AlphaZero v2 campaign (torch-free, Phase 4D.3B.4).

The official campaign is one owning process holding one exclusive lock on its
campaign directory. It runs every seed, the development selection and the
sealed final evaluation sequentially under one declaration. There is no resume:

* ``state.json`` is the single authority. It is replaced atomically (temporary
  file, fsync, rename, directory fsync) on every transition, so a crash leaves
  either the previous or the next complete state, never a torn one. The owner
  only replaces the exact bytes it last wrote; after a failed write it adopts
  whatever readers can now see.
* States advance along one fixed line: CREATED, RUNNING_SEED_<seed> for each
  declared seed in order, DEVELOPMENT_SELECTION_COMPLETE,
  RUNNING_FINAL_EVALUATION, COMPLETE. INCOMPLETE may follow any non-terminal
  state. COMPLETE and INCOMPLETE are terminal and never transition again:
  ``terminate`` is the one way into them, and ``mark_incomplete`` writes
  INCOMPLETE only if the visible state is still non-terminal.
* A non-terminal state with no live owner means the owner died. A later
  invocation marks it INCOMPLETE and refuses; it never repairs and continues.
* ``Budget`` measures monotonic seconds inside the owning process only. Limits
  are checked before every substantial unit and every search, move and
  optimizer update. Nothing is reconstructed across processes, so no leases.
  ``cross_completion_barrier`` latches the campaign's one authoritative
  endpoint; a stop requested after it is recorded but cannot change the outcome.
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

LAUNCH_CONTROL_VERSION = "connect4-alphazero-v2-launch-control-v3-terminal-commit"
STATE_FORMAT = "connect4-alphazero-v2-official-campaign-state-v2"
# Declaration tokens that must never authorize a campaign: Phase 4D.3B (launch-readiness review),
# Phase 4D.3B.1 (final launch review, NO GO), Phase 4D.3B.2 (final fail-closed launch review, NO GO) and
# Phase 4D.3B.3 (NO GO: a malformed COMPLETE record without seed 42 training time was accepted).
REJECTED_DECLARATION_TOKENS = ("2741314399741c1b20510b8a2beb42e0938b0aeafc9dd0c487201e57208867d8",
                               "8a8a52b100c7e32aa5e3e5b05469578084c8e12f37750ba81fee071990ff5e39",
                               "ed641aea2fef86a29000df55d1e46d9a32bf48dc3ec8f5dbcc0f8449668d0bbb",
                               "34b4d899639d59d1bd4408ecc505b7eb20cb4d9c2fdc12095402c246887ac390")
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
    """Replace a file atomically: a reader sees the old or the new bytes, never a mixture.

    The new bytes become visible at the rename. A failure before it removes the
    temporary file and leaves the old bytes; a failure after it (directory fsync)
    leaves the new bytes visible, so callers must re-read before deciding anything.
    """
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(temporary, "wb") as stream:
            stream.write(data)
            stream.flush()
            full_fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass  # already renamed, or never created
        raise
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
    """``state.json``: current state, transition history and the last counter snapshot.

    Every write first checks that the visible file still holds exactly the bytes
    this object last wrote or read, so nothing ever blindly replaces it. If a
    write fails, the object adopts whatever is now visible (old or new).
    """

    NAME = "state.json"

    def __init__(self, path, document, data):
        self.path, self.document, self.data = Path(path), document, data

    @classmethod
    def create(cls, directory, *, declaration_sha256, seeds):
        document = dict(format=STATE_FORMAT, launch_control=LAUNCH_CONTROL_VERSION,
                        declaration_sha256=declaration_sha256, seeds=list(seeds), state=CREATED,
                        owner=dict(pid=os.getpid(), host=socket.gethostname(), started_utc=utc_now()),
                        history=[dict(state=CREATED, utc=utc_now())], counters=None, outcome=None)
        path, data = Path(directory) / cls.NAME, view_bytes(document)
        write_once(path, data)
        return cls(path, document, data)

    @classmethod
    def load(cls, directory):
        path = Path(directory) / cls.NAME
        try:
            data = path.read_bytes()
            document = json.loads(data)
        except (OSError, ValueError) as error:
            raise StateCorrupt(f"{path} is missing or unreadable: {error}") from error
        cls.validate(document)
        return cls(path, document, data)

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
        if document["outcome"] is not None and document["outcome"].get("status") != document["state"]:
            raise StateCorrupt("state.json outcome status disagrees with its terminal state")

    @property
    def state(self):
        return self.document["state"]

    @property
    def terminal(self):
        return self.state in TERMINAL_STATES

    def refresh(self):
        """Adopt the state readers can see now. If it is unreadable, keep the cached copy: the next write
        then refuses, because the visible bytes are no longer the bytes this object holds."""
        try:
            visible = self.load(self.path.parent)
        except StateCorrupt:
            return
        self.document, self.data = visible.document, visible.data

    def _write(self, document):
        """Replace state.json, only if it still holds exactly the bytes this object last wrote or read."""
        self.validate(document)
        try:
            visible = self.path.read_bytes()
        except OSError as error:
            raise StateCorrupt(f"{self.path} is unreadable; refusing to replace it: {error}") from error
        if visible != self.data:
            raise StateCorrupt(f"{self.path} is not the document this owner last wrote; refusing to replace it")
        data = view_bytes(document)
        try:
            atomic_replace(self.path, data)
        except BaseException:
            self.refresh()  # the old or the new document may be visible now; only the visible one counts
            raise
        self.document, self.data = document, data

    def _require_nonterminal(self):
        if self.terminal:
            raise InvalidTransition(f"Campaign is {self.state}; terminal states never transition again")

    def _next(self, new_state, counters, outcome, info):
        document = json.loads(json.dumps(self.document))
        document["state"] = new_state
        document["history"].append(dict(info, state=new_state, utc=utc_now()))
        if counters is not None:
            document["counters"] = counters
        document["outcome"] = outcome
        return document

    def transition(self, new_state, *, counters=None, **info):
        """Durably enter the next running state in sequence. Terminal states are entered only by ``terminate``."""
        self._require_nonterminal()
        sequence = state_sequence(self.document["seeds"])
        expected = sequence[sequence.index(self.state) + 1]
        if new_state in TERMINAL_STATES or "outcome" in info:
            raise InvalidTransition(f"{new_state} is not a running state; use terminate() for {COMPLETE} or "
                                    f"{INCOMPLETE}")
        if new_state != expected:
            raise InvalidTransition(f"{self.state} -> {new_state} is not allowed (next is {expected} or "
                                    f"{INCOMPLETE})")
        self._write(self._next(new_state, counters, None, info))

    def terminate(self, terminal_state, *, outcome, counters=None, **info):
        """The one terminal-state transition. COMPLETE only from RUNNING_FINAL_EVALUATION, INCOMPLETE from any
        non-terminal state; neither has an outgoing transition. ``outcome['status']`` must name the state."""
        self._require_nonterminal()
        if terminal_state not in TERMINAL_STATES:
            raise InvalidTransition(f"{terminal_state} is not a terminal state")
        if terminal_state == COMPLETE and self.state != RUNNING_FINAL:
            raise InvalidTransition(f"{self.state} -> {COMPLETE} is not allowed (only from {RUNNING_FINAL})")
        if not isinstance(outcome, dict) or outcome.get("status") != terminal_state:
            raise InvalidTransition("A terminal state needs an outcome whose status names it")
        self._write(self._next(terminal_state, counters, outcome, info))

    def mark_incomplete(self, *, reason, counters=None, **info):
        """Conditionally end the campaign INCOMPLETE; the only operation error and interruption paths use.

        It re-reads the visible state first. If that is already terminal (COMPLETE or
        INCOMPLETE) nothing is written. Otherwise INCOMPLETE is built from the visible
        document, never from a stale copy. Returns the visible terminal state.
        """
        self.refresh()
        if not self.terminal:
            self.terminate(INCOMPLETE, counters=counters, outcome=dict(status=INCOMPLETE, reason=reason,
                                                                       during=self.state), **info)
        return self.state

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

    ``cross_completion_barrier`` takes the campaign's one authoritative endpoint.
    It latches that clock reading *before* reading the stop flag, so every stop
    request is either seen by the barrier or arrives after it. Python runs signal
    handlers between bytecodes of the main thread, whichever thread received the
    signal, so this ordering needs no signal masking. After the latch, elapsed
    time is frozen at the endpoint and stop requests are only recorded.
    """

    def __init__(self, limits, *, clock):
        self.limits, self.clock = limits, clock
        self.started = clock()
        self.training_accumulated, self.training_seed, self.training_started = {}, None, None
        self.phase_seconds = {}
        self.games_started = 0
        self.stop_reason = None
        self.barrier_time = None
        self.late_stop_requests = []

    def request_stop(self, reason="stop requested"):
        """Signal-handler safe: only records. After the completion barrier a request cannot change the outcome."""
        if self.barrier_time is not None:
            self.late_stop_requests.append(reason)
        else:
            self.stop_reason = reason

    def elapsed(self):
        return (self.clock() if self.barrier_time is None else self.barrier_time) - self.started

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

    def cross_completion_barrier(self, seeds):
        """Latch the authoritative endpoint, then decide completion eligibility from it alone.

        Limits are inclusive at completion, as for every acceptance check: elapsed
        seconds at the endpoint <= campaign_seconds, each seed's training seconds
        <= per_run_training_seconds, games started <= evaluation_games_ceiling.
        Raises CampaignStop/BudgetExhausted if not eligible; returns the account.
        """
        if self.barrier_time is not None:
            raise RuntimeError("The completion barrier is crossed once")
        if self.training_seed is not None:
            raise RuntimeError("Training time is still being measured at the completion barrier")
        self.barrier_time = self.clock()  # the latch: a stop request handled after this line is post-completion
        if self.stop_reason is not None:
            raise CampaignStop(self.stop_reason)
        if self.elapsed() > self.limits["campaign_seconds"]:
            raise BudgetExhausted("campaign wall-clock budget exceeded at the completion barrier")
        for seed in seeds:
            if self.training_seconds(seed) > self.limits["per_run_training_seconds"]:
                raise BudgetExhausted(f"seed {seed} collection+optimization budget exceeded at the completion "
                                      "barrier")
        if self.games_started > self.limits["evaluation_games_ceiling"]:
            raise BudgetExhausted("evaluation-game ceiling exceeded at the completion barrier")
        return self.snapshot(seeds)

    def snapshot(self, seeds):
        return dict(elapsed_seconds=self.elapsed(), phase_seconds=dict(sorted(self.phase_seconds.items())),
                    training_seconds={str(s): self.training_seconds(s) for s in seeds},
                    evaluation_games_started=self.games_started)
