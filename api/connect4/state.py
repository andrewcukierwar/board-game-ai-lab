"""Bounded, process-local sessions. Deploy with one Gunicorn worker only."""
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Lock
from time import monotonic
from uuid import uuid4

from games.connect4.connect4 import Connect4
from games.connect4.grounding.history import MoveRecord


class GameError(Exception):
    def __init__(self, code, message, status=400):
        self.code, self.message, self.status = code, message, status


@dataclass
class GameSession:
    players: list
    touched: float
    rng_seed: int | None = None
    game: Connect4 = field(default_factory=Connect4)
    revision: int = 0
    history: tuple[MoveRecord, ...] = ()
    explanation_requests: int = 0
    lock: Lock = field(default_factory=Lock)


class GameStore:
    def __init__(self, capacity=128, ttl=1800, clock=monotonic):
        if capacity < 1 or ttl <= 0:
            raise ValueError('Session capacity and TTL must be positive.')
        self.capacity, self.ttl, self.clock = capacity, ttl, clock
        self._games = {}
        self._lock = Lock()

    def _expire(self):
        # Caller holds the registry lock. Never remove a game being played.
        now = self.clock()
        for gid, session in list(self._games.items()):
            if now - session.touched >= self.ttl and session.lock.acquire(False):
                try:
                    # A request may have completed between the first check and lock acquisition.
                    if now - session.touched >= self.ttl:
                        del self._games[gid]
                finally:
                    session.lock.release()

    def create(self, players, replace_id=None, rng_seed=None):
        with self._lock:
            self._expire()
            old = self._games.get(replace_id)
            if old is not None and not old.lock.acquire(False):
                raise GameError('game_busy', 'A move is still running. Try again shortly.', 409)
            try:
                if old is None and len(self._games) >= self.capacity:
                    raise GameError('session_limit', 'The game server is full. Try again later.', 503)
                gid = str(uuid4())
                session = GameSession(players=players, touched=self.clock(), rng_seed=rng_seed)
                self._games[gid] = session
                if old is not None:
                    del self._games[replace_id]
                return gid, session
            finally:
                if old is not None:
                    old.lock.release()

    @contextmanager
    def access(self, gid):
        with self._lock:
            self._expire()
            session = self._games.get(gid)
            if session is None:
                raise GameError('session_not_found', 'This game expired or no longer exists. Start a new game.', 404)
            if not session.lock.acquire(False):
                raise GameError('game_busy', 'A move is still running. Try again shortly.', 409)
        try:
            yield session
        finally:
            session.touched = self.clock()
            session.lock.release()
