"""Exact opening book: precomputed W/D/L move values for early positions.

Every entry is a completed search of the independent C oracle
(``tests/victor_validation/c4_oracle.c``) run offline by
``tests/victor_validation/opening_book_builder.py``. Timeouts and node cut-offs
are recorded as ``unresolved`` and never become entries. Runtime lookup is pure
Python (no compiler) and fails closed:

- the whole file must match its schema and SHA-256 digest, otherwise the book is
  disabled;
- each hit replays its stored history and must reproduce the looked-up board,
  its legal columns and a value equal to the best move value, otherwise the hit
  is reported ``invalid`` and the caller falls back to its normal policy.

Keys are ``X + mask + BOTTOM`` on seven-bit columns (unique per board), reduced
to the smaller of the board and its mirror image. Values are mover-relative.
"""
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from .exact import BOTTOM, CENTER_ORDER, Bits, _mirror
from .position import Position

SCHEMA = 'victor-opening-book-v1'
BOOK_PATH = Path(__file__).with_name('data') / 'opening_book.json'
VALUE_CHARS = {'+': 1, '=': 0, '-': -1}
CHAR_VALUES = {v: k for k, v in VALUE_CHARS.items()}


class BookIntegrityError(ValueError):
    pass


def board_key(bits: Bits) -> int:
    mask = bits.pieces[0] | bits.pieces[1]
    return bits.pieces[0] + mask + BOTTOM


def canonical_key(bits: Bits) -> tuple[int, bool]:
    """(canonical key, whether the canonical orientation is the mirror image)."""
    key = board_key(bits)
    mirrored = _mirror(key)
    return (mirrored, True) if mirrored < key else (key, False)


def entries_digest(entries, unresolved) -> str:
    text = json.dumps([entries, unresolved], separators=(',', ':'))
    return hashlib.sha256(text.encode()).hexdigest()


@dataclass(frozen=True)
class BookLookup:
    """``status``: 'exact' (use it), 'absent', 'unresolved' (attempted, unknown),
    'invalid' (failed a structural check) or 'disabled' (no usable book)."""
    status: str
    value: int | None = None
    move_values: tuple[tuple[int, int], ...] = ()  # actual orientation, centre order
    sources: tuple[str, ...] = ()
    detail: str = ''

    @property
    def optimal_moves(self):
        return tuple(c for c, v in self.move_values if v == self.value)


def _replay(history):
    bits = Bits((0, 0), (0,) * 7, 0)
    for c in history:
        if not 0 <= c < 7 or bits.heights[c] >= 6 or bits.value is not None:
            return None
        bits = bits.drop(c)
    return None if bits.value is not None else bits


class OpeningBook:
    def __init__(self, data):
        if not isinstance(data, dict) or data.get('schema') != SCHEMA:
            raise BookIntegrityError('unsupported opening-book schema')
        entries, unresolved = data.get('entries'), data.get('unresolved')
        if not isinstance(entries, list) or not isinstance(unresolved, list):
            raise BookIntegrityError('malformed opening-book tables')
        if entries_digest(entries, unresolved) != data.get('digest'):
            raise BookIntegrityError('opening-book digest mismatch (stale or edited data)')
        self.sources = tuple(data.get('sources', ()))
        self.provenance = data.get('provenance', {})
        self.generation = data.get('generation', {})
        self.entries, self.unresolved = {}, {}
        try:
            for key, player, value, moves, history, source_mask in entries:
                self.entries[(int(key, 16), player)] = (value, moves, history, source_mask)
            for key, player, history, reason in unresolved:
                self.unresolved[(int(key, 16), player)] = (history, reason)
        except (TypeError, ValueError) as exc:
            raise BookIntegrityError(f'malformed opening-book row: {exc}') from None

    @classmethod
    def load(cls, path=BOOK_PATH):
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, ValueError) as exc:
            raise BookIntegrityError(f'cannot read opening book: {exc}') from None
        return cls(data)

    def __len__(self):
        return len(self.entries)

    def lookup(self, position: Position, *, exclude_sources=()) -> BookLookup:
        bits = Bits.from_position(position)
        if bits.value is not None:
            return BookLookup('absent', detail='terminal position')
        key, mirrored = canonical_key(bits)
        player = position.player_to_move
        row = self.entries.get((key, player))
        if row is None:
            if (key, player) in self.unresolved:
                return BookLookup('unresolved', detail=self.unresolved[(key, player)][1])
            return BookLookup('absent')
        value, moves, history, source_mask = row
        sources = tuple(s for i, s in enumerate(self.sources) if source_mask >> i & 1)
        if exclude_sources and set(sources) <= set(exclude_sources):
            return BookLookup('absent', detail='entry excluded by source filter')
        canonical = None
        if (isinstance(history, str) and len(history) == 42 - bits.remaining
                and all(ch in '1234567' for ch in history)):
            canonical = _replay([int(c) - 1 for c in history])  # stored 1-based
        if (canonical is None or board_key(canonical) != key or canonical.turn != player
                or not isinstance(moves, str) or len(moves) != 7 or value not in VALUE_CHARS):
            return BookLookup('invalid', detail='entry does not replay to this position')
        legal = {c for c in range(7) if canonical.heights[c] < 6}
        if {c for c, m in enumerate(moves) if m != 'x'} != legal or \
                any(moves[c] not in VALUE_CHARS for c in legal):
            return BookLookup('invalid', detail='entry legal columns disagree with the board')
        values = {c: VALUE_CHARS[moves[c]] for c in legal}
        if VALUE_CHARS[value] != max(values.values()):
            return BookLookup('invalid', detail='entry value is not its best move value')
        actual = {(6 - c if mirrored else c): v for c, v in values.items()}
        return BookLookup('exact', VALUE_CHARS[value],
                          tuple((c, actual[c]) for c in CENTER_ORDER if c in actual), sources)


@lru_cache(maxsize=1)
def default_book():
    """The packaged book, or None if it is missing or fails its integrity check."""
    try:
        return OpeningBook.load()
    except BookIntegrityError:
        return None


def lookup(position: Position, *, exclude_sources=()) -> BookLookup:
    book = default_book()
    if book is None:
        return BookLookup('disabled', detail='opening book missing or failed integrity check')
    return book.lookup(position, exclude_sources=exclude_sources)


def preferred_move(board, player_to_move, move_values, value, depth):
    """An optimal move; equal-value ties go to the bounded Negamax ranking.

    Shared by exact-search and opening-book selection (and by the book builder
    to follow Victor's own choices). Every candidate has the same proved W/D/L
    value, so the choice stays exact; ties among them prefer moves that are
    resilient against fallible opponents, then the centre-first order.
    """
    from games.connect4.agents.negamax_agent import NegamaxAgent
    best = [c for c, v in move_values if v == value]
    if len(best) == 1:
        return best[0]
    scores = NegamaxAgent(depth).score_moves(
        SimpleNamespace(board=board, current_player=player_to_move))
    return max(best, key=lambda c: scores[c])  # max keeps the first (centre) tie.


__all__ = ['BOOK_PATH', 'BookIntegrityError', 'BookLookup', 'OpeningBook', 'SCHEMA',
           'board_key', 'canonical_key', 'default_book', 'entries_digest', 'lookup',
           'preferred_move']
