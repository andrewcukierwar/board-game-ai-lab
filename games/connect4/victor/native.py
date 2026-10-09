"""Optional native move proofs, with no compiler or test imports at runtime.

Build explicitly: python -c 'from games.connect4.victor.native import build; build()'
The library is source/platform keyed and loaded from the package directory.
Missing/unsupported native code returns unavailable; the Python solver remains
fully functional. ctypes releases the GIL during the bounded search.
"""
from dataclasses import dataclass
from functools import lru_cache
import ctypes
import hashlib
from math import isfinite
from pathlib import Path
import platform
import subprocess
import sys
from time import monotonic

from .exact import Bits, CENTER_ORDER

SOURCE = Path(__file__).with_name('native_search.c')


def library_path():
    digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()[:16]
    return SOURCE.with_name(f'.native-{digest}-{sys.platform}-{platform.machine()}.so')


@dataclass(frozen=True)
class NativeBudget:
    nodes: int = 10_000_000
    seconds: float | None = 0.4
    table_entries: int = 1_048_576  # 16 MiB; replacement never implies a proof
    claimeven: bool = True
    serial: bool = False  # full-values control for experiments

    def __post_init__(self):
        for name in ('nodes', 'table_entries'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f'{name} must be a nonnegative integer')
        if self.nodes > 2**64 - 1:
            raise ValueError('native nodes must fit uint64')
        if self.table_entries > 4_194_304:
            raise ValueError('native table ceiling is 64 MiB')
        if type(self.serial) is not bool or type(self.claimeven) is not bool:
            raise ValueError('serial and claimeven must be boolean')
        if self.seconds is not None and (
                type(self.seconds) not in (int, float) or
                not isfinite(self.seconds) or self.seconds < 0):
            raise ValueError('seconds must be finite and nonnegative or None')


@dataclass(frozen=True)
class MoveProof:
    status: str
    intervals: tuple[tuple[int, int, int], ...]
    nodes: int
    cache_hits: int
    elapsed: float
    bound_hits: int = 0

    @property
    def lower(self):
        return max((lo for _, lo, _ in self.intervals), default=-1)

    @property
    def upper(self):
        return max((hi for _, _, hi in self.intervals), default=1)

    @property
    def optimal_moves(self):
        return tuple(c for c, lo, _ in self.intervals if lo >= self.upper)

    @property
    def admissible_moves(self):
        # Preserve a proved non-loss. Otherwise only remove proved losses while
        # an alternative can still be better. This is selection, not a proof.
        if self.lower >= 0:
            return tuple(c for c, lo, _ in self.intervals if lo == self.lower)
        candidates = tuple(c for c, _, hi in self.intervals if hi > -1)
        return candidates or tuple(c for c, _, _ in self.intervals)


@lru_cache(maxsize=1)
def _library():
    try:
        lib = ctypes.CDLL(str(library_path()))
        f = lib.victor_prove
        int_pointer = ctypes.POINTER(ctypes.c_int)
        f.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_int, ctypes.c_uint64,
                      ctypes.c_double, ctypes.c_uint64, int_pointer, ctypes.c_int,
                      int_pointer, int_pointer, ctypes.POINTER(ctypes.c_uint64)]
        f.restype = ctypes.c_int
        return lib
    except (OSError, AttributeError):
        return None


def available():
    """Whether a matching prebuilt library can be loaded (never compiles)."""
    return _library() is not None


def prove_moves(position, budget=NativeBudget(), order=CENTER_ORDER):
    if type(budget) is not NativeBudget:
        raise ValueError('budget must be NativeBudget')
    if (len(order) != 7 or set(order) != set(range(7)) or
            any(type(c) is not int for c in order)):
        raise ValueError('order must be a permutation of columns 0..6')
    start = monotonic()
    bits = Bits.from_position(position)
    if bits.value is not None:
        return MoveProof('terminal', (), 0, 0, monotonic() - start)
    lib = _library()
    if lib is None:
        return MoveProof('unavailable', (), 0, 0, monotonic() - start)
    lo, hi = (ctypes.c_int * 7)(), (ctypes.c_int * 7)()
    ordering = (ctypes.c_int * 7)(*order)
    stats = (ctypes.c_uint64 * 3)()
    status = lib.victor_prove(
        bits.pieces[bits.turn], bits.pieces[0] | bits.pieces[1],
        42 - bits.remaining, budget.nodes,
        -1.0 if budget.seconds is None else budget.seconds,
        budget.table_entries, ordering, int(budget.serial) + 2 * int(budget.claimeven),
        lo, hi, stats)
    if status == -1:
        return MoveProof('unavailable', (), 0, 0, monotonic() - start)
    intervals = tuple((c, lo[c], hi[c]) for c in order if lo[c] != 2)
    return MoveProof({0: 'partial', 1: 'optimal_move', 2: 'all_moves', 3: 'time_budget'}[status],
                     intervals, stats[0], stats[1], monotonic() - start, stats[2])


def build():
    """Explicit local/build-time compilation; never invoked by move selection."""
    import os
    import shutil
    import tempfile
    compiler = shutil.which(os.environ.get('CC', 'cc'))
    if compiler is None:
        raise RuntimeError('native search requires a C compiler at build time')
    target = library_path()
    with tempfile.TemporaryDirectory(dir=SOURCE.parent) as tmp:
        output = Path(tmp) / 'search.so'
        subprocess.run([compiler, '-O3', '-std=c11', '-fPIC', '-shared', str(SOURCE),
                        '-o', str(output)], check=True, capture_output=True)
        output.replace(target)
    _library.cache_clear()
    return target


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', action='store_true', required=True)
    parser.parse_args()
    print(build())
