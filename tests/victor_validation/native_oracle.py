"""Thin wrapper around the independent C WDL oracle (``c4_oracle.c``).

Benchmark/test tooling only: no Victor, engine or grounding imports. The binary
is compiled on demand with the system C compiler into a cache directory keyed by
the source hash (``VICTOR_ORACLE_CACHE`` or the system temp directory). Callers
must handle ``OracleUnavailable`` when no compiler exists.
"""
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

SOURCE = Path(__file__).with_name('c4_oracle.c')


class OracleUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class OracleResult:
    status: str  # 'exact' | 'unknown' | 'error'
    value: int | None  # mover-relative WDL
    move_values: dict[int, int]  # zero-based legal column -> mover-relative WDL
    nodes: int

    @property
    def optimal_moves(self):
        return tuple(c for c, v in self.move_values.items() if v == self.value)


def binary_path():
    digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()[:16]
    cache = Path(os.environ.get('VICTOR_ORACLE_CACHE', tempfile.gettempdir()))
    target = cache / f'victor-c4-oracle-{digest}'
    if target.exists():
        return target
    compiler = shutil.which(os.environ.get('CC', 'cc'))
    if compiler is None:
        raise OracleUnavailable('no C compiler available for the native oracle')
    cache.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(f'.{os.getpid()}.tmp')
    done = subprocess.run([compiler, '-O3', '-o', str(partial), str(SOURCE)],
                          capture_output=True, text=True)
    if done.returncode:
        raise OracleUnavailable(done.stderr.strip() or 'native oracle compilation failed')
    partial.replace(target)
    return target


def _parse(line):
    parts = line.split()
    if parts[0] != 'ok':
        return OracleResult('unknown' if parts[0] == 'unknown' else 'error', None, {},
                            int(parts[-1]))
    values = {c: int(v) for c, v in enumerate(parts[2:9]) if v != 'x'}
    return OracleResult('exact', int(parts[1]), values, int(parts[9]))


def solve_histories(histories, *, node_limit=50_000_000, timeout=None):
    """Exact WDL for each zero-based move history; shares one transposition table."""
    histories = [tuple(h) for h in histories]
    if not histories:
        return []
    if any(type(c) is not int or not 0 <= c < 7 for h in histories for c in h):
        raise ValueError('histories must contain zero-based columns 0..6')
    text = ''.join((''.join(str(c + 1) for c in h) or '.') + '\n' for h in histories)
    done = subprocess.run([str(binary_path()), str(int(node_limit))], input=text,
                          capture_output=True, text=True, timeout=timeout, check=True)
    lines = done.stdout.splitlines()
    if len(lines) != len(histories):
        raise RuntimeError('native oracle returned an unexpected number of results')
    return [_parse(line) for line in lines]


def solve_history(history, **kwargs):
    return solve_histories([history], **kwargs)[0]
