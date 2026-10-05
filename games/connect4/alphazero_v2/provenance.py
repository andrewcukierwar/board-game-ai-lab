"""Runtime and execution-source identity that bounds the exact-resume claim.

Exact continuation from a boundary is claimed only when (a) every recorded
runtime field that can change floating-point results or RNG algorithms is
identical, and (b) the *content* of every source file that can affect the
training trajectory is identical. A Git commit alone is not sufficient (a dirty
tree under the same commit can differ) and not necessary (documentation-only
commits leave execution unchanged), so the enforced identity is a content
digest of the execution closure; Git state is recorded as provenance only.
"""
import ctypes
import ctypes.util
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
V2_PACKAGE = "games/connect4/alphazero_v2"
# Repository modules outside the v2 package that the training trajectory imports.
# A test asserts that this list plus the package covers every repository module
# loaded by running and resuming a generation in a fresh interpreter.
EXECUTION_DEPENDENCIES = (
    "games/connect4/__init__.py",
    "games/connect4/agents/mcts_agent.py",
    "games/connect4/agents/mcts_nn_agent.py",
    "games/connect4/board.py",
    "games/connect4/connect4.py",
    "games/connect4/neural_mcts.py",
)
# Library thread controls read when native libraries initialize (Accelerate,
# OpenMP, MKL, OpenBLAS); they must be set before the interpreter starts.
THREAD_ENVIRONMENT = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                      "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
SOURCE_SCHEME = "sha256(json(sorted relative path -> file sha256))-v1"
# Reporting partition of the execution closure (both groups are always enforced together):
# (A) modules a GenerationRunner loads to collect, train, save and resume; (B) everything
# else in the closure: evaluation, solver, arena, statistics, packages and the launcher.
TRAINING_SOURCE_FILES = tuple(f"{V2_PACKAGE}/{name}.py" for name in (
    "__init__", "artifacts", "config", "data", "diagnostics", "generation", "network", "oracle",
    "provenance", "search", "selfplay", "training")) + EXECUTION_DEPENDENCIES
SOURCE_GROUPS = ("training", "evaluation_and_launch")


def execution_source_files():
    """Every file whose content can change the v2 training trajectory (repo-relative, sorted)."""
    package = {p.relative_to(REPO_ROOT).as_posix() for p in (REPO_ROOT / V2_PACKAGE).rglob("*.py")}
    return tuple(sorted(package | set(EXECUTION_DEPENDENCIES)))


def execution_source_identity():
    files = {}
    for relative in execution_source_files():
        files[relative] = hashlib.sha256((REPO_ROOT / relative).read_bytes()).hexdigest()
    combined = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return dict(scheme=SOURCE_SCHEME, sha256=combined, files=files)


def source_group(relative):
    return "training" if relative in TRAINING_SOURCE_FILES else "evaluation_and_launch"


def source_groups(files):
    """Per-group digests of an execution-closure file map (same scheme as the combined digest)."""
    groups = {}
    for name in SOURCE_GROUPS:
        members = {path: digest for path, digest in files.items() if source_group(path) == name}
        groups[name] = dict(files=sorted(members),
                            sha256=hashlib.sha256(json.dumps(members, sort_keys=True).encode()).hexdigest())
    return groups


def source_differences(recorded, current):
    """Sorted execution files whose content (or presence) differs."""
    before, after = recorded.get("files", {}), current.get("files", {})
    return sorted(name for name in set(before) | set(after) if before.get(name) != after.get(name))


def _git(*args):
    try:
        return subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True,
                              timeout=30, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return None


def git_provenance():
    """Recorded, never enforced: commit, tracked/untracked dirt and execution diff from HEAD."""
    files = execution_source_files()
    commit = _git("rev-parse", "HEAD")
    tracked_status = _git("status", "--porcelain=v1", "--untracked-files=no")
    untracked = _git("ls-files", "--others", "--exclude-standard")
    tracked_execution = _git("ls-files", "--", *files)
    diff = _git("diff", "--no-ext-diff", "--binary", "HEAD", "--", *files)
    if None in (commit, tracked_status, untracked, tracked_execution, diff):
        return dict(available=False)
    untracked_execution = sorted(set(files) - set(tracked_execution.split()))
    return dict(available=True, commit=commit.strip(), tracked_dirty=bool(tracked_status.strip()),
                untracked_files=len(untracked.split()), untracked_execution_files=untracked_execution,
                execution_diff_from_head_sha256=hashlib.sha256(diff.encode()).hexdigest() if diff else None,
                execution_matches_commit=not diff and not untracked_execution)


def source_identity(execution=None):
    return dict(execution=execution or execution_source_identity(), git=git_provenance())


def _sysctl_string(name):
    """In-process sysctlbyname (no subprocess, so sandboxes that block exec still identify the CPU)."""
    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
        size = ctypes.c_size_t(0)
        if libc.sysctlbyname(name.encode(), None, ctypes.byref(size), None, ctypes.c_size_t(0)) or not size.value:
            return None
        buffer = ctypes.create_string_buffer(size.value)
        if libc.sysctlbyname(name.encode(), buffer, ctypes.byref(size), None, ctypes.c_size_t(0)):
            return None
        return buffer.value.decode().strip() or None
    except (OSError, AttributeError, UnicodeDecodeError):
        return None


@lru_cache(maxsize=1)
def _cpu_model():
    try:
        if sys.platform == "darwin":
            probed = _sysctl_string("machdep.cpu.brand_string")
            if probed:
                return probed
            result = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                    text=True, timeout=10, check=True).stdout.strip()
            return result or None
        if sys.platform.startswith("linux"):
            with open("/proc/cpuinfo") as stream:
                for line in stream:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return platform.processor() or None


@lru_cache(maxsize=1)
def _static_runtime():
    return dict(
        python_implementation=platform.python_implementation(), python=platform.python_version(),
        torch=str(torch.__version__), torch_git_version=str(torch.version.git_version),
        torch_build_sha256=hashlib.sha256(torch.__config__.show().encode()).hexdigest(),
        numpy=str(np.__version__), platform=platform.platform(), machine=platform.machine(),
        cpu_model=_cpu_model(), cpu_capability=str(torch.backends.cpu.get_cpu_capability()))


def runtime_identity():
    """Every runtime field that can change numerics or RNG algorithms; all are enforced."""
    return dict(
        _static_runtime(),
        intra_op_threads=torch.get_num_threads(), inter_op_threads=torch.get_num_interop_threads(),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        mkldnn_enabled=bool(torch.backends.mkldnn.enabled),
        thread_environment={name: os.environ.get(name) for name in THREAD_ENVIRONMENT})


def unavailable_runtime_fields(identity):
    """Identity fields whose value could not be determined; such a runtime cannot be bound exactly.

    Two runtimes that both failed to identify (say) their CPU would otherwise compare equal.
    """
    missing = sorted(key for key, value in identity.items() if value is None)
    environment = identity.get("thread_environment")
    if isinstance(environment, dict):
        missing += sorted(f"thread_environment.{key}" for key, value in environment.items() if value is None)
    return missing


def runtime_differences(recorded, current):
    """Sorted keys whose values differ, including keys present on only one side."""
    return sorted(key for key in set(recorded) | set(current) if recorded.get(key, KeyError) != current.get(key, KeyError))


def configure_deterministic_runtime(threads=1):
    """Pin intra/inter-op threads and deterministic algorithms; return the runtime identity.

    Call once at process start, before any torch parallel work. Library thread
    environment variables must already be set by the launching shell, because
    native libraries read them at load time; mismatches are reported, not fixed.
    """
    if type(threads) is not int or threads < 1:
        raise ValueError("threads must be a positive integer")
    torch.set_num_threads(threads)
    if torch.get_num_interop_threads() != threads:
        try:
            torch.set_num_interop_threads(threads)
        except RuntimeError as error:
            raise RuntimeError("Inter-op threads are already fixed; configure the runtime at process start") from error
    torch.use_deterministic_algorithms(True)
    identity = runtime_identity()
    if (identity["intra_op_threads"], identity["inter_op_threads"], identity["deterministic_algorithms"]) != (
            threads, threads, True):
        raise RuntimeError("Deterministic runtime configuration did not take effect")
    return identity


def required_thread_environment(threads=1):
    return {name: str(threads) for name in THREAD_ENVIRONMENT}
