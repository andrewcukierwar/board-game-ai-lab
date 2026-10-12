"""Isolate one-shot Torch runtime configuration from the historical test suite.

Torch inter-op configuration cannot be reversed after initialization. Run the
experimental contract cases in their own interpreter, exactly like recovery.
"""
import os
from pathlib import Path
import subprocess
import sys


def test_isolated_experimental_contract_suite():
    environment = dict(os.environ, **{key: '1' for key in (
        'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
        'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')})
    result = subprocess.run([sys.executable, '-B', '-m', 'pytest', '-q',
                             'tests/alphazero_experimental_cases.py'],
                            cwd=Path(__file__).resolve().parents[1], env=environment,
                            capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    print(result.stdout.strip())
