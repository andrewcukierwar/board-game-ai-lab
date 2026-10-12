"""Run the ResNet cases in their own interpreter, like the experimental contract suite.

The cases pin one-thread deterministic Torch settings that cannot be reversed
in a shared test process.
"""
import os
from pathlib import Path
import subprocess
import sys


def test_isolated_resnet_suite():
    environment = dict(os.environ, **{key: '1' for key in (
        'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
        'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')})
    result = subprocess.run([sys.executable, '-B', '-m', 'pytest', '-q',
                             'tests/alphazero_resnet_cases.py'],
                            cwd=Path(__file__).resolve().parents[1], env=environment,
                            capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    print(result.stdout.strip())
