"""Immutable starting engine with a hash-verified shallow-checkout fallback."""
import hashlib
from pathlib import Path
import subprocess

BASELINE = '60b99b0d42145da149f433907585b809060c946d'
PATH = 'games/connect4/agents/negamax_agent.py'
COPY = 'docs/search-negamax-v3/mirror/direct-source.py'
SHA256 = 'f46461d73b240c645248d57884fee0b9917c1be289758efe381cf6458a9cb392'


def baseline_bytes():
    try:
        data = subprocess.check_output(['git','show',f'{BASELINE}:{PATH}'],
                                       stderr=subprocess.DEVNULL)
    except (OSError,subprocess.CalledProcessError):
        data = Path(COPY).read_bytes()
    if hashlib.sha256(data).hexdigest() != SHA256:
        raise ValueError('Negamax v3 baseline differs from immutable SHA-256')
    return data
