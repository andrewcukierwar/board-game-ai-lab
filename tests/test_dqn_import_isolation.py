"""Production imports and startup must work even if neural imports are blocked."""

from pathlib import Path
import subprocess
import sys


def test_api_starts_without_neural_modules():
    script = '''
import importlib.abc
import sys

class BlockNeural(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'torchvision', 'torchaudio'}:
            raise AssertionError('Production tried to import ' + fullname)
        if fullname.startswith('games.connect4.dqn.') or fullname.endswith('.dqn_agent'):
            raise AssertionError('Production tried to import ' + fullname)

sys.meta_path.insert(0, BlockNeural())
import games.connect4.dqn
from api.app import create_app
app = create_app({'TESTING': True, 'EXPLANATIONS_ENABLED': False})
assert app.test_client().get('/v1/connect4/health').status_code == 200
assert not any(n == 'torch' or n.startswith('torch.') for n in sys.modules)
print('API startup succeeded without neural imports')
'''
    result = subprocess.run([sys.executable, '-c', script], capture_output=True,
                            text=True, timeout=30, cwd=Path(__file__).resolve().parents[1])
    assert result.returncode == 0, result.stdout + result.stderr
