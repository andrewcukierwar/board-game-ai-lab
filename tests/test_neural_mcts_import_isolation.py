"""Backend startup must not import either optional neural family."""
from pathlib import Path
import subprocess
import sys


def test_api_starts_with_all_neural_imports_blocked():
    script = '''
import importlib.abc
import sys
class BlockNeural(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if (fullname.split('.')[0] in {'torch', 'torchvision', 'torchaudio'}
            or fullname.startswith('games.connect4.dqn.')
            or fullname in {'games.connect4.neural_mcts',
                            'games.connect4.agents.mcts_nn_agent',
                            'games.connect4.train_mcts_nn',
                            'games.connect4.neural_self_play',
                            'games.connect4.neural_evaluation',
                            'games.connect4.neural_value_target_audit',
                            'games.connect4.agents.dqn_agent'}):
            raise AssertionError('Production tried to import ' + fullname)
sys.meta_path.insert(0, BlockNeural())
from api.app import create_app
app = create_app({'TESTING': True, 'EXPLANATIONS_ENABLED': False})
assert app.test_client().get('/v1/connect4/health').status_code == 200
assert not any(n == 'torch' or n.startswith('torch.') for n in sys.modules)
print('API startup succeeded with all neural imports blocked')
'''
    result = subprocess.run([sys.executable, '-c', script], capture_output=True,
                            text=True, timeout=30, cwd=Path(__file__).resolve().parents[1])
    assert result.returncode == 0, result.stdout + result.stderr
