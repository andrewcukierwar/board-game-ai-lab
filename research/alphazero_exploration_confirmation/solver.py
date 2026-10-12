"""Bounded persistent access to the project's unchanged independent C oracle.

No solver enters neural search. Exact vectors, including reflection, are cached
by board/actor, never by history. Unknown results retain explicit reasons.
"""
import select
import subprocess
import time
from games.connect4.alphazero_v2.bitboard import BitboardState
from research.alphazero_exploration.analysis import canonical, mirror_bits
from .native_oracle import binary_path, _parse

class NativeSession:
    def __init__(self, nodes, seconds):
        self.nodes, self.seconds, self.process = nodes, seconds, None
    def close(self):
        if self.process is not None:
            self.process.kill()
            self.process.communicate()
            self.process = None
    def query(self, history, seconds=None):
        if self.process is None:
            self.process = subprocess.Popen([str(binary_path()), str(self.nodes)], stdin=subprocess.PIPE,
                                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        started = time.perf_counter()
        self.process.stdin.write((''.join(str(c+1) for c in history) or '.')+'\n')
        self.process.stdin.flush()
        ready, _, _ = select.select([self.process.stdout], [], [], self.seconds if seconds is None else seconds)
        if not ready:
            self.close()
            return dict(status='unknown', value=None, actions={}, reason='wall_timeout', nodes=None,
                        seconds=time.perf_counter()-started)
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError('Native oracle exited without a result')
        result = _parse(line)
        if result.status == 'error':
            raise ValueError(f'Native oracle rejected nonterminal history: {history}')
        return dict(status=result.status, value=result.value, actions=result.move_values,
                    reason=None if result.status == 'exact' else 'node_limit', nodes=result.nodes,
                    seconds=time.perf_counter()-started)

def orientation(history):
    state = BitboardState.from_game(__import__('games.connect4.alphazero_v2.oracle', fromlist=['engine_position']).engine_position(history))
    key = state.pieces
    return canonical(key), key != canonical(key)

def classify(before, after):
    """Both values belong to the original actor; None is never safe or erroneous."""
    if before is None or after is None:
        return 'unresolved'
    if after > before:
        raise ValueError('Exact action value exceeds exact optimal state value')
    if before == after:
        return 'neutral'
    return {(1,0):'win_draw', (1,-1):'win_loss', (0,-1):'draw_loss'}[before,after]

def decision_interval(lower, upper, margin=.05):
    if lower > margin:
        return 'MATERIAL HARM'
    if upper < margin:
        return 'NO MATERIAL HARM WITHIN THE DECLARED MARGIN'
    return 'INCONCLUSIVE'
