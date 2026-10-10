"""Research-only exact search variants generated from an immutable Git engine."""
import subprocess
import types

BASELINE = '60b99b0d42145da149f433907585b809060c946d'
AGENT_PATH = 'games/connect4/agents/negamax_agent.py'
VARIANTS = ('direct', 'mirror', 'mirror-selective', 'pvs')


def replace(source, old, new):
    assert source.count(old) == 1, old
    return source.replace(old, new)


MIRROR = '''
def reflect(bits):
    # Seven complete 7-bit chunks: sentinel geometry is preserved too.
    return (((bits & 127) << 42) | (((bits >> 7) & 127) << 35) |
            (((bits >> 14) & 127) << 28) | (bits & (127 << 21)) |
            (((bits >> 28) & 127) << 14) | (((bits >> 35) & 127) << 7) |
            ((bits >> 42) & 127))


def identity(state, depth, threshold=1):
    board = state.pieces[0] | (state.pieces[1] << 49)
    mirrored = False
    if depth >= threshold:
        reflected = reflect(state.pieces[0]) | (reflect(state.pieces[1]) << 49)
        mirrored = reflected < board
        if mirrored:
            board = reflected
    return board | (state.mover << 98) | (depth << 99), mirrored


def reflect_hint(move):
    return 6 - move if type(move) is int and 0 <= move < 7 else move
'''


def variant_source(variant, baseline=None):
    if baseline is None:
        baseline = subprocess.check_output(['git', 'show', f'{BASELINE}:{AGENT_PATH}']).decode()
    source = baseline
    if variant in ('mirror', 'mirror-selective'):
        source = replace(source,
            '    key = (state.pieces[0] | (state.pieces[1] << 49) |\n'
            '           (state.mover << 98) | (depth << 99))',
            f'    key, mirrored = identity(state, depth, {1 if variant == "mirror" else 3})')
        source = replace(source, '            hint = move',
                         '            hint = reflect_hint(move) if mirrored else move')
        source = replace(source, 'pack_entry(flag, best, best_move)',
                         'pack_entry(flag, best, reflect_hint(best_move) if mirrored else best_move)')
        source += MIRROR
    elif variant == 'pvs':
        source = replace(source, '    for col in state.ordered_moves(hint, tactical if depth > 1 else \'none\'):',
            '    first = True\n    for col in state.ordered_moves(hint, tactical if depth > 1 else \'none\'):')
        source = replace(source,
            '            value = -negamax(state, depth - 1, -beta, -alpha, table)',
            '            if first or alpha == -inf:\n'
            '                value = -negamax(state, depth - 1, -beta, -alpha, table)\n'
            '            else:\n'
            '                value = -negamax(state, depth - 1, -alpha - 1, -alpha, table)\n'
            '                if alpha < value < beta:\n'
            '                    value = -negamax(state, depth - 1, -beta, -alpha, table)')
        source = replace(source, '        if value > best:', '        first = False\n        if value > best:')
    elif variant != 'direct':
        raise ValueError(variant)
    return source


def diagnostic_source(source):
    source = replace(source, '        self.nodes = 0',
        '        self.leaves = self.terminals = self.probes = self.canonicalizations = self.reflected_hits = 0\n'
        '        self.nodes = 0')
    source = replace(source, '    if terminal is not None:\n        return terminal',
        '    if terminal is not None:\n        if table is not None:\n            table.terminals += 1\n        return terminal')
    source = replace(source, '    if depth == 0:\n        return state.heuristic()',
        '    if depth == 0:\n        if table is not None:\n            table.leaves += 1\n        return state.heuristic()')
    source = replace(source, '    alpha_original, beta_original = alpha, beta',
        '    if table is not None:\n        table.probes += 1\n'
        '    alpha_original, beta_original = alpha, beta')
    if 'key, mirrored = identity' in source:
        threshold = 3 if 'identity(state, depth, 3)' in source else 1
        source = replace(source, '    hint = None',
            f'    if table is not None and depth >= {threshold}:\n        table.canonicalizations += 1\n'
            '    hint = None')
        source = replace(source, '        table.hits += 1',
            '        table.reflected_hits += int(mirrored)\n        table.hits += 1')
    return source


def load_source(source, name='research'):
    module = types.ModuleType(name)
    exec(compile(source, name, 'exec'), module.__dict__)
    return module


def load_variant(variant, diagnostic=False):
    source = variant_source(variant)
    return load_source(diagnostic_source(source) if diagnostic else source, variant)
