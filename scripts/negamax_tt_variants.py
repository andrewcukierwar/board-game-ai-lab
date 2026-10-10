"""TT-only ablations generated from the immutable incremental-evaluation baseline.

Small asserted substitutions keep a single canonical search algorithm. The
resulting source is saved with benchmark evidence for independent inspection.
No benchmark outcome changes a search parameter or its evaluation.
"""
import types

from games.connect4.agents.negamax_tt import DirectMappedEntries
from scripts.pinned_git_source import pinned_source

BASELINE = 'f2f57b46c7817bb7324f4390da0bb345ae2fc7e4'
AGENT_PATH = 'games/connect4/agents/negamax_agent.py'
VARIANTS = ('baseline', 'packed-key', 'packed-entry')


def git_source(path=AGENT_PATH):
    return pinned_source(BASELINE, path)


def replace_once(source, old, new):
    assert source.count(old) == 1, old
    return source.replace(old, new)


def variant_source(variant, bounded=False):
    if variant not in VARIANTS:
        raise ValueError(variant)
    source = git_source().decode()
    if variant != 'baseline':
        source = replace_once(source, 'from math import inf',
            'from math import inf\nfrom games.connect4.agents.negamax_tt import pack_entry, unpack_entry')
        source = replace_once(source, 'key = (*state.pieces, state.mover, depth)',
            'key = (state.pieces[0] | (state.pieces[1] << 49) |\n'
            '           (state.mover << 98) | (depth << 99))')
    if variant == 'packed-entry':
        source = replace_once(source, 'flag, value, move = table.entries[key]',
                              'flag, value, move = unpack_entry(table.entries[key])')
        source = replace_once(source, 'table.entries[key] = (flag, best, best_move)',
                              'table.entries[key] = pack_entry(flag, best, best_move)')
    if bounded:
        source = replace_once(source, 'if table is not None and key in table.entries:',
            'entry = table.entries.get(key) if table is not None else None\n'
            '    if entry is not None:')
        source = replace_once(source,
            'flag, value, move = unpack_entry(table.entries[key])' if variant == 'packed-entry'
            else 'flag, value, move = table.entries[key]',
            'flag, value, move = unpack_entry(entry)' if variant == 'packed-entry'
            else 'flag, value, move = entry')
    return source


def load_variant(variant, capacity=None):
    source = variant_source(variant, bounded=capacity is not None)
    module = types.ModuleType(f'negamax_tt_{variant}_{capacity}')
    exec(compile(source, f'{BASELINE}:{AGENT_PATH}:{variant}:{capacity}', 'exec'), module.__dict__)
    if capacity is not None:
        original = module.SearchTable

        class BoundedSearchTable(original):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.entries = DirectMappedEntries(capacity)

        module.SearchTable = BoundedSearchTable
    return module
