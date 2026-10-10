"""Research-only complete decisions generated from the immutable Phase 3B.2 engine.

No public search path imports this module. Source substitutions are asserted;
score TT identity, bound classification, evaluation and restoration stay intact.
"""
import subprocess
import types

from scripts.negamax_tt_variants import replace_once

BASELINE = '2e79b15ecdb1967345a1e66593201f9803c89757'
AGENT_PATH = 'games/connect4/agents/negamax_agent.py'
VARIANTS = ('direct', 'hints', 'iterative', 'combined')
IDENTITY_MASK = (1 << 99) - 1


def git_source(path=AGENT_PATH):
    return subprocess.check_output(['git', 'show', f'{BASELINE}:{path}'])


def schedule(target, variant):
    if type(target) is not int or target < 1 or variant not in VARIANTS:
        raise ValueError('Invalid experiment configuration')
    if variant == 'direct' or target <= 2:
        return (target,)
    if variant == 'hints':
        return (max(1, target - 2), target)
    return tuple(range(2 if target % 2 == 0 else 1, target + 1, 2))


DRIVER = '''
IDENTITY_MASK = (1 << 99) - 1


def export_hints(table):
    # A value's move field is decoded ONLY to produce an ordering suggestion.
    # The resulting cache contains no scores, bounds, horizons or solved flags.
    hints = {}
    for key, entry in table.entries.items():
        move = unpack_entry(entry)[2]
        if type(move) is int and 0 <= move < 7:
            hints[key & IDENTITY_MASK] = move
    return hints


class ExperimentalAgent(NegamaxAgent):
    def score_moves(self, game):
        state = SearchState(game)
        if state.terminal_value(self.depth) is not None:
            raise ValueError('Cannot choose a move from a terminal position')
        hints = None
        self.last_iterations = []
        for horizon in SCHEDULE(self.depth, VARIANT):
            table, scores = SearchTable(), {}
            table.hints = hints
            for col in state.legal():
                state.play(col)
                try:
                    scores[col] = -negamax(state, horizon - 1, -inf, inf, table)
                finally:
                    state.undo(col)
            stats = dict(depth=horizon, nodes=table.nodes, entries=len(table.entries),
                         hits=table.hits, cutoffs=table.cutoffs)
            self.last_iterations.append(stats)
            if horizon != self.depth and VARIANT in ('hints', 'combined'):
                hints = export_hints(table)
        self.last_scores = scores
        self.last_stats = {name: sum(s[name] for s in self.last_iterations)
                           for name in ('nodes', 'entries', 'hits', 'cutoffs')}
        return scores


NegamaxAgent = ExperimentalAgent
'''


def variant_source(variant):
    if variant not in VARIANTS:
        raise ValueError(variant)
    source = git_source().decode()
    if variant == 'direct':
        return source
    if variant in ('hints', 'combined'):
        source = replace_once(source, 'self.entries = {}', 'self.entries = {}\n        self.hints = None')
        source = replace_once(source, '    best, best_move = -inf, None',
            '    if hint is None and table is not None and table.hints is not None:\n'
            '        suggestion = table.hints.get(key & ((1 << 99) - 1))\n'
            '        if (type(suggestion) is int and 0 <= suggestion < 7\n'
            '                and state.heights[suggestion] < 6):\n'
            '            hint = suggestion\n'
            '    best, best_move = -inf, None')
    # C needs a hints attribute for the common driver but no lookup in search.
    else:
        source = replace_once(source, 'self.entries = {}', 'self.entries = {}\n        self.hints = None')
    return source + DRIVER


def diagnostic_source(variant):
    source = variant_source(variant)
    source = replace_once(source, '    if depth == 0:\n        return state.heuristic()',
        '    if depth == 0:\n        if table is not None:\n            table.diagnostic["leaves"] += 1\n'
        '        return state.heuristic()')
    if variant in ('hints', 'combined'):
        source = replace_once(source, '        suggestion = table.hints.get',
            '        table.diagnostic["hint_lookups"] += 1\n        suggestion = table.hints.get')
        source = replace_once(source, '            hint = suggestion',
            '            table.diagnostic["legal_hints"] += 1\n            hint = suggestion\n'
            '        elif suggestion is not None:\n            table.diagnostic["invalid_hints"] += 1')
    source = replace_once(source,
        '    for col in state.ordered_moves(hint, tactical if depth > 1 else \'none\'):',
        '    moves = state.ordered_moves(hint, tactical if depth > 1 else "none")\n'
        '    hinted_first = hint is not None and moves[0] == hint\n'
        '    if table is not None and hinted_first:\n'
        '        if moves[0] != state.ordered_moves(None, tactical if depth > 1 else "none")[0]:\n'
        '            table.diagnostic["changed_first"] += 1\n'
        '    for index, col in enumerate(moves):')
    source = replace_once(source, '                table.cutoffs += 1',
        '                table.cutoffs += 1\n'
        '                table.diagnostic["first_move_cutoffs"] += int(index == 0)\n'
        '                table.diagnostic["hinted_first_cutoffs"] += int(index == 0 and hinted_first)')
    return source


DIAGNOSTICS = ('leaves', 'hint_lookups', 'legal_hints', 'invalid_hints',
               'changed_first', 'first_move_cutoffs', 'hinted_first_cutoffs')


def load_variant(variant, diagnostic=False):
    module = types.ModuleType(f'negamax_iterative_{variant}')
    module.SCHEDULE, module.VARIANT = schedule, variant
    source = diagnostic_source(variant) if diagnostic else variant_source(variant)
    exec(compile(source, f'{BASELINE}:{AGENT_PATH}:{variant}', 'exec'), module.__dict__)
    return module
