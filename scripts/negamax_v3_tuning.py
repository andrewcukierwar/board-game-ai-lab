"""Provisional bounded fast paths; selected only after strongest-engine profiling."""
import re
from scripts.negamax_v3_variants import replace


def terminal_parent_proof(source):
    """Only a drop from a verified nonterminal parent may skip the other side."""
    source=replace(source,'    def terminal_value(self, depth):',
                   '    def terminal_value(self, depth, previous_only=False):')
    source=replace(source,'        if has_four(self.pieces[self.mover]):',
                   '        if not previous_only and has_four(self.pieces[self.mover]):')
    source=replace(source,'def negamax(state, depth, alpha=-inf, beta=inf, table=None):',
                   'def negamax(state, depth, alpha=-inf, beta=inf, table=None, parent_checked=False):')
    source=replace(source,'    terminal = state.terminal_value(depth)',
                   '    terminal = state.terminal_value(depth, previous_only=parent_checked)')
    source,n=re.subn(r'negamax\(state, (depth - 1|self.depth - 1), ([^\n]+), table\)',
                     r'negamax(state, \1, \2, table, parent_checked=True)',source)
    assert n in (2,4), n # Direct: recursive + root. PVS: three recursive + root.
    return source


def trusted_tt(source):
    """Fast integer access under the same internal validity proof as packed keys."""
    source=replace(source,'    if table is not None and key in table.entries:',
                   '    entry = table.entries.get(key) if table is not None else None\n'
                   '    if entry is not None:')
    source=replace(source,'        flag, value, move = unpack_entry(table.entries[key])',
                   '        signed, flag, hint_code = entry >> 6, entry & 3, (entry >> 2) & 15\n'
                   '        value = signed // 2 if signed & 1 == 0 else -(signed // 2) - 1\n'
                   '        move = None if hint_code == 15 else hint_code\n'
                   '        if flag == 3:\n            raise ValueError("Invalid TT entry")')
    source=replace(source,'        if flag == EXACT:', '        if flag == 0:')
    source=replace(source,'        if flag == LOWER:', '        if flag == 1:')
    source=replace(source,'        flag = UPPER if best <= alpha_original else LOWER if best >= beta_original else EXACT',
                   '        flag = 2 if best <= alpha_original else 1 if best >= beta_original else 0')
    source=replace(source,'        table.entries[key] = pack_entry(flag, best, best_move)',
                   '        # Nonterminal interior search yields integer score and legal column.\n'
                   '        signed = 2 * best if best >= 0 else -2 * best - 1\n'
                   '        table.entries[key] = (signed << 6) | (best_move << 2) | flag')
    return source
