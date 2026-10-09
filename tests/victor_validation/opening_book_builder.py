"""Offline generator and verifier for Victor's exact opening book.

  estimate  count mirror-distinct positions per ply (no solving)
  build     solve the book tiers with the native C oracle under a hard wall budget
  verify    replay every entry, check parent/child consistency and re-solve a sample

  PYTHONPATH=.:tests .venv/bin/python -m victor_validation.opening_book_builder build \
      --output games/connect4/victor/data/opening_book.json

Tiers (an entry may belong to several):

  shallow    every position with at most ``--shallow-plies`` stones (default 2)
  closure    every position where Victor is to move, reached from the empty board
             when Victor follows the book as White or as Black and the opponent
             plays ANY legal move, up to ``--max-plies`` stones (default 8)
  benchmark  early positions (at most --max-plies stones) of the frozen
             371-position suite; IN-SAMPLE, reported separately

Exactness: a position becomes an entry only if the oracle printed ``ok`` (every
root child solved by a completed terminal-only search). ``unknown`` (node limit),
a killed process (wall budget) or any error is recorded as ``unresolved`` and
never stored with a value. Closure children are expanded only from exact entries.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import date
import hashlib
import json
import platform
from random import Random
import subprocess
import sys
from time import monotonic

from games.connect4.victor.exact import Bits
from games.connect4.victor.opening_book import (
    CHAR_VALUES, SCHEMA, OpeningBook, board_key, canonical_key, entries_digest, preferred_move,
)

from .native_oracle import SOURCE, binary_path, solve_histories

TIERS = ('shallow', 'closure', 'benchmark')
TIE_BREAK_DEPTH = 4  # Victor's fallback depth in the public and benchmark profiles.


def replay(history):
    bits = Bits((0, 0), (0,) * 7, 0)
    for c in history:
        if bits.heights[c] >= 6 or bits.value is not None:
            raise ValueError(f'illegal history {history}')
        bits = bits.drop(c)
    return bits


def node_limit(plies):
    return {0: 30_000_000_000, 1: 30_000_000_000, 2: 10_000_000_000,
            3: 6_000_000_000, 4: 4_000_000_000}.get(plies, 2_000_000_000)


class Builder:
    def __init__(self, args):
        self.args = args
        self.deadline = monotonic() + args.wall_budget
        self.solved = {}       # (key, player) -> dict(values, value, history, sources, nodes)
        self.unresolved = {}   # (key, player) -> dict(history, reason, sources)
        self.events = []
        binary_path()  # compile once before threads start

    def remaining(self):
        return self.deadline - monotonic()

    def _solve_chunk(self, histories):
        left = self.remaining()
        if left <= 1:
            return [('unresolved', 'wall_budget_exhausted_before_start', None)] * len(histories)
        try:
            results = solve_histories(histories, node_limit=node_limit(len(histories[0])),
                                      timeout=left)
        except subprocess.TimeoutExpired:
            return [('unresolved', 'wall_budget_exhausted', None)] * len(histories)
        except Exception as exc:  # Never turn a failure into a value.
            return [('unresolved', f'error:{type(exc).__name__}', None)] * len(histories)
        return [('exact', None, r) if r.status == 'exact' else
                ('unresolved', f'oracle_{r.status}_after_{r.nodes}_nodes', None) for r in results]

    def solve(self, items, label):
        """items: [(history, source)]; solves new canonical positions in parallel."""
        start = monotonic()
        todo, seen = {}, set()
        for history, source in items:
            bits = replay(history)
            key, mirrored = canonical_key(bits)
            ident = (key, bits.turn)
            for table in (self.solved, self.unresolved):
                if ident in table:
                    table[ident]['sources'].add(source)
            if ident in self.solved or ident in self.unresolved or ident in seen:
                if ident in todo:
                    todo[ident][2].add(source)
                continue
            seen.add(ident)
            todo[ident] = (tuple(history), mirrored, {source})
        # Siblings share one oracle process (and transposition table); cost is
        # dominated by shallow positions, so those go first and alone.
        groups = {}
        for ident, (history, mirrored, sources) in todo.items():
            parent = history[:-1] if len(history) >= 5 else history
            groups.setdefault(parent, []).append((ident, history, mirrored, sources))
        chunks = sorted(groups.values(), key=lambda g: len(g[0][1]))
        with ThreadPoolExecutor(self.args.workers) as pool:
            outcomes = list(pool.map(lambda g: self._solve_chunk([h for _, h, _, _ in g]), chunks))
        nodes = exact = 0
        for group, results in zip(chunks, outcomes):
            for (ident, history, mirrored, sources), (status, reason, r) in zip(group, results):
                canon = tuple(6 - c if mirrored else c for c in history)
                if status == 'exact':
                    values = ''.join(CHAR_VALUES[r.move_values[6 - c if mirrored else c]]
                                     if (6 - c if mirrored else c) in r.move_values else 'x'
                                     for c in range(7))
                    self.solved[ident] = dict(value=CHAR_VALUES[r.value], values=values,
                                              history=canon, sources=set(sources), nodes=r.nodes)
                    nodes += r.nodes
                    exact += 1
                else:
                    self.unresolved[ident] = dict(history=canon, reason=reason, sources=set(sources))
        self.events.append(dict(level=label, new_positions=len(todo), exact=exact,
                                unresolved=len(todo) - exact, oracle_nodes=nodes,
                                seconds=round(monotonic() - start, 1)))
        print(json.dumps(self.events[-1]), flush=True)

    def entry_moves(self, history):
        """Actual-orientation move values of a solved history, or None."""
        bits = replay(history)
        key, mirrored = canonical_key(bits)
        row = self.solved.get((key, bits.turn))
        if row is None:
            return None, None
        values = {}
        for c, ch in enumerate(row['values']):
            if ch != 'x':
                values[6 - c if mirrored else c] = {'+': 1, '=': 0, '-': -1}[ch]
        order = (3, 2, 4, 1, 5, 0, 6)
        return {'+': 1, '=': 0, '-': -1}[row['value']], tuple((c, values[c]) for c in order
                                                               if c in values)

    def expand(self, h, frontier, expanded_boards):
        """Victor's book move, then every non-terminal opponent reply."""
        value, moves = self.entry_moves(h)
        if moves is None or len(h) + 2 > self.args.max_plies:
            return  # unresolved positions are never expanded
        bits = replay(h)
        if board_key(bits) in expanded_boards:
            return  # same actual board via a transposition: same choice, same children
        expanded_boards.add(board_key(bits))
        move = preferred_move(bits.board, bits.turn, moves, value, TIE_BREAK_DEPTH)
        after = bits.drop(move)
        if after.value is not None:
            return
        for reply in range(7):
            if after.heights[reply] < 6 and after.drop(reply).value is None:
                frontier.setdefault(len(h) + 2, []).append(h + (move, reply))

    def build(self):
        args = self.args
        started = monotonic()
        level = {0: [()]}  # shallow tier: all histories up to --shallow-plies
        for plies in range(1, args.shallow_plies + 1):
            level[plies] = [h + (c,) for h in level[plies - 1] for c in range(7)
                            if not replay(h).heights[c] >= 6 and replay(h + (c,)).value is None]
        bench = []
        if args.suite:
            suite = json.load(open(args.suite))
            bench = sorted({tuple(p['history']) for p in suite['positions']
                            if p['plies'] <= args.max_plies})
        # Closure frontiers: Victor-to-move histories; White starts at ply 0,
        # Black at every ply-1 position. A closure position at ply p needs its
        # ply p-2 parent solved, so plies are solved in batches [0..2], [3,4],
        # [5,6], ...; independent benchmark positions fill the first batch.
        frontier = {0: [()], 1: [(c,) for c in range(7)]}
        closure, expanded_boards = set(), set()
        batches = [tuple(range(0, min(2, args.max_plies) + 1))]
        batches += [tuple(range(p, min(p + 1, args.max_plies) + 1))
                    for p in range(3, args.max_plies + 1, 2)]
        for batch in batches:
            extra = [(h, 'shallow') for p in batch for h in level.get(p, [])]
            if batch[0] == 0:
                extra += [(h, 'benchmark') for h in bench]
            # Expansion inside a batch (ply 0 -> 2) adds closure positions that
            # the shallow tier has already solved; loop until nothing is new.
            while True:
                pending = [h for p in batch for h in frontier.get(p, []) if h not in closure]
                if not pending and extra is None:
                    break
                self.solve([(h, 'closure') for h in pending] + (extra or []),
                           f'plies {batch[0]}-{batch[-1]}')
                extra = None
                for h in pending:
                    closure.add(h)
                    self.expand(h, frontier, expanded_boards)
        return self.write(started, closure, bench)

    def write(self, started, closure, bench):
        args = self.args
        entries = sorted([format(k, 'x'), p, row['value'], row['values'],
                          ''.join(str(c + 1) for c in row['history']),
                          sum(1 << TIERS.index(s) for s in row['sources'])]
                         for (k, p), row in self.solved.items())
        unresolved = sorted([format(k, 'x'), p, ''.join(str(c + 1) for c in row['history']),
                             row['reason']] for (k, p), row in self.unresolved.items())
        by_ply = {}
        for (k, p), row in self.solved.items():
            by_ply.setdefault(len(row['history']), [0, 0])[0] += 1
        for (k, p), row in self.unresolved.items():
            by_ply.setdefault(len(row['history']), [0, 0])[1] += 1
        compiler = subprocess.run(['cc', '--version'], capture_output=True, text=True).stdout
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True,
                                text=True).stdout.strip()
        data = dict(
            schema=SCHEMA,
            description='Exact W/D/L opening book. Values are mover-relative (+ win, = draw, '
                        '- loss, x full column) in canonical orientation; every entry is a '
                        'completed independent C-oracle search. NOT a complete opening book.',
            sources=list(TIERS),
            provenance=dict(oracle=str(SOURCE.relative_to(SOURCE.parents[2])),
                            oracle_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                            compiler=compiler.splitlines()[0] if compiler else None,
                            generator='tests/victor_validation/opening_book_builder.py',
                            base_commit=commit, generated=date.today().isoformat(),
                            python=platform.python_version(), machine=platform.machine()),
            generation=dict(max_plies=args.max_plies, shallow_plies=args.shallow_plies,
                            suite=args.suite, tie_break=f'NegamaxAgent depth {TIE_BREAK_DEPTH}',
                            workers=args.workers, wall_budget_seconds=args.wall_budget,
                            node_limits={str(p): node_limit(p) for p in range(args.max_plies + 1)},
                            elapsed_seconds=round(monotonic() - started, 1),
                            oracle_nodes=sum(r['nodes'] for r in self.solved.values()),
                            closure_histories=len(closure), benchmark_histories=len(bench),
                            levels=self.events),
            completion=dict(exact=len(entries), unresolved=len(unresolved),
                            by_ply={str(k): dict(exact=v[0], unresolved=v[1])
                                    for k, v in sorted(by_ply.items())},
                            by_source={s: sum(s in r['sources'] for r in self.solved.values())
                                       for s in TIERS},
                            complete_closure=not any('closure' in r['sources']
                                                     for r in self.unresolved.values())),
            entries=entries, unresolved=unresolved,
            digest=entries_digest(entries, unresolved))
        with open(args.output, 'w') as f:
            json.dump(data, f, separators=(',', ':'))
            f.write('\n')
        print(json.dumps(dict(data['completion'], elapsed=data['generation']['elapsed_seconds'])))


def estimate(args):
    """Distinct mirror-reduced non-terminal positions per ply (enumeration only)."""
    level = {canonical_key(Bits((0, 0), (0,) * 7, 0))[0]: Bits((0, 0), (0,) * 7, 0)}
    for plies in range(args.max_plies + 1):
        print(json.dumps(dict(plies=plies, distinct=len(level))), flush=True)
        nxt = {}
        for bits in level.values():
            for c in range(7):
                if bits.heights[c] < 6:
                    child = bits.drop(c)
                    if child.value is None:
                        nxt.setdefault(canonical_key(child)[0], child)
        level = nxt


def verify(args):
    """Structural replay of every entry, minimax consistency, oracle re-solve sample."""
    data = json.load(open(args.book))
    book = OpeningBook(data)  # digest and schema
    from games.connect4.victor.position import Position
    problems, rows = [], {}
    for key, player, value, values, history, mask in data['entries']:
        bits = replay([int(c) - 1 for c in history])
        if canonical_key(bits) != (int(key, 16), False) or bits.turn != player:
            problems.append(('key', key))
        hit = book.lookup(Position.from_board(bits.board, player))
        if hit.status != 'exact':
            problems.append(('lookup', key, hit.status))
        mirrored = replay([7 - int(c) for c in history])
        mhit = book.lookup(Position.from_board(mirrored.board, player))
        if mhit.status != 'exact' or {(6 - c, v) for c, v in mhit.move_values} != set(hit.move_values):
            problems.append(('mirror', key))
        rows[(int(key, 16), player)] = (bits, value, values)
    # Minimax consistency wherever a child of an entry is itself an entry.
    linked = 0
    for (key, player), (bits, value, values) in rows.items():
        for c, ch in enumerate(values):
            if ch == 'x':
                continue
            child = bits.drop(c)
            if child.value is not None:
                continue
            ckey, _ = canonical_key(child)
            other = rows.get((ckey, child.turn))
            if other is not None:
                linked += 1
                if {'+': 1, '=': 0, '-': -1}[other[1]] != -{'+': 1, '=': 0, '-': -1}[ch]:
                    problems.append(('minimax', key, c))
    sample = Random(args.seed).sample(data['entries'], min(args.sample, len(data['entries'])))
    sample = [r for r in sample if len(r[4]) >= args.min_resolve_plies]
    start = monotonic()
    resolved = solve_histories([[int(c) - 1 for c in r[4]] for r in sample],
                               node_limit=args.node_limit)
    mismatched = unknown = 0
    for row, r in zip(sample, resolved):
        if r.status != 'exact':
            unknown += 1
            continue
        values = ''.join(CHAR_VALUES[r.move_values[c]] if c in r.move_values else 'x'
                         for c in range(7))
        if values != row[3] or CHAR_VALUES[r.value] != row[2]:
            mismatched += 1
            problems.append(('oracle', row[0]))
    out = dict(entries=len(rows), structural_and_mirror_ok=not any(
                   p[0] in ('key', 'lookup', 'mirror') for p in problems),
               minimax_links_checked=linked, resolved_sample=len(sample),
               resolved_unknown=unknown, resolved_mismatched=mismatched,
               resolve_seconds=round(monotonic() - start, 1), problems=problems[:20])
    print(json.dumps(out))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    e = sub.add_parser('estimate')
    e.add_argument('--max-plies', type=int, default=8)
    b = sub.add_parser('build')
    b.add_argument('--output', required=True)
    b.add_argument('--max-plies', type=int, default=8)
    b.add_argument('--shallow-plies', type=int, default=2)
    b.add_argument('--suite', default='docs/victor-performance/suite.json')
    b.add_argument('--workers', type=int, default=8)
    b.add_argument('--wall-budget', type=float, default=1200.0,
                   help='hard wall-clock limit in seconds; unfinished work stays unresolved')
    v = sub.add_parser('verify')
    v.add_argument('--book', default='games/connect4/victor/data/opening_book.json')
    v.add_argument('--sample', type=int, default=300)
    v.add_argument('--min-resolve-plies', type=int, default=0)
    v.add_argument('--seed', type=int, default=1)
    v.add_argument('--node-limit', type=int, default=30_000_000_000)
    args = parser.parse_args(argv)
    if args.command == 'estimate':
        estimate(args)
    elif args.command == 'build':
        Builder(args).build()
    else:
        sys.exit(1 if verify(args)['problems'] else 0)


if __name__ == '__main__':
    main()
