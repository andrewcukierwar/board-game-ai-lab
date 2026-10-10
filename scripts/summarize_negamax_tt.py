"""Render complete, separate TT experiment comparisons from audited raw evidence."""
import json
from pathlib import Path
import statistics

from scripts import benchmark_negamax_tt as bench

ROOT = bench.ROOT
MIB = 1048576


def load_rows(directory, experiment):
    return [json.loads(line) for line in (directory / f'{experiment}-results.jsonl').read_text().splitlines()]


def table(headers, rows):
    lines = ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
    return '\n'.join(lines + ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def retained(data):
    return data['memory'][0]['retained']['total_bytes']


def peak(data):
    return max(m['traced_peak_bytes'] for m in data['memory'])


def by_condition(rows):
    return {(r['position'], r['depth']): r for r in rows}


def performance_matrix(rows, variant, reference):
    lookup = by_condition(rows)
    positions = list(dict.fromkeys(r['position'] for r in rows))
    output = []
    for name in positions:
        cells = []
        for depth in (4, 6, 8, 10):
            v = lookup[name, depth]['variants']
            old, new = v[reference]['median_wall_seconds'] * 1000, v[variant]['median_wall_seconds'] * 1000
            cells.append(f'{old:.3f} → {new:.3f} / {new/old:.3f}×')
        output.append([name, *cells])
    return table(['Position', 'D4', 'D6', 'D8', 'D10'], output)


def summarize(directory=ROOT):
    a, b = load_rows(directory, 'A'), load_rows(directory, 'B')
    mixed = directory / 'mixed-index'
    b2 = load_rows(mixed, 'B')
    aa = json.loads((directory / 'A-analysis.json').read_text())
    ba = json.loads((directory / 'B-analysis.json').read_text())
    b2a = json.loads((mixed / 'B-analysis.json').read_text())
    assert aa['selected'] == 'packed-entry' and ba['selected'] is None and b2a['selected'] is None
    assert json.loads((directory / 'audit.json').read_text())['all_checks_passed']
    assert json.loads((mixed / 'audit.json').read_text())['all_checks_passed']
    fixture_config = json.loads((directory / 'manifest.json').read_text())
    tail = by_condition(a)['post-hoc-tail', 10]['variants']
    baseline, compact = tail['baseline'], tail['packed-entry']
    output = []
    add = output.append
    add('''# Phase 3B.2: Negamax transposition-table storage

October 9, 2026. Laptop repository `andrewcukierwar/board-game-ai-lab`, branch
`search/negamax-v2`. Immutable baseline:
[f2f57b46c7817bb7324f4390da0bb345ae2fc7e4](https://github.com/andrewcukierwar/board-game-ai-lab/commit/f2f57b46c7817bb7324f4390da0bb345ae2fc7e4).

**Accept packed full-identity keys and packed integer entries. Keep public
storage unbounded.** Compact storage meets the declared memory/latency criteria.
Both bounded replacement strategies preserve exact decisions but fail default
eligibility. A and B are independent comparisons; their latency samples and
references are never pooled. No public depth/preset, MCTS, API, frontend,
production environment, training worktree or previous evidence changes.

## Architecture and correctness

Before: fresh `SearchTable` per complete decision; dict keyed by the tuple
`(X_bits, O_bits, mover, remaining_depth)` and storing `(flag, score, hint)`.
One shared table serves every legal root move. Normal choose_move releases it.
After: the same dict, table lifecycle, search algorithm and incremental evaluator,
with a single integer key and integer value. SearchState, terminal logic,
ordering, pruning, root loop, public interfaces and last_stats fields are unchanged.
The production diff consists of the key expression, entry encode/decode and an
import. The selected production bodies match the Git-derived measured variant.

Key, with arbitrary-precision nonnegative remaining depth d:

```python
key = X | (O << 49) | (mover << 98) | (d << 99)
```

Seven columns × seven bits occupy addresses 0..48; playable cells use at most
bit 47 and sentinels remain empty. Both bitboards fit below 2**49. The fields
are disjoint: masks recover X and O, bit 98 recovers mover, and `key >> 99`
recovers the entire depth. This proves injectivity for every valid internal
board (and even arbitrary 49-bit field combinations), both movers and any
nonnegative depth. No truncated/Zobrist hash or symmetry canonicalization is
used as identity. Python hash collisions remain possible; dictionary/full-slot
key equality prevents false hits. Tests force a genuine Python hash collision
using distinct depths separated by `sys.hash_info.modulus`.

Entry:

```python
z = 2 * score if score >= 0 else -2 * score - 1
entry = (z << 6) | (hint_code << 2) | flag_code
```

EXACT/LOWER/UPPER map to 0/1/2; flag code 3 is rejected. Legal hints 0..6
retain their values; 15 represents None. Invalid codes 7..14 round-trip and
are ignored by existing legal-hint validation. Unsupported encoder hints
(including 99) are rejected; the ordering function still ignores arbitrary
invalid hints such as 99. Round trips cover all supported hints, flags and
signed scores, including ±(1,000,000 + 10**100). Zigzag is a bijection from
signed integers to nonnegative integers. Arbitrary precision avoids overflow
for every supported positive search depth; Victor's int8 terminal-only WDL
representation is never reused. Packed value zero is a valid entry, not absence.

Terminal precedence, fast-win/slower-win scores and draw semantics are exact.
Alpha/beta bounds are classified using the original pre-probe window exactly
as before. A bound remains a bound; its move is only an ordering hint. Depth
and mover stay in the full key. Full-window searches complete every root move,
including alternatives to an immediate win, and CENTER_ORDER insertion preserves
historical root ties. Exceptions restore every detached state field, including
incremental window counts, score and history, and leave the caller unchanged.

Independent array minimax verifies quiet/tactical boards, wins, defenses,
near-full boards and draws at tractable depths. Tests cover all bound types,
narrow-window reuse, transpositions, multiple depths/movers, absent/invalid/full
column hints, huge signed values, collisions and forced one-slot eviction.
The identity test visits every legal opening history through five plies plus
200 seeded legal play/undo trajectories, testing more than 100,000 distinct
board/mover/depth identities. This complements the algebraic field-separation
proof; finite collision tests alone are not a proof over all boards.

## Design, provenance and measurement

[DESIGN.md](DESIGN.md) was written before measurements. A uses all 20 immutable
Phase 3B.1 histories: eleven Phase 3A quiet/tactical/endgame boards, eight seeded
boards, and the separately identified post hoc tail `[1,4,6,0,6]`. Four additional
nonterminal boards at 10/14/18/22 plies use seed 20261010 with terminal rejection
only. No fixture selection examines optimization outcomes. All 24 boards run
at depths 4/6/8/10 (96 conditions); the tail is excluded from acceptance aggregates.

A acceptance: exact scores/moves/counters/leaves; ≥30% summed reachable-TT saving
on non-tail conditions with ≥2,000 baseline entries; wall geometric and summed
median ratios ≤1.08; per-condition ratio ≤1.20 or added ≤0.25 ms; traced peak
growth ≤5% or ≤32 KiB. Select the eligible variant saving most retained memory.
The tail also satisfies the per-condition latency/peak limits.

B acceptance: ≥30% summed reachable saving where the non-tail reference exceeds
16,384 entries; wall geometric and summed ratios ≤1.05; per-condition ratio
≤1.20 or added ≤0.25 ms; every expensive reference (≥50 ms, including tail)
latency ≤1.20× and nodes ≤1.50×; peak growth ≤5% or ≤32 KiB. If no budget passes,
public behavior remains unbounded. These are fixed engineering thresholds,
not statistical population claims or production endpoint percentiles.

The baseline is executed directly from Git's immutable object database.
Assertion-checked TT-only source substitutions generate A variants; all three
complete sources are saved beside the manifest. Source/design/runtime/fixture
hashes freeze before each run and drift/overwrite/result mismatches fail closed.
64 table constructions per implementation stabilize CPython split instance
dictionaries, including fresh workers. Seven complete unwrapped choose_move
wall/CPU samples follow one discarded warmup, with rotating variant order and
fresh tables. Ordinary timings include creation, all root work and table release.
Game creation, instrumentation, retained sizing and fresh-worker launch are outside.
A separately counted run verifies heuristic leaves; two separate tracemalloc
runs retain the table diagnostically. One fresh subprocess per condition/variant
reports before/retained RSS and RUSAGE_SELF high water. macOS ru_maxrss is bytes;
Linux is normalized from KiB. All saved decisions are checked, including their
own deterministic counters for bounded variants.

Reachable retained bytes count unique Python objects from the SearchTable itself,
its attribute dict, TT storage and referenced keys/entries. Shared cached integers
and strings count once: traverse each dictionary record's key before value,
then surrounding table attributes. Classes, module globals/code and unrelated
interpreter objects are outside this logical instance graph. `traced_current`
and `traced_peak` measure Python allocation during the diagnostic decision;
shared preexisting objects/import allocations need not appear there. RSS and
high water include the interpreter, imports and allocator; they are not TT bytes.
The harness imports common helper metadata for every variant, so fresh RSS does
not separately estimate a production helper-import cost. No RSS ceiling follows
from bounded entry count or traced peak. Normal search releases the table; retained
measurements are deliberate diagnostics, not evidence of a leak.

The process was not isolated from other laptop activity. An extended correctness
run briefly overlapped A; no samples were selectively discarded. Rotating matched
samples, CPU measurements and variability evidence describe that limitation.
B1 and B2 ran serially without backend tests overlapping them. See manifest/runtime
metadata for Python, platform, CPU count, host load and hardware; results can vary
with interpreter, allocator, frequency and scheduling.

The first preflight failed solely because JSON converted tuple score pairs to
lists. [serialization-preflight/](serialization-preflight/) preserves the original
harness, declaration, source snapshots and failure log (no completed result row).
The corrected harness uses JSON-native ordered pairs and has a round-trip regression
test. The same design/fixtures/thresholds were redeclared before the full run.
''')
    add('## Experiment A — unbounded storage')
    summary = []
    for name, c in aa['comparisons'].items():
        summary.append([name, f"{(1-c['retained_large_ratio'])*100:.1f}%",
            f"{(c['geometric_wall_ratio']-1)*100:+.2f}%", f"{(c['summed_wall_ratio']-1)*100:+.2f}%",
            f"{(c['geometric_cpu_ratio']-1)*100:+.2f}%", f"{(c['worst_wall_ratio']-1)*100:+.2f}%", 'yes'])
    add(table(['Variant', 'Retained saving (28 large conditions)', 'Wall geometric overhead',
               'Summed median overhead', 'CPU geometric overhead', 'Worst non-tail overhead', 'Eligible'], summary))
    add('Packed keys alone are acceptable but save less. The combined entry encoding is selected by the predeclared memory rule. Its measurable CPU cost is an accepted tradeoff, not a speedup claim. The exact incremental evaluator from Phase 3B.1 remains unchanged; the earlier 1.83× evaluation speedup is preserved structurally rather than re-estimated here.')
    summary = []
    for depth in (4, 6, 8, 10):
        rows = [r for r in a if r['depth'] == depth and r['group'] != 'post-hoc-diagnostic']
        summary.append([depth, *[f"{sum(r['variants'][n]['median_wall_seconds'] for r in rows)*1000:.3f}" for n in bench.VARIANTS],
            f"{aa['comparisons']['packed-entry']['depth_wall_ratios'][str(depth)]:.4f}×"])
    add(table(['Depth', 'Baseline summed wall ms', 'Packed key ms', 'Packed entry ms', 'Selected geometric ratio'], summary))
    for name in ('packed-key', 'packed-entry'):
        add(f'### {name}: all per-position/depth latency comparisons')
        add('Every cell is baseline → variant milliseconds / slowdown ratio. Ratios below one are faster. All regressions, including tiny/endgame and tail rows, remain visible.')
        add(performance_matrix(a, name, 'baseline'))
    add('### Real TT populations, memory and process measurements')
    rows = []
    for r in a:
        if r['depth'] != 10:
            continue
        v = r['variants']
        rows.append([r['position'], v['baseline']['counted']['stats']['entries'],
            f"{retained(v['baseline'])/MIB:.3f} → {retained(v['packed-key'])/MIB:.3f} → {retained(v['packed-entry'])/MIB:.3f}",
            f"{peak(v['baseline'])/MIB:.3f} → {peak(v['packed-entry'])/MIB:.3f}",
            f"{v['baseline']['rss']['rss_retained_bytes']/MIB:.2f} → {v['packed-entry']['rss']['rss_retained_bytes']/MIB:.2f}",
            f"{v['baseline']['rss']['high_water_bytes']/MIB:.2f} → {v['packed-entry']['rss']['high_water_bytes']/MIB:.2f}"])
    add(table(['D10 position', 'Entries (all identical)', 'TT MiB baseline → key → entry', 'Traced peak MiB before → after',
               'Retained process RSS MiB before → after', 'Process high water MiB before → after'], rows))
    add('All depths, both trace repetitions, traced-current bytes, RSS-before measurements and raw timings are in [A-results.jsonl](A-results.jsonl) and [performance.csv](performance.csv). These real-population savings are not extrapolations from single shallow entry objects.')
    categories = sorted(set().union(*(v['memory'][0]['retained']['categories'] for v in tail.values())))
    add(table(['Tail retained category', 'Tuple baseline bytes', 'Packed key bytes', 'Packed entry bytes'],
        [[category, *[tail[n]['memory'][0]['retained']['categories'].get(category, 0) for n in bench.VARIANTS]]
         for category in categories]))
    layout = next(r for r in json.loads((directory / 'dictionary-layout.json').read_text())
                  if r['position'] == 'post-hoc-tail' and r['depth'] == 10 and r['variant'] == 'baseline')
    add(f"The tail dictionary itself is {layout['allocated_bytes']:,} bytes in all three variants: "
        f"{layout['headers_bytes']} bytes of headers, {layout['hash_index_bytes']:,} bytes of hash indexes, "
        f"{layout['occupied_dense_bytes']:,} bytes of occupied dense records and {layout['spare_dense_bytes']:,} "
        f"bytes of unused dense-record reserve ({layout['dense_capacity']:,} available records / "
        f"{layout['hash_index_slots']:,} sparse slots). Capacity is inferred from the exact CPython 3.11 "
        '64-bit general-key table size formula and checked against measured sys.getsizeof for every A table. '
        'Local pycore_dict.h establishes the header/record/index widths; a regression test checks several '
        'resize boundaries. There are no deletions in A, so dense-record count equals occupancy. The hash '
        'index is metadata for the entire allocated table; it is not counted again as extra unused bytes. '
        '[dictionary-layout.json](dictionary-layout.json) preserves every inference. Shared scalars explain '
        'why score/hint/category totals cannot be obtained by multiplying shallow sizes by entry count. '
        'SearchTable object and calibrated attribute dictionary are included explicitly.')
    add(f"Tail bytes per entry fall from {retained(baseline)/baseline['counted']['stats']['entries']:.1f} "
        f"to {retained(compact)/compact['counted']['stats']['entries']:.1f}. Retained TT "
        f"{retained(baseline)/MIB:.2f} → {retained(compact)/MIB:.2f} MiB "
        f"({(1-retained(compact)/retained(baseline))*100:.1f}% saving); traced peak "
        f"{peak(baseline)/MIB:.2f} → {peak(compact)/MIB:.2f} MiB. Median tail wall "
        f"{baseline['median_wall_seconds']*1000:.1f} → {compact['median_wall_seconds']*1000:.1f} ms "
        f"({(compact['median_wall_seconds']/baseline['median_wall_seconds']-1)*100:+.1f}%). "
        'Do not compare this matched result causally with the previous phase’s 723.4 ms from another run. '
        'All tail variants retain 423,857 nodes, 119,958 entries, 47,944 hits, 93,664 loop cutoffs and '
        '236,149 heuristic leaves, with move 4 and best heuristic score −4. This is not a solved forced loss.')
    micros = json.loads((directory / 'A-microbenchmark.json').read_text())
    micro_summary = []
    for name in ('packed-key', 'packed-entry'):
        for kind in ('creation', 'lookup'):
            ratios = []
            for row in micros:
                if row['position'] == 'post-hoc-tail':
                    continue
                data = row['variants']
                ratios.append(statistics.median(s[kind]['wall_seconds'] for s in data[name]) /
                              statistics.median(s[kind]['wall_seconds'] for s in data['baseline']))
            micro_summary.append([name, kind, f'{bench.geometric(ratios):.3f}×'])
    add('### Microbenchmarks')
    add(table(['Variant', 'Loop', 'Geometric wall ratio to tuple baseline'], micro_summary))
    add('[A-microbenchmark.json](A-microbenchmark.json) records seven rotating samples of 20,000 operations on real depth-six TT identities per fixture, after a warmup. The creation loop includes key construction plus hash/checksum; the successful lookup loop includes membership and retrieval, without entry decoding. Common loop/index overhead remains. These are explanatory microbenchmarks, not pure instruction timings or acceptance metrics; complete choose_move timings decide acceptance.')
    add('## Experiment B — independent deterministic replacement')
    add('''B starts only after A completion and uses accepted packed keys/entries for its
unbounded reference. Parallel fixed-size key/value lists implement direct mapping.
A probe compares the complete arbitrary-width stored key before returning its
entry. Different-key writes evict; same-key writes count as replacements. Every
store rechecks its slot after recursive descendants may have replaced it; bounds
are overwritten with the current complete entry, never merged with a foreign key.
Misses resume normal search, without budget exceptions, unknown values or partial
root results. Capacities are slot counts and occupancy may be lower.

B1 was declared initially: `hash(full_key) % capacity`. B2 is a separate,
supplemental strategy declared in [mixed-index/DESIGN.md](mixed-index/DESIGN.md)
after interim B1 opening results exposed clustering. It changes only indexing to
the high log2(capacity) bits of `(hash(key) * 11400714819323198485) & (2**64-1)`.
Truncation chooses a slot; identity is never truncated. Full root equality and
all original thresholds/fixtures/samples remain mandatory. B2 does not retroactively
replace B1 evidence or enter A acceptance. The supplemental manifest freezes its
runner/design/input hashes before any B2 measurements. Python integer hashing
is deterministic on this recorded runtime.

Both strategies add Python get/store dispatch and, for B2, mixing arithmetic.
Measured total cost therefore includes dispatch/indexing/allocation/release as
well as work caused by losing cache entries. Occupancy, evictions and extra nodes
separately quantify cache effects; ratios do not isolate eviction CPU alone.
''')
    summary = []
    for label, analysis in [('B1 modulo', ba), ('B2 mixed', b2a)]:
        for capacity, c in analysis['comparisons'].items():
            summary.append([label, capacity, f"{c['geometric_wall_ratio']:.3f}×", f"{c['summed_wall_ratio']:.3f}×",
                f"{c['geometric_cpu_ratio']:.3f}×", f"{(1-c['retained_large_ratio'])*100:.1f}%",
                ', '.join(k for k, ok in c['checks'].items() if not ok), 'no'])
    add(table(['Strategy', 'Capacity', 'Wall geometric ratio', 'Summed median ratio', 'CPU ratio',
               'Retained saving (large reference)', 'Failed criteria', 'Default eligible'], summary))
    for label, rows, analysis, path in [('B1 modulo', b, ba, ''), ('B2 mixed', b2, b2a, 'mixed-index/')]:
        add(f'### {label}: per-depth aggregate ratios')
        add(table(['Capacity', 'D4', 'D6', 'D8', 'D10'], [[capacity, *[
            f"{c['depth_wall_ratios'][str(d)]:.3f}×" for d in (4, 6, 8, 10)]]
            for capacity, c in analysis['comparisons'].items()]))
        add(f'### {label}: expensive D10 search behavior')
        selected_names = ('empty', 'near-opening', 'midgame-wide', 'supplement-main-003', 'seeded-04', 'post-hoc-tail')
        rows10 = [r for r in rows if r['depth'] == 10 and r['position'] in selected_names]
        probes = by_condition([json.loads(line) for line in (directory / path / 'B-probe-rates.jsonl').read_text().splitlines()])
        data_rows = []
        for row in rows10:
            ref = row['variants']['unbounded']
            for name, data in row['variants'].items():
                s, st = data['counted']['stats'], data['counted']['storage']
                data_rows.append([row['position'], name, f"{data['median_wall_seconds']*1000:.1f}",
                    f"{data['median_wall_seconds']/ref['median_wall_seconds']:.3f}×",
                    s['entries'], st['evictions'], st['replacements'], s['nodes'],
                    f"{s['nodes']/ref['counted']['stats']['nodes']:.3f}×",
                    s['nodes'] - ref['counted']['stats']['nodes'],
                    f"{100*probes[row['position'], row['depth']]['variants'][name]['hit_per_probe']:.2f}%",
                    f"{retained(data)/MIB:.3f}", f"{peak(data)/MIB:.3f}",
                    f"{data['rss']['rss_retained_bytes']/MIB:.2f}", f"{data['rss']['high_water_bytes']/MIB:.2f}"])
        add(table(['Position', 'Slots', 'Wall ms', 'Wall ratio', 'Occupancy', 'Evictions', 'Same-key replacements',
            'Nodes', 'Node ratio', 'Extra nodes', 'Hits/probe', 'TT MiB', 'Peak MiB', 'RSS MiB', 'High water MiB'], data_rows))
        worst = max(((row, name, data) for row in rows for name, data in row['variants'].items()
                    if name != 'unbounded'), key=lambda item: item[2]['median_wall_seconds'])
        max_sample = max((sample['wall_seconds'], row['position'], row['depth'], name)
            for row in rows for name, data in row['variants'].items() if name != 'unbounded'
            for sample in data['samples'])
        add(f"Largest observed {label} bounded median: {worst[2]['median_wall_seconds']*1000:.1f} ms "
            f"at {worst[0]['position']} D{worst[0]['depth']} / {worst[1]} slots. Largest individual "
            f"timed sample: {max_sample[0]*1000:.1f} ms at {max_sample[1]} D{max_sample[2]} / "
            f"{max_sample[3]} slots. These are observed maxima, not worst-case guarantees.")
        for capacity in ('16384', '32768', '65536'):
            add(f'### {label}, {capacity} slots: all position/depth latencies')
            add(performance_matrix(rows, capacity, 'unbounded'))
        add(f'Complete occupancy, eviction/replacement counts, extra nodes, all counters, leaves, retained/current/peak memory, before/after RSS and high water appear in [{path}B-results.jsonl]({path}B-results.jsonl) and [{path}performance.csv]({path}performance.csv).')
    add('''Actual cache-hit rates use hits divided by TT probes, excluding terminal nodes and heuristic leaves. Separate instrumented runs in A/B-probe-rates.jsonl and mixed-index/B-probe-rates.jsonl reproduce the original vectors/counters/leaves; their timings never enter medians. CSV also reports the simpler hits/nodes ratio. [index-distribution.json](index-distribution.json) maps the complete unbounded D10 populations to each index scheme/capacity, retaining occupied-slot and maximum collision-bucket counts. This is a diagnostic of slot distribution, not a replacement for the measured bounded search.

**Keep bounded replacement experimental; do not enable it publicly.** B1's
low-bit clustering causes severe underoccupancy and repeated work; its result
must not be treated as the inherent cost of all bounded tables. B2 tests a better
distribution independently, but still fails the unchanged combined criteria.
Fixed slot lists also preallocate memory and impose construction/release costs
on tiny trees. A 65,536-slot table can allocate more than an unbounded tiny TT.
Smaller budgets trade memory for eviction/recomputation; a larger budget can
save too little or retain dispatch costs. Neither policy establishes a process
memory ceiling. Default storage remains an unbounded compact dictionary.

## Validation and limitations
''')
    backend = (directory / 'backend-tests.txt').read_text()
    summary_line = next(line for line in reversed(backend.splitlines()) if 'passed' in line)
    add(f'Complete backend result: **{summary_line.strip()}**. [backend-tests.txt](backend-tests.txt) preserves the full output. The suite includes Negamax/array oracle and incremental evaluation, MCTS, public API validation, concurrency gates, rate limits, history/replay, and evidence/provenance contracts. PyTorch-dependent skips retain the existing unavailable-dependency limitation.')
    audit = json.loads((directory / 'audit.json').read_text())
    add(f"[audit.json](audit.json) checks all {audit['saved_decisions_checked']:,} saved A/B1 decisions, "
        f"the {audit['previous_phase_vectors_checked']} reused prior-phase score/move/counter/leaf vectors, "
        'complete condition coverage, exact self-counter determinism, two-repeat retained sizes, capacities '
        'and source/hash/AST integrity. [mixed-index/audit.json](mixed-index/audit.json) independently '
        'checks B2 against the same copied immutable A input and selected production bodies. '
        'All 96 integrated-production conditions match selected A scores, moves, counters and leaves. '
        'The primary runners save 11,616 distinct decisions across A/B1/B2 (7,392 timed plus 4,224 diagnostic), '
        'excluding warmups/reference setup/microbenchmarks. The copied A input under mixed-index is '
        'not a second A experiment and is never double-counted in conclusions.')
    add('''Backend runs use dotenv disabled, explanations disabled and an empty API key;
the global fixture forbids live HTTPS provider calls. No paid/provider calls,
production operations, merges or deployments occurred. Frontend/shared public
interfaces are unchanged, so frontend tests were not run. Python compilation
and Git whitespace checks pass. Earlier completed phase evidence is untouched.

These are fixed-depth laptop engineering results, with finite boards/samples,
one Python/runtime and one fresh RSS observation per condition. No strength,
endpoint throughput, concurrent RSS limit or worst-case latency claim follows.
Finite oracle depths are tractable; deeper comparisons use the pinned validated
algorithm plus exact vector/counter parity. B2 is explicitly a supplemental,
interim-evidence-motivated experiment, not an initially preregistered strategy.

## Reproduction

Use a new output directory; runners refuse evidence overwrite. Preserve the
immutable baseline commit in Git. Full source snapshots/hashes are recorded.

```sh
mkdir -p /tmp/negamax-tt-replay/mixed-index
cp docs/search-negamax-v2/transposition-table/DESIGN.md /tmp/negamax-tt-replay/
cp docs/search-negamax-v2/transposition-table/mixed-index/DESIGN.md /tmp/negamax-tt-replay/mixed-index/
PYTHON_DOTENV_DISABLED=1 EXPLANATIONS_ENABLED=false OPENAI_API_KEY= .venv/bin/python -m scripts.benchmark_negamax_tt declare --directory /tmp/negamax-tt-replay
PYTHON_DOTENV_DISABLED=1 EXPLANATIONS_ENABLED=false OPENAI_API_KEY= .venv/bin/python -m scripts.benchmark_negamax_tt A --directory /tmp/negamax-tt-replay
PYTHON_DOTENV_DISABLED=1 EXPLANATIONS_ENABLED=false OPENAI_API_KEY= .venv/bin/python -m scripts.benchmark_negamax_tt B --directory /tmp/negamax-tt-replay
PYTHON_DOTENV_DISABLED=1 EXPLANATIONS_ENABLED=false OPENAI_API_KEY= .venv/bin/python -m scripts.benchmark_negamax_tt_mixed declare --directory /tmp/negamax-tt-replay/mixed-index --parent /tmp/negamax-tt-replay
PYTHON_DOTENV_DISABLED=1 EXPLANATIONS_ENABLED=false OPENAI_API_KEY= .venv/bin/python -m scripts.benchmark_negamax_tt_mixed run --directory /tmp/negamax-tt-replay/mixed-index
PYTHON_DOTENV_DISABLED=1 EXPLANATIONS_ENABLED=false OPENAI_API_KEY= .venv/bin/python -m scripts.diagnose_negamax_tt --directory /tmp/negamax-tt-replay
PYTHON_DOTENV_DISABLED=1 EXPLANATIONS_ENABLED=false OPENAI_API_KEY= .venv/bin/python -m scripts.audit_negamax_tt --directory /tmp/negamax-tt-replay
PYTHON_DOTENV_DISABLED=1 EXPLANATIONS_ENABLED=false OPENAI_API_KEY= .venv/bin/python -m scripts.audit_negamax_tt --directory /tmp/negamax-tt-replay/mixed-index
PYTHON_DOTENV_DISABLED=1 EXPLANATIONS_ENABLED=false OPENAI_API_KEY= .venv/bin/python -m pytest tests -q -rs
```

RSS uses read-only process inspection (`ps`) and may need a local sandbox
exception. No service or network access is needed for benchmarking. Reports/CSV
are generated from complete saved evidence; no result-dependent fixture filters.
`python -m scripts.summarize_negamax_tt` regenerates this report after audit and
backend log creation. [timing-stability.json](timing-stability.json) retains
sample variability; [runtime-end.json](runtime-end.json) records host metadata.

## Recommended Phase 3B.3

Test move-only cross-depth hints and iterative deepening as separate ablations.
A hint cache may omit depth but must include both full boards and mover, validate
legality, and store only a move: it must never supply another depth's score or
bound. Keep score TT identity depth-sensitive. Compare direct fixed-depth search,
move-only hints, iterative deepening, then their combination on the same frozen
quiet/tactical/endgame distribution and separate tail. Continue all root moves
to exact scores at the requested final depth and preserve historical ties.
Predeclare complete-decision latency thresholds, ordering/node/leaf changes,
memory cost and exception rollback; oracle/vector correctness and the full backend
suite remain mandatory. Do not change depth limits or public agent settings.

Published phase commit: [search/negamax-v2 head](https://github.com/andrewcukierwar/board-game-ai-lab/commit/search/negamax-v2).
The final delivery records the immutable commit SHA and verified remote equality;
this branch link can move with subsequent phases.
''')
    (directory / 'REPORT.md').write_text('\n\n'.join(output) + '\n')


if __name__ == '__main__':
    summarize()
