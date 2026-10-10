"""Render the report from audited complete evidence, without selecting favorable rows."""
import argparse
import json
from pathlib import Path
import re
import statistics

from scripts import benchmark_negamax_iterative as bench
from scripts.negamax_iterative_variants import VARIANTS, BASELINE


def generate(directory):
    config = json.loads((directory / 'manifest.json').read_text())
    bench.check_source(config, directory)
    audit = json.loads((directory / 'audit.json').read_text())
    for p, expected in audit['evidence_hashes'].items():
        assert bench.digest((directory / p).read_bytes()) == expected, p
    rows = [json.loads(l) for l in (directory / 'results.jsonl').read_text().splitlines()]
    analysis = json.loads((directory / 'analysis.json').read_text())
    assert analysis == bench.analyze(rows)
    assert analysis['selected'] == 'direct', 'An eligible candidate requires integration before reporting'
    cross = json.loads((directory / 'cross-hint-effectiveness.json').read_text())
    assert cross['script_sha256'] == bench.digest(Path('scripts/diagnose_negamax_iterative.py').read_bytes())
    for v, expected in cross['generated_source_hashes'].items():
        assert bench.digest((directory / f'{v}-cross-diagnostic.py').read_bytes()) == expected
    preflight = json.loads((directory / 'preflight.json').read_text())
    end = json.loads((directory / 'runtime-end.json').read_text())
    d12 = json.loads((directory / 'depth12-preflight.json').read_text())
    rss = json.loads((directory / 'rss.json').read_text())
    non_tail = [r for r in rows if r['group'] != 'post-hoc-diagnostic']
    tests = (directory / 'backend-tests.txt').read_text()
    assert not re.search(r'\b[1-9]\d* failed\b', tests), 'Backend failures must be resolved'
    test_result = re.findall(r'\d+ passed[^\n]*', tests)[-1]
    lines = []

    def add(value=''):
        lines.extend(value.splitlines())
        lines.append('')

    def table(headers, data):
        add('| ' + ' | '.join(headers) + ' |\n| ' + ' | '.join('---' for _ in headers) + ' |\n' +
            '\n'.join('| ' + ' | '.join(str(x) for x in row) + ' |' for row in data))

    def ms(r, v):
        return r['variants'][v]['median_wall_seconds'] * 1000

    def ratio(r, v):
        return ms(r, 'direct') / ms(r, v)

    add('# Phase 3B.3: iterative deepening and cross-depth move-only hints')
    add(f'October 9, 2026. Existing laptop checkout, branch `search/negamax-v2`; immutable baseline '
        f'[{BASELINE}](https://github.com/andrewcukierwar/board-game-ai-lab/commit/{BASELINE}).')
    add('**Reject B/hints, C/iterative and D/combined as production defaults. Keep A/direct unchanged.** '
        'Cross-depth hints reduce work on several expensive trees, including the prior tail, '
        'but do not meet the predeclared distribution, regression and memory requirements. '
        'Iterative deepening without reuse adds work and gives identical final-iteration counters. '
        'The experiments remain research scripts; no production module imports them.')
    add('## Design and implementation')
    add('[DESIGN.md](DESIGN.md) and [manifest.json](manifest.json) were frozen before performance measurement. '
        'A runs the byte-identical baseline source loaded with `git show`. B searches target−2, then target; '
        'C searches 2/4/…/target without cross-depth hints; D uses that schedule with hints from its '
        'immediately preceding completed iteration. Odd targets use 1/3/…/target; targets ≤2 have one '
        'iteration. Two-ply increments align horizon parity and limit preparatory overhead. '
        'No schedules were tuned after observing results.')
    add('B/D export a new dictionary of **full X board, full O board and mover → integer column only** '
        'from the previous TT. The packed identity is `X | (O << 49) | (mover << 98)`. '
        'Export decodes only the move field and discards every score, bound and horizon. Both exact and '
        'bound entries may suggest a searched move. The next score TT is fresh and remains keyed by '
        'full identity plus remaining depth; its EXACT/LOWER/UPPER interpretation and original-window '
        'classification are unchanged. No previous horizon can return a value or close a window. '
        'A same-depth TT move has priority over a cross-depth suggestion; immediate wins remain first. '
        'Strict integer/range/non-full-column checks reject invalid hints. Replacing snapshots avoids '
        'accumulating older horizons. Export and release costs are included in decision timings.')
    add('All legal root moves are searched with full windows in historical `(3,2,4,1,5,0,6)` order '
        'at every iteration, preserving exact vectors and center-first root ties. No anytime returns, '
        'aspiration, new pruning, skipped roots, time limits or changed final evaluations were introduced. '
        'Terminal-before-leaf ±(1,000,000 + remaining depth), the exact incremental 69-window heuristic, '
        'play/undo and exception restoration retain the baseline bodies. Generated source snapshots '
        'for all normal and diagnostic variants are included here.')
    add('Victor inspection: `victor/exact.py` uses terminal-only WDL, game-rule forced-move pruning, '
        'mirror identities and bounds; the native solver also uses bounded proof intervals. '
        'Its bitboard geometry informs the existing immediate-win ordering. WDL score reuse, proof '
        'pruning and budget interruptions are incompatible with this heuristic horizon contract '
        'and were not transferred.')
    add('## Distribution, timing and resource preflight')
    add('All 24 Phase 3B.2 histories are reused unchanged: eleven established Phase 3A boards, eight '
        'Phase 3B.1 seeded boards, four Phase 3B.2 additions and the separately labeled POST HOC '
        '`[1,4,6,0,6]` tail. Four new legal nonterminal histories use seed 20261011 at 6/10/18/26 plies, '
        'uniform legal moves, terminal/duplicate rejection only. No position selection uses outcomes. '
        'There are 112 depth/position conditions, 108 non-tail conditions. Broad-tree classification '
        'uses A alone: depth 8/10, ≥2,000 nodes and ≥35% heuristic leaves. '
        'The inherited `double-threat-loss` fixture permits an own immediate win; a separate test '
        'history `[1,0,3,0,5,0,1,6,3,6,5,6]` verifies an actual forced root loss.')
    add(f'Initial preflight: {preflight["elapsed_seconds"]:.1f}s, largest untraced decision '
        f'{max(v["normal"]["wall_seconds"] for r in preflight["rows"] for v in r["variants"].values()):.3f}s, '
        f'largest traced peak {max(v["memory"]["traced_peak_bytes"] for r in preflight["rows"] for v in r["variants"].values())/1048576:.2f} MiB. '
        f'Conservative 2× main projection {preflight["main_projected_seconds"]:.1f}s was within the declared '
        f'7,200s main allowance. Actual main runtime {end["elapsed_seconds"]:.1f}s. '
        'Preflight observations are excluded from final estimates. The declared overall compute allowance '
        'is 8,400s (7,200 main + 600 diagnostics/audit + 600 validation/report).')
    add('Each condition has one discarded warm-up and seven serial timed decisions per variant, '
        'rotating order by repetition and condition. Wall and process CPU cover complete choose_move, '
        'state initialization, every intermediate/root search, hint export, table construction and '
        'release. Game creation, variant loading and diagnostics are outside timing. Tables receive '
        '64 constructor warmups to stabilize CPython shared instance dictionaries. Separate counted '
        'and two traced decisions measure leaves/ordering and memory. The final table alone is retained '
        'for memory sizing; no artificial accumulation of intermediate tables inflates peak.')
    add('Runtime: ' + '; '.join(config.get('hardware', [])) + f'; {config["runtime"]["python"].splitlines()[0]}; '
        f'{config["runtime"]["platform"]}; initial load averages {config["runtime"]["load"]}. '
        'Other host activity was not stopped. These are finite laptop engineering comparisons, '
        'not randomized population estimates, game-strength results, public endpoint throughput '
        'or worst-case guarantees.')
    add('## Four-way complete-decision comparison')
    data = [['A/direct', '1.000', '1.000', '1.000', '1.000', 'reference']]
    for v, a in analysis['comparisons'].items():
        data.append([v, f'{a["geometric_speedup"]:.3f}×', f'{a["broad_speedup"]:.3f}×',
                     f'{a["summed_latency_ratio"]:.3f}', f'{a["summed_cpu_ratio"]:.3f}', 'reject'])
    table(['Variant', 'All-condition geometric speedup', 'Broad geometric speedup',
           'Sum median wall / A', 'Sum median CPU / A', 'Decision'], data)
    add('Speedup is A/new: above 1 is faster. Ratios new/A: below 1 is cheaper. '
        'Tail and D12 are excluded from acceptance aggregates. Summed medians give each declared '
        'condition one complete decision; geometric means give each condition equal log weight.')
    table(['Depth', 'A summed medians ms', 'B ms / geometric speedup', 'C ms / geometric speedup',
           'D ms / geometric speedup'], [[d,
        f'{sum(ms(r,"direct") for r in non_tail if r["depth"]==d):.3f}',
        *[f'{sum(ms(r,v) for r in non_tail if r["depth"]==d):.3f} / '
          f'{analysis["comparisons"][v]["depth_speedups"][str(d)]:.3f}×' for v in VARIANTS[1:]]]
        for d in (4, 6, 8, 10)])
    table(['Variant', 'Broad conditions faster', 'New generalization geometric speedup',
           'Broad retained bytes / A', 'Failed declared gates'], [[v,
        f'{a["broad_faster_fraction"]*100:.1f}% of {a["broad_conditions"]}',
        f'{a["generalization_speedup"]:.3f}×', f'{a["broad_retained_ratio"]:.3f}',
        ', '.join(k for k, passed in a['checks'].items() if not passed)]
        for v, a in analysis['comparisons'].items()])
    add('Eligibility required all exact checks; broad ≥1.15× and ≥75% faster; overall ≥1.05× and '
        'wall sum ≤90%/CPU sum ≤95%; each condition ≤20% slowdown or ≤0.25ms added; expensive '
        'conditions including tail ≤20% slowdown and ≤25% extra total nodes; every retained/peak '
        '≤25% growth or ≤64KiB added; broad retained sum ≤15% growth. No thresholds were relaxed.')
    add('## Every position and depth')
    add('Each cell is complete-decision median milliseconds; B/C/D include A/new speedup. '
        'Full CPU times, sample ranges, counters, leaves, hint metrics and memory are in '
        '[performance.csv](performance.csv); all raw samples are in [results.jsonl](results.jsonl).')
    table(['Position', 'Depth', 'A ms', 'B ms / speedup', 'C ms / speedup', 'D ms / speedup'],
          [[r['position'], r['depth'], f'{ms(r,"direct"):.3f}',
            *[f'{ms(r,v):.3f} / {ratio(r,v):.3f}×' for v in VARIANTS[1:]]] for r in rows])
    add('## Final iteration versus total work and ordering effectiveness')
    add('A’s node count is its single final iteration. Other cells show final / TOTAL nodes; '
        'total includes every preparatory search. Iteration TT entry sums count entries retained '
        'at each iteration end and are not simultaneous cache occupancy. C’s final nodes, hits, '
        'entries and cutoffs equal A exactly in every condition; its extra work buys no final ordering.')
    table(['Position D10', 'A nodes', 'B final / total', 'C final / total', 'D final / total'],
          [[r['position'], r['variants']['direct']['counted']['stats']['nodes'],
            *[f'{r["variants"][v]["counted"]["iterations"][-1]["nodes"]} / '
              f'{r["variants"][v]["counted"]["stats"]["nodes"]}' for v in VARIANTS[1:]]]
           for r in rows if r['depth'] == 10])
    records = [r for r in cross['records'] if r['depth'] in (8, 10) and r['position'] != 'post-hoc-tail']
    table(['Variant D8/D10 pooled', 'Lookups', 'Legal cross hints', 'Cross hint first',
           'Cross hint changes first', 'Cutoff on cross hinted first'], [[v,
        *[sum(r['totals'][k] for r in records if r['variant'] == v)
          for k in ('hint_lookups', 'legal_hints', 'cross_first', 'cross_changed_first', 'cross_first_cutoffs')]]
        for v in ('hints', 'combined')])
    add('The supplemental [cross-hint-effectiveness.json](cross-hint-effectiveness.json) distinguishes '
        'cross-depth suggestions from same-depth TT hints. Its source is archived and its complete '
        'scores/counters match primary evidence. All instrumentation is excluded from timing. '
        'A legal hint or first-move cutoff is not itself causal proof of saved work; compare final '
        'and total nodes/leaves and complete time. Primary changed-first/hinted-cutoff counters include '
        'both kinds of ordering hint; supplemental cross counters disambiguate them.')
    def metric_total(v, key):
        total = 0
        for r in non_tail:
            if r['depth'] not in (8, 10):
                continue
            counted = r['variants'][v]['counted']
            if key == 'final_nodes':
                total += counted['iterations'][-1]['nodes']
            elif key == 'nodes':
                total += counted['stats']['nodes']
            elif key == 'final_leaves':
                total += counted['diagnostics'][-1]['leaves']
            else:
                total += counted['diagnostic_totals']['leaves']
        return total

    table(['Variant D8/D10', 'Total nodes / A', 'Final nodes / A', 'Total leaves / A', 'Final leaves / A'],
          [[v, *[f'{metric_total(v,key)/metric_total("direct",key):.3f}'
                  for key in ('nodes','final_nodes','leaves','final_leaves')]] for v in VARIANTS])
    add('## Memory and expensive searches')
    add('Retained memory sums unique reachable Python objects, attributing score TT first, hint '
        'dictionary/keys/moves second, then table overhead. Shared objects count once; dictionary '
        'allocated capacity is included. Two repetitions match retained sizes exactly. Traced peak '
        'includes intermediate snapshot export overlap; traced current includes result objects. '
        'These are different from fresh-process RSS, which includes interpreter, imports and allocator. '
        'Production releases all caches after each decision; diagnostic retention is for sizing only.')
    def memory_value(r, v, kind):
        data = r['variants'][v]['memory']
        return max(m['retained']['total_bytes'] if kind == 'retained' else m['traced_peak_bytes'] for m in data)

    table(['Variant non-tail', 'Sum retained / A', 'Maximum retained growth / A',
           'Maximum peak growth / A', 'Largest peak MiB'], [[v,
        f'{sum(memory_value(r,v,"retained") for r in non_tail)/sum(memory_value(r,"direct","retained") for r in non_tail):.3f}',
        f'{max(memory_value(r,v,"retained")/memory_value(r,"direct","retained") for r in non_tail):.3f}',
        f'{max(memory_value(r,v,"peak")/memory_value(r,"direct","peak") for r in non_tail):.3f}',
        f'{max(memory_value(r,v,"peak") for r in non_tail)/1048576:.3f}'] for v in VARIANTS])
    table(['Variant', 'Memory gate failures (condition / retained or peak)'], [[v,
        ', '.join(f'{r["position"]}/D{r["depth"]}/{kind}' for r in non_tail
                  for kind in ('retained', 'peak')
                  if memory_value(r,v,kind) > 1.25 * memory_value(r,'direct',kind)
                  and memory_value(r,v,kind)-memory_value(r,'direct',kind) > 65536) or 'none']
        for v in VARIANTS[1:]])
    table(['Position D10', 'Variant', 'Wall ms / speedup', 'Retained MiB', 'Hint KiB', 'Peak MiB', 'Fresh RSS MiB'],
          [[r['position'], v, f'{ms(r,v):.3f} / {ratio(r,v):.3f}×',
            f'{r["variants"][v]["memory"][0]["retained"]["total_bytes"]/1048576:.3f}',
            f'{r["variants"][v]["memory"][0]["retained"]["categories"].get("hint_cache",0)/1024:.1f}',
            f'{max(m["traced_peak_bytes"] for m in r["variants"][v]["memory"])/1048576:.3f}',
            str(round(next(s['result']['rss_after_bytes'] or s['result']['high_water_bytes']
                           for s in rss if s['position']==r['position'] and s['variant']==v)/1048576,2))]
           for r in rows if r['depth']==10 and r['position'] in bench.PREFLIGHT_IDS for v in VARIANTS])
    add('Fresh RSS shows after-search retained snapshots when `ps` is available; otherwise the table '
        'uses OS high water. [rss.json](rss.json) records before/after/high-water fields and parity. '
        'RSS observations are single fresh-process diagnostics, not replicated memory claims.')
    if all(r['result']['rss_after_bytes'] is None for r in rss):
        add('In this run `ps` process inspection was unavailable in the sandbox, so all sixteen '
            'fresh-process measurements report OS `ru_maxrss` high water. Before/after RSS fields '
            'are null in the raw evidence; no before/after RSS reduction is claimed.')
    regressions = []
    for v in VARIANTS[1:]:
        failures = [r for r in non_tail if ms(r,v)>1.2*ms(r,'direct') and ms(r,v)-ms(r,'direct')>.25]
        regressions.append([v, len(failures), ', '.join(f'{r["position"]}/D{r["depth"]}' for r in failures)])
    table(['Variant', 'Per-condition latency gate failures', 'All failed conditions'], regressions)
    add('Improvements on the prior high-cost tail are useful research evidence, but tail success '
        'cannot override broad-board or memory failures. Shallow, tactical and terminal-heavy searches '
        'can finish before hints amortize preparation. The empty opening is a material counterexample '
        'to treating iterative deepening as universally faster. No adaptive fixture-specific strategy '
        'was added.')
    add('## Experimental depth 12')
    if d12['passed']:
        extra = [json.loads(l) for l in (directory / 'depth12-results.jsonl').read_text().splitlines()]
        add(f'Empty direct D12 preflight completed in {d12["probe"]["wall_seconds"]:.3f}s, '
            f'high-water RSS {d12["probe"]["high_water_bytes"]/1048576:.2f} MiB, '
            f'conservative diagnostic projection {d12["projected_seconds"]:.1f}s ≤600s. '
            'All four declared D12 diagnostic positions completed, with seven matched samples and '
            'one traced run per variant. Depth 12 remains unexposed by public API/caps.')
        table(['Position D12', 'A ms', 'B ms / speedup', 'C ms / speedup', 'D ms / speedup'],
              [[r['position'], f'{ms(r,"direct"):.3f}',
                *[f'{ms(r,v):.3f} / {ratio(r,v):.3f}×' for v in VARIANTS[1:]]] for r in extra])
        table(['Position D12', 'Variant', 'Final / total nodes', 'Retained / peak MiB'],
              [[r['position'], v,
                f'{r["variants"][v]["counted"]["iterations"][-1]["nodes"]} / {r["variants"][v]["counted"]["stats"]["nodes"]}',
                f'{r["variants"][v]["memory"][0]["retained"]["total_bytes"]/1048576:.2f} / '
                f'{r["variants"][v]["memory"][0]["traced_peak_bytes"]/1048576:.2f}']
               for r in extra for v in VARIANTS])
    else:
        add('D12 broad four-way measurement was rejected by the declared resource gate. '
            + (f'Empty probe completed in {d12["probe"]["wall_seconds"]:.3f}s; high-water RSS '
               f'{d12["probe"]["high_water_bytes"]/1048576:.2f} MiB; projected four-way diagnostic '
               f'cost {d12["projected_seconds"]:.1f}s exceeded the available criterion.' if 'probe' in d12
               else d12['reason']) + ' The completed probe is retained in [depth12-preflight.json](depth12-preflight.json). '
               'No partial search result was used. This is budget feasibility, not proof that D12 is generally infeasible.')
    add('## Correctness, validation and evidence audit')
    add(f'All {audit["verified_saved_decisions"]:,} saved main/preflight/RSS/D12 decisions have identical '
        'ordered final scores and selected moves across variants and deterministic counters within '
        'each variant. [root-parity.json](root-parity.json) records every complete comparison vector '
        'and final counters. All 96 prior Phase 3B.2 score/move/counter/leaf vectors match direct A. '
        f'Independent unpruned array minimax verified {audit["independent_oracle_vectors"]} vectors '
        f'and {audit["independent_oracle_variant_decisions"]} variant decisions: every manifest board '
        'at 1/2/3 plus nearly full draw prefixes at 4/6/8/10/12. '
        '[oracle-vectors.json](oracle-vectors.json) preserves expected scores. Deeper broad checks '
        'use the immutable validated baseline, not an unpruned depth-10 oracle.')
    add('The 109 added tests cover true forced losses and faster wins, required replies, Phase 4 '
        'unsafe-cache reproduction, transpositions, differing mover/horizon semantics, every '
        'retained bound against array truth, full draws and near-full boards, booleans/floats/strings/'
        'foreign/full-column hints, ties, repeated decisions, incremental state, recursive/root/'
        'final-iteration/export exception rollback, source scope and exclusive evidence writes. '
        'Primary and supplemental instruments preserve self counters; no invalid generated hints '
        'occurred. C has exactly A’s final counters, while B/D counters legitimately change.')
    add(f'Full backend: **{test_result}**. [backend-tests.txt](backend-tests.txt) contains full output. '
        'The suite preserves prior Negamax/MCTS tests and checks public API validation/caps, search '
        'concurrency, rate limits, history/replay and provenance. Runs use '
        '`PYTHON_DOTENV_DISABLED=1 EXPLANATIONS_ENABLED=false OPENAI_API_KEY=`; the global fixture '
        'forbids live HTTPS providers. PyTorch skips retain the existing dependency limitation. '
        'No paid API calls, public interface or frontend changes; frontend tests were unnecessary. '
        'Python compilation and Git whitespace checks pass. Production agent bytes equal baseline; '
        'previous phase reports, canonical datasets and protected services/models are untouched.')
    add('[audit.json](audit.json) verifies exact coverage, source/design/runtime hashes, all saved '
        'score/counter vectors, repeated retained sizes, previous-phase parity and oracle results. '
        'The report is generated from complete evidence. Negative findings apply to these concrete '
        'target−2/even-increment schedules and previous-iteration snapshots, not every possible '
        'iterative/hint design. No sampling uncertainty or universal strength improvement is claimed.')
    add('## Reproduction and next phase')
    add('Use this checkout and immutable baseline objects; choose a fresh output directory. Runners '
        'refuse to overwrite evidence. Copy DESIGN first, then declare. Run serially so benchmarks '
        'do not compete with tests. Source/hash drift fails closed. `ps` RSS is optional; high water '
        'is retained. Run depth12 only when its preflight saved `passed: true`.')
    add('```sh\nmkdir -p /tmp/negamax-iterative-replay\n'
        'cp docs/search-negamax-v2/iterative-deepening/DESIGN.md /tmp/negamax-iterative-replay/\n'
        'export PYTHON_DOTENV_DISABLED=1 EXPLANATIONS_ENABLED=false OPENAI_API_KEY=\n'
        '.venv/bin/python -m scripts.benchmark_negamax_iterative declare --directory /tmp/negamax-iterative-replay\n'
        '.venv/bin/python -m scripts.benchmark_negamax_iterative preflight --directory /tmp/negamax-iterative-replay\n'
        '.venv/bin/python -m scripts.benchmark_negamax_iterative run --directory /tmp/negamax-iterative-replay\n'
        '.venv/bin/python -m scripts.benchmark_negamax_iterative rss --directory /tmp/negamax-iterative-replay\n'
        '.venv/bin/python -m scripts.benchmark_negamax_iterative depth12-preflight --directory /tmp/negamax-iterative-replay\n'
        '# If the depth12 gate passes, run its declared four-way diagnostic:\n'
        '# .venv/bin/python -m scripts.benchmark_negamax_iterative depth12 --directory /tmp/negamax-iterative-replay\n'
        '.venv/bin/python -m scripts.diagnose_negamax_iterative --directory /tmp/negamax-iterative-replay\n'
        '.venv/bin/python -m scripts.audit_negamax_iterative --directory /tmp/negamax-iterative-replay\n'
        '.venv/bin/python -m pytest tests -q -rs > /tmp/negamax-iterative-replay/backend-tests.txt\n'
        '.venv/bin/python -m scripts.summarize_negamax_iterative --directory /tmp/negamax-iterative-replay\n```')
    add('Recommended next phase: evaluate collision-free mirror canonicalization of the score TT '
        'as a separately predeclared experiment, including exact column remapping for hints, stable '
        'root ties, all bound semantics and complete-decision overhead. Symmetry can reduce duplicate '
        'work without paying for preparatory horizons; bit manipulation and canonicalization costs '
        'must still earn their place on this same distribution. Keep iterative deepening and '
        'cross-depth hints experimental unless a new independently declared policy meets correctness '
        'and broad regression/memory gates.')
    with (directory / 'REPORT.md').open('x') as stream:
        stream.write('\n'.join(lines))
    return dict(report=str(directory / 'REPORT.md'), selected='direct')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=bench.ROOT)
    print(json.dumps(generate(parser.parse_args().directory), indent=2))
