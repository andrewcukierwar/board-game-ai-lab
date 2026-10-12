"""Presentation only: markdown tables, static SVG figures and resource accounting from the analysis output.

Computes no statistic of its own beyond sums and ratios of already-published
numbers. Not part of the frozen source set; nothing measured depends on it.
"""
import argparse
from pathlib import Path

from .common import NAMES, OUTPUT, RESULTS, SEEDS, read, write

LABEL = {'cnn': 'CNN', 'resnet3x64': 'ResNet 3x64', 'resnet5x64': 'ResNet 5x64'}
# Categorical slots 1-3 of the validated reference palette (all-pairs safe), fixed per architecture.
COLOR = {'cnn': '#2a78d6', 'resnet3x64': '#eb6834', 'resnet5x64': '#1baf7a'}
SURFACE, INK, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e6e5e0'
COMPARISONS = (('equal_updates_400', '400 updates'), ('equal_updates_800', '800 updates'),
               ('equal_updates_1600', '1,600 updates'), ('equal_updates_3000', '3,000 updates (primary)'),
               ('equal_seconds_60', '60 s'), ('equal_seconds_120', '120 s'), ('equal_seconds_240', '240 s'),
               ('equal_seconds_matched', 'largest common time'), ('equal_wall_final', 'final models (primary)'))


def f(value, digits=4):
    return '—' if value is None else f'{value:.{digits}f}'


def table(header, rows):
    lines = ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    lines += ['| ' + ' | '.join(str(c) for c in row) + ' |' for row in rows]
    return '\n'.join(lines)


def interval(entry, digits=4):
    return f"{entry['difference']:+.{digits}f} [{entry['low']:+.{digits}f}, {entry['high']:+.{digits}f}]"


# Figures -----------------------------------------------------------------------------------------

def ticks(low, high, count=5):
    span = high - low
    raw = span / count
    magnitude = 10 ** __import__('math').floor(__import__('math').log10(raw))
    step = next(m * magnitude for m in (1, 2, 2.5, 5, 10) if m * magnitude >= raw)
    first = __import__('math').ceil(low / step) * step
    values = []
    while first <= high + 1e-12:
        values.append(round(first, 10))
        first += step
    return values


def number(value):
    return f'{value:,.0f}' if abs(value) >= 100 or value == int(value) else f'{value:g}'


def panel(x0, y0, width, height, title, x_label, series, y_range=None, x_max=None):
    """One small multiple: series is [(name, [(x, y), ...], color, label)]."""
    xs = [x for _, points, _, _ in series for x, _ in points]
    ys = [y for _, points, _, _ in series for _, y in points]
    x_hi = x_max or max(xs)
    y_lo, y_hi = y_range or (min(ys), max(ys))
    pad = (y_hi - y_lo) * 0.06 or 0.01
    y_lo, y_hi = y_lo - pad, y_hi + pad
    left, right, top, bottom = 52, 96, 26, 34
    pw, ph = width - left - right, height - top - bottom
    sx = lambda x: x0 + left + pw * x / x_hi
    sy = lambda y: y0 + top + ph * (1 - (y - y_lo) / (y_hi - y_lo))
    out = [f'<text x="{x0 + left}" y="{y0 + 15}" font-size="12" font-weight="600" fill="{INK}">{title}</text>']
    for tick in ticks(y_lo, y_hi):
        out.append(f'<line x1="{x0 + left}" x2="{x0 + left + pw}" y1="{sy(tick):.1f}" y2="{sy(tick):.1f}" stroke="{GRID}" stroke-width="1"/>')
        out.append(f'<text x="{x0 + left - 6}" y="{sy(tick) + 3.5:.1f}" font-size="10" text-anchor="end" fill="{MUTED}">{number(tick)}</text>')
    for tick in ticks(0, x_hi, 4):
        out.append(f'<text x="{sx(tick):.1f}" y="{y0 + top + ph + 14}" font-size="10" text-anchor="middle" fill="{MUTED}">{number(tick)}</text>')
    out.append(f'<line x1="{x0 + left}" x2="{x0 + left + pw}" y1="{y0 + top + ph}" y2="{y0 + top + ph}" stroke="{MUTED}" stroke-width="1"/>')
    out.append(f'<text x="{x0 + left + pw / 2}" y="{y0 + height - 4}" font-size="10" text-anchor="middle" fill="{MUTED}">{x_label}</text>')
    ends = []
    for name, points, color, label in series:
        clipped = [(x, min(max(y, y_lo), y_hi)) for x, y in points if x <= x_hi * 1.0001]
        path = ' '.join(f'{sx(x):.1f},{sy(y):.1f}' for x, y in clipped)
        out.append(f'<polyline points="{path}" fill="none" stroke="{color}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
        ex, ey = clipped[-1]
        out.append(f'<circle cx="{sx(ex):.1f}" cy="{sy(ey):.1f}" r="4" fill="{color}" stroke="{SURFACE}" stroke-width="2"/>')
        ends.append([sy(ey), sx(ex), color, label])
    # Direct end labels in ink, spread apart with a thin leader so they never collide.
    ends.sort()
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 12)
    column = x0 + left + pw + 10
    for y, x, color, label in ends:
        out.append(f'<line x1="{x + 5:.1f}" x2="{column - 2}" y1="{y:.1f}" y2="{y:.1f}" stroke="{color}" stroke-width="1"/>')
        out.append(f'<text x="{column}" y="{y + 3.5:.1f}" font-size="10" fill="{INK}">{label}</text>')
    return '\n'.join(out)


def figure(path, title, subtitle, panels, columns, legend, panel_size=(430, 230)):
    width = columns * panel_size[0] + 20
    rows = -(-len(panels) // columns)
    height = rows * panel_size[1] + 78
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
           f'font-family="-apple-system, BlinkMacSystemFont, Segoe UI, Helvetica, Arial, sans-serif" role="img">',
           f'<title>{title}</title>', f'<rect width="{width}" height="{height}" fill="{SURFACE}"/>',
           f'<text x="14" y="22" font-size="14" font-weight="600" fill="{INK}">{title}</text>',
           f'<text x="14" y="39" font-size="11" fill="{MUTED}">{subtitle}</text>']
    x = 14
    for label, color in legend:
        out.append(f'<line x1="{x}" x2="{x + 18}" y1="55" y2="55" stroke="{color}" stroke-width="2" stroke-linecap="round"/>')
        out.append(f'<text x="{x + 24}" y="58.5" font-size="11" fill="{INK}">{label}</text>')
        x += 40 + 6.2 * len(label)
    for index, make in enumerate(panels):
        out.append(make(10 + (index % columns) * panel_size[0], 66 + (index // columns) * panel_size[1], *panel_size))
    out.append('</svg>')
    Path(path).write_text('\n'.join(out) + '\n')


def figures(learning, output):
    curves = learning['curves']
    legend = [(LABEL[n], COLOR[n]) for n in NAMES]

    def series(seed, getter, x_key, skip_zero=True):
        result = []
        for n in NAMES:
            rows = curves[str(seed)][n]['milestones']
            points = [(r[x_key], getter(r)) for r in rows if not (skip_zero and r['updates'] == 0)]
            result.append((n, points, COLOR[n], f'{LABEL[n]} {points[-1][1]:.3f}'))
        return result

    combined = lambda r: r['novel']['policy_ce'] + r['novel']['value_mse']
    values = [combined(r) for s in SEEDS for n in NAMES for r in curves[str(s)][n]['milestones'] if r['updates'] >= 50]
    shared = (min(values), max(values))
    panels = []
    for seed in SEEDS:
        panels.append(lambda x, y, w, h, seed=seed: panel(x, y, w, h, f'Seed {seed} · by optimizer updates', 'optimizer updates',
                                                          series(seed, combined, 'updates'), shared))
        panels.append(lambda x, y, w, h, seed=seed: panel(x, y, w, h, f'Seed {seed} · by optimizer seconds', 'measured optimizer seconds',
                                                          series(seed, combined, 'train_seconds'), shared))
    figure(output / 'figure-heldout-combined.svg', 'Held-out combined loss on novel positions (lower is better)',
           'Policy cross-entropy + value MSE against CNN-generated replay targets; from 50 updates onward. Labels give the final value.',
           panels, 2, legend)
    for key, name, title in (('policy_ce', 'figure-heldout-policy.svg', 'Held-out policy cross-entropy on novel positions (lower is better)'),
                             ('value_mse', 'figure-heldout-value.svg', 'Held-out value MSE on novel positions (lower is better)')):
        getter = lambda r, key=key: r['novel'][key]
        values = [getter(r) for s in SEEDS for n in NAMES for r in curves[str(s)][n]['milestones'] if r['updates'] >= 50]
        shared_key = (min(values), max(values))
        figure(output / name, title, 'By measured optimizer seconds, from 50 updates onward. Labels give the final value.',
               [lambda x, y, w, h, seed=seed, getter=getter, shared_key=shared_key: panel(
                   x, y, w, h, f'Seed {seed}', 'measured optimizer seconds', series(seed, getter, 'train_seconds'), shared_key)
                for seed in SEEDS], 2, legend)
    # Training probe against novel held-out, one panel per architecture and seed.
    panels = []
    gap_values = [v for s in SEEDS for n in NAMES for r in curves[str(s)][n]['milestones'] if r['updates'] >= 50
                  for v in (combined(r), r['train_probe']['policy_ce'] + r['train_probe']['value_mse'])]
    gap_range = (min(gap_values), max(gap_values))
    for seed in SEEDS:
        for n in NAMES:
            def make(x, y, w, h, seed=seed, n=n):
                rows = [r for r in curves[str(seed)][n]['milestones'] if r['updates'] >= 50]
                held = [(r['updates'], combined(r)) for r in rows]
                probe = [(r['updates'], r['train_probe']['policy_ce'] + r['train_probe']['value_mse']) for r in rows]
                return panel(x, y, w, h, f'{LABEL[n]} · seed {seed}', 'optimizer updates',
                             [('held', held, COLOR[n], f'held-out {held[-1][1]:.3f}'),
                              ('probe', probe, MUTED, f'training {probe[-1][1]:.3f}')], gap_range)
            panels.append(make)
    figure(output / 'figure-train-vs-heldout.svg', 'Training-probe versus novel held-out combined loss',
           'Colored line: novel held-out positions. Gray line: 4,096 fixed training positions, eval mode. Shared y-axis.',
           panels, 3, [('novel held-out (architecture color)', COLOR['cnn']), ('training probe', MUTED)], panel_size=(330, 210))
    panels = []
    for suite, label in (('solved', 'solved300 optimal-action accuracy'), ('tactical', 'tactical200 accuracy')):
        getter = lambda r, suite=suite: r[suite]['accuracy']
        values = [getter(r) for s in SEEDS for n in NAMES for r in curves[str(s)][n]['milestones']]
        limits = (min(values), max(values))
        for seed in SEEDS:
            panels.append(lambda x, y, w, h, seed=seed, getter=getter, label=label, limits=limits: panel(
                x, y, w, h, f'{label} · seed {seed}', 'measured optimizer seconds',
                series(seed, getter, 'train_seconds', skip_zero=False), limits))
    figure(output / 'figure-suite-accuracy.svg', 'Raw-network accuracy on the development suites (higher is better)',
           'Argmax of the legal-masked policy, no search. Exact labels, independent of the CNN teacher. Labels give the final value.',
           panels, 2, legend)


# Tables ------------------------------------------------------------------------------------------

def benchmark_tables(b):
    parts = ['## Inference\n']
    for path, title in (('predict_batch', 'Full `predict_batch` path (encoding, forward, legal masking)'),
                        ('forward', 'Bare forward pass on a prepared tensor')):
        rows = []
        for batch in (1, 8, 16, 32):
            cell = b['inference']['latency'][f'{path}:{batch}']
            for n in NAMES:
                c = cell[n]
                rows.append([batch, LABEL[n], f(c['latency_ms'], 3), f"{f(c['latency_ms_min'], 3)}–{f(c['latency_ms_max'], 3)}",
                             f(c['p95_ms'], 3), f"{c['positions_per_second']:,.0f}",
                             f"{c['cost_ratio_to_cnn']:.2f}x ({c['cost_ratio_min']:.2f}–{c['cost_ratio_max']:.2f})"])
        parts.append(f'**{title}.** Median over seven repetitions of the per-call mean.\n\n' + table(
            ['Batch', 'Architecture', 'Latency ms', 'Range over repetitions', 'p95 ms', 'Positions/s', 'Cost vs CNN (range)'], rows) + '\n')
    rows = [[batch, *(f"{b['inference']['memory'][f'{n}:{batch}']['peak_rss_mib']:.1f}" for n in NAMES)] for batch in (1, 8, 16, 32)]
    rows.append(['import only', *(f"{b['inference']['memory'][f'{n}:1']['import_rss_mib']:.1f}" for n in NAMES)])
    rows.append(['weights + statistics, MB', *(f"{b['inference']['memory'][f'{n}:1']['state_megabytes']:.2f}" for n in NAMES)])
    parts.append('**Peak resident memory, MiB** (fresh process per cell).\n\n' + table(['Batch', *(LABEL[n] for n in NAMES)], rows) + '\n')
    c = b['collection']
    rows = [
        ['Complete games', *(c[n]['games'] for n in NAMES)],
        ['Mean game length, plies (min–max)', *(f"{c[n]['mean_game_length']:.1f} ({c[n]['min_game_length']}–{c[n]['max_game_length']})" for n in NAMES)],
        ['Total plies', *(f"{c[n]['plies']:,}" for n in NAMES)],
        ['Total simulations', *(f"{c[n]['simulations']:,}" for n in NAMES)],
        ['Neural positions', *(f"{c[n]['neural_positions']:,}" for n in NAMES)],
        ['Terminal leaves (no forward)', *(f"{c[n]['terminal_leaves']:,}" for n in NAMES)],
        ['Wall seconds', *(f(c[n]['wall_seconds'], 1) for n in NAMES)],
        ['Complete games per second', *(f(c[n]['games_per_second'], 3) for n in NAMES)],
        ['Neural positions per second', *(f"{c[n]['neural_positions_per_second']:,.0f}" for n in NAMES)],
        ['Simulations per second', *(f"{c[n]['simulations_per_second']:,.0f}" for n in NAMES)],
        ['ms per neural position', *(f(1e3 * c[n]['seconds_per_neural_position'], 4) for n in NAMES)],
        ['Cost per neural position vs CNN', *(f"{c[n]['seconds_per_neural_position_ratio_to_cnn']:.2f}x" for n in NAMES)],
        ['ms per ply (256 simulations)', *(f(1e3 * c[n]['seconds_per_ply'], 1) for n in NAMES)],
        ['Cost per ply vs CNN', *(f"{c[n]['seconds_per_ply_ratio_to_cnn']:.2f}x" for n in NAMES)],
        ['Games per second at 30 plies (normalized)', *(f(c[n]['length_normalized_games_per_second'], 3) for n in NAMES)],
        ['Mean inference batch occupancy', *(f(c[n]['mean_batch_occupancy'], 2) for n in NAMES)],
        ['Forward calls at full batch 8', *(f"{100 * c[n]['full_batch_share_of_calls']:.1f}%" for n in NAMES)],
        ['Inference share of wall time', *(f"{100 * c[n]['inference_share_of_wall']:.1f}%" for n in NAMES)],
        ['Peak resident memory, MiB', *(f(c[n]['peak_rss_mib'], 1) for n in NAMES)],
        ['Outcomes X / O / draw', *(f"{c[n]['outcomes']['0']} / {c[n]['outcomes']['1']} / {c[n]['outcomes']['-1']}" for n in NAMES)],
    ]
    parts.append('## Collection\n\n' + table(['Pool-eight, 256 simulations, 32 matched game identities', *(LABEL[n] for n in NAMES)], rows) + '\n')
    rows = [[LABEL[n], block['game_ids'], f(block['wall_seconds'], 1), f(1e3 * block['seconds_per_neural_position'], 4),
             f(block['mean_batch'], 2)] for n in NAMES for block in c[n]['by_block']]
    parts.append('**By block** (order CNN, 3x64, 5x64 for games 0–15; reversed for 16–31).\n\n' + table(
        ['Architecture', 'Game ids', 'Wall s', 'ms per neural position', 'Mean batch'], rows) + '\n')
    rows = [[size, *(f"{c[n]['batch_sizes'].get(str(size), 0):,}" for n in NAMES)] for size in range(1, 9)]
    parts.append('**Forward calls by batch size.**\n\n' + table(['Batch size', *(LABEL[n] for n in NAMES)], rows) + '\n')
    t = b['training']
    stage_names = list(t['cnn']['staged_ms'])
    rows = [['`trainer.step` mean ms', *(f(t[n]['update_ms'], 2) for n in NAMES)],
            ['by pass', *(' / '.join(f(v, 2) for v in t[n]['update_ms_by_pass']) for n in NAMES)],
            ['median / p95 ms', *(f"{f(t[n]['update_median_ms'], 2)} / {f(t[n]['update_p95_ms'], 2)}" for n in NAMES)],
            ['Cost vs CNN', *(f"{t[n]['cost_ratio_to_cnn']:.2f}x" for n in NAMES)],
            ['Examples per second', *(f"{t[n]['examples_per_second']:,.0f}" for n in NAMES)],
            ['Staged replica total ms', *(f(t[n]['staged_total_ms'], 2) for n in NAMES)]]
    rows += [[f'· {stage} ms (share)', *(f"{f(t[n]['staged_ms'][stage], 2)} ({100 * t[n]['shares'][stage]:.1f}%)" for n in NAMES)]
             for stage in stage_names]
    rows += [['Forward only, eval mode ms', *(f(t[n]['eval_forward_ms'], 2) for n in NAMES)],
             ['Forward only, train mode ms', *(f(t[n]['train_forward_ms'], 2) for n in NAMES)],
             ['Peak resident memory, MiB', *(f(t[n]['peak_rss_mib'], 1) for n in NAMES)]]
    parts.append('## Training update, batch 128\n\n' + table(['', *(LABEL[n] for n in NAMES)], rows) + '\n')
    rows = []
    for key, label in (('update_ms_without', 'Update ms with batch norm removed'), ('update_overhead_ms', 'Batch-norm cost per update, ms'),
                       ('update_overhead_share', 'Share of the update'), ('forward_overhead_ms', '· in the forward pass, ms'),
                       ('backward_overhead_ms', '· in the backward pass, ms'), ('optimizer_overhead_ms', '· in the optimizer step, ms'),
                       ('eval_forward_overhead_ms', 'Eval-mode forward cost of batch norm at 128, ms'),
                       ('train_minus_eval_forward_ms', 'Train-mode minus eval-mode forward, ms')):
        rows.append([label, *(f"{100 * t[n]['batch_norm'][key]:.1f}%" if key.endswith('share') else f(t[n]['batch_norm'][key], 2)
                              for n in ('resnet3x64', 'resnet5x64'))])
    parts.append('**Measured batch-norm overhead** (same network with every batch norm replaced by the identity).\n\n' + table(
        ['', 'ResNet 3x64', 'ResNet 5x64'], rows) + '\n')
    return '\n'.join(parts)


def learning_tables(learning, comparisons, detail, common=3000):
    curves = learning['curves']
    parts = ['## Fixed-data learning\n']
    rows = [[seed, LABEL[n], f"{curves[str(seed)][n]['updates']:,}", f(curves[str(seed)][n]['train_seconds'], 1),
             f(1e3 * curves[str(seed)][n]['seconds_per_update'], 2), f(curves[str(seed)][n]['evaluation_seconds'], 1),
             f(curves[str(seed)][n]['wall_seconds'], 1)] for seed in SEEDS for n in NAMES]
    parts.append(table(['Seed', 'Architecture', 'Updates', 'Optimizer s', 'ms per update', 'Evaluation s', 'Wall s'], rows) + '\n')
    for seed in SEEDS:
        for n in NAMES:
            rows = []
            for r in curves[str(seed)][n]['milestones']:
                rows.append([f"{r['updates']:,}", f(r['train_seconds'], 1), f(r['novel']['policy_ce']), f(r['novel']['value_mse']),
                             f(r['novel']['policy_ce'] + r['novel']['value_mse']), f(r['novel']['top1'], 3),
                             f(r['train_probe']['policy_ce']), f(r['train_probe']['value_mse']),
                             f(r['generalization_gap']['policy_ce']), f(r['generalization_gap']['value_mse']),
                             f(r['solved']['accuracy'], 3), f(r['solved']['value_mse'], 3), f(r['tactical']['accuracy'], 3)])
            parts.append(f'**{LABEL[n]}, seed {seed}.**\n\n' + table(
                ['Updates', 'Opt. s', 'Novel CE', 'Novel MSE', 'Novel CE+MSE', 'Novel top-1', 'Train CE', 'Train MSE', 'Gap CE', 'Gap MSE',
                 'solved300 acc', 'solved300 value MSE', 'tactical200 acc'], rows) + '\n')
    parts.append('## Contrasts, ResNet minus CNN\n\nDifference [95% family-bootstrap interval]. Negative favors the ResNet for losses; '
                 'positive favors it for accuracies.\n')
    for label, title in COMPARISONS:
        if label not in comparisons:
            continue
        entry = comparisons[label]
        rows = []
        for seed in SEEDS:
            rows.append([seed, *(f"{f(entry['checkpoints'][str(seed)][n]['combined'])} @ {entry['checkpoints'][str(seed)][n]['updates']:,.0f}"
                                 for n in NAMES), *(f"{entry['checkpoints'][str(seed)][n]['solved_accuracy']:.3f} / "
                                                    f"{entry['checkpoints'][str(seed)][n]['tactical_accuracy']:.3f}" for n in NAMES)])
        parts.append(f'### {title}\n\n' + table(['Seed', *(f'{LABEL[n]} CE+MSE @ updates' for n in NAMES),
                                                 *(f'{LABEL[n]} solved / tactical acc' for n in NAMES)], rows) + '\n')
        for name in ('resnet3x64', 'resnet5x64'):
            block = entry['resnets'][name]
            rows = []
            for metric, title_m in (('combined', 'Novel CE+MSE (primary)'), ('policy_ce', 'Novel policy CE'),
                                    ('value_mse', 'Novel value MSE'), ('top1', 'Novel top-1 agreement')):
                h = block['held_out']
                rows.append([title_m, *(interval(h[str(s)][metric]) for s in SEEDS), interval(h['pooled'][metric]), block['verdicts'][metric]])
            for suite in ('solved', 'tactical'):
                if suite in block:
                    for metric, e in block[suite].items():
                        rows.append([f'{suite} {metric}', *(interval(e[str(s)], 3) for s in SEEDS), interval(e['pooled'], 3),
                                     block['verdicts'][f'{suite}_{metric}']])
                else:
                    for metric, e in block['suite_point_differences'][suite].items():
                        rows.append([f'{suite} {metric}', *(f'{e[str(s)]:+.3f}' for s in SEEDS),
                                     f'{sum(e.values()) / len(e):+.3f}', 'point only'])
            parts.append(f'**{LABEL[name]} minus CNN.**\n\n' + table(
                ['Metric', *(f'Seed {s}' for s in SEEDS), 'Pooled', 'Verdict'], rows) + '\n')
    parts.append('## Batch norm\n')
    for name in ('resnet3x64', 'resnet5x64'):
        for seed in SEEDS:
            rows = []
            for r in curves[str(seed)][name]['milestones']:
                bn = r['batch_norm']
                rows.append([f"{r['updates']:,}", f(bn['eval_mode']['policy_ce']), f(bn['train_mode']['policy_ce']),
                             f(bn['mode_gap']['policy_ce']), f(bn['mode_gap']['value_mse']), f(bn['mean_abs_value_difference']),
                             f(bn['mean_policy_kl_eval_to_train']), f(100 * bn['choice_disagreement'], 1) + '%',
                             f(bn['standardized_mean_shift']), f(bn['abs_log_variance_ratio'])])
            parts.append(f'**{LABEL[name]}, seed {seed}** (training probe, 4,096 positions).\n\n' + table(
                ['Updates', 'CE eval mode', 'CE train mode', 'CE gap', 'MSE gap', 'Mean abs Δvalue', 'Policy KL', 'Choice changes',
                 'Mean shift (σ)', 'Abs log var ratio'], rows) + '\n')
    rows = []
    for name in ('resnet3x64', 'resnet5x64'):
        for seed in SEEDS:
            final = str(curves[str(seed)][name]['updates'])
            layers = detail[str(seed)][name][final]['batch_norm_layers']
            variances = [v['running_var_mean'] for v in layers.values()]
            rows.append([LABEL[name], seed, final, len(layers), f(min(v['running_var_min'] for v in layers.values()), 5),
                         f(min(variances), 3), f(max(variances), 3), f(max(v['running_var_max'] for v in layers.values()), 2),
                         f(max(v['running_mean_abs'] for v in layers.values()), 3),
                         f(max(v['standardized_mean_shift'] for v in layers.values())), f(max(v['abs_log_variance_ratio'] for v in layers.values())),
                         all(v['updates'] == int(final) for v in layers.values())])
    parts.append('**Running statistics of the final models.**\n\n' + table(
        ['Architecture', 'Seed', 'Updates', 'Layers', 'Smallest channel variance', 'Smallest layer-mean variance', 'Largest layer-mean variance',
         'Largest channel variance', 'Largest layer mean abs running mean', 'Worst mean shift (σ)', 'Worst abs log var ratio', 'Counters = updates'], rows) + '\n')
    parts.append('## Gradient stability and calibration\n')
    rows = []
    for seed in SEEDS:
        for n in NAMES:
            run = curves[str(seed)][n]
            windows = [r['optimization'] for r in run['milestones'] if r['optimization']]
            total = sum(w['updates'] for w in windows)
            first, last = windows[0], windows[-1]
            rows.append([seed, LABEL[n], f(first['gradient_norm_mean'], 2), f(last['gradient_norm_mean'], 2),
                         f(max(w['gradient_norm_max'] for w in windows), 2), f(max(w['gradient_norm_p95'] for w in windows), 2),
                         f(100 * sum(w['clipped_share'] * w['updates'] for w in windows) / total, 2) + '%',
                         f(last['combined_loss_mean']), f(last['combined_loss_stdev'])])
    parts.append(table(['Seed', 'Architecture', 'Mean gradient norm, first 50', 'Mean gradient norm, last interval', 'Largest norm',
                        'Largest interval p95', 'Updates clipped at 5', 'Training loss, last interval', 'Its spread'], rows) + '\n')
    rows = []
    for seed in SEEDS:
        for n in NAMES:
            run = curves[str(seed)][n]
            for updates in sorted({common, run['updates']}):
                r = next(m for m in run['milestones'] if m['updates'] == updates)
                rows.append([seed, LABEL[n], f'{updates:,}', f(r['calibration']['policy_ece']), f(r['calibration']['value_error']),
                             f(r['calibration']['mean_abs_value'], 3), f(100 * r['calibration']['value_saturated_share'], 1) + '%',
                             f(r['solved']['policy_ece']), f(r['solved']['value_error'])])
    parts.append(table(['Seed', 'Architecture', 'Updates', 'Held-out policy ECE', 'Held-out value calibration error', 'Mean abs value',
                        'Abs value > 0.95', 'solved300 policy ECE', 'solved300 value calibration error'], rows) + '\n')
    return '\n'.join(parts)


def consistency_tables(benchmarks, learning):
    """Descriptive, written after the run: how stable the timings and the late checkpoints were."""
    import statistics
    pilot = read(RESULTS / 'pilot.json')
    curves = learning['curves']
    parts = ['## Post hoc descriptive checks\n\nNot declared in DESIGN.md. No interval, no verdict.\n']
    rows = []
    for n in NAMES:
        runs = [1e3 * curves[str(s)][n]['seconds_per_update'] for s in SEEDS]
        steps = [1e3 * curves[str(s)][n]['step_seconds'] / curves[str(s)][n]['updates'] for s in SEEDS]
        rows.append([LABEL[n], f(pilot['training'][n]['trainer_step']['mean_ms'], 2),
                     ' / '.join(f(v, 2) for v in benchmarks['training'][n]['update_ms_by_pass']),
                     ' / '.join(f(v, 2) for v in runs), ' / '.join(f(v, 2) for v in steps),
                     f(1e3 * pilot['collection'][n]['wall_seconds'] / pilot['collection'][n]['neural_positions'], 4),
                     ' / '.join(f(1e3 * b['seconds_per_neural_position'], 4) for b in benchmarks['collection'][n]['by_block'])])
    cnn = [statistics.fmean(curves[str(s)]['cnn']['seconds_per_update'] for s in SEEDS)]
    parts.append('**Timing consistency.** Milliseconds per batch-128 update and per pool-eight neural position, measured at different times.\n\n'
                 + table(['Architecture', 'Pilot update', 'Benchmark update, pass 1 / 2', 'Learning run incl. sampling, seed 42 / 314159',
                          'Learning run trainer.step only, seed 42 / 314159', 'Pilot search position', 'Collection block 1 / 2'], rows) + '\n')
    rows = [[LABEL[n], *(f"{curves[str(s)][n]['seconds_per_update'] / curves[str(s)]['cnn']['seconds_per_update']:.3f}x" for s in SEEDS),
             f"{statistics.fmean(curves[str(s)][n]['seconds_per_update'] for s in SEEDS) / cnn[0]:.3f}x"] for n in NAMES]
    parts.append('**Update cost against the CNN inside the learning runs** (thousands of updates each).\n\n'
                 + table(['Architecture', 'Seed 42', 'Seed 314159', 'Mean'], rows) + '\n')
    rows = []
    for seed in SEEDS:
        for n in NAMES:
            late = [r for r in curves[str(seed)][n]['milestones'] if 1200 <= r['updates'] <= 3000]
            cell = [seed, LABEL[n], len(late)]
            for getter in (lambda r: r['novel']['policy_ce'], lambda r: r['novel']['value_mse'],
                           lambda r: r['novel']['policy_ce'] + r['novel']['value_mse'], lambda r: r['solved']['accuracy'],
                           lambda r: r['tactical']['accuracy']):
                values = [getter(r) for r in late]
                cell.append(f'{statistics.fmean(values):.4f} ± {statistics.pstdev(values):.4f} ({min(values):.3f}–{max(values):.3f})')
            rows.append(cell)
    parts.append('**Checkpoint-to-checkpoint variability, 1,200 to 3,000 updates** (five checkpoints per run: mean ± spread, range). '
                 'The bootstrap intervals above describe one checkpoint each and do not include this variation.\n\n'
                 + table(['Seed', 'Architecture', 'Checkpoints', 'Novel policy CE', 'Novel value MSE', 'Novel CE+MSE',
                          'solved300 accuracy', 'tactical200 accuracy'], rows) + '\n')
    return '\n'.join(parts)


def resources(run, benchmarks, learning):
    status = read(run / 'status.json')
    pilot, smoke = read(RESULTS / 'pilot.json'), read(RESULTS / 'smoke.json')
    design = read(RESULTS / 'design.json')
    stages = [dict(stage=s['stage'], wall_seconds=s['wall_seconds'], outcome=s['outcome']) for s in status['stages']]
    measured = sum(s['wall_seconds'] for s in stages)
    total = status['elapsed_seconds'] + pilot['wall_seconds'] + smoke['wall_seconds']
    return dict(
        status=status['status'], design_commit=status['design_commit'], stages=stages, run_elapsed_seconds=status['elapsed_seconds'],
        run_stage_seconds=measured, pilot_seconds=pilot['wall_seconds'], smoke_seconds=smoke['wall_seconds'],
        total_measured_seconds=total, total_measured_minutes=total / 60, target_seconds=design['budget']['target_seconds'],
        hard_maximum_seconds=design['budget']['hard_maximum_seconds'], run_ceiling_seconds=design['budget']['run_ceiling_seconds'],
        projected_run_seconds=design['budget']['projected_seconds'], within_hard_maximum=total <= design['budget']['hard_maximum_seconds'],
        children_cpu_seconds=status['stages'][-1]['children_cpu_seconds_so_far'],
        children_peak_rss_mib=status['stages'][-1]['children_peak_rss_mib_so_far'],
        learning={seed: dict(wall_seconds=v['wall_seconds'], cpu_seconds=v['cpu_seconds'], peak_rss_mib=v['peak_rss_mib'],
                             load_seconds=v['load_seconds'],
                             optimizer_seconds=sum(a['train_seconds'] for a in v['architectures'].values()),
                             evaluation_seconds=sum(a['evaluation_seconds'] for a in v['architectures'].values()))
                  for seed, v in learning['summaries'].items()},
        optimizer_updates=sum(c['updates'] for seed in learning['curves'].values() for c in seed.values()),
        collection_games=sum(benchmarks['collection'][n]['games'] for n in NAMES),
        collection_simulations=sum(benchmarks['collection'][n]['simulations'] for n in NAMES),
        official_preservation=status['official_preservation'], phase4d4g_preservation=status['phase4d4g_preservation'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default=str(OUTPUT / 'run'))
    parser.add_argument('--results', default=str(RESULTS))
    args = parser.parse_args()
    run, results = Path(args.run), Path(args.results)
    benchmarks, learning = read(results / 'benchmarks.json'), read(results / 'learning.json')
    comparisons, detail = read(results / 'comparisons.json'), read(results / 'endpoint-detail.json')
    figures(learning, results)
    (results / 'tables.md').write_text('# Phase 4D.4H-B result tables\n\nGenerated by `report.py` from the analysis output; '
                                       'do not edit by hand.\n\n' + benchmark_tables(benchmarks) + '\n'
                                       + learning_tables(learning, comparisons, detail) + '\n'
                                       + consistency_tables(benchmarks, learning))
    write(results / 'resources.json', resources(run, benchmarks, learning))


if __name__ == '__main__':
    main()
