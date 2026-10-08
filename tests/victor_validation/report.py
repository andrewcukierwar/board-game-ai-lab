"""Reproduce the bounded report; stdout only, never rewrite baselines on failure.

Run: PYTHONPATH=tests .venv/bin/python -m victor_validation.report
Add --include-generated to print every generated replay and observation.
"""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import sys

from .strategic_harness import compare_history, generated_histories

FIXTURES = Path(__file__).with_name('positions.json')


def digest(records):
    return sha256(json.dumps(records, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def summarize(records):
    counts = Counter((r['cover_status'], r['white_value']) for r in records)
    execution = [r['execution'] for r in records if r['execution']]
    return {
        'positions': len(records), 'unique_boards': len({tuple(r['board']) for r in records}),
        'counts': [{'cover_status': s, 'white_value': v, 'count': n}
                   for (s, v), n in sorted(counts.items())],
        'potential_counterexamples': [r for r in records if r['potential_counterexample']],
        'execution_statuses': dict(sorted(Counter(r['status'] for r in execution).items())),
        'max_oracle_positions': max((r['oracle_positions'] for r in records), default=0),
        'max_execution_positions': max((r['visited_positions'] for r in execution), default=0),
        'execution_edges': {key: sum(r[key] for r in execution) for key in (
            'white_edges', 'forced_reply_edges', 'spare_reply_edges', 'proactive_pair_edges')},
        'records_sha256': digest(records),
    }


def build_report(inputs, *, include_generated=False):
    generated = []
    batches = []
    for spec in inputs['generator_specs']:
        records = [compare_history(m) for m in generated_histories(**spec)]
        batches.append({'spec': spec, 'summary': summarize(records)})
        generated.extend(records)
    curated = []
    for case in inputs['cases']:
        for reflected in (False, True):
            moves = [6 - c for c in case['moves']] if reflected else case['moves']
            curated.append({'id': case['id'] + ('-reflected' if reflected else ''),
                            **compare_history(moves)})
    source = [2, 3, 3, 3, 3, 3, 3, 4]
    source_checks = [compare_history(m) for m in (source, [6 - c for c in source])]
    covered = next(c['moves'] for c in inputs['cases'] if c['id'] == 'claim-base-draw')
    uncovered = next(c['moves'] for c in inputs['cases'] if c['id'] == 'immediate-white-win')
    resource_checks = [
        {'id': 'cover-positive-cutoff', **compare_history(covered, cover_budget=1)},
        {'id': 'cover-zero-cutoff', **compare_history(uncovered, cover_budget=0)},
        {'id': 'oracle-cutoff', **compare_history(covered, oracle_budget=1)},
        {'id': 'execution-cutoff', **compare_history(covered, execution_budget=1)},
    ]
    report = {'generated_batches': batches, 'generated_summary': summarize(generated),
              'curated': curated, 'curated_summary': summarize(curated),
              'source_checks': source_checks, 'resource_checks': resource_checks}
    if include_generated:
        report['generated'] = generated
    return report


if __name__ == '__main__':
    inputs = json.loads(FIXTURES.read_text())['inputs']
    print(json.dumps(build_report(inputs, include_generated='--include-generated' in sys.argv), indent=2))
