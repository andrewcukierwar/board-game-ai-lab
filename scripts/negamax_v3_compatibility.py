"""Certify two metadata-only revisions without relaxing frozen search/timing code.

Older measurements retain their exact tool bytes. Only the baseline-loader
substitution, hash-validator call/import and extra metadata hash paths below
are recognized. Every other source byte must match the archived tool.
"""
import hashlib
import json
from pathlib import Path

ROOT=Path('docs/search-negamax-v3/tooling-compatibility')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def approved_revision(path,old):
    source=old.decode()
    def once(a,b):
        nonlocal source
        assert source.count(a)==1,(path,a)
        source=source.replace(a,b)
    if path=='scripts/negamax_v3_variants.py':
        once('import subprocess\n','from scripts.negamax_v3_pinned import baseline_bytes\n')
        once("baseline = subprocess.check_output(['git', 'show', f'{BASELINE}:{AGENT_PATH}']).decode()",
             'baseline = baseline_bytes().decode()')
    elif path=='scripts/benchmark_negamax_v3.py':
        once("ROOT = Path('docs/search-negamax-v3')",
             "from scripts.negamax_v3_compatibility import verify_source_hash\n\nROOT = Path('docs/search-negamax-v3')")
        once("'games/connect4/board.py', 'scripts/with_benchmark_lock.sh')",
             "'games/connect4/board.py', 'scripts/with_benchmark_lock.sh',\n"
             "              'scripts/negamax_v3_pinned.py', 'scripts/negamax_v3_compatibility.py')")
        once('        assert digest(Path(p).read_bytes()) == expected, p',
             '        verify_source_hash(p, expected)')
    else:
        raise ValueError('Unapproved compatibility role: '+path)
    return source.encode()


def verify_source_hash(path,expected):
    current=Path(path).read_bytes()
    if digest(current)==expected:
        return
    certificate=json.loads((ROOT/'certificate.json').read_text())
    record=certificate['files'].get(path)
    assert record is not None,'Unapproved source drift: '+path
    assert record['old_sha256']==expected,'Unknown original revision: '+path
    old=(ROOT/record['archive']).read_bytes()
    assert digest(old)==expected,'Corrupt archived tool: '+path
    assert current==approved_revision(path,old),'Measured code or unapproved metadata changed: '+path
    assert digest(current)==record['new_sha256']
    for helper,sha in certificate['helpers'].items():
        assert digest(Path(helper).read_bytes())==sha,'Compatibility helper drift: '+helper
