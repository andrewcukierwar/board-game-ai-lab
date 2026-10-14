"""Post-run hash-only preservation, inherited-file and checkpoint audit; no experiment, no sealed parsing."""
import subprocess, sys, time
from .common import *


def main():
    audit = preservation()
    changed = hc.git(REPO, 'diff', '--name-only', BASE, 'HEAD').splitlines()
    status = hc.git(REPO, 'status', '--porcelain').splitlines()
    allowed = ('research/alphazero_tactical_guard_confirmation/', 'docs/phase4d4j-b-alphazero-tactical-guard-confirmation.md')
    touched = changed + [line[3:] for line in status]
    outside = [p for p in touched if not p.startswith(allowed)]
    inherited = hc.git(REPO, 'ls-tree', '-r', '--name-only', BASE).splitlines()
    working = set(hc.git(REPO, 'diff', '--name-only', BASE).splitlines())  # working tree versus base blobs
    blob_changes = sorted(working & set(inherited))
    freeze = read(RESULTS / 'freeze.json')
    checkpoints = [dict(path=e['path'], ok=sha(e['path']) == e['sha256']) for e in freeze['checkpoints']]
    guard_ok = sha(GUARD) == GUARD_SHA256
    env = hc.thread_environment()
    tests = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', 'tests/test_connect4_mcts.py',
                            'tests/test_alphazero_v2_bitboard.py', '-k', 'not complete_selfplay and not capture'],
                           cwd=REPO, env=env, capture_output=True, text=True, timeout=300)
    result = dict(preservation_intact=audit['intact'], scopes={k: v['files'] for k, v in audit['scopes'].items()},
                  changed_in_scopes={k: v['changed'] for k, v in audit['scopes'].items() if v['changed']},
                  official_files=audit['official_files'], outside_allowed_paths=outside, inherited_tracked_files=len(inherited),
                  inherited_blob_changes=blob_changes, checkpoints=checkpoints, guard_unchanged=guard_ok,
                  regression_tests=dict(returncode=tests.returncode, tail=tests.stdout.strip().splitlines()[-1:]),
                  no_merge=True, no_deploy=True, no_training=True, no_sealed_contents=True,
                  epoch=time.time(), charged_seconds=time.time() - TASK_ORIGIN_EPOCH)
    result['status'] = 'PASS' if (audit['intact'] and not outside and not blob_changes and all(c['ok'] for c in checkpoints)
                                  and guard_ok and tests.returncode == 0) else 'FAIL'
    write(RESULTS / 'final-audit.json', result); print(result['status'], result['regression_tests'], result['charged_seconds'])


if __name__ == '__main__':
    main()
