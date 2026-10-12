"""Post-run adjudication of the supervisor's INTEGRITY_FAILURE status, then the frozen analysis.

The supervisor's Phase 4D.4G check compared the worktree commit as well as
the output inventory. During the run a separate session committed Phase
4D.4G's final results in that worktree, so the commit changed and the check
failed. status.json is left exactly as the supervisor wrote it. This tool
re-derives, from Git and from file hashes, whether anything this phase is
required to preserve changed, writes that evidence, and only if every
condition holds calls the unmodified frozen analyze.analyze(). It is not part
of the frozen source set and was written after the run.
"""
import json
import subprocess

from . import analyze
from .common import OUTPUT, RESULTS, SEEDS, preserve, read, sha, source_map, write
from .validate import PHASE4D4G, PHASE4D4G_HEAD, phase4d4g_inventory

RUN = OUTPUT / 'run'


def git(*arguments):
    return subprocess.run(['git', '-C', str(PHASE4D4G), *arguments], check=True, capture_output=True, text=True).stdout.strip()


def main():
    status = read(RUN / 'status.json')
    before, after = status['phase4d4g_before'], status['phase4d4g_after']
    current = phase4d4g_inventory()
    head = current['head']
    commits = git('rev-list', '--reverse', f'{PHASE4D4G_HEAD}..{head}').split()
    changed = git('diff', '--name-only', PHASE4D4G_HEAD, head).splitlines()
    ancestor = subprocess.run(['git', '-C', str(PHASE4D4G), 'merge-base', '--is-ancestor', PHASE4D4G_HEAD, head]).returncode == 0
    freeze = read(RESULTS / 'freeze.json')
    conditions = {
        'every stage completed with return code 0': all(s['outcome'] == 'COMPLETE' and s['returncode'] == 0 for s in status['stages'])
        and len(status['stages']) == 5,
        'the only failed check is the Phase 4D.4G comparison': status['status'] == 'INTEGRITY_FAILURE'
        and status['official_preservation'] == 'PASS' and status['phase4d4g_preservation'] == 'FAIL',
        'all 150 official files are byte-identical now': len(preserve()) == 150,
        'Phase 4D.4G output inventory is byte-identical before, after and now':
            before['inventory_sha256'] == after['inventory_sha256'] == current['inventory_sha256']
            and before['files'] == after['files'] == current['files'] == 151,
        'the only field that differs between before and after is the commit':
            {k for k in before if before[k] != after[k]} == {'head'},
        'the new Phase 4D.4G commit descends from the frozen one': ancestor and before['head'] == PHASE4D4G_HEAD,
        'the new commits change no package source': not any(p.startswith('games/') for p in changed),
        'the Phase 4D.4G worktree is clean': current['clean'],
        'the frozen sources, design and dataset of this phase are unchanged': source_map() == freeze['files'],
    }
    record = dict(
        supervisor_status=status['status'], status_file_sha256=sha(RUN / 'status.json'), before=before, after=after, current=current,
        new_commits=[dict(commit=c, author=git('show', '-s', '--format=%an', c), date=git('show', '-s', '--format=%aI', c),
                          subject=git('show', '-s', '--format=%s', c)) for c in commits],
        changed_paths=len(changed), changed_top_level=sorted({p.split('/')[0] for p in changed}),
        run_start_epoch=status['start_epoch'], run_end_epoch=status['end_epoch'], conditions=conditions,
        finding=('The supervisor reported INTEGRITY_FAILURE because the Phase 4D.4G worktree commit changed during the run. '
                 'A separate session committed that phase\'s final analysis there. No Phase 4D.4G checkpoint, boundary or '
                 'execution evidence file changed, and this phase has no code path that writes to that worktree.'),
        consequence=('The same session was computing on this machine during the inference, training-benchmark, collection and '
                     'seed-42 learning stages, so those timings were not taken on an otherwise idle machine.'),
        adjudication='ANALYZE' if all(conditions.values()) else 'DO_NOT_ANALYZE')
    write(RESULTS / 'integrity-adjudication.json', record)
    print(json.dumps(dict(conditions=conditions, adjudication=record['adjudication'], commits=record['new_commits']), indent=1))
    if record['adjudication'] != 'ANALYZE':
        raise SystemExit('Preservation could not be established; the run is not analyzed')
    analyze.analyze(RUN, RESULTS, read(RESULTS / 'design.json'), SEEDS)


if __name__ == '__main__':
    main()
