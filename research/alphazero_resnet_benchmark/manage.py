"""Detached, bounded supervisor for the measured Phase 4D.4H-B run.

`launch` returns after a startup handshake; no interactive session has to stay
open. Stages run strictly one after another on one thread each. There is one
absolute deadline, no retry and no workload reduction: a failed or late stage
ends the run as incomplete and leaves its logs in place.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time

from .common import OUTPUT, PYTHON, REPO, RESULTS, SEEDS, enforce, preserve, read, runtime, thread_environment, write
from .validate import phase4d4g_inventory

RUN = OUTPUT / 'run'
STATUS = RUN / 'status.json'
PACKAGE = 'research.alphazero_resnet_benchmark'


def stage_commands():
    commands = [(stage, [PYTHON, '-B', '-m', f'{PACKAGE}.stages', stage]) for stage in ('inference', 'training', 'collection')]
    commands += [(f'learning-seed-{seed}', [PYTHON, '-B', '-m', f'{PACKAGE}.learn', str(seed)]) for seed in SEEDS]
    return commands


def git(*arguments):
    return subprocess.run(['git', '-C', str(REPO), *arguments], check=True, capture_output=True, text=True).stdout.strip()


def launch():
    runtime()
    design = read(RESULTS / 'design.json')
    enforce()
    if RUN.exists():
        raise SystemExit('A run directory already exists; this design permits exactly one launch')
    if git('status', '--porcelain', '--', 'research/alphazero_resnet_benchmark', 'games', 'docs'):
        raise SystemExit('Commit the frozen design before launching')
    commit = git('rev-parse', 'HEAD')
    if git('rev-parse', 'origin/phase4d4h-b-resnet-benchmark') != commit:
        raise SystemExit('Push the prospective design commit before launching')
    preserve()
    RUN.mkdir(parents=True)
    started = time.time()
    deadline = started + design['budget']['run_ceiling_seconds']
    write(RUN / 'launch.json', dict(design_commit=commit, start_epoch=started, deadline_epoch=deadline,
                                    phase4d4g_before=phase4d4g_inventory()))
    with (RUN / 'supervisor.log').open('ab') as log:
        process = subprocess.Popen([PYTHON, '-B', '-m', f'{PACKAGE}.manage', 'supervise'], cwd=REPO,
                                   env=thread_environment(), stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                   start_new_session=True)
    for _ in range(100):
        if STATUS.exists():
            break
        time.sleep(0.1)
    else:
        raise SystemExit('Supervisor did not start')
    print(json.dumps(dict(pid=process.pid, design_commit=commit, deadline_epoch=deadline, status=str(STATUS))))


def supervise():
    launch_record = read(RUN / 'launch.json')
    deadline = launch_record['deadline_epoch']
    status = dict(launch_record, pid=os.getpid(), status='RUNNING', stages=[])
    write(STATUS, status)
    try:
        before = preserve()
        for name, command in stage_commands():
            left = deadline - time.time()
            record = dict(stage=name, start_epoch=time.time(), log=str(RUN / f'{name}.log'))
            status['stages'].append(record)
            write(STATUS, status)
            if left <= 5:
                record['outcome'] = 'NOT_STARTED_DEADLINE'
                status['status'] = 'STOPPED_INCOMPLETE'
                break
            with open(record['log'], 'ab') as log:
                child = subprocess.Popen([*command, '--deadline', str(deadline)], cwd=REPO, env=thread_environment(),
                                         stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
                record['pid'] = child.pid
                write(STATUS, status)
                try:
                    code = child.wait(timeout=left)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                    code = 'deadline'
            usage = __import__('resource').getrusage(__import__('resource').RUSAGE_CHILDREN)
            record.update(end_epoch=time.time(), returncode=code, children_cpu_seconds_so_far=usage.ru_utime + usage.ru_stime,
                          children_peak_rss_mib_so_far=usage.ru_maxrss / 2 ** 20)
            record['wall_seconds'] = record['end_epoch'] - record['start_epoch']
            record['outcome'] = 'COMPLETE' if code == 0 else 'STOPPED_DEADLINE' if code == 'deadline' else 'FAILED'
            write(STATUS, status)
            if code != 0:
                status['status'] = 'STOPPED_INCOMPLETE'
                break
        else:
            status['status'] = 'COMPLETE_PENDING_ANALYSIS'
        status['official_preservation'] = 'PASS' if preserve() == before else 'FAIL'
        status['phase4d4g_after'] = phase4d4g_inventory()
        status['phase4d4g_preservation'] = 'PASS' if status['phase4d4g_after'] == launch_record['phase4d4g_before'] else 'FAIL'
        if 'FAIL' in (status['official_preservation'], status['phase4d4g_preservation']):
            status['status'] = 'INTEGRITY_FAILURE'
    except BaseException as error:
        status['status'] = 'SUPERVISOR_FAILURE'
        status['error'] = repr(error)
        raise
    finally:
        status['end_epoch'] = time.time()
        status['elapsed_seconds'] = status['end_epoch'] - status['start_epoch']
        write(STATUS, status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('launch', 'supervise', 'status'))
    args = parser.parse_args()
    if args.command == 'launch':
        launch()
    elif args.command == 'supervise':
        supervise()
    else:
        print(json.dumps(read(STATUS), indent=1) if STATUS.exists() else 'not launched')


if __name__ == '__main__':
    main()
