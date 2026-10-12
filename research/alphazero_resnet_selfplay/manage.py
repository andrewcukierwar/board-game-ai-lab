"""Detached, bounded supervisor for the Phase 4D.4H-C run.

    launch    verify the freeze, record the launch, start the supervisor in its own session and return
    status    print the durable ledger summary
    cancel    request cooperative cancellation (workers stop at their next check; nothing is retried)

Training runs strictly one generation at a time, each in a fresh interpreter
that strictly reloads the previous boundary, so the measured cost of one arm
is never disturbed by another task of this run. Evaluation starts only after
the last training generation and runs in a small pool of single-thread
workers; its results are deterministic and its timing is not part of any
comparison.

There is one absolute deadline, no retry, no extra generation and no workload
reduction. A failed or late training task ends the run as incomplete. A failed
evaluation task is recorded and never retried; independent evaluation tasks
still run. `launch --smoke` runs a tiny disposable design in a separate
directory to exercise this exact code path before the freeze.
"""
import argparse
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys
import time
import traceback

from .common import (ARMS, BRANCH, OUTPUT, PACKAGE, PYTHON, REPO, RESULTS, arm_label, compare_protected,
                     describe_external_commits, enforce, git, preserve, protected_inventory, read, runtime, sha,
                     thread_environment, worktree_observations, write)

RECOVERY = ('STOP. Nothing is retried automatically and no further generation may be run under this declaration. '
            'Published boundaries are complete generations: inspect one with '
            '`python -m games.connect4.alphazero_v2.experimental_cli inspect --boundary <path> --sha256 <sha>` from this '
            'worktree with the five thread variables set to 1. A partially run generation leaves no boundary and is '
            'discarded. Any continuation needs a new prospective declaration.')


def run_root(smoke):
    return OUTPUT / ('smoke' if smoke else 'run')


def log(message):
    print(f'{time.strftime("%Y-%m-%dT%H:%M:%S")} {message}', flush=True)


def launch(smoke):
    from . import prepare
    runtime()
    root = run_root(smoke)
    if root.exists():
        raise SystemExit(f'{root} already exists; this design permits exactly one launch')
    if smoke:
        design, commit = prepare.smoke_design(), git(REPO, 'rev-parse', 'HEAD')
        root.mkdir(parents=True)
        write(root / 'design.json', design)
        initial = prepare.initial_boundaries(design, root / 'arms')
    else:
        freeze = enforce()
        design, initial = read(RESULTS / 'design.json'), freeze['initial']
        if git(REPO, 'status', '--porcelain', '--', 'research/alphazero_resnet_selfplay', 'games', 'docs', 'tests'):
            raise SystemExit('Commit the frozen design before launching')
        commit = git(REPO, 'rev-parse', 'HEAD')
        if git(REPO, 'rev-parse', f'origin/{BRANCH}') != commit:
            raise SystemExit('Push the prospective design commit before launching')
        for label, record in initial.items():
            if sha(record['path']) != record['sha256']:
                raise SystemExit(f'Initial boundary {label} differs from the freeze')
        root.mkdir(parents=True)
        shutil.copyfile(RESULTS / 'design.json', root / 'design.json')
    preserve()
    started = time.time()
    write(root / 'launch.json', dict(smoke=smoke, design_commit=commit, start_epoch=started,
                                     deadline_epoch=started + design['budget']['run_ceiling_seconds'],
                                     initial=initial, protected_before=protected_inventory(),
                                     worktrees_before=worktree_observations()))
    with (root / 'supervisor.log').open('ab') as stream:
        process = subprocess.Popen([PYTHON, '-B', '-m', f'{PACKAGE}.manage', 'supervise', *(['--smoke'] if smoke else [])],
                                   cwd=REPO, env=thread_environment(), stdin=subprocess.DEVNULL, stdout=stream,
                                   stderr=stream, start_new_session=True)
    for _ in range(150):
        if (root / 'status.json').exists():
            break
        if process.poll() is not None:
            raise SystemExit('Supervisor exited during startup; inspect supervisor.log')
        time.sleep(0.2)
    else:
        os.killpg(process.pid, signal.SIGTERM)
        raise SystemExit('Supervisor did not start; it was stopped')
    print(json.dumps(dict(pid=process.pid, design_commit=commit, root=str(root),
                          deadline_epoch=started + design['budget']['run_ceiling_seconds'])))


class Supervisor:
    def __init__(self, smoke):
        self.root = run_root(smoke)
        self.launch = read(self.root / 'launch.json')
        self.design = read(self.root / 'design.json')
        self.deadline = self.launch['deadline_epoch']
        self.cancel = self.root / 'CANCEL'
        self.running = []
        self.boundaries = {label: {'0': dict(path=r['path'], sha256=r['sha256'], weights_sha256=r['weights_sha256'])}
                           for label, r in self.launch['initial'].items()}
        self.status = dict(smoke=smoke, status='RUNNING', phase='starting', pid=os.getpid(),
                           design_commit=self.launch['design_commit'], start_epoch=self.launch['start_epoch'],
                           deadline_epoch=self.deadline, tasks=[], boundaries=self.boundaries,
                           work={arm_label(*arm): dict(games=0, updates=0, generations=0) for arm in ARMS},
                           retries=0, recovery=RECOVERY)
        self.save()

    def save(self):
        self.status['elapsed_seconds'] = time.time() - self.status['start_epoch']
        write(self.root / 'status.json', self.status)

    def stop_reason(self):
        if time.time() >= self.deadline:
            return 'hard deadline reached'
        if self.cancel.exists():
            return 'cooperative cancellation requested'
        return None

    def start(self, kind, seed, name, generation=None, extra=()):
        label = f'{arm_label(seed, name)}' + (f'-g{generation}' if generation is not None else '') + f'-{kind}'
        output = self.root / 'tasks' / label
        record = dict(kind=kind, label=label, seed=seed, architecture=name, generation=generation, state='STARTED',
                      start_epoch=time.time(), output=str(output), log=str(self.root / 'logs' / f'{label}.log'))
        (self.root / 'logs').mkdir(exist_ok=True)
        (self.root / 'tasks').mkdir(exist_ok=True)
        command = [PYTHON, '-B', '-m', f'{PACKAGE}.worker', kind, '--root', str(self.root), '--output', str(output),
                   '--deadline', str(self.deadline - 5), '--seed', str(seed), '--name', name, *map(str, extra)]
        if generation is not None:
            command += ['--generation', str(generation)]
        stream = open(record['log'], 'xb')
        child = subprocess.Popen(command, cwd=REPO, env=thread_environment(), stdin=subprocess.DEVNULL, stdout=stream,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        stream.close()
        record['pid'] = child.pid
        self.status['tasks'].append(record)
        self.running.append((child, record))
        self.save()
        log(f'started {label} pid {child.pid}')
        return child, record

    def finish(self, child, record):
        self.running = [(c, r) for c, r in self.running if c is not child]
        output = Path(record['output'])
        record.update(end_epoch=time.time(), returncode=child.returncode)
        record['wall_seconds'] = record['end_epoch'] - record['start_epoch']
        complete = output / 'task-complete.json'
        if child.returncode == 0 and complete.exists():
            record.update(state='COMPLETE', resources=read(complete))
        else:
            record.update(state='FAILED', recovery=RECOVERY)
            if (output / 'progress.json').exists():
                record['partial_progress'] = read(output / 'progress.json')
        self.save()
        log(f'{record["state"]} {record["label"]} in {record["wall_seconds"]:.1f}s')
        return record['state'] == 'COMPLETE'

    def terminate(self, grace=20):
        """Workers stop cooperatively at their next check; anything left is signalled, then killed."""
        until = time.time() + grace
        for child, _ in list(self.running):
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for child, record in list(self.running):
            try:
                child.wait(timeout=max(0.1, until - time.time()))
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            self.finish(child, record)

    def wait_one(self):
        """Block until one running child exits; stop everything on deadline or cancellation."""
        while True:
            for child, record in self.running:
                if child.poll() is not None:
                    return self.finish(child, record)
            reason = self.stop_reason()
            if reason:
                self.terminate()
                raise TimeoutError(reason) if 'deadline' in reason else InterruptedError(reason)
            time.sleep(0.2)

    def train(self, seed, name, generation):
        label = arm_label(seed, name)
        source = self.boundaries[label][str(generation - 1)]
        child, record = self.start('train', seed, name, generation, ('--boundary', source['path'], '--sha256', source['sha256']))
        if not self.wait_one():
            raise RuntimeError(f'Training task failed: {record["label"]}')
        done = read(Path(record['output']) / 'complete.json')
        if sha(done['boundary']) != done['sha256'] or done['generation'] != generation:
            raise ValueError('Published boundary differs from its completion record')
        self.boundaries[label][str(generation)] = dict(path=done['boundary'], sha256=done['sha256'],
                                                       weights_sha256=done['after']['model'])
        work = self.status['work'][label]
        work['games'] += done['result']['games']
        work['updates'] += done['result']['updates']
        work['generations'] = generation
        arm = self.design['arms'][name]
        if (work['games'], work['updates']) != (generation * arm['games_per_generation'], generation * arm['optimizer_updates']):
            raise ValueError('Cumulative work differs from the frozen budget')
        record['timing'] = done['timing']
        self.save()

    def evaluation_tasks(self):
        plan, final = self.design['evaluation'], self.design['generations']
        tasks = [('matches', seed, name, final) for seed, name in ARMS]
        tasks += [('positions', seed, name, final) for seed, name in ARMS]
        tasks += [('dynamics', seed, name, None) for seed, name in ARMS]
        for generation in sorted((g for g in plan['generations'] if g != final), reverse=True):
            tasks += [('matches', seed, name, generation) for seed, name in ARMS]
            tasks += [('positions', seed, name, generation) for seed, name in ARMS]
        return tasks

    def evaluate(self):
        pending, failed = self.evaluation_tasks(), 0
        while pending or self.running:
            while pending and len(self.running) < self.design['evaluation']['workers']:
                kind, seed, name, generation = pending.pop(0)
                label = arm_label(seed, name)
                if kind == 'dynamics':
                    listing = self.root / 'tasks' / f'{label}-boundaries.json'
                    write(listing, [dict(generation=int(g), **{k: v for k, v in b.items() if k != 'weights_sha256'})
                                    for g, b in sorted(self.boundaries[label].items(), key=lambda item: int(item[0]))])
                    self.start(kind, seed, name, None, ('--boundaries', listing))
                else:
                    source = self.boundaries[label][str(generation)]
                    self.start(kind, seed, name, generation, ('--boundary', source['path'], '--sha256', source['sha256']))
            failed += not self.wait_one()
        if failed:
            raise RuntimeError(f'{failed} evaluation task(s) failed; none was retried')

    def integrity(self, official_before):
        """Strict end-of-run checks. Every failure is recorded by name; none is overridden."""
        checks = {}
        try:
            checks['official_campaign_150_files'] = 'PASS' if preserve() == official_before else 'FAIL'
        except BaseException as error:
            checks['official_campaign_150_files'] = f'FAIL: {error!r}'
        comparison = compare_protected(self.launch['protected_before'], protected_inventory())
        checks['protected_experiment_bytes'] = 'PASS' if comparison['intact'] else 'FAIL'
        mismatched = [f'{label}/g{g}' for label, rows in self.boundaries.items() for g, b in rows.items()
                      if not Path(b['path']).exists() or sha(b['path']) != b['sha256']]
        checks['checkpoint_hashes'] = 'PASS' if not mismatched else f'FAIL: {mismatched}'
        if self.status['smoke']:
            checks['frozen_sources_runtime_design'] = 'NOT APPLICABLE (smoke)'
        else:
            try:
                enforce()
                same = sha(self.root / 'design.json') == sha(RESULTS / 'design.json')
                checks['frozen_sources_runtime_design'] = 'PASS' if same else 'FAIL: run design differs'
            except BaseException as error:
                checks['frozen_sources_runtime_design'] = f'FAIL: {error!r}'
        self.status['integrity'] = checks
        self.status['protected_trees'] = comparison['trees']
        # Reported separately and never a failure by itself: commits other sessions made meanwhile.
        self.status['external_commits'] = describe_external_commits(self.launch['worktrees_before'], worktree_observations())
        return all(v == 'PASS' or v.startswith('NOT APPLICABLE') for v in checks.values())

    def run(self):
        def interrupted(signum, frame):
            raise InterruptedError(f'supervisor received signal {signum}')
        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        official_before = None
        try:
            official_before = preserve()
            self.status['phase'] = 'training'
            for generation, order in enumerate(self.design['order'], start=1):
                for seed, name in order:
                    self.train(seed, name, generation)
            self.status['training_end_epoch'] = time.time()
            self.status['phase'] = 'evaluation'
            self.save()
            self.evaluate()
            self.status['status'] = 'COMPLETE_PENDING_ANALYSIS'
        except BaseException as error:
            self.status.update(status='STOPPED_INCOMPLETE', error=repr(error), traceback=traceback.format_exc())
            log(f'stopped: {error!r}')
        finally:
            if self.running:
                self.terminate()
            self.status['phase'] = 'finished'
            try:
                intact = self.integrity(official_before if official_before is not None else {})
            except BaseException as error:
                intact = False
                self.status['integrity_error'] = repr(error)
            if not intact:
                self.status['status_before_integrity'] = self.status['status']
                self.status['status'] = 'INTEGRITY_FAILURE'
            usage = resource.getrusage(resource.RUSAGE_CHILDREN)
            self.status.update(end_epoch=time.time(), children_cpu_seconds=usage.ru_utime + usage.ru_stime,
                               children_peak_rss_mib=usage.ru_maxrss / (2 ** 20 if sys.platform == 'darwin' else 2 ** 10))
            self.save()
            log(f'final status {self.status["status"]} after {self.status["elapsed_seconds"]:.1f}s')
        return 0 if self.status['status'] == 'COMPLETE_PENDING_ANALYSIS' else 1


def summarize(status):
    tasks = status['tasks']
    return dict(status=status['status'], phase=status['phase'], elapsed_minutes=round(status['elapsed_seconds'] / 60, 2),
                minutes_to_deadline=round((status['deadline_epoch'] - time.time()) / 60, 2),
                tasks_complete=sum(t['state'] == 'COMPLETE' for t in tasks), tasks_failed=sum(t['state'] == 'FAILED' for t in tasks),
                tasks_running=[t['label'] for t in tasks if t['state'] == 'STARTED'], work=status['work'],
                integrity=status.get('integrity'), error=status.get('error'))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=('launch', 'supervise', 'status', 'cancel'))
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    root = run_root(args.smoke)
    if args.command == 'launch':
        launch(args.smoke)
    elif args.command == 'supervise':
        runtime()
        raise SystemExit(Supervisor(args.smoke).run())
    elif args.command == 'status':
        print(json.dumps(summarize(read(root / 'status.json')), indent=1) if (root / 'status.json').exists() else 'not launched')
    else:
        (root / 'CANCEL').touch()
        print('cancellation requested; workers stop at their next check and nothing is retried')


if __name__ == '__main__':
    main()
