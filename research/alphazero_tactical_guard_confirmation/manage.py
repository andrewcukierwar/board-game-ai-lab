"""Detached managed lifecycle: one attempt, frozen task list, hard deadline, no retries."""
import json, os, signal, subprocess, sys, time, shutil
from .common import *
PACKAGE = 'research.alphazero_tactical_guard_confirmation'


def stop(child):
    for sig, wait in ((signal.SIGTERM, 3), (signal.SIGKILL, 3)):
        try: os.killpg(child.pid, sig)
        except ProcessLookupError: return
        try: child.wait(timeout=wait); return
        except subprocess.TimeoutExpired: pass


def run_parallel(tasks, deadline):
    children = []
    try:
        for name, module, extra in tasks:
            stream = (OUTPUT / f'{name}.log').open('x')
            children.append((name, stream, subprocess.Popen([sys.executable, '-m', f'{PACKAGE}.{module}', *extra], cwd=REPO,
                env=hc.thread_environment(), stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)))
        while any(c.poll() is None for _, _, c in children):
            Deadline(deadline)()
            failed = [n for n, _, c in children if c.poll() not in (None, 0)]
            if failed: raise RuntimeError(f'{failed} failed; no retry')
            write(OUTPUT / 'status.json', dict(status='RUNNING', running=[n for n, _, c in children if c.poll() is None], deadline_epoch=deadline))
            time.sleep(1)
        failed = [n for n, _, c in children if c.returncode]
        if failed: raise RuntimeError(f'{failed} failed; no retry')
    except BaseException:
        for _, _, c in children: stop(c)
        raise
    finally:
        for _, s, _ in children: s.close()


def main():
    runtime(); enforce(); design = read(RESULTS / 'design.json'); deadline = design['compute_deadline_epoch']
    if hc.git(REPO, 'status', '--porcelain') != '' or hc.git(REPO, 'rev-parse', '--abbrev-ref', 'HEAD') != BRANCH:
        raise ValueError('Clean requested branch required')
    head = hc.git(REPO, 'rev-parse', 'HEAD')
    remote = subprocess.run(['git', 'ls-remote', 'origin', 'refs/heads/' + BRANCH], cwd=REPO, capture_output=True, text=True, check=True, timeout=30).stdout.split()
    if not remote or remote[0] != head: raise ValueError('Prospective commit must be pushed')
    if read(RESULTS / 'prelaunch-review.json')['disposition'] != 'PASS': raise ValueError('Prelaunch review must pass')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / 'attempt.json').open('x') as f:
        json.dump(dict(commit=head, started_epoch=time.time(), compute_deadline_epoch=deadline, no_retry=True), f); f.flush(); os.fsync(f.fileno())
    started = time.time(); phases = []
    try:
        tasks = [(f'{t["seed"]}-negamax{t["depth"]}', 'worker', ['--seed', str(t['seed']), '--depth', str(t['depth']), '--deadline', str(deadline)]
                  + (['--suites'] if t['suites'] else [])) for t in design['tasks']]
        run_parallel(tasks, deadline); phases.append(dict(name='tasks', seconds=time.time() - started))
        shutil.copytree(OUTPUT / 'tasks', RESULTS / 'tasks')
        audit = preservation(); write(RESULTS / 'preservation.json', audit); assert audit['intact']
        t = time.time(); run_parallel([('analyze', 'analyze', [])], deadline); phases.append(dict(name='analyze', seconds=time.time() - t))
        enforce()
        write(RESULTS / 'execution.json', dict(status='COMPLETE_PENDING_RESULTS_REVIEW', prospective_commit=head, tasks=len(tasks),
            matches=design['matches'], regression_searches=design['regression_searches'], phases=phases, started_epoch=started,
            finished_epoch=time.time(), task_origin_epoch=TASK_ORIGIN_EPOCH, charged_seconds=time.time() - TASK_ORIGIN_EPOCH,
            hard_maximum_seconds=HARD_MAXIMUM_SECONDS, retries=0))
        write(OUTPUT / 'status.json', dict(status='COMPLETE_PENDING_RESULTS_REVIEW', phases=phases))
    except BaseException as e:
        write(OUTPUT / 'failure.json', dict(error=repr(e), phases=phases, epoch=time.time()))
        write(OUTPUT / 'status.json', dict(status='FAILED', error=repr(e), phases=phases)); raise


if __name__ == '__main__':
    main()
