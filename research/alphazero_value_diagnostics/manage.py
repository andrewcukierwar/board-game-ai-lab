"""Detached bounded manager: exactly 48 tasks, three owned workers, no retries."""
import os
from pathlib import Path
import signal
import subprocess
import time
from .common import RESULTS, REPO, PYTHON, ARMS, VARIANTS, runtime, enforce, read, write, preservation, prior


def tasks():
    jobs=[]
    for arm in ARMS:
        for variant in VARIANTS:
            for task in ('positions','matches'):
                jobs.append(dict(task=task,arm=arm,generation=6,variant=variant,label=f'{arm}-g6-{variant}-{task}'))
        jobs.append(dict(task='positions',arm=arm,generation=0,variant='A',label=f'{arm}-g0-A-positions'))
        jobs.append(dict(task='values',arm=arm,generation=6,variant='A',label=f'{arm}-values'))
    assert len(jobs)==48
    return jobs


def main():
    runtime();freeze=enforce()
    plan=read(RESULTS/'design.json')
    head=prior.git(REPO,'rev-parse','HEAD')
    remote=prior.git(REPO,'ls-remote','origin','refs/heads/phase4d4i-a-alphazero-value-diagnostics')
    if not remote or remote.split()[0]!=head or prior.git(REPO,'status','--porcelain'):
        raise ValueError('Measured work requires clean, pushed prospective HEAD')
    started=time.time(); deadline=started+plan['run_ceiling_seconds']
    queue=tasks(); live=[]; finished=[]
    before=read(RESULTS/'protected-before.json')
    status=dict(status='RUNNING',pid=os.getpid(),prospective_commit=head,remote_at_launch=remote,
                started_epoch=started,deadline_epoch=deadline,planned_tasks=len(queue),planned_matches=1600,
                planned_roots=15072,completed=[],failed=[],active=[])
    def update():
        status.update(completed=finished.copy(),active=[j['label'] for _,j,_ in live],elapsed_seconds=time.time()-started)
        write(RESULTS/'status.json',status)
    def cancel(*_):
        raise InterruptedError('Manager cancellation requested')
    signal.signal(signal.SIGTERM,cancel);signal.signal(signal.SIGINT,cancel)
    try:
        while queue or live:
            if time.time()>=deadline or (RESULTS/'CANCEL').exists():
                raise TimeoutError('Hard stop reached; no retries')
            while queue and len(live)<3:
                job=queue.pop(0)
                log=(RESULTS/f'{job["label"]}.log').open('w')
                command=[PYTHON,'-m','research.alphazero_value_diagnostics.worker',job['task']]
                for name in ('arm','generation','variant','label'):
                    command.extend([f'--{name}',str(job[name])])
                command.extend(['--deadline',str(deadline)])
                process=subprocess.Popen(command,cwd=REPO,env=prior.thread_environment(),stdout=log,stderr=subprocess.STDOUT,
                                         start_new_session=True)
                live.append((process,job,log))
            for process,job,log in list(live):
                rc=process.poll()
                if rc is None:
                    continue
                log.close();live.remove((process,job,log))
                if rc!=0:
                    status['failed'].append(dict(label=job['label'],returncode=rc))
                    raise RuntimeError(f'Task failed: {job["label"]}; no retries')
                if not read(RESULTS/'tasks'/job['label']/'complete.json')['complete']:
                    raise RuntimeError('Missing task completion')
                finished.append(job['label'])
            update();time.sleep(1)
        status['status']='COMPLETE'
    except BaseException as error:
        status['status']='STOPPED';status['reason']=repr(error)
    finally:
        for process,job,log in live:
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
        for process,job,log in live:
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=3)
            log.close()
        live.clear()
        status['finished_epoch']=time.time()
        status['remaining_tasks']=[j['label'] for j in queue]
        try:
            enforce()
            status['preservation']=preservation(before)
            if not status['preservation']['intact']:
                status['status']='INTEGRITY_FAILURE'
        except BaseException as error:
            status['status']='INTEGRITY_FAILURE';status['integrity_error']=repr(error)
        update()

if __name__=='__main__':
    main()
