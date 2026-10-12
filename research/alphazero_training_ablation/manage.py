"""Detached supervisor: sequential tasks, durable ledger, owned process watchdog."""
import argparse
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path
from .common import ROOT, REPO, OUTPUT, PYTHON, THREADS, read, write, sha, enforce, preserve

ORDER=((42,'control',21),(42,'treatment',21),(314159,'treatment',21),(314159,'control',21),
       (42,'treatment',22),(42,'control',22),(314159,'control',22),(314159,'treatment',22))

def execute(command, seconds, log, *, cooperate=True):
    """Kill only this new child's process group. Hard deadline includes shutdown."""
    with Path(log).open('xb') as stream:
        child=subprocess.Popen(command,cwd=REPO,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            child.wait(timeout=max(0.01,seconds-3))
        except BaseException:
            os.killpg(child.pid,signal.SIGTERM)
            try: child.wait(timeout=min(2,max(.01,seconds)))
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL); child.wait()
            raise
        if child.returncode: raise RuntimeError(f'Child exit {child.returncode}: {log}')

def run():
    from games.connect4.alphazero_v2.provenance import configure_deterministic_runtime
    configure_deterministic_runtime(1); freeze=enforce(); preserve()
    readiness=read(ROOT/'results/readiness.json')
    if not readiness['estimate']['feasible']: raise ValueError('Readiness infeasible')
    if sha(ROOT/'results/readiness.json')!=freeze['readiness_sha256']: raise ValueError('Readiness changed')
    git=lambda *args: subprocess.check_output(['git',*args],cwd=REPO,text=True).strip()
    design=git('rev-parse','HEAD')
    if git('status','--porcelain'): raise ValueError('Commit all changes before launch')
    if git('rev-parse','@{upstream}') != design: raise ValueError('Prospective commit must be pushed')
    started=time.monotonic(); deadline=started+10800
    runroot=OUTPUT/'execution'; runroot.mkdir(exist_ok=False)
    ledger=dict(status='RUNNING',pid=os.getpid(),design_commit=design,start_epoch=time.time(),
        deadline_epoch=time.time()+10800,wall_ceiling_seconds=10800,order=ORDER,tasks=[],primary_games=0,primary_updates=0)
    current={(int(s),a):dict(v) for s,record in readiness['seeds'].items() for a,v in record['arms'].items()}
    def save():
        ledger['elapsed_seconds']=time.monotonic()-started; write(runroot/'status.json',ledger)
    save()
    def interrupted(signum, frame):
        raise InterruptedError(f'Supervisor signal {signum}; stop owned task and retain evidence')
    signal.signal(signal.SIGTERM,interrupted)
    signal.signal(signal.SIGINT,interrupted)
    def task(kind,key,label,limit=10800,extra=()):
        enforce()
        # Reserve 30 seconds for final official inventory and durable shutdown.
        remaining=deadline-time.monotonic()-30; available=min(limit,remaining)
        if available<=4: raise TimeoutError('Primary deadline reached')
        source=current[key]; output=runroot/label
        command=[PYTHON,'-B','-m','research.alphazero_training_ablation.worker',kind,
            '--boundary',source['path'],'--sha256',source['sha256'],'--output',str(output),
            '--deadline',str(time.monotonic()+available-3),*extra]
        attempt=dict(task=kind,key=key,label=label,output=str(output),log=str(runroot/f'{label}.log'),
                     state='STARTED',start_elapsed=time.monotonic()-started,limit_seconds=available)
        ledger['tasks'].append(attempt); save()
        try:
            execute(command,available,attempt['log'])
            complete=read(output/'task-complete.json'); attempt.update(state='COMPLETE',resources=complete)
            if kind=='train':
                record=read(output/'complete.json')
                current[key]=dict(path=record['boundary'],sha256=record['sha256'])
                ledger['primary_games']+=record['result']['games']; ledger['primary_updates']+=record['result']['updates']
                if ledger['primary_games']>512 or ledger['primary_updates']>512: raise ValueError('Primary work exceeds budget')
            save()
        except BaseException as error:
            attempt.update(state='FAILED',error=repr(error),recovery='STOP; no automatic retry. Strictly inspect published boundary; never use live attempted runner.')
            if (output/'progress.json').exists():
                attempt['partial_work_lower_bound']=read(output/'progress.json')
                attempt['partial_work_upper_bound']=dict(games=64,updates=64,
                    reason='Cancellation can occur after an operation and before its progress callback; no retry permitted')
            if (output/'complete.json').exists(): attempt['published_boundary']=read(output/'complete.json')
            save(); raise
    try:
        # Each parent's identical model is evaluated once per seed, reused explicitly for both arms.
        for seed in (42,314159): task('positions',(seed,'control'),f'parent-{seed}')
        for seed,arm,generation in ORDER:
            task('train',(seed,arm),f'{seed}-{arm}-g{generation}',1200)
            task('positions',(seed,arm),f'{seed}-{arm}-g{generation}-evaluation')
        for seed,arm in ((42,'control'),(42,'treatment'),(314159,'treatment'),(314159,'control')):
            parent=readiness['seeds'][str(seed)]['arms']['control']
            task('matches',(seed,arm),f'{seed}-{arm}-strength',extra=('--parent',parent['path'],
                 '--parent-sha256',parent['sha256'],'--seed',str(seed)))
        if (ledger['primary_games'],ledger['primary_updates'])!=(512,512): raise ValueError('Incomplete primary budgets')
        ledger['status']='COMPLETE_PENDING_FINAL_ANALYSIS'
    except BaseException as error:
        ledger['status']='STOPPED_INCOMPLETE'; ledger['error']=repr(error); ledger['traceback']=traceback.format_exc()
    finally:
        try: preserve(); ledger['official_preservation']='PASS'
        except BaseException as error: ledger['official_preservation']=repr(error); ledger['status']='INTEGRITY_FAILURE'
        ledger['last_boundaries']={f'{s}-{a}':v for (s,a),v in current.items()}; save()
    return 0 if ledger['status']=='COMPLETE_PENDING_FINAL_ANALYSIS' else 1

def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('launch','run','status')); args=parser.parse_args()
    if args.command=='run': raise SystemExit(run())
    if args.command=='status':
        print(__import__('json').dumps(read(OUTPUT/'execution/status.json'),indent=2)); return
    # The supervisor gets its own session and durable output, independent of this shell/Codex.
    if (OUTPUT/'execution').exists() or (OUTPUT/'supervisor.log').exists(): raise ValueError('Launch already attempted')
    env=os.environ.copy(); env.update({k:'1' for k in THREADS}); env['PYTHONDONTWRITEBYTECODE']='1'
    with (OUTPUT/'supervisor.log').open('xb') as stream:
        process=subprocess.Popen([PYTHON,'-B','-m','research.alphazero_training_ablation.manage','run'],
            cwd=REPO,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    write(OUTPUT/'launch.json',dict(pid=process.pid,log=str(OUTPUT/'supervisor.log'),command='manage run',launched_epoch=time.time()))
    # Short startup handshake only, never wait for training.
    until=time.monotonic()+20
    while time.monotonic()<until:
        if (OUTPUT/'execution/status.json').exists(): print(f'Supervisor running: PID {process.pid}'); return
        if process.poll() is not None: raise RuntimeError('Supervisor startup failed; inspect supervisor.log')
        time.sleep(.2)
    os.killpg(process.pid,signal.SIGTERM)
    try: process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid,signal.SIGKILL); process.wait()
    raise RuntimeError('Startup handshake failed; supervisor stopped')

if __name__=='__main__': main()
