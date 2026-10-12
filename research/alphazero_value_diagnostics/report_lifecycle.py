"""Additional integrity gate and absolute total-resource watchdog; original protocol unchanged.

No training, retries, expanded evaluation, Git mutations or fixture access.
Runs the original measured manager followed by the fixed analysis and fresh
independent reproduction within the original 3600-second total cap.
"""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
RESULTS=ROOT/'results'
RECORDS=REPO/'experiment-output/phase4d4i-a-lifecycle'
PYTHON='/Users/andrewcukierwar/.venvs/board-game-alphazero-m4/bin/python'
PACKAGE='research.alphazero_value_diagnostics'
THREADS=('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')


def read(path):
    return json.loads(path.read_text())


def write(path,value):
    temp=path.with_suffix('.tmp')
    with temp.open('w') as stream:
        json.dump(value,stream,indent=1,sort_keys=True,allow_nan=False);stream.write('\n')
        stream.flush();os.fsync(stream.fileno())
    os.replace(temp,path)


def verify_hashes(gate):
    for path,digest in gate['supplemental_sources'].items():
        assert hashlib.sha256((REPO/path).read_bytes()).hexdigest()==digest,path
    freeze_path=RESULTS/'freeze.json'
    assert hashlib.sha256(freeze_path.read_bytes()).hexdigest()==gate['original_freeze_sha256']
    for path,digest in read(freeze_path)['files'].items():
        assert hashlib.sha256((REPO/path).read_bytes()).hexdigest()==digest,path


def verify():
    gate=read(RESULTS/'integrity-resolution.json')
    assert gate['launch_disposition']=='PASS' and gate['clean_reviewer_no_inherited_history']
    verify_hashes(gate)
    assert not (RESULTS/'lifecycle.json').exists(),'No retries: lifecycle already attempted'
    assert not (RECORDS/'lifecycle.json').exists(),'No retries: ignored attempt ledger already exists'
    assert not (RESULTS/'status.json').exists(),'No retries: manager already attempted'
    assert gate['prelaunch_charge_seconds']==180 and gate['hard_total_seconds']==3600
    return gate


def main():
    gate=verify()
    started=time.time()
    # Thirty seconds reserved for owned-child cleanup, counted within the total cap.
    deadline=started+gate['hard_total_seconds']-gate['prelaunch_charge_seconds']-30
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',**{k:'1' for k in THREADS})
    RECORDS.mkdir(parents=True,exist_ok=True)
    ledger=dict(status='RUNNING',pid=os.getpid(),started_epoch=started,absolute_total_deadline=deadline+30,
                prelaunch_charge_seconds=180,hard_total_seconds=3600,phases=[],retries=0)
    child=None; log=None
    def interrupted(*_):
        raise InterruptedError('Lifecycle cancellation')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    def stop_child():
        if child is not None and child.poll() is None:
            # The unchanged manager owns cleanup of its worker process groups.
            try:
                os.killpg(child.pid,signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(child.pid,signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait(timeout=3)
    phases=(('measured','manage',2700),('analysis','analyze',240),
            ('independent-arithmetic','independent_clean_review',240),
            ('independent-model-rerun','independent_clean_rerun',240))
    try:
        write(RECORDS/'lifecycle.json',ledger)
        for label,module,ceiling in phases:
            verify_hashes(gate)
            if time.time()>=deadline or (RESULTS/'CANCEL').exists():
                raise TimeoutError('Original total measured cap reached')
            if label!='measured' and read(RESULTS/'status.json')['status']!='COMPLETE':
                raise RuntimeError('Incomplete measured work; no analyses or retries')
            phase=dict(name=label,started_epoch=time.time(),ceiling_seconds=ceiling)
            ledger['phases'].append(phase);write(RECORDS/'lifecycle.json',ledger)
            log=(RECORDS/f'lifecycle-{label}.log').open('w')
            phase_deadline=min(deadline,phase['started_epoch']+ceiling)
            command=[PYTHON,'-m',f'{PACKAGE}.{module}']
            if label=='independent-model-rerun':
                command.extend(['--deadline',str(phase_deadline)])
            child=subprocess.Popen(command,cwd=REPO,env=env,
                                   stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            phase['pid']=child.pid
            phase_deadline=min(deadline,phase['started_epoch']+ceiling)
            while child.poll() is None:
                if time.time()>=phase_deadline or (RESULTS/'CANCEL').exists():
                    raise TimeoutError(f'{label}: bounded deadline/cancel; no retry')
                time.sleep(.5)
            phase.update(returncode=child.returncode,finished_epoch=time.time(),
                         seconds=time.time()-phase['started_epoch'])
            log.close();log=None
            if child.returncode!=0:
                raise RuntimeError(f'{label}: failed; no retry')
            child=None;write(RECORDS/'lifecycle.json',ledger)
        verify_hashes(gate)
        ledger['status']='COMPLETE_PENDING_FINAL_REPORT'
    except BaseException as error:
        ledger['status']='STOPPED';ledger['reason']=repr(error)
    finally:
        stop_child()
        if log is not None:log.close()
        ledger['finished_epoch']=time.time()
        ledger['postlaunch_seconds']=ledger['finished_epoch']-started
        ledger['charged_total_seconds']=180+ledger['postlaunch_seconds']
        ledger['hard_cap_satisfied']=ledger['charged_total_seconds']<=3600
        write(RECORDS/'lifecycle.json',ledger)
        # Reporting starts only after the unchanged manager's clean/pushed launch check.
        write(RESULTS/'lifecycle.json',ledger)
        import shutil
        for console in RECORDS.glob('lifecycle-*.log'):
            shutil.copyfile(console,RESULTS/console.name)

if __name__=='__main__':
    main()
