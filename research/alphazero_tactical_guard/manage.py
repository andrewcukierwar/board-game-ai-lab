"""Detached managed lifecycle, one attempt, hard deadline, no automatic retries."""
import os,signal,subprocess,sys,time,shutil
from .common import *
PACKAGE='research.alphazero_tactical_guard'


def run(module,extra,deadline,log):
    with log.open('x') as stream:
        child=subprocess.Popen([sys.executable,'-m',PACKAGE+'.'+module,*extra],cwd=REPO,env=hc.thread_environment(),
            stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while child.poll() is None:
                Deadline(deadline)();time.sleep(.25)
            if child.returncode:raise RuntimeError(f'{module} failed: {child.returncode}; no retry')
        except BaseException:
            try:os.killpg(child.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:child.wait(timeout=3)
            except subprocess.TimeoutExpired:
                try:os.killpg(child.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                child.wait(timeout=3)
            raise


def main():
    runtime();freeze=enforce();design=read(RESULTS/'design.json');deadline=design['absolute_deadline_epoch']-180
    if hc.git(REPO,'status','--porcelain')!='' or hc.git(REPO,'rev-parse','--abbrev-ref','HEAD')!=BRANCH:
        raise ValueError('Clean requested branch required')
    head=hc.git(REPO,'rev-parse','HEAD')
    remote=subprocess.run(['git','ls-remote','origin','refs/heads/'+BRANCH],cwd=REPO,capture_output=True,text=True,check=True,timeout=30).stdout.split()
    if not remote or remote[0]!=head:raise ValueError('Prospective commit must be pushed')
    if read(RESULTS/'prelaunch-review.json')['disposition']!='PASS':raise ValueError('Prelaunch review must pass')
    OUTPUT.mkdir(parents=True,exist_ok=True);attempt=OUTPUT/'attempt.json'
    with attempt.open('x') as f:
        import json
        json.dump(dict(commit=head,started_epoch=time.time(),absolute_deadline=design['absolute_deadline_epoch'],no_retry=True),f)
        f.flush();os.fsync(f.fileno())
    phases=[]
    try:
        for seed in (42,314159):
            enforce();started=time.time()
            write(OUTPUT/'status.json',dict(status='RUNNING',seed=seed,phases=phases,deadline_epoch=deadline))
            run('worker',['--seed',str(seed),'--deadline',str(deadline)],deadline,OUTPUT/f'{seed}.log')
            phases.append(dict(name=str(seed),seconds=time.time()-started))
        shutil.copytree(OUTPUT/'tasks',RESULTS/'tasks')
        audit=preservation();assert audit['intact'];write(RESULTS/'preservation.json',audit)
        for module in ('analyze','independent_verify'):
            enforce();started=time.time();run(module,['--results','--output',str(RESULTS/'independent-verification.json')] if module=='independent_verify' else [],deadline,OUTPUT/f'{module}.log')
            phases.append(dict(name=module,seconds=time.time()-started))
        enforce();design=read(RESULTS/'design.json')
        write(RESULTS/'execution.json',dict(status='COMPLETE_PENDING_RESULTS_REVIEW',prospective_commit=head,
            fixture_searches=2512,fixture_paired_records=1256,matches=320,phases=phases,
            started_epoch=read(attempt)['started_epoch'],finished_epoch=time.time(),
            task_origin_epoch=design['task_origin_epoch'],charged_seconds=time.time()-design['task_origin_epoch'],
            hard_maximum_seconds=2700,absolute_deadline_epoch=design['absolute_deadline_epoch'],retries=0))
        write(OUTPUT/'status.json',dict(status='COMPLETE_PENDING_RESULTS_REVIEW',phases=phases))
    except BaseException as e:
        write(OUTPUT/'failure.json',dict(error=repr(e),phases=phases,epoch=time.time()))
        write(OUTPUT/'status.json',dict(status='FAILED',error=repr(e),phases=phases));raise
if __name__=='__main__':main()
