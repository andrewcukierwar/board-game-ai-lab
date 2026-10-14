"""One detached, no-retry lifecycle with absolute timeout and pushed-design gate."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from .common import *
from games.connect4.alphazero_v2.experimental import digest

PACKAGE='research.alphazero_value_regularization'


def run(command,deadline,log):
    remaining=deadline-time.time()
    if remaining<=0:raise TimeoutError('Lifecycle ceiling')
    with log.open('x') as stream:
        child=subprocess.Popen(command,env=hc.thread_environment(),cwd=REPO,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            code=child.wait(timeout=remaining)
            if code:raise RuntimeError(f'Task failed (code{code}): {log.name}')
        except BaseException:
            import signal
            try:os.killpg(child.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:child.wait(timeout=3)
            except subprocess.TimeoutExpired:
                try:os.killpg(child.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                child.wait(timeout=3)
            raise


def main():
    start=time.time();runtime();freeze=enforce();design=read(RESULTS/'design.json')
    if hc.git(REPO,'status','--porcelain')!='' or hc.git(REPO,'rev-parse','--abbrev-ref','HEAD')!=BRANCH:
        raise ValueError('Launch requires clean requested branch')
    head=hc.git(REPO,'rev-parse','HEAD')
    remote=subprocess.run(['git','ls-remote','origin','refs/heads/'+BRANCH],cwd=REPO,capture_output=True,text=True,timeout=30,check=True).stdout.split()
    if not remote or remote[0]!=head:raise ValueError('Prospective HEAD not pushed')
    if sha(ROOT/'independent-prelaunch-review.md')!=freeze['prelaunch_review_sha256'] or sha(RESULTS/'prelaunch-review.json')!=freeze['prelaunch_review_evidence_sha256']:
        raise ValueError('Reviewed evidence changed')
    if read(RESULTS/'prelaunch-review.json')['disposition']!='PASS':raise ValueError('Independent review not PASS')
    (OUTPUT/'lifecycle').mkdir(parents=True,exist_ok=True)
    # Permanent no-restart ledger: every attempt, even a failure, consumes this experiment.
    write(OUTPUT/'lifecycle'/'attempt.json',dict(prospective_commit=head,started_epoch=start,design_sha256=sha(RESULTS/'design.json'),no_retry=True))
    deadline=start+design['resource']['run_ceiling_seconds'];total_deadline=start+2454
    phases=[]
    try:
        for seed in design['seeds']:
            jobs=[('initialize',0,None,0),('evaluate',0,'control',0)]
            for prev,step in zip(design['milestones'],design['milestones'][1:]):
                for arm in ('control','treatment'):
                    jobs.extend([('train',step,arm,prev),('evaluate',step,arm,step)])
            for task,updates,arm,prev in jobs:
                enforce();name=f'{seed}-{arm or "both"}-{updates}-{task}'
                command=[sys.executable,'-m',PACKAGE+'.worker',task,'--seed',str(seed),'--updates',str(updates),
                         '--previous',str(prev),'--deadline',str(deadline)]
                if arm:command+=['--arm',arm]
                before=time.time();run(command,deadline,OUTPUT/'lifecycle'/f'{name}.log')
                phases.append(dict(name=name,wall_seconds=time.time()-before))
                write(OUTPUT/'lifecycle'/f'{name}.complete.json',phases[-1])
                if task=='evaluate' and arm=='treatment':
                    a=read(OUTPUT/'records'/f'{seed}-control-{updates}-train.json')
                    b=read(OUTPUT/'records'/f'{seed}-treatment-{updates}-train.json')
                    if a['trace']!=b['trace'] or a['replay_sha256']!=b['replay_sha256'] or a['counters']!=b['counters']:
                        raise ValueError('Paired training examples/augmentation/replay/counts differ')
        # Publish raw evidence only after full ten-task completion. All new files only.
        shutil.copytree(OUTPUT/'tasks',RESULTS/'tasks')
        shutil.copytree(OUTPUT/'records',RESULTS/'records')
        integrity=preservation()
        if not integrity['intact']:raise ValueError('Historical preservation failed')
        write(RESULTS/'preservation.json',integrity)
        for module in ('analyze','independent_verify'):
            enforce();before=time.time();run([sys.executable,'-m',PACKAGE+'.'+module]+(['--root',str(RESULTS)] if module=='independent_verify' else []),total_deadline,OUTPUT/'lifecycle'/f'{module}.log')
            phases.append(dict(name=module,wall_seconds=time.time()-before))
        enforce()
        elapsed=time.time()-start
        resources=dict(complete=False,computation_complete=True,prospective_commit=head,phases=phases,wall_seconds=elapsed,
            prelaunch_charge_seconds=240,charged_seconds=elapsed+240,hard_maximum_seconds=2700,
            measured_optimizer_updates=4*design['updates_per_arm'],evaluation_tasks=10,search_roots=6280,matches=800,
            no_new_selfplay=True,no_retry=True)
        if resources['charged_seconds']>2700:raise TimeoutError('Total ceiling exceeded')
        write(RESULTS/'resources.json',resources)
        before=time.time()
        run([sys.executable,'-m',PACKAGE+'.report'],total_deadline,OUTPUT/'lifecycle'/'report.log')
        phases.append(dict(name='report',wall_seconds=time.time()-before))
        enforce()
        resources.update(complete=True,phases=phases,wall_seconds=time.time()-start,charged_seconds=time.time()-start+240)
        if resources['charged_seconds']>2700:raise TimeoutError('Reporting exceeded total ceiling')
        write(RESULTS/'resource-final.json',resources)
        write(OUTPUT/'lifecycle'/'complete.json',resources)
    except BaseException as e:
        write(OUTPUT/'lifecycle'/'failure.json',dict(error=repr(e),phases=phases,wall_seconds=time.time()-start,charged_seconds=time.time()-start+240))
        raise

if __name__=='__main__':main()
