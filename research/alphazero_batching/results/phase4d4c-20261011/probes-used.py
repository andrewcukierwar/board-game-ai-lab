from pathlib import Path
import json, os, subprocess
root=Path('/Volumes/Mini SSD/code/board-game-ai-lab-m4/experiment-output/phase4d4c-worktree')
python='/Users/andrewcukierwar/.venvs/board-game-alphazero-m4/bin/python'
campaign='/Volumes/Mini SSD/code/board-game-ai-lab-m4/experiment-output/phase4d3c-alphazero-v2-official-m4-20261008-110950'
env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONHASHSEED='0')
for mode,condition,games in [('fixed',None,16),('profile',None,8),('confirmation',None,16)]+[('memory',c,16) for c in ('serial','1','2','4','8')]:
    spent=sum(json.loads(p.read_text())['elapsed_seconds'] for p in (root/'experiment-output').glob('phase4d4c-*/status.json'))
    remaining=1800-spent
    if remaining<20: raise RuntimeError('Aggregate measurement budget exhausted')
    name=f'phase4d4c-{mode}'+(f'-{condition}' if condition else '')+'-20261011'
    command=[python,'-m','research.alphazero_batching.benchmark','--campaign',campaign,'--output',f'experiment-output/{name}','--mode',mode,'--games',str(games),'--seconds',str(min(180,remaining-5))]
    if condition: command += ['--condition',condition]
    print('Starting',name,'remaining aggregate seconds',round(remaining,2),flush=True)
    with (root/'experiment-output'/f'{name}-console.txt').open('x') as stream:
        result=subprocess.run(command,cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT)
    print('Finished',name,'exit',result.returncode,flush=True)
    if result.returncode: raise RuntimeError('Probe failed; retained output: '+name)
