"""Postexecution evidence packaging only; no checkpoint queries or training."""
import shutil
from .common import ROOT, OUTPUT, read, write, sha

def main():
    execution=OUTPUT/'execution'
    destination=ROOT/'results/execution-evidence'
    destination.mkdir(parents=True,exist_ok=False)
    manifest={}
    for p in sorted(execution.rglob('*')):
        if not p.is_file() or p.suffix not in ('.json','.log'): continue
        # Completed rows plus durable attempts; partial evaluations retained separately on disk.
        if p.name.endswith('-partial.json'): continue
        target=destination/p.relative_to(execution); target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(p,target); manifest[str(target.relative_to(ROOT))]=sha(target)
    for name in ('launch.json','supervisor.log'):
        p=OUTPUT/name
        if p.exists():
            target=destination/name; shutil.copyfile(p,target); manifest[str(target.relative_to(ROOT))]=sha(target)
    write(ROOT/'results/evidence-manifest.json',dict(files=manifest,
        binary_policy='Boundaries remain ignored; file/state hashes and absolute recovery paths are in sidecars',
        status=read(execution/'status.json')['status']))

if __name__=='__main__': main()
