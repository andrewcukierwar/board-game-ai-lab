"""Post-execution evidence packaging only: copies JSON and log evidence, never a checkpoint; queries no model."""
import argparse
import shutil

from .common import OUTPUT, RESULTS, ROOT, read, sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    run, destination = OUTPUT / 'run', RESULTS / 'run-evidence'
    destination.mkdir(parents=True, exist_ok=False)
    manifest = {}
    for path in sorted(run.rglob('*')):
        if not path.is_file() or path.suffix not in ('.json', '.log') or '.tmp' in path.name:
            continue
        target = destination / path.relative_to(run)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        manifest[str(target.relative_to(ROOT))] = sha(target)
    status = read(run / 'status.json')
    boundaries = {f'{label}/generation-{int(g):04d}': dict(path=b['path'], sha256=b['sha256'], weights_sha256=b['weights_sha256'])
                  for label, rows in status['boundaries'].items() for g, b in rows.items()}
    write(RESULTS / 'evidence-manifest.json', dict(
        files=manifest, status=status['status'], boundaries=boundaries,
        binary_policy='Boundaries stay local and ignored by git; their file and weight hashes and local paths are listed here'))
    print(len(manifest), 'files,', sum((ROOT / k).stat().st_size for k in manifest) / 1e6, 'MB')


if __name__ == '__main__':
    main()
