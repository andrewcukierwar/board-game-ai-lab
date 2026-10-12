"""Copy finalized raw experiment folders into the committed results tree.

Game files are gzip-compressed (deterministic header, round trip verified);
everything else is copied byte-for-byte. Packaged run folders are never
overwritten; only artifact_manifest.json is regenerated.
"""
import argparse
import gzip
from pathlib import Path
import shutil

from research.alphazero_baseline import audit


def package(runs, destination):
    registry = []
    for name, source in runs.items():
        target = destination / name
        target.mkdir(parents=True, exist_ok=False)
        for path in sorted(source.iterdir()):
            if path.suffix == '.jsonl':
                with (target / f'{path.name}.gz').open('xb') as raw, \
                        gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0) as stream:
                    stream.write(path.read_bytes())
                if gzip.open(target / f'{path.name}.gz', 'rb').read() != path.read_bytes():
                    raise RuntimeError('Compressed evidence does not round-trip')
            elif path.is_file():
                shutil.copyfile(path, target / path.name)
        console = source.with_name(source.name + '-console.txt')
        if console.exists():
            shutil.copyfile(console, destination / f'{name}-console.txt')
        status = audit.read(source / 'status.json') if (source / 'status.json').exists() else {}
        registry.append(dict(name=name, raw_folder=source.name, complete=status.get('complete'),
                             elapsed_seconds=status.get('elapsed_seconds'), stop_reason=status.get('stop_reason'),
                             error=status.get('error')))
    return registry


def manifest(destination):
    return {str(p.relative_to(destination)): dict(bytes=p.stat().st_size, sha256=audit.sha(p))
            for p in sorted(destination.rglob('*')) if p.is_file() and p.name != 'artifact_manifest.json'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--run', action='append', default=[], metavar='NAME=PATH')
    parser.add_argument('--manifest-only', action='store_true')
    args = parser.parse_args()
    if not args.manifest_only:
        runs = dict(item.split('=', 1) for item in args.run)
        registry = package({k: Path(v) for k, v in runs.items()}, args.destination)
        audit.write(args.destination / 'run_registry.json', dict(runs=registry))
    path = args.destination / 'artifact_manifest.json'
    path.unlink(missing_ok=True)
    audit.write(path, manifest(args.destination))


if __name__ == '__main__':
    main()
