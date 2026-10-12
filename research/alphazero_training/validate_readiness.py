"""Read-only official import/readiness screen: zero collection, zero updates.

Run as python -m research.alphazero_training.validate_readiness with all five
thread environment variables = 1. Stores bulky converted boundaries only in
an explicitly new scratch output directory; commits JSON evidence separately.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

from games.connect4.alphazero_v2.artifacts import file_sha256
from games.connect4.alphazero_v2.config import V2Config
from games.connect4.alphazero_v2.experimental import (ExperimentalConfig, ORIGINAL, EXTENDED,
                                                     load_experimental, digest)
from games.connect4.alphazero_v2.experimental_import import read_historical, fork_verified
from games.connect4.alphazero_v2.provenance import configure_deterministic_runtime, source_identity


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--official',required=True)
    parser.add_argument('--historical-checkout',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--evidence',required=True)
    args=parser.parse_args()
    runtime=configure_deterministic_runtime(1)
    official=Path(args.official).resolve()
    output=Path(args.output).resolve(); output.mkdir(parents=True,exist_ok=False)
    evidence=Path(args.evidence)
    results={}
    for seed in (42,314159):
        start=time.perf_counter()
        gen=official/f'runs/seed-{seed}/generations/generation-0020'
        parent=gen/'generation-0020.resume.pt'
        summary=json.loads((gen/'summary.json').read_text())
        # SHA comes from the archived generation manifest, not an invented hash.
        artifact=summary['files']['resume']
        sha=artifact['sha256']
        assert file_sha256(parent)==sha
        verified=read_historical(parent,expected_sha256=sha,historical_checkout=args.historical_checkout)
        old=V2Config.from_dict(verified['original']['config'])
        config=ExperimentalConfig(replace(old,games_per_generation=64,max_generations=22),64)
        arms={name:fork_verified(verified,config,parent_sha256=sha,schedule=schedule,
                     lineage=f'phase4d4f-readiness-seed-{seed}',arm=name)
              for name,schedule in (('control',ORIGINAL),('treatment',EXTENDED))}
        initial={name:r.initial_state_sha256() for name,r in arms.items()}
        assert len(set(initial.values()))==1
        records={}
        for name,r in arms.items():
            directory=output/f'seed-{seed}'/name; directory.mkdir(parents=True)
            path=directory/'fork-0020.pt'
            file_hash=r.save(path)
            loaded=load_experimental(path,expected_sha256=file_hash,expected_config=config)
            assert loaded.initial_state_sha256()==r.initial_state_sha256()
            assert r.counters()==verified['original']['counters']
            assert r.replay.digest()==verified['verified_replay']
            assert r.factory.hash(r.model)==verified['original']['learner_weights_sha256']
            assert digest(r.trainer.optimizer.state_dict())==digest(verified['original']['optimizer_state_dict'])
            assert r.trainer.steps==verified['original']['counters']['trainer_steps']
            records[name]=dict(path=str(path),file_sha256=file_hash,initial_state_sha256=initial[name],
                               model_hash=r.factory.hash(r.model),optimizer_digest=digest(r.trainer.optimizer.state_dict()),
                               replay_digest=r.replay.digest(),counters=r.counters(),schedule=r.schedule.to_dict(),
                               replay_generations=list(r.replay.generations), replay_games=r.replay.games,
                               replay_positions=len(r.replay),augmentation_counts=r._content()['augmentation'] | {'rng':'verified_equal'},
                               global_rng_digest=digest(r.global_rng))
        results[str(seed)]=dict(parent_file_sha256=sha,parent_source=verified['original']['source'],
                               parent_runtime=verified['original']['runtime'],arms=records,
                               identical_initial_states=True,collection_games=0,optimizer_updates=0,
                               validation_seconds=time.perf_counter()-start)
        print(f'Seed {seed}: strict historical fork and both arm reloads PASS',flush=True)
    evidence.write_text(json.dumps(dict(runtime=runtime,source=source_identity(),seeds=results),indent=2)+'\n')


if __name__=='__main__': main()
