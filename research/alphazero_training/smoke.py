"""Disposable one-generation scratch smoke; never accepts an archived model."""
import argparse
import json
from pathlib import Path

from games.connect4.alphazero_v2.config import V2Config
from games.connect4.alphazero_v2.experimental import ExperimentalConfig, ExperimentalRunner, load_experimental, digest
from games.connect4.alphazero_v2.provenance import configure_deterministic_runtime, source_identity


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--evidence',required=True)
    args=parser.parse_args()
    runtime=configure_deterministic_runtime(1)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    config=ExperimentalConfig(V2Config(seed=44046,self_play_simulations=2,games_per_generation=8,
        batch_size=2,replay_generations=2,replay_max_games=16,max_generations=1),1)
    runner=ExperimentalRunner(config,lineage='phase4d4f-disposable-smoke',arm='scratch')
    initial=runner.save(out/'initial.pt')
    result=runner.run_generation(out/'complete.pt')
    restored=load_experimental(out/'complete.pt',expected_sha256=result['boundary_sha256'])
    if runner.initial_state_sha256()!=restored.initial_state_sha256():
        raise ValueError('Smoke resume state differs')
    evidence=dict(config=config.to_dict(),runtime=runtime,source=source_identity(),initial_file_sha256=initial,
        completed=result,round_trip_identical=True,model_hash=runner.factory.hash(runner.model),
        optimizer_digest=digest(runner.trainer.optimizer.state_dict()),replay_digest=runner.replay.digest(),
        counters=runner.counters(),global_rng_digest=digest(runner.global_rng))
    Path(args.evidence).write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps(dict(round_trip_identical=True,counters=runner.counters(),resources=result['resources']),indent=2))


if __name__=='__main__':main()
