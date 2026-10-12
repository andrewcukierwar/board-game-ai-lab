"""Freeze design, source closure, checkpoint hashes, runtime and protected bytes."""
from games.connect4.alphazero_v2.provenance import execution_source_identity
from .common import ROOT, RESULTS, REPO, entries, protected, sha, read, write, runtime, source_map, prior
from .evaluate import openings


def main():
    identity=runtime()
    assert read(RESULTS/'validation.json')['passed']
    pilot=read(RESULTS/'pilot.json')
    assert pilot['contingency_seconds']<2700
    ledger=entries()
    for entries_ in ledger.values():
        for e in entries_:
            assert sha(e['path'])==e['sha256']
    plan=dict(simulations=256,exploration=1.41,search_seed=4404901,match_seed=4404902,
              analysis_seed=4410903,resamples=10000,openings=[r['id'] for r in openings()],
              roots=15072,matches=1600,workers=3,run_ceiling_seconds=2700,hard_total_seconds=3600,
              training=False,selfplay_collection=False,variants=dict(A=['learned',1.],B=['learned',0.],
                    C=['uniform',1.],D=['uniform',0.],E=['learned',.5]))
    write(RESULTS/'design.json',plan)
    before=protected()
    write(RESULTS/'protected-before.json',before)
    write(RESULTS/'freeze.json',dict(files=source_map(),runtime=identity,checkpoints=ledger,
          execution_source=execution_source_identity(),official_files=len(prior.preserve()),
          protected_inventory_sha256=sha(RESULTS/'protected-before.json'),
          fixture_sha256=sha(REPO/'research/alphazero_resnet_selfplay/fixtures.json'),
          openings_sha256=sha(REPO/'games/connect4/alphazero_v2/frozen/openings-development.json')))
    print('Prospective freeze complete')

if __name__=='__main__':
    main()
