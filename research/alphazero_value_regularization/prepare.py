"""Prepare immutable membership/design then freeze reviewed source/runtime bytes."""
import argparse
import time
from .common import *
from .data import membership


def design():
    pilot=read(RESULTS/'pilot.json')
    n=pilot['selected_updates']
    membership_value=membership()
    write(RESULTS/'membership.json',membership_value)
    from research.alphazero_resnet_selfplay.evaluate import openings
    write(RESULTS/'design.json',dict(schema=1,lineage=LINEAGE,base_commit=BASE,seeds=[42,314159],
        arms=dict(control=0.0,treatment=0.2),updates_per_arm=n,milestones=[0,n//2,n],
        learning_rate=3e-4,weight_decay=1e-4,batch_size=128,max_grad_norm=5.,
        training='retained g5 replay (generations1..5), no data collection',
        heldout='all new generation6 complete games from own-seed frozen CNN checkpoint, excluded from training',
        primary=dict(endpoint='treatment-minus-control novel heldout realized-outcome MSE at final milestone',
            practical_margin=0.02,success='both seed differences <= -0.02 and both seed 95% CI upper bounds < 0; pooled descriptive only'),
        uncertainty=dict(replicates=10000,seed=4411403,method='paired percentile bootstrap whole opening-family clusters; all complete-game rows within family retained; position-weighted MSE',
            interpretation='conditional on frozen seeds/data, no training-seed population inference',multiplicity='none; final primary endpoint only'),
        safeguards=dict(tactical=dict(endpoint='each immediate-win/safe-response raw-PUCT category at final milestone vs control and initial',
                pass_point=-0.03,pass_lower=-0.10,fail_point=-0.05,fail_upper=0.),
            deep=dict(endpoint='deep128 exact mistakes final treatment-minus-control and treatment-minus-initial',
                fail_point=0.05,fail_lower=0.),
            strength=dict(endpoint='paired negamax3 score final treatment-minus-control and treatment-minus-initial',
                pass_point=-0.05,pass_lower=-0.15,fail_point=-0.05,fail_upper=0.,supporting=True,
                interpretation='40 paired opening families is underpowered for small differences; cannot establish noninferiority')),
        evaluation=dict(simulations=256,search_seed=4404901,match_seed=4404902,opponent_depth=3,
            tactical=200,solved=300,deep=128,openings=[x['id'] for x in openings()],matches_per_task=80,
            raw_output_scale=1.,policy_namespace='phase4d4h-c-policy',search_namespace='phase4d4h-c-evaluation',match_namespace='phase4d4h-c-strength'),
        resource=dict(target_minutes=[20,30],hard_maximum_seconds=2700,prelaunch_charge_seconds=240,
            run_ceiling_seconds=2360,postprocessing_reserve_seconds=100,workers=1,
            no_retry=True,no_extension=True,no_additional_arms=True),pilot_sha256=sha(RESULTS/'pilot.json'),
        membership_sha256=sha(RESULTS/'membership.json'),parents=[parent_entry(s,g) for s in (42,314159) for g in (5,6)]))
    write(RESULTS/'protected-before.json',protected())
    print('Prospective memberships and design prepared',flush=True)


def freeze():
    review=read(RESULTS/'prelaunch-review.json')
    validation=read(RESULTS/'validation.json')
    if review['disposition']!='PASS' or not validation['passed']:
        raise ValueError('Independent prelaunch review and focused tests must pass')
    write(RESULTS/'freeze.json',dict(files=source_map(),runtime=runtime_identity(),
        parents=read(RESULTS/'design.json')['parents'],prelaunch_review_sha256=sha(ROOT/'independent-prelaunch-review.md'),
        prelaunch_review_evidence_sha256=sha(RESULTS/'prelaunch-review.json'),validation_sha256=sha(RESULTS/'validation.json')))
    print('Prospective freeze recorded',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('task',choices=['design','freeze']);args=p.parse_args()
    runtime();globals()[args.task]()
