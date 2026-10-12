# Clean independent completed-results review

Disposition: **PASS for completed evaluation, numerical reproducibility, the
actual-path preservation audit and the qualified interpretation below.** The
original canonical-benchmark inventory covers zero files at an incorrect path;
it remains a reporting defect and is not evidence. The supplemental actual-path
audit, independently verified below, resolves the preservation question for its
stated baseline paths. The earlier sealed-access incident remains disclosed;
neither review success nor byte preservation erases it.

Reviewer: `/root/clean_prelaunch_review`, October12,2026. This review used the
completed new-package outputs, frozen source/hash metadata, checkpoint hashes
and permitted nonsealed development fixtures. It did not inspect sealed fixture
contents, the raw incident response or the old substantive reviewer report, invoke the old
review script, train a model, or run additional measured model evaluation.
The final approved byte-preservation check used Git object identities/hashing
only, including sealed paths; no fixture payload was printed or interpreted.

## Completed evidence and independent verification

The run uses prospective commit83c365777f79a4883c70ac8cfcaca1a76755d160.
All48 planned tasks completed:24 position tasks containing15072 roots,20 match
tasks containing1600 completed games, and4 value tasks. No task failed and no
replacement/retry occurred. Independent file-count checks reproduce these totals.
The original55-file source map, runtime, original freeze digest, supplemental
script hashes and all12 accessed g0/g5/g6 checkpoint file hashes still match.

Fresh independent arithmetic compared640 means, including all48 pooled
factorial contrasts, with maximum discrepancy0. Fresh model reproduction
completed exactly2400 roots:200 tactical positions x A/B/E x each of4 arms.
Every stored comparison matched exact actions, visits, root priors and values.
The already-loaded replay also passed160752 independent label/encoding/key/
repetition/novelty checks, with no additional model inference for that audit.
These counts include checks of the same examples in multiple diagnostic datasets;
they are not160752 independent positions. Reproduction is same-host and retains
the unchanged canonical search; it is not external replication.

The lifecycle completed all four phases successfully. Recorded phase wall times
are864.3847s main work,19.6839s analysis,22.7230s independent arithmetic and
90.3463s independent model reproduction. The reproduction's own elapsed measure
is89.0770s. Total charged time is1177.1641s, including180s conservative prelaunch
charge, below3600s. The lifecycle status COMPLETE_PENDING_FINAL_REPORT correctly
means computation/review phases finished while final reporting is still pending.
The separately retained final-resource record charges another120s conservatively
for reporting/audit work, bringing the final charge to1297.1641s (21.62 minutes),
also below3600s, with zero additional model queries or training updates.

Official150-file preservation and all nonempty G/H-B/H-C output, source and
evidence inventories report intact with no changed launch-time files. Those
recorded checks support their stated scopes. The zero-file canonical-benchmarks
entry supports no substantive preservation conclusion and remains a path/inventory
defect. The separate final-integrity audit now compares170 baseline-tracked
game/model/test/application paths, including all11 actual frozen-package paths,
against starting commit3b563e1753ad93a1315b7d0f88a7cc45e1ee7359. This reviewer
independently recomputed baseline Git blob identities and current-file Git hashes
for all170 paths; every identity matches the recorded audit and baseline. A Git
tree filename listing also confirms the11 frozen paths are the complete tracked
baseline frozen-package set. This does not reopen the semantic fixture contents
or make the original empty inventory valid. Independent final checks also match
all28 checkpoint SHA256 identities,55 frozen source hashes and3 supplemental
script hashes. Original status/inventories were retained unchanged.

The final report's15 Markdown tables have consistent column counts. Independent
checks reproduce20 tactical rows,20 exact-accuracy rows,24 class-denominator/WDL
downgrade rows,20 match W/D/L/score rows and40 color-score rows, within stated
rounding. The reporting supplement's120 class-selected-action count vectors were
recomputed from existing evidence and agree. Its code aggregates fixed saved
endpoints only; no new model query, intervention selection or fixture membership
change occurs. Sensitive-position rankings are retrospective presentation of the
full measured set and do not create new independent confirmation evidence.

## What the experiment establishes

All table entries below are percentages. A=learned policy/value, B=learned
policy/zero value, C=uniform policy/learned value, D=uniform policy/zero value,
E=learned policy/half value. Pooled means equally weight the two fixed training
seeds within architecture. Match score gives a win1, draw0.5, loss0.

| Architecture / endpoint | A | B | C | D | E |
|---|---:|---:|---:|---:|---:|
| CNN tactical accuracy |93.25|98.00|94.75|100.00|97.00|
| ResNet tactical accuracy |94.00|97.75|97.25|100.00|96.75|
| CNN solved300 optimal accuracy |77.00|76.00|76.67|74.00|77.33|
| ResNet solved300 optimal accuracy |76.33|76.50|75.33|74.00|77.17|
| CNN deep128 optimal accuracy |75.78|71.48|73.83|66.41|75.78|
| ResNet deep128 optimal accuracy |67.58|71.48|66.02|66.41|68.36|
| CNN match score |51.875|31.875|44.375|11.875|47.813|
| ResNet match score |35.938|29.375|35.625|11.875|43.750|

Zeroing or halving the learned value improves total tactical accuracy on every
checkpoint. Fixed-analysis pooled B-A effects are CNN+4.75 percentage points
(95% cluster interval[+1.991,+8.259]) and ResNet+3.75[+1.636,+6.395].
E-A effects are CNN+3.75[+1.733,+6.250] and ResNet+2.75[+1.099,+4.847].
E fixes26 baseline tactical errors across800 checkpoint-position observations
without introducing a new tactical correctness error. B fixes35 and introduces
one; its net gain is34. These are paired observations on reused development
boards, not counts of independent situations.

The categories identify different intervention effects. C achieves100/100
immediate wins for every checkpoint; its safe-response accuracy is91/88 for
CNN seeds42/314159 and92/97 for ResNet. B safe-response accuracy rises from
90/87 to96/98 for CNN and90/93 to97/98 for ResNet. E reaches96/93 for CNN
and96/95 for ResNet. Thus learned priors also obstruct some immediate wins at
the fixed budget, while value changes recover more mandatory safe responses.
No claim about why training produced those outputs follows automatically.
D and every generation0 A-only negative control reach100% tactical accuracy.
This small tactical suite is readily saturated by near-neutral search outputs
and is insufficient as a standalone strength criterion.

Zeroing values loses match score on all four checkpoints, despite improving
tactics. The seed results make the architecture dependence of E visible:

| Arm | A score | B score | B-A pp | E score | E-A pp |
|---|---:|---:|---:|---:|---:|
| CNN42 |57.500|28.750|−28.750|49.375|−8.125|
| CNN314159 |46.250|35.000|−11.250|46.250|0.000|
| ResNet42 |30.625|27.500|−3.125|38.125|+7.500|
| ResNet314159 |41.250|31.250|−10.000|49.375|+8.125|

CNN B-A pooled match effect is−20.000pp[−29.375,−10.313]. ResNet B-A
is−6.5625pp[−16.5625,+3.4375]; both seed point estimates are negative,
but the pooled interval includes zero. With uniform priors, D-C is−32.5pp
[−40.938,−24.063] for CNN and−23.75pp[−32.188,−15.000] for ResNet.
With zero value, uniformizing priors gives D-B=−20.0pp[−28.133,−11.563]
for CNN and−17.5pp[−25.313,−9.375] for ResNet. Thus both learned outputs
retain match usefulness; the100%-tactical D control scores only11.875%.

The match factorial interaction A-B-C+D is−12.5pp[−23.438,−2.188] for
CNN and−17.1875pp[−28.125,−5.938] for ResNet. Effects of an output depend
on which other output is present; simple separate blame of one head misses that
interaction. The effect is a diagnostic of changing frozen outputs inside this
search, not a decomposition of training causes.

Halving values has architecture-dependent strength point estimates. CNN E-A
is−4.0625pp[−10.9375,+2.500]; ResNet is+7.8125pp[0,+15.625]. Both
ResNet colors improve in both seeds, so its point gain is not limited to one
color. The fixed primary ResNet interval touches zero and does not demonstrate
a confirmed strength improvement. Independent resampling produces
[+0.3125,+15.9375], illustrating Monte Carlo sensitivity at the endpoint.
Similarly, CNN C-A's fixed match interval ends at zero while the independent
interval ends at−0.3125pp. Reporting a different binary conclusion because
one bootstrap realization crosses zero would overstate these diagnostic data.
No half-value adoption, architectural promotion or universal attenuation rule
is justified by this study.

There is no consistent deeper optimal-action gain. Every pooled intervention
contrast on solved300 and deep128 has a fixed95% interval containing zero.
CNN E's unchanged pooled deep accuracy conceals opposite seed changes:
−3.90625pp at seed42 and+3.90625pp at seed314159. ResNet E gains only
0.78125pp in each seed. Solved300 contains100 theoretically lost rows where
all legal WDL labels are−1, so their optimal-action accuracy cannot establish
action quality. Deep128 contains70 wins,58 draws and no losses; the suites
must not be pooled as interchangeable distributions.

## Value evidence and limits of explanations

G6 replay realized-outcome MSE is below the constant-zero baseline in every arm:
CNN42 .5500 vs .7952, CNN314159 .5473 vs .8221, ResNet42 .4607 vs .8703,
ResNet314159 .5239 vs .8548. These compare different arms' own replay sets;
cross-architecture training MSE cannot itself rank generalization.

G5 prequential MSE on its newly generated g6 data is CNN42 .7012 vs zero .8178,
CNN314159 .6684 vs .6650, ResNet42 .9724 vs .9027 and ResNet314159 .9512
vs .8462. Excess over zero is respectively−.11656[−.22754,−.02391],
+.00334[−.08342,+.06724],+.06971[−.03183,+.17701],+.10494
[−.02565,+.23890]. Only CNN42's conditional interval clearly favors its
learned value. Both ResNet point estimates are worse than zero but their excess
intervals include zero. G6-on-g7 prequential evidence remains unavailable.
Training-to-prequential gaps can reflect generalization problems, distribution
changes and imperfect-policy target nonstationarity; they do not prove overfit.

G6 solved300 exact-value MSE is CNN42 .6471, CNN314159 .6888, ResNet42 .8032,
ResNet314159 .8618, compared with zero .6667. The paired pooled ResNet-minus-CNN
difference is+.16450[+.04979,+.28495]. On deep128 the corresponding means are
.5795/.5922/.4799/.4843 versus zero .546875; paired architecture difference
−.10378[−.31364,+.06574] includes zero. ResNet's solved mean predictions
are+.1815/+.3338 despite a zero mean exact target, and its lost-position MSE
is1.3033/1.7148. Deep128's positive target composition makes any general
architecture-wide optimism/accuracy conclusion especially unsafe.

G6 replay saturation above|v|=.95 is0.98%/2.57% for CNN and14.49%/14.76%
for ResNet; high-confidence shares above.8 are9.61%/15.14% and34.40%/33.53%.
These associations support studying value reliability, but do not identify
confidence as a causal failure mechanism. G6 identical-position outcome-noise
floors are .1559/.1804 for CNN and .1691/.1610 for ResNet, with586/593 and
385/385 conflicting repeated positions. Labels passed the independent actor/
history audit; a conflict among outcomes of identical boards is not a sign bug.
The empirical floor is descriptive of this realized dataset and not an oracle
Bayes-risk estimate for all play.

Exact-versus-realized overlap is only4 replay examples for CNN42, with1 target
disagreement, and zero for the other three arms. This cannot establish that
training systematically uses mismatched targets. Optimal-play WDL and the
realized result of imperfect play can legitimately differ on the same board.

All intervals are conditional fixed-network/family resampling, with only two
training seeds per architecture, reused development fixtures,28 deep clusters,
40 opening families and no multiplicity adjustment. Calibration, repetition,
ply and terminal-distance strata are descriptive. Replay clustering by mirrored
first-eight-ply prefix does not recover original generator trajectory identities.
There is no sealed generalization test or causal diagnosis of how training went
wrong. The incident-isolation attestations and their limits remain those stated
in the clean prelaunch review.

## A justified follow-up, without adopting a change

One prospective CNN value-head weight-decay comparison,1e-3 versus the current
1e-4, is defensible as a targeted engineering probe while retaining useful values.
It is not an identified optimum or a proven cure for overfitting. Start with a
fixed-data comparison using identical frozen replay examples, a prospectively
frozen held-out game/family/novel-position split, and identical minibatch/reflection
order, with no new self-play collection. Held-out examples must remain excluded
from training. Keep target definitions, optimizer steps, learning rate, search
and compute matched, and leave shared-trunk/policy decay unchanged. During full-network
training, altered value-head decay can still affect shared-trunk gradients; this
is one changed regularizer rather than a claim that only head weights can change.
Prequential novel-board error and matched playing strength should judge the
probe, with tactical and exact-outcome safeguards. No such change was implemented
or evaluated here. Universal zeroing/halving is unsupported by the CNN strength
results, and the ResNet half-value signal warrants restraint rather than adoption.
