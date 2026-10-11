# Negamax v3 — completed research and recovery

2026-10-10, America/New_York. All declared studies complete. No merge/deployment.

## Verified starting scope

- Starting main SHA:60b99b0d42145da149f433907585b809060c946d.
- Current branch:research/negamax-v3; starting main is ancestor.
- Worktree/root:/Users/andrewcukierwar/Documents/GitHub/board-game-ai-lab-negamax-v3.
- Origin:https://github.com/andrewcukierwar/board-game-ai-lab.git.
- Initial clean HEAD exactly starting main; no new worktree/branch switch.
- Other MCTS/AlphaZero/Services worktrees, public infrastructure, containers,
  Tailscale, Render, main and deployment untouched.
- Canonical v2 evidence/tests retained; API/public limits/frontend unchanged.
- Only changed game production file:games/connect4/agents/negamax_agent.py.
- Worktree-local CPython3.11.17 .venv, requirements-dev +Pillow/tqdm; Torch absent.

## Phase status, hypotheses and immutable decisions

A COMPLETE: full mirror and remaining-depth>=3 selective mirror REJECTED.
Hypothesis:symmetry-orbit score reuse offsets reflection/hint-remapping cost.
Fixed broad/general/held-out latency gates failed; all exact/memory gates passed.
Full0.884xgeometric,wall1.124*A;selective0.966x,wall1.010*A.
128conditions/4224decisions/111oracle/112inherited. DESIGN/REPORT in mirror/.

B COMPLETE: PVS REJECTED as default. Integer scout/full verification hypothesis.
1.011xgeometric,1.086xbroad,wall0.886*A,retainedbroad0.898*A; fails geometric,
broad and immediate-win/D8 regression gates. Exact2816decisions/111oracle/
112inherited. DESIGN/REPORT in pvs/. No mirror combination justified.

C COMPLETE: profile strongest original direct; two independent bounded trials.
Terminal-parent proof ACCEPTED AND INTEGRATED: geometric1.135x,broad1.138x,
wall/CPU0.878*A,heldout1.130x,retained1.000*A,nogatedregressions. All trees,
leaves,terminals,probes and full TT dictionaries identical on128conditions.
Trusted TT fast path REJECTED:1.019xgeometric,wall0.979*A,too small for gates.
4224exactdecisions/111oracle/112inherited. profile/ and tuning/ designs/reports.

D COMPLETE: sixhistories atD10/D12; original vs accepted terminalproof.
264exactdecisions/33oracle;12completeTT/treeequalities;4historicalD12 vectors/
counters identical. AcceptedD12empty0.592s,near0.548s,seeded1.766s,tail2.907s.
Largest retained54.52MiB,tracedpeak56.49MiB, excludes RSS/interpreter overhead.
391.7s measuredbatchesplusauditswithin900s. Server2/4/8multipliers are planning
assumptions only, not measured MacMini throughput. No strength/cap change.
DESIGN/REPORT in deeper/.

Tooling COMPLETE: exact Git-or-literal-SHA copy fallback for shallow checkouts.
Original generator/runner bytes archived, only enumerated non-timed metadata
substitutions permitted; altered timing/gates, unknown revisions/corrupt bytes
fail closed. All prior manifests verify, missing-Git variants byte-identical.
147focusedtests. tooling-compatibility/REPORT.md and certificate preserve provenance.

E COMPLETE: initial-root-symmetry policy ELIGIBLE RESEARCH-ONLY, not promoted.
Coreoriginalcohortgeo1.012x,wall0.971*A;asymgeo0.992x;12broadsymmetricconditions
1.864x,100%faster;corememory0.962*A,nogatefailures. New prospective targeted+
neutrality gates, not relaxed PhaseA thresholds. Sixnew legal symmetric roots,
bothmovers, froze before629sresourceforecastpassed. 5016exactdecisions/129oracle/
112inherited;ALL124asymTT/treeequalities. 42focusedtests. Source/state/bounds/hints/
ties/restoration tested. DESIGN/REPORT in root-mirror/. Production stays terminal
proof; conditional-mode integration and historical counter-ablation scope are
future, separately scoped work.

All acceptance thresholds were fixed before timing. Rejected results retained.
No incomplete measured condition. Expensive tail remains a labeled diagnostic.
No paid LLM API, strength inference, native rewrite or tactical pruning.

## Current strongest engines and validation

- Strongest production-path engine: terminal-parent-proof51bd80e.
- Production commit full SHA:51bd80ed65413fe0d14dc6e6b4d04d60a3381fb6.
- Strongest eligible research strategy: that engine +root-symmetry conditional
  mirror policy, sources in root-mirror; deliberately research-only by design.
- Production bytes equal tuning/terminal-proof-source.py exactly.
- Last validated functional Git SHA:211ac78c8b0272730e6aa3264df98441be9c80a9.
- Last pushed SHA before final documentation:211ac78c8b0272730e6aa3264df98441be9c80a9,
  verified origin. Final docs tip resolves via gitHEAD/remote command below;
  self-referential commit SHA cannot be embedded inside that same commit.
- Final backend:2916passed,15Torch-dependent skips,123.95s under shared lock.
- Integration backend:2867passed,15skips,126.21s under shared lock.
- Providers disabled using exact requested envcommand; global HTTPS guard intact.
- 16,544recorded decision vectors/moves and raw evidence hashes audited across
  allfive studies. Independent arrayoracle and old goldenvectors remain active.
- Compilation/whitespace/scope pass. No frontend changes/checks needed.
- GitHub workflow triggers main/PRs; no research-branch CI run was triggered or
  dispatched. Local full suite and explicit missing-Git compatibility tested.
- Known blockers:none. No owned benchmark/validation child left running.

## Principal durable checkpoints

See validation/checkpoints.txt for full historical SHAs/messages.
A design244b0c5,freezee45cc91,completedb133403.
B design59b996f,completed1e515a9.
Profile92a272f,Cdesignee9ec57,Cevidence116c3e2,production51bd80e.
Ddesign5e79a7d,completed0f742c2.
Toolingcompatibilityf5651b7.
Edesignc38792d,resource6fdb058,completed211ac78.
WIP pushes preserve partial runs; all completed result pushes verified.

## Exact next action and recovery

The research goal is complete. Recommended next optional research is scoped
conditional-symmetry integration preserving historical counter-ablation checks,
or a bounded play/undo improvement with a measured whole-decision hypothesis.
No sufficiently small further hypothesis justifies beginning now; codec gains
were too small, state rewrites and native/pruning changes need their own scope.

1. Stay in this worktree/branch; verify pwd/root/status/startingSHAancestor.
2. Read chosen DESIGN/REPORT and create a NEW phase/evidence directory. Preserve
   all current raw files/source snapshots/criteria; never overwrite them.
3. Predeclare exact strongest-source SHA, datasets, criteria and compute budget;
   commit/push before timing. Optional integration needs separate full backend.
4. Use mandatory shared lock; wait for normal release, never remove active owner.
5. Commit/push independently validated outcomes, verify origin, update this journal.

```sh
git status --short
git branch --show-current
git merge-base --is-ancestor 60b99b0d42145da149f433907585b809060c946d HEAD
git rev-parse HEAD
git ls-remote origin refs/heads/research/negamax-v3
# Measured runs and full suites ONLY through the shared wrapper/wait helper:
scripts/wait_benchmark_slot.sh COMMAND...
# Missing declared conditions only; no completed file overwritten:
scripts/wait_benchmark_slot.sh .venv/bin/python -m scripts.resume_negamax_v3 --phase PHASE --start START --stop STOP
```

Shared directory exactly:$HOME/.cache/bgai-laptop-benchmark.lock. Atomic mkdir,
owner PID/branch/start/worktree, same-shell reliable wait/trap; SIGTERM cannot
release while foreground child remains. Isolated4lock lifecycle tests pass.
Every measured warmup/calibration included. Released between bounded batches;
MCTS pilots/confirmation ownership respected. Reproduction uses fresh --directory
with copied DESIGN/input/sources/manifest parameters; root/deeper custom selected
histories/depths must be preserved. Source metadata certificates validate frozen
old bytes without changing timed or acceptance functions. No merge or deployment.
