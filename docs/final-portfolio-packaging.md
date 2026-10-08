# Final portfolio packaging and verification

Prepared October 8, 2026 on `final-portfolio-packaging`. This is documentation and evidence packaging for the completed public Connect 4 system; no gameplay, search, evaluation, provider, dependency, or deployment code was changed.

## Repository safety

- Fetched `origin` before edits. `main` and `origin/main` both resolved to `c7d0ce65e0a2a23dc6f398164429bd1f2162228f`; the ancestry check passed.
- Starting working tree was clean and `final-portfolio-packaging` did not exist; created that branch.
- No commit, push, merge, deployment, live provider call, training, or new canonical benchmark was performed.
- `phase4d3b-alphazero-v2-preflight`, frozen AlphaZero artifacts, and historical technical reports were left intact.

## Preserved canonical run

Source: `benchmarks/connect4/runs/canonical-season-v1-2026-10-08T03-38-38-499Z-82b7a4b5/`.

Destination: `benchmarks/connect4/results/canonical-season-v1/`.

Copied `evaluation.json`, `games.csv`, `summary.csv`, and `run-metadata.json` byte-for-byte. Compared all source/destination bytes and file SHA-256 hashes; the [benchmark report](../benchmarks/connect4/results/canonical-season-v1/README.md) publishes the four file hashes. The existing ignore rule covers only transient `runs/`; no ignore changes or arbitrary run directories are needed.

The CLI verifier passed on the original before use, the stable copy after copying, and again after packaging. Every check reports 112/112 complete games and evidence SHA-256:

```text
e5845905b29c158cebed8579cd0a858b3a873a289d2b2e754c29601b7c9e407d
```

All 112 game backend manifests and runtime metadata declare the frozen source commit. The top-level CLI generator commit is null, as documented by the export contract; that distinction and its attestation limits are preserved and explained. Raw evidence was not edited or regenerated.

Independent aggregation directly from raw completed games also checked all entrant W/D/L and score rates, every pairwise outcome, per-pair 2/2 color balance, 59 Red wins / 47 Yellow wins / 6 draws, and 3,057 plies. Runtime statistics were recomputed from metadata. Every supplied benchmark figure agrees with the artifacts.

## Presentation decisions

The rewritten [root README](../README.md) opens with the purpose, live application, implemented methods, and observed result. It separates exploration, canonical results, architecture, agents, reproducibility, LLM grounding, engineering, local setup, verification, deeper documentation, and limitations. Historical phase names are confined to links near the bottom.

A GitHub-compatible Mermaid diagram describes browser interfaces, the revision-bound API, session locks, public agents, immutable evidence, replay, evaluation exports, and post-hoc analysis. The detailed benchmark report carries exact configs, intervals, Elo methodology, pairwise checks, timings, provenance, file hashes, and verification commands.

Only two existing application captures are retained in `docs/assets/`, with [explicit capture context](assets/README.md). Images of fixture Season rankings/digests were excluded. The project plan now has a current completion summary above its preserved historical record; the benchmark methodology guide links to the completed results above its historical planning notes.

Scientific limits are visible beside the root results: deterministic repeated trajectories, fixed empty-board start, four games per pair, pool/order-relative Elo, and descriptive bootstrap intervals for observed score rate. The report records that each deterministic Negamax pair has two oriented trajectories, each repeated twice. Backend runtime versions were not captured; hardware-specific timing is not a portability claim.

## Verification actually run

| Check | Result |
| --- | --- |
| Full backend `pytest tests -q` | 501 passed; 14 existing optional PyTorch skips |
| Frontend `npm test` | 398 passed, no skips |
| Default Vite production build | PASS |
| Render production build with explicit HTTPS API origin | PASS |
| Separate-origin local production build | PASS |
| Chromium full regression | 122 passed |
| Firefox configured product/export smoke | 33 passed |
| WebKit configured product/export smoke | 33 passed |
| Export smoke within those browser runs | 12 passed across three engines |
| Canonical verifier and source/destination byte/hash comparison | PASS; final check repeated after packaging |
| Independent raw-game aggregation | PASS |
| Relative documentation links and `git diff --check` | PASS |

The API was isolated on port 8002 with explanations disabled and an empty provider key; the production UI was served on port 4173 with exact-origin CORS. Existing user processes on port 8000 were left alone. Provider responses used by tests are mocked. Initial browser attempts could not launch because the default cache was empty; using the existing cache documented in the Phase 5E report resolved setup. Those launch failures were environment setup failures, not executed product assertions.

Commands (repository root, unless indicated):

```sh
EXPLANATIONS_ENABLED=false OPENAI_API_KEY='' .venv/bin/python -m pytest tests -q
# ui/:
npm test
npm run build
VITE_API_BASE=https://board-game-ai-lab.onrender.com npm run build:render
VITE_API_BASE=http://127.0.0.1:8002 npm run build
# Root, isolated local backend:
EXPLANATIONS_ENABLED=false OPENAI_API_KEY='' \
  CORS_ALLOWED_ORIGINS=http://127.0.0.1:4173 \
  .venv/bin/gunicorn api.app:app --bind 127.0.0.1:8002 --workers 1 --threads 4
# ui/, separate terminal:
npm run preview -- --host 127.0.0.1 --port 4173 --strictPort
PLAYWRIGHT_BROWSERS_PATH=/private/tmp/phase5a-browsers \
  PLAYWRIGHT_BASE_URL=http://127.0.0.1:4173 \
  PLAYWRIGHT_API_URL=http://127.0.0.1:8002 \
  npm run test:e2e -- --project=chromium --project=firefox --project=webkit
# Root:
node ui/scripts/verify-evaluation-export.mjs \
  benchmarks/connect4/results/canonical-season-v1/evaluation.json
git diff --check
```

## Files and release boundary

- `README.md`: public portfolio landing page and Mermaid architecture.
- `project_plan.md`: current completion status above unchanged historical context.
- `benchmarks/connect4/canonical-season-v1.md`: completed-result link and historical planning label.
- `benchmarks/connect4/results/canonical-season-v1/README.md`: detailed verified report.
- Four raw files in that result directory: exact canonical copies.
- `docs/assets/README.md`, `connect4-play.png`, `match-lab-replay.png`: two curated captures and provenance/context.
- `docs/final-portfolio-packaging.md`: this verification/release record.

**GO for packaging review/merge and subsequent resume preparation.** Local verification passed, including all 188 browser executions in 2.7 minutes and final canonical integrity checks. Deployment remains subject to confirming the live source/image and completing release smoke checks. Review/merge and public deployment freshness/smoke verification are subsequent release actions. Exact deployed commit/image identity was not re-inspected here. Resume editing is a subsequent task; no resume was edited. Missing backend runtime metadata is a disclosed limitation of the frozen evidence, not a field to retroactively fabricate.
