# Phase 3B.1: explanation quality and contribution audit

Implemented locally on `phase3b-llm-explanations`, starting from a clean tree at `c5fc0f3`, October 3, 2026. The user reports that the original three modes worked locally with GPT-6 Luna and reasoning effort `none`. This refinement makes no new provider requests, changes no credentials or production settings, and does not push, merge or deploy.

**Subsequent Phase 3C status:** the user reports successful real GPT-6 Luna testing of the revised implementation committed at `45894d1`. The unverified-live statements below describe this historical refinement's verification boundary. See [the current production-readiness review](phase3c-readiness.md) and [disabled-first rollout procedure](deployment.md). Phase 3C makes no new paid requests.

## What the LLM contributes

Before this refinement, `prepare_evidence()` sent a detached board, coordinate conventions, up to four recent move records, readable deterministic facts, curated knowledge, unknowns/analysis limits and the optional untrusted question. For a hypothesis it sent the simulated resulting board. It did not send the complete history's boards, game ID, credentials or request headers as model content.

The provider made one Responses request with strict Structured Outputs. Its parsed return was exactly `fact_ids` (1–12) and `concept_ids` (1–3). Free text, unsupported fields/IDs and model citations were rejected. The provider retained neither usage metadata nor response timing. The backend displayed the selected fact order **plus every omitted fact**, selected catalog paragraphs, all reference preconditions/limitations, and four fixed methodology paragraphs. Thus the model could reorder a long report and select references, but could neither condense the visible evidence nor join facts into a causal explanation. Generic tactics/framework entries were supplied by default, and a concept selection was mandatory even on quiet openings. These constraints explain the small practical contribution observed in testing.

Afterward the same single call selects a `focus_id`, 1–3 supporting fact IDs and 0–1 concept IDs. A focus is a deterministic, evidence-referenced relationship with trusted wording, named squares, supporting fact IDs, analyzer paths and explicit concept connections. The model can select among multiple immediate wins, inaccessible completion squares or risky alternatives, and order the relevant evidence to answer the optional question. Decisive tactical conclusions take precedence, so it cannot replace a forced-loss explanation with an unrelated opening observation. The backend validates the ID selection and its relationships, safely omits irrelevant allowed supporting facts, restores missing support, and selects a relevant catalog fallback if necessary. Full facts remain available separately.

The model now controls the primary paragraph's verified focus and visible evidence emphasis. It still does not author unrestricted prose, invent relationships, calculate new sequences, evaluate optimality, or report an agent's internal rationale. Wording and tactical reasoning remain deterministic. On a position with only one meaningful focus, including the b1→b2 forced placement, its editorial contribution is deliberately small. This is a constrained educational selector, not a new solver.

## Historical live measurements

Inspected repository artifacts and the available local Phase 3B test/review artifacts. They contain mocked verification/build material, not saved live request/response or usage logs. The three successful modes are user-reported observations. Historical provider request counts, latency, token usage and cache-hit counts cannot be recovered from the available evidence; no measurements are estimated. Source inspection shows successful identical requests reuse the cache without another provider attempt. Mocked regressions verify this behavior, including the reconstructed revision-36 scenario.

## Primary-source grounding

Read the local ignored `research/references/allis-1988-connect4.pdf` directly, including Chapters 3–8. Visually inspected original diagrams 3.2, 3.9 and 6.1 and the §7.4 compatibility table. The original §3.1 discussion and diagram 3.2 (especially p.17) explain how a placement on b1/f1 lets the opponent occupy b2/f2. §3.4 (pp.21–24) explains immediate tactics; diagram 3.9 illustrates distinct playable threats. Chapters 5–8 retain the formal coverage, compatibility and evaluation-region conditions. Existing curated citation metadata is preserved: the supplied edition has 91 PDF pages, with contents-based thesis numbers matching 1-based PDF pages.

Default concepts now follow verified relationships, rather than generic keyword overlap. Gravity accessibility links to `winning_square`; mandatory defense and exhaustive next-reply losses link to `tactics`. Quiet openings show no unrelated concept. Explicitly requested rule terminology can appear in detailed reference material, always reference-only. No formal rule application, Zugzwang control or long-term result is inferred. Shortened the catalog's two displayed paraphrases; preserved the original threat-perspective convention in limitations. Every displayed thesis citation comes from the existing catalog.

The PDF remains a local reference, excluded from Git and Docker build contexts. Research extraction/rendering used temporary files and macOS PDFKit. No PDF or PDF tooling is an application dependency, and no thesis content is redistributed.

## Before and after examples

These are reproducible deterministic examples using mocked selections, not transcripts of the unavailable live responses. The revision-36 board is a legally replayed reconstruction of the reported mechanism, not the original saved game.

| Scenario | Before: separate observations in the full report | After: primary explanation |
| --- | --- | --- |
| Revision 36, only Column 2 legal | “Player 1 moves next. Legal columns: 2.” “Immediate winning columns for that player: none.” “Playing Column 2 allows an immediate winning reply in Column(s) 2.” Other completion squares, concepts and caveats follow. | “Column 2 is Player 1's only available move, but playing there places the piece on b1. This makes b2 accessible to Player 2, who can immediately play there to complete four in a row.” |
| Quiet opening | Revision, legal columns, two empty win/threat lists, a defensive-status sentence, generic concepts and methodology. | “Player 1 moves next. Neither player has an immediate winning move on this board.” No unrelated Allis concept. |
| Mandatory defense at b4 | Separate opponent threat list and mandatory-column fact. | “Player 1 must play Column 2 at b4 to block the opponent's immediate win. Every other legal move allows a winning reply.” |
| Hypothetical Column 3 ignores that defense | Simulated move, next turn, immediate winning columns, square changes and before/after lists. | “If Player 1 plays Column 3, the piece lands on c2. Player 2 can now win immediately by playing Column 2 at b4.” |

For the reconstructed revision-36 position, the original renderer produced nine visible fact paragraphs, plus selected concept text and caveats. The revised default view has the primary paragraph, at most three supporting observations, and the winning-square concept connected explicitly to b2. All original facts remain in **Detailed analysis**.

## Presentation and safeguards

The panel presents **Primary explanation**, **Key tactical evidence**, and a single **Relevant Allis concept** when helpful. **Detailed analysis** and **Methodology and limitations** use native collapsed `details` elements. References and research-integrity disclosures remain accessible. Verified square coordinates add small b1/b2-style labels and outlines to existing board buttons; they do not insert pieces, change legal moves or intercept clicks. New requests, gameplay changes, restart and cleanup clear them. Hypothetical analysis retains its explicit mode/column/revision label and game immutability.

Provider transport, deadline watchdog/socket cleanup, GPT-6 Luna / `none` defaults, backend credentials, quota/concurrency/duplicate handling, cache identity/TTL, revision checks, exact hypothetical-column matching and cancellation remain intact. No API route, game engine, gameplay agent, training, Mancala or Render configuration changes were made.

## Changed-file inventory

| Area | Files and purpose |
| --- | --- |
| Explanation composition | `api/connect4/explanation_focus.py` (new verified relationships); `api/connect4/explanations.py` (strict selection schema, relevance checks, concise/detailed response fields). |
| Curated knowledge | `games/connect4/grounding/knowledge.py` (concise winning-square/tactics paraphrases, retained perspective disclosure and citations). |
| Presentation | `ui/legacy/explanations.js` (hierarchy, disclosures, response validation, highlights/lifecycle); `ui/legacy/connect4.js` (named-square metadata); `ui/src/pages/Connect4.jsx` (short panel introduction); `ui/connect4/connect4.css` (summary, disclosures and labels). |
| Regression coverage | `tests/test_connect4_explanations.py`; `ui/tests/explanations.test.js`; `ui/e2e/explanations.spec.js`. Existing defaults test now isolates token/timeout defaults from ignored local environment configuration. |
| Local-reference protection | `.gitignore`; `.dockerignore` (exclude local research PDFs). |
| Documentation | `docs/phase3b1-quality.md` (this audit); `docs/llm-explanations.md` (response/model boundary); `docs/allis-grounding.md` (primary-source recheck); `docs/phase3b-review.md` (historical review link); `README.md`; `docs/quickstart.md`; `project_plan.md` (current status/usage). |

## Verification

All provider responses are mocked. Explanation backend tests forbid live HTTPS requests; browser explanation requests are intercepted. Local browser/API servers explicitly disable explanations and clear the key through their process environment, without editing credentials.

| Check | Result |
| --- | --- |
| Backend | **238 passed** on host Python 3.11 and **238 passed** in a disposable Linux production API container. Linux initially emitted a read-only pytest-cache warning from the intentionally read-only test mount; the final run uses a temporary cache directory. |
| Frontend/controller | **29 passed**. |
| Chromium | **15 passed** against separate-origin production hosting, **15** against Docker/Nginx, **15** against Vite development: **45 executions**. Includes complete Random/Negamax games, all explanation modes, stale/cancel behavior, exact hypothetical binding, native expand/collapse and verified-square highlighting/clearing. |
| Builds/runtime | Default, separate-origin and Render-mode Vite production builds passed; local API/UI images rebuilt and Compose health passed. Render-mode output was only built, never used for remote requests. API runtime remains key-free/disabled with gpt-6-luna/none and contains no `.env` or PDF; runtime dependency check passed. A real disabled-endpoint/gameplay smoke passed against the final container. |
| Visual check | Inspected the browser-rendered panel: summary, evidence, citation, collapsed sections and b1/b2 labels are readable and correctly located. |
| Integrity | `git diff --check` passed. Thesis PDF remains ignored/untracked and is excluded from Docker context. |

New regressions cover the b1→b2 consequence in position, hypothetical and last-move modes; quiet openings; mandatory defense; immediate opponent replies; source selection; multiple legitimate focus selections; rejected unsupported relationships/prose/references; safe omission of unrelated allowed selections; cache reuse; newer analysis surviving a late older result; disclosures and coordinate validation. Existing deadline, quotas, duplicate protection, revision/history immutability and gameplay regressions remain green.

The first host run exposed an existing defaults test's dependence on local token configuration; it now clears token/timeout environment settings alongside model/effort when testing defaults. The first browser launch used an absent default cache; subsequent suites used the existing installed Chromium under `/private/tmp/board-game-ai-lab-browsers`. No provider call was involved in either issue.

## Remaining limits

The revised schema and editorial selections have not been tested against the live provider, as requested. Historical live measurements and the original revision-36 history are unavailable. Explanations cannot answer questions beyond supplied evidence or prove deeper strategy; constrained wording remains a deliberate tradeoff. Existing process-local game/cache/quota reset behavior, CPU work before cache/quota checks, OS DNS timing boundary, proxy identity considerations and dependency advisory backlog remain as documented in the earlier review. No broader operational change is included.
