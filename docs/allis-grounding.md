# Phase 3A: deterministic Connect 4 grounding

This Phase 3A layer records moves and produces verifiable, revision-bound evidence. It does not itself call an LLM or choose moves. The separate [Phase 3B service and panel](llm-explanations.md) now consume it locally. The gameplay endpoints and their JSON response shapes are unchanged. Random and Negamax still select moves exactly as before.

## Use

```python
from api.connect4.evidence import get_explanation_context

context = get_explanation_context(
    app.extensions['connect4_games'], game_id, expected_revision=7,
)
```

The internal service uses the existing session store and checks the requested revision under its lock. It copies the board/configuration and captures the immutable history, releases the lock, then builds evidence for that captured revision. A later move does not change the payload's revision or contents. A busy game, expired/replaced/missing game, or stale revision raises the same `GameError` categories used by gameplay. No new HTTP route is introduced in Phase 3A.

For offline analysis:

```python
from games.connect4.grounding import analyze_position, analyze_move

facts = analyze_position(board, player_to_move=0)
move_facts = analyze_move(board, player=0, column=3)
```

These functions copy rather than mutate the board. Inputs must be 6-by-7 matrices of `" "`, `"X"`, `"O"`, with valid gravity, alternating-player counts, and a consistent side to move. Dual winners and wins that cannot be explained by removing a topmost last stone are rejected. These checks are not an exhaustive proof of historical reachability; the session context builder additionally replays the complete recorded game from empty, checking every transition and stopping at terminal play.

## Coordinates and turn semantics

| Field | Convention |
| --- | --- |
| `board[row_index][column]` | Matrix rows 0–5 run **top to bottom**; columns 0–6 run left to right. |
| Square object | `{row_index: 5, column: 3, row: 1, name: "d1"}` is the bottom center square. |
| Named square / parity | Allis's columns a–g and rows **1–6 from bottom to top**. `row = 6 - row_index`. Odd/even always refers to this row, never the array index. |
| Player | `0` / X / Player 1 / Allis's White moves first; `1` / O / Player 2 / Black moves second. |
| Move number | One placed stone = one move (ply), starting at 1. Allis's written game lists instead pair White and Black moves. |
| Revision | Starts at 0; increases by one only on a successful committed move. |
| Terminal side to move | The engine's toggled player is retained for consistency. No legal moves are offered after a win/draw. |

## Evidence schema (version 1.0)

All public grounding functions return JSON-ready values. Full field examples are generated and validated by `tests/test_connect4_evidence.py`. No prompt text or provider-specific format is embedded.

| Payload field | Meaning |
| --- | --- |
| `schema_version` | `"1.0"`. |
| `provenance` | Game ID, captured revision, deterministic post-hoc method, replay verification, and `agent_reasoning_available: false`. |
| `coordinates` | Matrix, named-square, player and move-number conventions. |
| `position` | Detached board, side to move, player configurations and actual outcome. |
| `move_history` | Complete ordered move records for this session through the captured revision. |
| `confirmed_tactical_facts` | Current board analysis described below. |
| `last_move_facts` | Verified effects and legal alternatives from the previous position, or `null` before the first move. |
| `supported_allis_rule_applications` | **Always `[]` in Phase 3A.** None of the nine formal rules is asserted. |
| `general_strategic_observations` | Bounded contextual observations linked to concept IDs. They are explicitly marked `context_only`, never treated as a result proof. |
| `unknown_or_unproven` | Missing agent rationale, deeper evaluation/optimality, formal-rule/Zugzwang proofs, and VictorAgent limitations. |
| `analysis_limits` | Search horizon and meanings of survival, square changes and unsupported formal rules. |
| `knowledge` | Primary-source metadata, pagination note, retrieval policy and relevant curated entries. |

Each immutable `MoveRecord` contains `move_number`, `revision_before`, `revision`, `player`, agent type/depth, selected `column`, immutable before/after matrices, outcome status and winner. Its detached JSON representation nests configuration in `agent` and outcome in `outcome`. Outcomes are `{status: "ongoing"|"win"|"draw", winner: 0|1|null}`. Winner is `null` except for an actual win.

Recording takes place after a successful candidate move and before session commit, under the existing lock. Rejections, stale revisions, full columns, failed agents and retries do not append records. Each game stores at most 42 records. History shares session lifetime: replacement starts fresh and removes the old session; idle expiration or process restart loses it. Capacity, TTL and single-worker/instance requirements are unchanged. History is internal and is not added to existing gameplay responses.

## Tactical facts and their limits

The analyzer enumerates all 69 four-square lines and all legal columns. It reports:

- **Winning lines and actual outcome.** Terminal boards have no legal actions or immediate winning moves.
- **Winning squares for each player.** Every entry has a completion square, bottom-based parity, gravity playability, and the four-square groups it would complete. Multiple groups sharing a square produce one square entry. Unsupported squares remain geometric patterns; they are not legal winning moves.
- **Immediate winning columns for the side to move.** Opponent playable winning squares describe what would win if the opponent could move on the unchanged board. They do not assign the turn to the opponent.
- **Every legal alternative.** Includes landing square, immediate outcome, every opponent immediate winning reply, and `avoids_immediate_loss`. This last field means only that the opponent cannot win on the next move. It is not a long-term safety or optimality claim.
- **Defensive status.** `mandatory_block` is emitted only when there is no immediate win for the mover, an opponent immediate threat exists, and exactly one legal move avoids an immediate loss. The unique column is recorded. If every legal move loses on the next reply, status is `unavoidable_loss_next_reply` with no recommended block. Winning now takes precedence over blocking. Other statuses are `no_immediate_threat`, `immediate_win_available`, `defensive_options`, and `not_applicable_terminal`.
- **Last-move effects.** The played square, actual outcome, whether it was an immediate win or mandatory block, and created/removed/newly playable winning squares for both players. Changes compare square sets, not every line's identity. A removed square may have been occupied to complete a win; a newly playable square may have been an existing geometric threat supported by the move. Terminal-board patterns remain geometric and are not future actions.

No multi-turn threat-space search, fork scoring, deeper forced-win search, move ranking, game-theoretic value, or strategic rule proof is performed. Agent choice and the analyzer's post-hoc tactical description have separate provenance. Negamax has not been instrumented to expose its reasoning and is not said to use Allis's rules.

## Primary source and pagination

Victor Allis (1988), *A Knowledge-Based Approach of Connect-Four: The Game Is Solved: White Wins*, [full thesis](https://tromp.github.io/c4/connect4_thesis.pdf).

The actual PDF was read for this implementation, including nomenclature; threat and parity analysis; Zugzwang; all nine formal definitions; rule interactions; application scope; and evaluation. The local PDF/text/rendering tools used for research are not project dependencies or runtime ingestion. The thesis itself is not redistributed in this repository.

**Pagination:** `thesis_pages` records page numbers assigned by the thesis contents, not a guessed offset from the cover. In this supplied 91-page PDF these equal the **1-based PDF page numbers**. Thus §6.1 starts on thesis page 36 / PDF page 36 / zero-based PDF index 35. §7.4 is thesis page 50 / PDF page 50 / index 49. The rendered body pages have no visible printed page folios, so references do not pretend otherwise. Every entry explicitly stores thesis page ranges, 1-based PDF page ranges, and zero-based PDF index ranges. Ranges are inclusive.

## Curated knowledge and formal-rule boundary

`games/connect4/grounding/knowledge.py` contains 15 short paraphrased entries, each with a stable ID, name, explanation, preconditions, limitations, chapter/section/page references, source URL, evidence mapping, and applicability status. The six conceptual entries cover coordinates, winning squares, tactics, parity, Zugzwang, and the rule framework. The nine rule entries are **reference-only**, with implementation status `not_implemented`. That status describes the explanation catalog, which deliberately does not use the separate research implementation of all nine rules in `games/connect4/victor/` ([report](victor-nine-rule-implementation.md)).

| Rule | Thesis / 1-based PDF pages | Key local requirement; see catalog for solutions and limitations |
| --- | --- | --- |
| Claimeven (§6.1) | 36–37 | Empty adjacent vertical pair with even upper square. |
| Baseinverse (§6.2) | 37–38 | Two distinct directly playable squares; relevant groups require both. |
| Vertical (§6.3) | 38–39 | Empty adjacent vertical pair with odd upper square (standalone rule). |
| Aftereven (§6.4) | 39–40 | Applying player's group can be completed using Claimeven squares; the timing solution must cover every missing-square column above the group. |
| Lowinverse (§6.5) | 40–41 | Empty vertical pairs in two different columns, both upper squares odd. |
| Highinverse (§6.6) | 41–42 | Empty three-square segments in two columns, both tops even; extra bottom/top solutions require the corresponding bottom to be playable. |
| Baseclaim (§6.7) | 42–43 | Three distinct playable squares with assigned roles and an even square immediately above the second. |
| Before (§6.8) | 43–45 | Applying player's unblocked group, no missing square on the top row, with the appropriate component pairs. |
| Specialbefore (§6.9) | 45–46 | Before-type setup with an internal playable square and extra playable square in another column, with the special response substitution. |

These are local requirements, **not sufficient position-level applicability checks**. Chapters 5, 7 and 8 matter:

- §5.4 (pp.34–35) requires an appropriate covering set of solutions. Failure to obtain it leaves the position unresolved, rather than proving a loss.
- Chapter 6 (p.36) specifies opponent-to-move evaluation and White's restricted region. §§8.1–8.4 (pp.51–57) explain Black's attempted coverage and White's odd-threat/threat-combination setups; §9.2 (pp.58–59) describes the permitted evaluation region and solution coverage.
- §7.2 (p.49) identifies Baseinverse and Vertical as Zugzwang-independent. That does not mean arbitrary local instances prove a draw or combine without conflicts.
- §5.3 (pp.33–34) shows why sharing a triggering square can demand incompatible replies. The §7.4 table (p.50) uses pair-specific constraints: square disjointness, no Claimeven below an inverse, column-wise disjointness/equality, and inverse column-set restrictions. Multiple listed conditions must all hold; Specialbefore adds special-square restrictions. Disjoint patterns alone are not a universal compatibility test.

Because Phase 3A does not implement those evaluation-region, compatibility and coverage checks, it deliberately does not label any geometric pattern as a supported formal application. This also prevents a simplistic odd/even count or empty pair from becoming an invented Zugzwang or Claimeven proof.

Retrieval is deterministic. Context always includes coordinates, tactical limitations and the rule framework. Winning-square and parity entries are added when the current board or last-move changes contain winning-square evidence. Callers may request particular reference concepts with `concept_ids`, but retrieving a rule never asserts it applies. `retrieve_knowledge` rejects unknown IDs, deduplicates requests, returns catalog order and detaches values from the catalog. `knowledge_catalog()` exposes the entire curated reference for review. No embeddings, database, or PDF access is required.

## Existing VictorAgent review

The experimental `games/connect4/agents/victor_agent.py` is not the VICTOR described in the thesis. Its A1/A2/B/C/D labels are prototype-specific, not Allis's nine strategic rules. Its five-square scanning can omit four-square edge patterns; the proof-number root-selection path can return without expanding an empty root, and its position hash omits the side to move. None of this supports its old claim of a complete solution. That claim and the label attribution were corrected in documentation only; behavior was preserved. Repairing or exposing this agent is outside Phase 3A. The grounding module neither imports nor relies on it.

## Regression checks

Run `python -m pytest -q` from the repository root and the existing UI tests/build/browser suite documented in the README. New tests cover:

- Allis diagram 3.9 (§3.4, p.22) from its actual move sequence: two separate playable winning squares.
- Allis diagram 3.1 (§3.1, pp.16–17) from its actual 24-ply sequence: many geometric completion squares, none immediately playable.
- Vertical/horizontal/both diagonal wins; mandatory defense; win-over-block; duplicate lines sharing one square; a block at e2 that exposes the opponent's win at e3; full columns, draws and terminal behavior.
- Differential checks against legal moves and winning replies executed by the existing game engine on deterministic sample games.
- Immutable snapshots, complete replay, isolation, atomic rejection/agent failure, revision mismatch, concurrency, replacement, expiry, restart and evidence captured before a concurrent later move.
- Explicit absence of formal-rule claims even on an empty board with many superficially plausible pairs; reference-only retrieval and stable citation metadata.

Phase 3B now adds a separate, disabled-by-default explanation endpoint and frontend panel. See [its contract, constrained composition boundary, configuration, tests and live-test proposal](llm-explanations.md). The Phase 3A evidence schema and unsupported-formal-rule boundary remain unchanged; no paid calls or public rollout were performed.

## Phase 3B.1 primary-source recheck (October 3, 2026)

Consulted the local ignored `research/references/allis-1988-connect4.pdf` directly, extracting Chapters 3–8 and visually checking original diagrams 3.2, 3.9 and 6.1 and the §7.4 compatibility table. This edition has 91 PDF pages; existing catalog section/page metadata is retained. §3.1 (especially diagram 3.2, p.17) explicitly describes how filling b1 or f1 lets Black occupy the second-row winning square. This supports the explanatory distinction between geometric completion and gravity playability. §3.4 (pp.21–24) supports immediate forcing tactics, including independent playable threats in diagram 3.9. Chapters 5–8 retain the coverage, compatibility and evaluation-region requirements for formal rule proofs.

Shortened the catalog's winning-square and immediate-tactics paraphrases for display. The threat-perspective convention remains in the winning-square limitations. No formal rule implementation or citation source was added. Phase 3A's context retrieval/schema is unchanged; Phase 3B.1's presentation layer selects only concepts linked to verified explanatory relationships. Quiet openings omit unrelated concepts; requested rule terminology remains reference material inside details. The revision-36 regression is a legally replayed reconstruction of the reported b1→b2 mechanism, not a claim to reproduce the unavailable live history or thesis diagram.

The local reference is excluded from Git and Docker build contexts. Research extraction/rendering used temporary files and macOS PDFKit; PDF tools and the thesis are not application dependencies.
