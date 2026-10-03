# Phase 4A.1 — Standalone MCTS correctness

October 3, 2026. Local branch `phase4a-mcts`, based on freshly fetched
`origin/main` at `5d12be2f526a9fdd4702ac4d7493704794fc929c`. The working tree
was clean before branching. The previous local branch was preserved.

## Correctness convention

`Connect4.make_move()` advances `current_player` immediately, including after
a winning move. The old backpropagation rewarded the next player to move at
each child. Because the parent maximizes child UCB, this rewarded the parent's
opponent and inverted its objective.

Each node now records `player_just_moved = 1 - game_state.current_player`.
`wins` is accumulated reward for that player: 1 for a win, 0.5 for a draw,
0 for a loss. All children therefore score outcomes for their parent's player
to move. Maximizing child UCB is correct at every depth, including the opponent's
turn. The root follows the same previous-player convention; its reward is not
used to choose a move, while its visit count supports UCB exploration.

Selection, expansion, random rollout and backpropagation remain distinct. Each
tree node owns a detached game state, and rollouts copy their input. Final move
selection uses the highest child visit count, with seeded random tie-breaking.

## Standalone contract and tactical guards

```python
from random import Random
from games.connect4.connect4 import Connect4
from games.connect4.agents.mcts_agent import MCTSAgent

agent = MCTSAgent(1000, rng=Random(42))
column = agent.choose_move(Connect4())
```

- The default remains 1,000 simulations, and positional `MCTSAgent(1000)` works.
  Non-integers, including booleans, raise `TypeError`; zero/negative integers
  raise `ValueError`. There is no unbounded-search option.
- Finished games or positions without legal moves raise `ValueError` from
  `choose_move()`. Terminal nodes are never expanded. Exhausted rollouts stop
  without choosing from an empty list. The Connect4 engine returns `-1` for draws.
- Expansion, UCB ties, random rollouts and final ties all use the injected RNG.
  Without injection, each agent owns a `random.Random()` instance. Reproducibility
  requires fresh generators with the same seed and the same sequence of calls;
  reusing an agent advances its generator state.
- Before search, an explicit root guard takes an immediate win. Otherwise it
  excludes moves permitting an immediate opponent win if any alternative exists.
  This checks the actual resulting board, including newly supplied supporting
  pieces. If every move loses on the next reply, all legal moves remain eligible.
- The tactical scan checks at most seven own moves and seven opponent replies
  per move. An immediate win bypasses simulation. Otherwise the search runs
  exactly the configured simulation count, even if only one candidate survives.
  These root guards are distinct from MCTS and guarantee only these immediate
  tactics. They do not establish the playing strength of random rollouts.

## Files and verification

- `games/connect4/agents/mcts_agent.py`: correct reward perspective, bounded
  configuration validation, RNG injection, terminal/exhaustion handling and root
  tactical checks. Removed unused imports. No engine changes or ML dependency.
- `tests/test_connect4_mcts.py`: 71 deterministic cases covering both players,
  win/loss/draw accounting, alternating tree perspectives, UCB exploration,
  final visit-count selection, legal tactical fixtures, root guards, nearly full
  boards, terminal reuse, input isolation, seeded statistics and small budgets.
  Scripted rollouts/selection isolate contracts where appropriate; tactical
  assertions test guards rather than stochastic playing strength.
- `docs/phase4a1-mcts.md`: this implementation and handoff record.

Verification with Python 3.11.17 in the existing local environment:

```sh
EXPLANATIONS_ENABLED=false OPENAI_API_KEY= .venv/bin/python -m pytest -q
git diff --check
```

**311 passed** (240 existing backend tests plus 71 new MCTS tests), in 5.51 seconds.
The existing suite also passed independently before the new tests were added:
240 passed. Explanation-provider interactions remained mocked; the explanation
test module blocks live HTTPS requests. No paid API calls were made.

A separate fresh-process import hook rejected every attempted `torch` import.
With that hook active, Flask initialized, Random and Negamax each played through
the API, and standalone MCTS chose a move. PyTorch was absent from loaded modules.
The API did not import standalone MCTS and continued rejecting the `mcts` agent
configuration. Production dependencies were unchanged. `git diff --check` passed.

A single local empty-board timing check, using fresh `Random(42)` generators:

| Simulation limit | Elapsed seconds |
| --- | --- |
| 1 | 0.008 |
| 100 | 0.129 |
| 500 | 0.400 |
| 1000 | 0.725 |

These are informal local observations, not a performance benchmark, strength
comparison, latency guarantee or estimate for Render Free.

## Limits and Phase 4A.2 handoff

The standalone agent is ready for Phase 4A.2 integration work. It is still a
random-rollout search with finite sampling error; beyond the root guards it can
miss tactics, especially with small budgets. It assumes valid alternating-play
Connect4 states and does not add arbitrary-board validation. The tests do not
claim perfect play or a formal strength improvement.

Search is synchronous, stores copied boards and creates a fresh tree per move.
Its simulation budget bounds work but is not a wall-clock deadline. Large
positive integer budgets remain allowed for standalone use; public presets,
latency/concurrency limits and deployment-environment checks belong to Phase 4A.2.
No API/UI integration, production dependency or configuration change, neural-agent
work, commit, push, merge, deployment or workflow trigger was performed.
