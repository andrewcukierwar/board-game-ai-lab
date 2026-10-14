# Application screenshots

Real captures of the application UI, with no generated imagery, retouching, or invented visual elements. They were taken on October 14, 2026 from this release's frontend (Vite dev server, WebKit, 1440 px viewport). The frontend used an isolated local staging API (`127.0.0.1`, never the public service) with `ALPHAZERO_ENABLED=true`, `VICTOR_RESEARCH_ENABLED=true` and explanations disabled. AlphaZero availability and every AI move shown came from that real backend; no response was mocked.

| Image | Context |
| --- | --- |
| [home.png](home.png) | Homepage first viewport. |
| [connect4-play.png](connect4-play.png) | Play against AlphaZero (Red, moving first) with the tactical safeguard On; the human replied in columns 4 and 3. Neither side has an immediate win. Before requesting analysis. |
| [match-lab-replay.png](match-lab-replay.png) | Match Lab: AlphaZero (safeguard On, Red) against Tactical MCTS at 400 maximum simulations, stepped nine moves and reviewing move 7. |
| [tournament-setup.png](tournament-setup.png) | Tournament setup showing the recommended eight-entrant default field with both optional agents available. |

All images are interface examples, **not benchmark evidence**. A single game says nothing about relative strength. The canonical results are in the [verified report](../../benchmarks/connect4/results/canonical-season-v1/README.md).
