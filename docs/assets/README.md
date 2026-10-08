# Application screenshots

Two curated PNGs copied from existing ignored Playwright capture paths. They show the real application UI, with no generated imagery, retouching, or invented visual elements. The current source was checked by rerunning the capture tests and Chromium regression, plus Firefox/WebKit product smoke checks; see [verification](../final-portfolio-packaging.md).

| Stable image | Capture source | Context |
| --- | --- | --- |
| [connect4-play.png](connect4-play.png) | `ui/playwright-report/portfolio/chromium/connect4-active-1440.png` | `polish.spec.js`: real local Random gameplay at 1440 px, before requesting analysis. The panel shows available analysis controls, not a live provider response. |
| [match-lab-replay.png](match-lab-replay.png) | `ui/playwright-report/phase5a/03-desktop-replay.png` | `match-lab.spec.js`: actual rendered Match Lab at 1440 px, inspecting move 4 using controlled legal mock API responses. Agent labels demonstrate configuration, not an independently executed Negamax/MCTS result. |

Both images are interface examples, **not canonical benchmark evidence**. The canonical results are in the [verified report](../../benchmarks/connect4/results/canonical-season-v1/README.md). Season export screenshots with test-fixture rankings/digests were deliberately excluded to avoid confusing them with that run. Full-page images are behind an expandable section in the root README to keep the landing page concise.
