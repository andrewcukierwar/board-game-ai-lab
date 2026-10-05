# Phase 4D.3B.1 launch-control evidence

- `head-reproduction/`: the step-1 reproductions of launch-readiness findings B1–B4 and S1–S4. They exercise
  the **pre-fix** launcher API, so run them only on a checkout of `d5696cb` (or `a663454`):

      PYTHONPATH=. python research/phase4d3b1-mutation/head-reproduction/reproduce_findings.py
      PYTHONPATH=. python research/phase4d3b1-mutation/head-reproduction/reproduce_extra.py

  `*-results.json` are the recorded outputs. Source edits are virtual (monkeypatched reads), and all
  campaigns use temporary directories.
- `mutate_launch.py`: the 42-mutation sweep over source and runtime binding, durable transitions, deadline
  and budget persistence, and old- and resumed-token verification. `mutation-launch-results.txt` records
  42 killed and 0 survived. The script edits sources in place, so never run it while another process
  imports the package.

See docs/phase4d3b1-launch-control-fixes.md.
