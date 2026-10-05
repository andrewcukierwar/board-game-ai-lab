# Phase 4D.3B mutation evidence

Each mutation is one plausible defect applied as an exact string replacement; the named test files
are then run and the source is restored (a backup of the original is written first, so a hard kill
cannot lose it). KILLED means at least one test failed.

    PYTHONPATH=. python research/phase4d3b-mutation/mutate.py tests/test_alphazero_v2.py [NAMES...]
    PYTHONPATH=. python research/phase4d3b-mutation/mutate_m2.py \
        tests/test_alphazero_v2_evaluation_core.py,tests/test_alphazero_v2_campaign.py [NAMES...]

Results: `mutation-m1-results.txt` (Milestone 1 hostile review), `mutation-m2-*-results.txt`
(Milestone 2). The first Milestone 2 pass was stopped deliberately after three results; the
remaining mutations were rerun in two chunks. Survivors were then followed by new tests and reruns:
`negamax_first_move_ties`, `solver_bounds_exact`, `package_no_exclusion`, `scan_safe_ignores_win` and
`tactical_no_family_average` are now killed. Two are equivalent mutants: `draw_not_zero` (the window
heuristic of a full board without four is always 0) and `solver_forced_ignored` (removes only a
double-threat shortcut; search still proves the loss).

`runtime_probe.py INTRA INTER DET` reproduces the thread/determinism trajectory probe.
These scripts edit source files in place: never run them while another process imports the package.
