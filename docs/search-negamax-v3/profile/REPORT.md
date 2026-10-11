# Strongest direct-engine profile

Declaration92a272f; optimized baseline60b99b0, exact source/script/design digests
in manifest. Eight complete D10 decisions over four declared costly histories,
under shared lock,7.479s profiled runtime. All vectors/counters match original
primary data; audit.json. This is explanatory cProfile evidence, not latency.

| Function | Calls | Self time | Fraction of profiled self |
| --- | --- | --- | --- |
| negamax | 1,809,760 | 1.761s | 23.6% |
| play | 1,809,760 | 1.491s | 20.0% |
| has_four | 3,619,536 | 1.188s | 15.9% |
| undo | 1,809,760 | 1.178s | 15.8% |
| terminal_value | 1,809,768 | 0.467s | 6.3% |
| winning_squares | 255,004 | 0.288s | 3.9% |
| ordered_moves | 565,142 | 0.232s | 3.1% |
| pack_entry/unpack_entry | 565,142/215,758 | 0.205s | 2.8% |

Choose terminal-parent proof: every child was dropped from a verified
nonterminal parent, so checking the unchanged side for four is redundant.
Choose a smaller, lower-confidence TT access/codec experiment: search-body
lookup/storage plus helper dispatch/validation/tuple/string costs may compound;
helper fractions alone do not promise a benefit. Both keep trees identical.
Play/undo is larger but already uses exact delta membership loops and cached
scores; no simple additional measured hypothesis was selected for it. No
speculative pruning/state rewrite. Tuning DESIGN gates apply to complete
unprofiled decisions, not these shares. Runtime/lock owner and full function
self/cumulative records in profile.json; laptop hardware.txt in parent.
