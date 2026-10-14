# Phase 4D.4I-B

A research-only, fixed replay experiment. Read DESIGN.md and the phase report before invoking anything. Historical checkpoints/sources remain immutable. Never read sealed fixture contents.

Use the original H-C Python interpreter and all five library thread controls set to1. Preparation, disposable validation and independent prelaunch review precede source freeze and clean pushed launch. `python -m research.alphazero_value_regularization.manage` executes one detached lifecycle with no retries and2700s total computational cap. It refuses dirty/unpushed sources or any prior attempt. Outputs/checkpoints live only under this phase’s ignored `experiment-output/phase4d4i-b`; complete evidence is copied into results after all tasks succeed. Incomplete tasks are never analyzed.

`checkpoint.py` defines a new version1 contract; it does not write historical resume contracts or invent new generations. `objective.py` calls original loss unchanged and adds the one declared full-network output penalty. Independently authored `independent_verify.py` verifies complete saved evidence without model queries. Final independent interpretation follows complete evidence.
