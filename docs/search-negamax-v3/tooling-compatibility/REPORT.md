# Exact source recovery for shallow/squashed checkouts

2026-10-10, after completed Phase D measurements. CI uses shallow checkout;
new research variants originally required git-show starting60b99b0 even when
that object was unavailable. The source loader now tries Git, then the already
committed immutable mirror/direct-source.py copy. Both paths must match literal
SHA-256 f46461d73b240c645248d57884fee0b9917c1be289758efe381cf6458a9cb392.
Bad Git bytes fail closed instead of silently substituting another engine.

Original variant-generator and runner bytes are archived here, with old/new
SHA fingerprints in certificate.json. Existing frozen manifests keep ORIGINAL
hash values. The metadata guard permits only enumerated text substitutions:
pinned-loader import/call; hash-validator import/call; extra helper hash paths.
Every other byte must match the archived tools, and helper bytes are also pinned.
Measured decision/run/memory/analysis/audit/report bodies and all thresholds are
unchanged. This is a metadata/loader revision, not a remeasurement or acceptance
threshold adjustment. Archive and all historical evidence remain untouched.

147 focused tests pass in9.01s, including legacy provenance, independent exact
search checks, pinned fallback and compatibility guards. Tests demonstrate a
forged new fingerprint cannot authorize an altered geometric threshold; unknown
roles/revisions and corrupt original archives fail. Audit verifies every prior
mirror/PVS/tuning/deeper manifest, plus all four regular variant sources with
Git forcibly unavailable. Production agent is unchanged from validated51bd80e.
No live service, public limits or old canonical v2 files changed.

Reproduce:

```sh
PYTHON_DOTENV_DISABLED=1 EXPLANATIONS_ENABLED=false OPENAI_API_KEY= .venv/bin/python -m pytest tests/test_negamax_v3_pinned.py tests/test_negamax_v3_compatibility.py tests/test_connect4_negamax_v3.py tests/test_negamax_v3_tuning.py tests/test_search_pinned_sources.py -q -rs
```

Run any prior phase's check/audit against its preserved data at the branch tip;
only these certified non-timed substitutions are compatible. Unrecognized source
drift still fails closed. New declarations hash the current helpers as well.
