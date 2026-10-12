# Sealed-access incident: metadata-only scope attestation

Author: the original independent reviewer agent, `/root/independent_review`.
This document reports access scope and provenance only. It does not reproduce
sealed labels, positions, moves, values, counts of theoretical outcomes or other
protected fixture content. No sealed fixture was reopened to prepare it. The
reviewer has stopped substantive review/evaluation; a clean reviewer must perform
the replacement independent audit.

## Offending call and scope

The relevant shell command in the functions.exec call was:

```text
rg --files research/alphazero_value_diagnostics; rg -n 'def load|predict|terminal|value|rng|seed|optimal|cluster' research/alphazero_resnet_selfplay/adapters.py research/alphazero_resnet_selfplay/evaluate.py games/connect4/alphazero*
```

Working directory was the new phase4d4i-a-worktree. The same functions.exec call
also read the prior H-C independent-results-review.md explicitly. The unsafe
part was the broad `games/connect4/alphazero*` directory scope. The tool call ID
was `call_9e42512ca12748c29c6a2bdaf9e73a29`. Its input SHA-256 is
`5d3659113a7466147f749c4f759252ba781806d1acf2a28995333e2483bf735e`.

Metadata-only filename enumeration confirms these three sealed files lie under
that recursive search scope:

- `games/connect4/alphazero_v2/frozen/openings-sealed.json`
- `games/connect4/alphazero_v2/frozen/solved-sealed.json`
- `games/connect4/alphazero_v2/frozen/tactical-sealed.json`

The `rg` process therefore had all three files within its traversal scope. I
cannot prove exactly how many bytes it read from each without a syscall trace,
which does not exist. The retained returned output contains source references
for **solved-sealed.json only**. No other sealed filename appeared in that
offending returned output. Absence from the truncated output does not establish
that the other two files were not read by the process or absent from its full
untruncated matches.

## Retained-output exposure

I used a safe parser of my own JSONL transcript which emitted filenames, counts,
line-reference numbers and hashes only. It did not print or copy protected text.
The offending returned output is transcript record 49, with recorded timestamp
`2026-10-12T22:00:07.084Z`. It is **truncated**: the retained output has three
truncation-marker mentions, and its outer marker reports original token count
13,643. It is not the complete `rg` result. The serialized returned output has
41,443 characters and SHA-256
`593cd687b1fcb82af91fa490d0c122d49c136c9dac3d4088c942327f53b1e988`.
This hash covers the whole returned output, including the prior report and
wrappers; it is not a hash of a sealed fixture.

There are **149 distinct retained source-line references** to solved-sealed.json,
from source line **30511** through **33400**, intermittently rather than as one
continuous block. The exhaustive source-line-reference metadata is:

```text
30511 30518 30521 30569 30576 30579 30633 30639 30642 30696 30702 30705
30747 30757 30760 30802 30812 30815 30865 30874 30877 30927 30936 30939
30991 30997 31000 31052 31058 31061 31107 31118 31121 31167 31178 31181
31229 31239 31242 31290 31300 31303 31355 31362 31365 31417 31424 31427
31471 31482 31485 31529 31540 31543 31589 31598 31601 31647 31656 31659
31711 31719 31722 31774 31782 31785 31829 31840 31843 31887 31898 31901
31947 31956 31959 32005 32014 32017 32061 32071 32074 32118 32128 32131
32179 32188 32191 32239 32248 32251 32293 32303 32306 32348 32358 32361
32409 32417 32420 32468 32476 32479 32523 32533 32536 32580 32590 32593
32641 32649 32652 32700 32708 32711 32755 32764 32767 32811 32820 32823
32871 32879 32882 32930 32938 32941 32983 32993 32996 33038 33048 33051
33097 33104 33107 33153 33160 33163 33209 33220 33223 33269 33280 33283
33331 33340 33343 33391 33400
```

These counts describe the retained transcript representation. They do not
reconstruct material elided by either truncation layer, and cannot prove which
portion the model internalized. The already-exposed text cannot be unlearned.

Transcript source, read only through the metadata parser:
`/Users/andrewcukierwar/.codex/sessions/2026/10/12/rollout-2026-10-12T17-59-45-01a13220-fd18-7512-a6c1-16e059766b7d.jsonl`.

## Other reads, storage and transmission

I found no separate subsequent sealed-fixture content read in my task history.
Subsequent source reads were restricted to explicit Python/source paths and the
nonsealed H-C/G fixtures and development openings. For this attestation I
enumerated sealed filenames as metadata and parsed the already-retained task
transcript without emitting protected contents. This transcript parsing does
inspect the existing tool-output record mechanically; it does not reopen the
fixtures or deliver its protected text to the model again.

The tool/service automatically retained the offending output in the task
transcript and delivered it into my context. Thus protected content was stored
by that automatic logging path; I cannot attest that no copy exists in service
logs. I made no manual copy/export of protected content into repository files,
messages, scripts or other artifacts. Messages sent to the lead agent identified
the incident and filename only; they did not contain protected fixture text.
No external recipient, solver or model evaluation received it from me. The
attestation does not claim to erase the automatic transcript.

## Authored files and influenced decisions after exposure

I authored only these files under the new package:

- `independent-prelaunch-review.md`: original source review, explicit incident
  disclosure and limits; its review disposition must be superseded by clean review.
- `independent_review.py`: a not-yet-executed independent arithmetic/board review
  script. It imports no phase implementation; it is not clean-review evidence and
  should not substitute for the replacement reviewer's independent work.
- `independent-incident-attestation.md`: this metadata-only document.

No manual file was created outside this new package. No checkpoint, historical
artifact, canonical benchmark, frozen execution source or evaluation membership
was modified by me. No package __pycache__ file was present when I checked during
this attestation. The runtime/task transcript itself is maintained automatically.

After exposure I independently validated nonsealed fixture labels and forty
development opening histories, tested wrappers with synthetic fixed-value
adapters, reviewed source/sign/RNG/replay/statistical contracts, and supplied
prelaunch findings to the lead agent. My material implementation recommendation
was to enforce the local development-opening hash in source_map; the lead agent
implemented it. I also recommended explicit distinctions between fixed-model
family uncertainty and seed uncertainty, replay outcomes and exact WDL,
mirror-canonical repetition and exact-position outcome variance, and generation5
prequential evidence versus unavailable generation6-on-generation7 evidence.

I did not choose checkpoint identities, fixture membership, scales, match count,
search budget, seeds or statistical methods using sealed content. Those choices
came from the user's request, prior published phases and the lead's prospective
design. Nevertheless, my post-exposure review and advice came from a reviewer
whose context had received protected text, so a declaration of non-use alone
cannot establish independence. The replacement clean reviewer must reconsider
the source and methodology without relying on my substantive conclusions.

No measured model/fixture evaluation, training or self-play collection was run by
me after the incident. My synthetic sign/prior tests and nonsealed fixture-rule
validation are correctness checks, not measured checkpoint outcomes. I have
stopped further substantive review/evaluation as instructed. This is a factual
scope attestation; it does not by itself resolve authorization to launch.
