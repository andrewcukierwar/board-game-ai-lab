"""Connect 4 AlphaZero v2 research path (Phases 4D.3A/4D.3B); inert on import.

Separately versioned from the v1 W/D/L research code: scalar tanh value,
legal-logit masking, visit targets independent of action temperature, frozen
generations, bounded replay, persistent AdamW and resumable boundaries, plus
model-blind evaluation packages, a corrected reference Negamax, an exact
oracle, a paired arena and a bounded campaign launcher. The launcher's run
commands require an explicit authorization token. Nothing here is imported by
the agent factory, API or UI.
"""
