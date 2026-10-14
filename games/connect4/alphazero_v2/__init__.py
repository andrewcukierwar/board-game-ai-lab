"""Connect 4 AlphaZero v2 research path (Phase 4D.3A); inert on import.

Separately versioned from the v1 W/D/L research code: scalar tanh value,
legal-logit masking, visit targets independent of action temperature, frozen
generations, bounded replay, persistent AdamW and resumable boundaries.
Phase 4D.4K-A adds the research bitboard PUCT path, the verified tactical root
guard and ``production.py`` (strict inference-artifact loading and fixed
decisions), used only by the opt-in API agent; no CLI or UI integration lives here.
"""
