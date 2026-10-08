"""Phase 6A: local Allis candidates and conditional coverage, NOT a solver.

No imports from or integration with VictorAgent, the API, or LLM evidence schemas.
"""
from .evidence import CandidateEvidence, CandidateReport, analyze_candidates
from .geometry import ALL_GROUPS, Group, Square
from .position import Player, Position
from .rules import (
    Prerequisites, RuleCandidate, RuleName, ThesisReference, enumerate_baseinverses,
    enumerate_candidates, enumerate_claimevens, enumerate_verticals,
)

__all__ = [
    'ALL_GROUPS', 'CandidateEvidence', 'CandidateReport', 'Group', 'Player', 'Position',
    'Prerequisites', 'RuleCandidate', 'RuleName', 'Square', 'ThesisReference',
    'analyze_candidates', 'enumerate_baseinverses', 'enumerate_candidates',
    'enumerate_claimevens', 'enumerate_verticals',
]
