"""Local Allis candidates and bounded compatible coverage, NOT a game solver.

No imports from or integration with VictorAgent, the API, or LLM evidence schemas.
"""
from .compatibility import compatible, required_constraints
from .coverage import (
    BlackEvaluationContext, CoverageWitness, CoveringSetResult, SearchStatus,
    black_evaluation_context, search_covering_set,
)
from .evidence import CandidateEvidence, CandidateReport, analyze_candidates
from .geometry import ALL_GROUPS, Group, Square
from .position import Player, Position
from .rules import (
    Prerequisites, RuleCandidate, RuleName, ThesisReference, enumerate_baseinverses,
    enumerate_candidates, enumerate_claimevens, enumerate_verticals,
)
from .verification import VerificationStatus, WitnessVerification, verify_coverage_witness

__all__ = [
    'ALL_GROUPS', 'CandidateEvidence', 'CandidateReport', 'Group', 'Player', 'Position',
    'Prerequisites', 'RuleCandidate', 'RuleName', 'Square', 'ThesisReference',
    'analyze_candidates', 'enumerate_baseinverses', 'enumerate_candidates',
    'enumerate_claimevens', 'enumerate_verticals',
    'BlackEvaluationContext', 'CoverageWitness', 'CoveringSetResult', 'SearchStatus',
    'VerificationStatus', 'WitnessVerification', 'black_evaluation_context',
    'compatible', 'required_constraints', 'search_covering_set', 'verify_coverage_witness',
]
