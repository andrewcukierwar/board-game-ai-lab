"""Allis coverage research and a bounded functional research solver.

All nine Allis rules (§§6.1–6.9) and the full §7.4 matrix are available through
``analyze_nine_rules`` / ``search_nine_rule_cover`` / ``verify_nine_rule_witness``
(docs/victor-nine-rule-implementation.md). Those results are coverage evidence
only. The CL/BI/VE certificate and its executable responses stay restricted to
those three rules.
No imports from or integration with VictorAgent, the API, or LLM evidence schemas.
"""
from .compatibility import compatible, failed_constraints, required_constraints
from .composite import (
    Component, CompositeCandidate, SolutionClause, enumerate_all_candidates,
    enumerate_composites, prerequisite_failures, solution_clauses,
)
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
from .nine_rules import (
    NINE_RULE_OBLIGATIONS, NineRuleAssignment, NineRuleReport, NineRuleSearchResult,
    NineRuleWitness, RuleEvidence, analyze_nine_rules, search_nine_rule_cover,
)
from .nine_rule_verification import NineRuleVerification, verify_nine_rule_witness
from .solver import SolverBudget, SolverResult, VictorSolver, analyze_position, select_move
from .exact import ExactResult, SearchBudget, solve_exact
from .white import WhiteContext, white_evaluation_contexts, search_white_covers
from .execution import NineRulePolicy, PolicyAudit

__all__ = [
    'ALL_GROUPS', 'CandidateEvidence', 'CandidateReport', 'Group', 'Player', 'Position',
    'Prerequisites', 'RuleCandidate', 'RuleName', 'Square', 'ThesisReference',
    'analyze_candidates', 'enumerate_baseinverses', 'enumerate_candidates',
    'enumerate_claimevens', 'enumerate_verticals',
    'BlackEvaluationContext', 'CoverageWitness', 'CoveringSetResult', 'SearchStatus',
    'VerificationStatus', 'WitnessVerification', 'black_evaluation_context',
    'compatible', 'required_constraints', 'search_covering_set', 'verify_coverage_witness',
    'Component', 'CompositeCandidate', 'SolutionClause', 'enumerate_all_candidates',
    'enumerate_composites', 'prerequisite_failures', 'solution_clauses', 'failed_constraints',
    'NINE_RULE_OBLIGATIONS', 'NineRuleAssignment', 'NineRuleReport', 'NineRuleSearchResult',
    'NineRuleWitness', 'NineRuleVerification', 'RuleEvidence', 'analyze_nine_rules',
    'search_nine_rule_cover', 'verify_nine_rule_witness',
    'SolverBudget', 'SolverResult', 'VictorSolver', 'analyze_position', 'select_move',
    'ExactResult', 'SearchBudget', 'solve_exact', 'WhiteContext',
    'white_evaluation_contexts', 'search_white_covers', 'NineRulePolicy', 'PolicyAudit',
]
