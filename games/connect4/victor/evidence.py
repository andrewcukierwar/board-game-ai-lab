"""Conditional local coverage, explicitly separated from any future proof system."""
from dataclasses import dataclass, field
from typing import Literal

from .geometry import Group
from .position import Player, Position, validate_player
from .rules import RuleCandidate, ThesisReference, enumerate_candidates


@dataclass(frozen=True)
class CandidateEvidence:
    candidate: RuleCandidate
    conditional_solved_groups: tuple[Group, ...]


@dataclass(frozen=True)
class CandidateReport:
    """Board-bound research evidence. No outcome, safety, or move recommendation.

    A manually assembled report is untrusted data, just as an enumerated report
    is not a certificate. Future verification must recompute all evidence.
    """

    position: Position
    defender: Player
    opponent_groups: tuple[Group, ...]
    evidence: tuple[CandidateEvidence, ...]
    status: Literal['candidates_only'] = field(default='candidates_only', init=False)
    method: str = field(default='allis-1988-local-candidates-v1', init=False)
    source: str = field(default=(
        'Victor Allis (1988), A Knowledge-Based Approach of Connect-Four: '
        'The Game Is Solved: White Wins'), init=False)
    source_url: str = field(default='https://tromp.github.io/c4/connect4_thesis.pdf', init=False)
    framework_references: tuple[ThesisReference, ...] = field(default=(
        ThesisReference('5.4', (34, 35)), ThesisReference('6', (36, 36)),
        ThesisReference('7.2', (49, 49)), ThesisReference('7.4', (50, 50)),
        ThesisReference('8.1', (51, 51)), ThesisReference('8.2–8.4', (51, 57)),
    ), init=False)
    unverified_obligations: tuple[str, ...] = field(default=(
        'opponent-to-move evaluation and permitted threat region',
        'joint strategic applicability, including Zugzwang-dependent reasoning',
        'pairwise compatibility under section 7.4',
        'complete coverage of relevant opponent groups',
        'independent position-proof certificate verification',
    ), init=False)


def analyze_candidates(position: Position, defender: Player) -> CandidateReport:
    """Attach conditional coverage against defender's opponent, on the whole board.

    Both defenders may be inspected regardless of turn. This deliberately does
    NOT certify the Chapter 6/8 evaluation context, especially White's region.
    Empty-coverage candidates are retained. Terminal positions have no candidates.
    """
    validate_player(defender)
    opponent: Player = 1 if defender == 0 else 0
    groups = position.potential_groups(opponent)
    evidence = tuple(CandidateEvidence(candidate, tuple(
        g for g in groups if candidate.coverage_squares.issubset(g.squares)))
        for candidate in enumerate_candidates(position))
    return CandidateReport(position, defender, groups, evidence)
