"""The verified CL/BI/VE fragment of Allis §7.4 (PDF p.50).

This checks intrinsic shapes and response-square conflicts, not position validity
or joint strategic applicability. Unknown matrix entries fail closed.
"""
from types import MappingProxyType

from .contracts import CompatibilityConstraint
from .geometry import Square
from .rules import RuleCandidate, RuleName

CL, BI, VE = RuleName.CLAIMEVEN, RuleName.BASEINVERSE, RuleName.VERTICAL
# Explicit unordered dispatch: adding a rule must not implicitly enable its pairs.
_CONSTRAINTS = MappingProxyType({
    frozenset(pair): (CompatibilityConstraint.DISJOINT_SQUARES,)
    for pair in ((CL, CL), (CL, BI), (CL, VE), (BI, BI), (BI, VE), (VE, VE))
})


def required_constraints(first: RuleName, second: RuleName) -> tuple[CompatibilityConstraint, ...]:
    if type(first) is not RuleName or type(second) is not RuleName:
        raise ValueError('compatibility requires RuleName identifiers')
    try:
        return _CONSTRAINTS[frozenset((first, second))]
    except KeyError:
        raise ValueError('rule compatibility is not implemented for this pair') from None


def checked_candidate(candidate: RuleCandidate) -> RuleCandidate:
    """Reconstruct untrusted intrinsic geometry; no board or coverage assertions."""
    if type(candidate) is not RuleCandidate or type(candidate.squares) is not tuple:
        raise ValueError('candidate must be a canonical RuleCandidate')
    for square in candidate.squares:
        if type(square) is not Square:
            raise ValueError('candidate square must be a Square')
        Square(square.row_index, square.column)
    rebuilt = RuleCandidate(candidate.rule, candidate.squares)
    if rebuilt != candidate:
        raise ValueError('candidate square roles/order are not canonical')
    return rebuilt


def compatible(first: RuleCandidate, second: RuleCandidate) -> bool:
    """Symmetric; duplicates conflict. Unsupported or malformed shapes raise ValueError."""
    first, second = checked_candidate(first), checked_candidate(second)
    constraints = required_constraints(first.rule, second.rule)
    return all(_CHECKS[constraint](first, second) for constraint in constraints)


_CHECKS = MappingProxyType({
    CompatibilityConstraint.DISJOINT_SQUARES:
        lambda first, second: first.affected_squares.isdisjoint(second.affected_squares),
})
