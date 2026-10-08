"""Research-only executable refinement of the CL/BI/VE theorem's sigma_R.

No solver, heuristic, public agent, or game-value claim. Re-verify a private
certificate snapshot and replay the complete continuation on EVERY selection.
Physical columns are 0..6; squares use thesis rows (a1..g6, bottom first).
"""
from dataclasses import dataclass, field, replace
from enum import Enum

from . import certificate as C

STRATEGY_ID = 'sigma-r-even-row-lowest-column-v1'
# Pin literals: a future verifier/theorem change needs a strategy review too.
SUPPORTED_IDENTITIES = (
    'victor-strategic-certificate-v1', 'cl-bi-ve-black-nonloss-v1',
    'connect4-standard-7x6-v1', 'allis1988-cl-bi-ve-v1',
    'allis1988-s7.4-cl-bi-ve-disjoint-v1',
)


class StrategyStatus(str, Enum):
    MOVE_SELECTED = 'legal_move_selected'
    INVALID_CERTIFICATE = 'invalid_certificate'
    UNSUPPORTED = 'unsupported_version_or_context'
    ILLEGAL_CONTINUATION = 'illegal_continuation'
    STRATEGY_VIOLATED = 'previously_violated_strategy'
    TERMINAL = 'terminal_position'
    UNKNOWN = 'unknown_resource_cutoff'
    INVARIANT_FAILURE = 'unexpected_invariant_failure'


class ResponseKind(str, Enum):
    FORCED = 'forced_response'
    SPARE = 'even_row_spare'


@dataclass(frozen=True, slots=True)
class StrategyRequest:
    certificate: C.StrategicCertificate
    continuation: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ExecutedMove:
    ply: int  # zero-based, relative to the certified position
    player: int
    column: int
    square: str
    response: ResponseKind | None
    triggering_rule: C.RuleInstance | None
    newly_retired: tuple[C.RuleInstance, ...]


@dataclass(frozen=True, slots=True)
class ContinuationState:
    board: tuple[tuple[str, ...], ...]
    board_digest: str
    player_to_move: int
    continuation: tuple[int, ...]
    trace: tuple[ExecutedMove, ...]
    active_rules: tuple[C.RuleInstance, ...]
    retired_rules: tuple[C.RuleInstance, ...]


@dataclass(frozen=True, slots=True)
class StrategyResult:
    status: StrategyStatus
    findings: tuple[C.Finding, ...] = ()
    column: int | None = None
    square: str | None = None
    response: ResponseKind | None = None
    triggering_rule: C.RuleInstance | None = None
    state: ContinuationState | None = None
    certificate_digest: str | None = None
    initial_board_digest: str | None = None
    starting_history_backed: bool | None = None
    strategy_id: str = field(default=STRATEGY_ID, init=False)
    theorem_id: str = field(default=SUPPORTED_IDENTITIES[1], init=False)
    schema_id: str = field(default=SUPPORTED_IDENTITIES[0], init=False)
    ruleset_id: str = field(default=SUPPORTED_IDENTITIES[2], init=False)
    rule_model_id: str = field(default=SUPPORTED_IDENTITIES[3], init=False)
    compatibility_model_id: str = field(default=SUPPORTED_IDENTITIES[4], init=False)
    game_theoretic_certification: str = field(
        default='not_certified_pending_independent_review', init=False)


def _sequence(value, limit):
    if type(value) not in (list, tuple) or len(value) > limit:
        raise ValueError(f'expected list/tuple with at most {limit} entries')
    return tuple(value)


def _snapshot(cert):
    """Bounded structural copy BEFORE verification; frozen records can hold lists.

    No arbitrary deepcopy/serialization hooks. The verifier still checks every
    scalar, shape, identity and hypothesis of this copy. No caller containers
    are used after verification. Arbitrary concurrent object.__setattr__ attacks
    inside this Python process are outside this data-input trust boundary.
    """
    if type(cert) is not C.StrategicCertificate:
        raise ValueError('expected an exact StrategicCertificate, never a saved verdict')
    board = tuple(_sequence(row, 7) for row in _sequence(cert.board, 6))
    rules = []
    for rule in _sequence(cert.rules, 21):
        if type(rule) is not C.RuleInstance:
            raise ValueError('expected exact RuleInstance entries')
        rules.append(C.RuleInstance(rule.kind, _sequence(rule.squares, 2)))
    replay, assignments = cert.replay, cert.assignments
    return replace(cert, board=board, rules=tuple(rules),
                   replay=None if replay is None else _sequence(replay, 42),
                   assignments=None if assignments is None else tuple(
                       _sequence(a, 2) for a in _sequence(assignments, 69)))


def _cell(board, square):
    return board[6 - int(square[1])]['abcdefg'.index(square[0])]


def _landing(board, column):
    return next((f'{"abcdefg"[column]}{row}' for row in range(1, 7)
                 if board[6 - row][column] == C.EMPTY), None)


def _terminal(board):
    return (any(all(board[6 - r][c] == stone for c, r in group)
                for stone in (C.WHITE, C.BLACK) for group in C.GROUPS)
            or all(C.EMPTY not in row for row in board))


def _active(board, rule):
    needed = rule.squares[1:] if rule.kind == C.CLAIMEVEN else rule.squares
    return all(_cell(board, s) != C.BLACK for s in needed)


def _invariant(board, rules, coverage):
    """At White turns (also after a terminal Black move), check L1/L4 explicitly.

    Check ALL initial groups covered by a retired rule, hence also any valid
    producer assignment, rather than trusting stored retirement flags.
    """
    for rule, groups in zip(rules, coverage):
        if rule.kind == C.CLAIMEVEN and _cell(board, rule.squares[0]) == C.BLACK:
            return f'Black occupied Claimeven lower {rule.squares[0]}'
        if _active(board, rule):
            if any(_cell(board, s) != C.EMPTY for s in rule.squares):
                return f'active rule has an occupied square: {rule.label}'
        elif any(not any(_cell(board, s) == C.BLACK for s in g.split('-'))
                 for g in groups):
            return f'retired rule no longer blocks its covered groups: {rule.label}'
    return None


def _response(board, rules, white_square):
    """Board is AFTER White's move; active tests still use only Black stones."""
    live = tuple(r for r in rules if _active(board, r))
    touched = tuple(r for r in live if white_square in r.squares)
    if len(touched) > 1:
        raise ValueError('conflicting mandatory responses')
    if touched:
        rule = touched[0]
        a, b = rule.squares
        if rule.kind != C.BASEINVERSE and white_square != a:
            raise ValueError('White reached an active upper square')
        reply = b if white_square == a else a
        if reply != _landing(board, 'abcdefg'.index(reply[0])):
            raise ValueError('mandatory response is not a landing square')
        return reply, ResponseKind.FORCED, rule
    forbidden = {r.squares[0] for r in live if r.kind == C.CLAIMEVEN}
    for column in range(7):  # Fixed physical ordering; deliberately not reflection equivariant.
        square = _landing(board, column)
        if square and int(square[1]) % 2 == 0 and square not in forbidden:
            return square, ResponseKind.SPARE, None
    raise ValueError('no permitted even-row spare (parity invariant failed)')


def select_black_move(request: StrategyRequest, *,
                      certificate_work_budget: int = C.DEFAULT_WORK_BUDGET,
                      replay_budget: int = 42) -> StrategyResult:
    """Select only at a nonterminal Black turn after a complete compliant replay.

    Prior Black spares MUST match this refinement's lowest-column tie-break,
    even if another spare would obey the broader theorem. Budgets are explicit:
    verifier work units and continuation plies; a cutoff supplies no move.
    Inputs are data, not authorities. Output traces are never accepted as input.
    """
    for name, budget in (('certificate_work_budget', certificate_work_budget),
                         ('replay_budget', replay_budget)):
        if type(budget) is not int or budget < 0:
            raise ValueError(f'{name} must be a nonnegative integer')
    if type(request) is not StrategyRequest:
        return StrategyResult(StrategyStatus.UNSUPPORTED,
                              (C.Finding('request.type', 'expected exact StrategyRequest'),))
    try:
        cert = _snapshot(request.certificate)
    except (ValueError, TypeError, AttributeError) as exc:
        return StrategyResult(StrategyStatus.INVALID_CERTIFICATE,
                              (C.Finding('certificate.snapshot', str(exc)),))
    verification = C.verify_certificate(cert, work_budget=certificate_work_budget)
    if verification.status is not C.CertificateStatus.HYPOTHESES_VERIFIED:
        status = {C.CertificateStatus.REJECTED: StrategyStatus.INVALID_CERTIFICATE,
                  C.CertificateStatus.UNSUPPORTED: StrategyStatus.UNSUPPORTED,
                  C.CertificateStatus.UNKNOWN: StrategyStatus.UNKNOWN}[verification.status]
        return StrategyResult(status, verification.findings)
    identities = (cert.schema, cert.theorem, cert.ruleset, cert.rule_model,
                  cert.compatibility_model)
    if (identities != SUPPORTED_IDENTITIES
            or verification.strategy_formulation != 'sigma-r-permissive-spare-v1'
            or verification.verifier_id != 'victor-independent-certificate-verifier-6b4a-v1'):
        return StrategyResult(StrategyStatus.UNSUPPORTED,
                              (C.Finding('version.strategy', 'unsupported strategy boundary'),))

    def result(status, code=None, detail='', **kwargs):
        return StrategyResult(
            status, () if code is None else (C.Finding(code, detail),),
            certificate_digest=verification.certificate_digest,
            initial_board_digest=verification.evidence.board_digest,
            starting_history_backed=verification.history_backed, **kwargs)

    history = request.continuation
    if (type(history) is not tuple or len(history) > verification.evidence.empty_squares
            or any(type(c) is not int or not 0 <= c <= 6 for c in history)):
        return result(StrategyStatus.ILLEGAL_CONTINUATION, 'continuation.schema',
                      'continuation must be a tuple of exact columns 0..6, within remaining cells')
    board = [list(row) for row in cert.board]
    rules, coverage = cert.rules, verification.evidence.rule_coverage
    trace, pending = [], None
    for ply, column in enumerate(history):
        if ply >= replay_budget:
            return result(StrategyStatus.UNKNOWN, 'resource.replay_budget',
                          f'replay budget {replay_budget} exhausted at ply {ply}')
        if _terminal(board):
            return result(StrategyStatus.ILLEGAL_CONTINUATION, 'continuation.after_terminal',
                          f'ply {ply} follows the first terminal position')
        player = ply % 2
        if player == 0:
            error = _invariant(board, rules, coverage)
            if error:
                return result(StrategyStatus.INVARIANT_FAILURE, 'invariant.white_turn', error)
        square = _landing(board, column)
        if square is None:
            return result(StrategyStatus.ILLEGAL_CONTINUATION, 'continuation.full_column',
                          f'ply {ply}: physical column {column} is full')
        response, trigger = None, None
        if player == 1:
            expected, response, trigger = pending
            if square != expected:
                return result(StrategyStatus.STRATEGY_VIOLATED, 'strategy.prior_black_move',
                              f'ply {ply}: played {square}; required {response.value} at {expected}'
                              + (f' for {trigger.label}' if trigger else ''))
        before = tuple(r for r in rules if _active(board, r))
        board[6 - int(square[1])][column] = C.WHITE if player == 0 else C.BLACK
        newly_retired = tuple(r for r in before if not _active(board, r))
        trace.append(ExecutedMove(ply, player, column, square, response, trigger, newly_retired))
        if player == 0:
            # Under H1-H4 and the checked prefix, even the triggering White move
            # cannot win before its response. Preserve a failure, never select.
            if _terminal(board):
                return result(StrategyStatus.INVARIANT_FAILURE, 'invariant.white_terminal',
                              f'White reached terminal at ply {ply}; continuation={history!r}')
            try:
                pending = _response(board, rules, square)
            except ValueError as exc:
                return result(StrategyStatus.INVARIANT_FAILURE, 'invariant.response', str(exc))
        else:
            error = _invariant(board, rules, coverage)
            if error:
                return result(StrategyStatus.INVARIANT_FAILURE, 'invariant.black_response', error)
    frozen = tuple(tuple(row) for row in board)
    turn = len(history) % 2
    state = ContinuationState(
        frozen, C.board_digest(frozen, turn), turn, history, tuple(trace),
        tuple(r for r in rules if _active(board, r)),
        tuple(r for r in rules if not _active(board, r)))
    if _terminal(board):
        return result(StrategyStatus.TERMINAL, state=state)
    if turn != 1:
        return result(StrategyStatus.UNSUPPORTED, 'context.white_to_move',
                      'selection needs a continuation ending after White, with Black to move',
                      state=state)
    square, response, trigger = pending
    return result(StrategyStatus.MOVE_SELECTED, column='abcdefg'.index(square[0]),
                  square=square, response=response, triggering_rule=trigger, state=state)
