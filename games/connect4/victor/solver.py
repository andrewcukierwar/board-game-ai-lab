"""One research solver API: exact, established bounds, coverage, and legal play.

No public VictorAgent/factory/API integration. Values are always mover-relative;
bounds identify their player explicitly. Heuristics never become proof leaves.
"""
from dataclasses import dataclass, field
from types import SimpleNamespace

from games.connect4.agents.negamax_agent import NegamaxAgent

from .certificate import CertificateStatus, StrategicCertificate, verify_certificate
from .certificate_producer import certificate_from_witness
from .coverage import SearchStatus, search_covering_set
from .exact import Bits, ExactResult, SearchBudget, solve_exact
from .execution import NineRulePolicy, PolicyAudit
from .nine_rule_verification import verify_nine_rule_witness
from .nine_rules import NineRuleSearchResult, search_nine_rule_cover
from .position import Position
from .strategy import StrategyRequest, StrategyStatus, select_black_move
from .verification import VerificationStatus
from .white import WhiteCover, search_white_covers, white_evaluation_contexts


@dataclass(frozen=True)
class SolverBudget:
    exact: SearchBudget = field(default_factory=SearchBudget)
    cover_nodes: int = 10_000
    white_contexts: int = 8
    strategic_children: int = 7
    fallback_depth: int = 4
    policy_audit: SearchBudget = field(default_factory=lambda: SearchBudget(
        nodes=20_000, seconds=0.25, max_remaining=10, table_entries=20_000))

    def __post_init__(self):
        if type(self.exact) is not SearchBudget or type(self.policy_audit) is not SearchBudget:
            raise ValueError('search budgets must be SearchBudget')
        for name in ('cover_nodes', 'white_contexts', 'strategic_children'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f'{name} must be a nonnegative integer')
        if self.strategic_children > 7:
            raise ValueError('strategic_children must be <=7')
        if type(self.fallback_depth) is not int or not 1 <= self.fallback_depth <= 6:
            raise ValueError('fallback_depth must be integer 1..6')


@dataclass(frozen=True)
class SolverResult:
    position: Position
    move: int | None
    move_kind: str
    exact_value: int | None
    bound: str | None
    reason: str
    exact: ExactResult
    black_cover: NineRuleSearchResult | None = None
    white_covers: tuple[WhiteCover, ...] = ()
    white_context_count: int = 0
    certificate: StrategicCertificate | None = None
    policy_audit: PolicyAudit | None = None
    # Child cover is bound to the board AFTER the returned Black move.
    continuation_cover: NineRuleSearchResult | None = None
    continuation_white_covers: tuple[WhiteCover, ...] = ()

    @property
    def justified_move(self):
        return self.move_kind in ('exact', 'terminal_win', 'forced_defense',
                                  'strategic_nonloss', 'verified_policy_nonloss')


def _certificate(position, budget):
    cover = search_covering_set(position, node_budget=budget.cover_nodes)
    if cover.witness is None:
        return None
    cert = certificate_from_witness(cover.witness)
    verdict = verify_certificate(cert)
    return cert if verdict.status is CertificateStatus.HYPOTHESES_VERIFIED else None


def analyze_position(board, player_to_move, budget: SolverBudget = SolverBudget()):
    """Evaluate and select on a detached basic-valid standard-board snapshot.

    Historical reachability is not established without a game replay. Exact
    values apply to the mathematical position. Whole-board Black covers require
    White to move; restricted White contexts require Black to move.
    """
    if type(budget) is not SolverBudget:
        raise ValueError('budget must be SolverBudget')
    p = Position.from_board(board, player_to_move)
    bits = Bits.from_position(p)
    exact = solve_exact(p, budget.exact)
    if p.terminal:
        return SolverResult(p, None, 'terminal', exact.value, None, 'game over', exact)
    if exact.status == 'exact':
        return SolverResult(p, exact.best_move, 'exact', exact.value, None,
                            'completed terminal-only search', exact)
    wins = bits.winning(player_to_move)
    if wins:
        return SolverResult(p, wins[0], 'terminal_win', 1, None,
                            'legal move completes four', exact)
    safe = tuple(c for c in bits.legal() if bits.drop(c).value == 0 or
                 not bits.drop(c).winning(1 - player_to_move))
    if not safe:
        return SolverResult(p, bits.legal()[0], 'exact', -1, None,
                            'every legal move allows an immediate opponent win', exact)
    # A mandatory defense is a proved necessary move, not a non-loss guarantee.
    forced = len(safe) == 1 and bool(bits.winning(1 - player_to_move))
    # Reuse the established bounded non-neural heuristic agent. Scores are
    # ordering/fallback evidence ONLY, never exact or strategic proof leaves.
    scores = NegamaxAgent(budget.fallback_depth).score_moves(
        SimpleNamespace(board=p.board, current_player=player_to_move))
    safe = tuple(sorted(safe, key=lambda c: -scores[c]))
    move, kind = safe[0], 'forced_defense' if forced else 'heuristic'
    reason = ('only move avoiding immediate defeat' if forced else
              f'depth-{budget.fallback_depth} Negamax heuristic with deterministic center ties')
    black_cover, white_covers, cert, audit, child_cover = None, (), None, None, None
    bound, white_count, white_child_covers = None, 0, ()
    if budget.cover_nodes:
        if player_to_move == 0:
            black_cover = search_nine_rule_cover(p, node_budget=budget.cover_nodes)
            if black_cover.witness:
                verdict = verify_nine_rule_witness(p, black_cover.witness)
                if verdict.status is not VerificationStatus.VERIFIED_UNCERTIFIED:
                    raise RuntimeError('producer/verifier disagreement')
                cert = _certificate(p, budget)
                if cert:
                    bound = 'Black >= draw (established CL/BI/VE theorem on this position)'
                else:
                    audit = NineRulePolicy(black_cover.witness).audit(budget.policy_audit)
                    if audit.status == 'verified_policy_nonloss':
                        bound = 'Black >= draw (complete adversarial replay of concrete policy)'
            if not forced:
                for c in safe[:budget.strategic_children]:
                    child = Position.from_board(bits.drop(c).board, 1)
                    if not white_evaluation_contexts(child):
                        continue
                    covers = search_white_covers(child, node_budget=budget.cover_nodes,
                                                context_budget=budget.white_contexts)
                    if any(cover.status is SearchStatus.FOUND for cover in covers):
                        move, kind, white_child_covers = c, 'exploratory_white_context', covers
                        reason = 'move reaches a restricted White threat cover; outcome remains conditional'
                        break
        else:
            contexts = white_evaluation_contexts(p)
            white_count = len(contexts)
            white_covers = search_white_covers(p, node_budget=budget.cover_nodes,
                                              context_budget=budget.white_contexts)
            # A Black move to a verified three-rule cover is demonstrably
            # non-losing. Otherwise prefer a nine-rule child cover as research.
            children = tuple((c, Position.from_board(bits.drop(c).board, 0))
                             for c in safe[:budget.strategic_children])
            for c, child in children:
                child_cert = _certificate(child, budget)
                if child_cert:
                    move, kind, cert = c, 'strategic_nonloss', child_cert
                    bound = 'Black >= draw (move reaches established CL/BI/VE theorem)'
                    reason = 'child certificate independently checked; sigma_R can continue'
                    break
            if cert is None:
                for c, child in children:
                    cover = search_nine_rule_cover(child, node_budget=budget.cover_nodes)
                    if cover.witness is None:
                        continue
                    policy = NineRulePolicy(cover.witness)
                    audit = policy.audit(budget.policy_audit)
                    if audit.status in ('refuted', 'unsupported_policy'):
                        continue
                    child_cover = cover
                    move, kind = c, 'exploratory_nine_rule'
                    reason = 'move reaches verified coverage; concrete response policy remains conditional'
                    if audit.status == 'verified_policy_nonloss':
                        kind = 'verified_policy_nonloss'
                        bound = 'Black >= draw (move reaches completely audited concrete policy)'
                    break
    return SolverResult(p, move, kind, None, bound, reason, exact, black_cover,
                        white_covers, white_count, cert, audit, child_cover, white_child_covers)


def select_move(board, player_to_move, budget: SolverBudget = SolverBudget()):
    """Return the same inspectable result as analyze_position, including its move."""
    return analyze_position(board, player_to_move, budget)


class VictorSolver:
    """Game session retaining original rule instances across Black responses.

    Calls still take exact board/turn. A mismatch discards the old policy rather
    than applying responses to another game. Certificates are reverified by the
    established strategy on every response. Conditional policies retain their
    audited/unaudited label; tactical wins and exact solving take precedence.
    """
    def __init__(self, budget: SolverBudget = SolverBudget()):
        if type(budget) is not SolverBudget:
            raise ValueError('budget must be SolverBudget')
        self.budget = budget
        self.expected = self.cert = self.policy = self.audit = None
        self.history = ()
        self.last_result = None

    def select_move(self, board, player_to_move):
        p = Position.from_board(board, player_to_move)
        bits = Bits.from_position(p)
        continuation = None
        if player_to_move == 1 and self.expected is not None:
            continuation = next((self.history + (c,) for c in self.expected.legal()
                                 if self.expected.drop(c) == bits), None)
        # Exact and immediate wins have priority over a retained response plan.
        if continuation is not None and not p.terminal:
            exact = solve_exact(p, self.budget.exact)
            if exact.status == 'exact' or bits.winning(1):
                self.cert = self.policy = self.expected = self.audit = None
                self.last_result = SolverResult(p,
                    exact.best_move if exact.status == 'exact' else bits.winning(1)[0],
                    'exact' if exact.status == 'exact' else 'terminal_win',
                    exact.value if exact.status == 'exact' else 1, None,
                    'completed terminal-only search' if exact.status == 'exact' else
                    'legal move completes four', exact)
                return self.last_result
            if exact.status != 'exact' and not bits.winning(1):
                if self.cert:
                    d = select_black_move(StrategyRequest(self.cert, continuation))
                    c = d.column if d.status is StrategyStatus.MOVE_SELECTED else None
                    kind, bound = 'strategic_nonloss', 'Black >= draw (retained CL/BI/VE sigma_R)'
                else:
                    d = self.policy.select(continuation)
                    c = d.column if d.status == 'selected' else None
                    checked = self.audit and self.audit.status == 'verified_policy_nonloss'
                    kind = 'verified_policy_nonloss' if checked else 'exploratory_nine_rule'
                    bound = 'Black >= draw (completely audited retained policy)' if checked else None
                tactical_survival = (c is not None and c in bits.legal() and
                    (bits.drop(c).value is not None or not bits.drop(c).winning(0)))
                if tactical_survival:
                    self.history = continuation + (c,)
                    self.expected = bits.drop(c)
                    self.last_result = SolverResult(p, c, kind, None, bound,
                        'response replays original rule instances from exact anchor', exact,
                        certificate=self.cert, policy_audit=self.audit)
                    return self.last_result
        self.cert = self.policy = self.expected = self.audit = None
        result = analyze_position(board, player_to_move, self.budget)
        if player_to_move == 1 and result.move is not None:
            if result.certificate:
                self.cert = result.certificate
            elif result.continuation_cover and result.move_kind in (
                    'exploratory_nine_rule', 'verified_policy_nonloss'):
                self.policy = NineRulePolicy(result.continuation_cover.witness)
                self.audit = result.policy_audit
            if self.cert or self.policy:
                self.expected = bits.drop(result.move)
                self.history = ()
        self.last_result = result
        return result

    def choose_move(self, game):
        result = self.select_move(game.board, game.current_player)
        if result.move is None:
            raise ValueError('cannot choose from terminal position')
        return result.move
