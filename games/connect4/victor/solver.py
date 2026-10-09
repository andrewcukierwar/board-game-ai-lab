"""One research solver API: exact, established bounds, coverage, and legal play.

This module has no API integration; the opt-in, flag-gated public adapter is
``games/connect4/agents/victor_research_agent.py``. Values are always
mover-relative; bounds identify their player explicitly. Heuristics never
become proof leaves.
"""
from dataclasses import dataclass, field, replace
from math import isfinite
from time import monotonic
from types import SimpleNamespace

from games.connect4.agents.negamax_agent import WIN_SCORE, NegamaxAgent

from .certificate import CertificateStatus, StrategicCertificate, verify_certificate
from .certificate_producer import certificate_from_witness
from .coverage import SearchStatus, search_covering_set
from .exact import Bits, ExactResult, SearchBudget, solve_exact
from .execution import NineRulePolicy, PolicyAudit
from .nine_rule_verification import verify_nine_rule_witness
from .nine_rules import ALL_RULES, NineRuleSearchResult, _checked_rules, search_nine_rule_cover
from .native import MoveProof, NativeBudget, available as native_available, prove_moves
from . import opening_book
from .opening_book import preferred_move
from .rules import RuleName
from .position import Position
from .strategy import StrategyRequest, StrategyStatus, select_black_move
from .verification import VerificationStatus
from .white import WhiteCover, search_white_covers, white_evaluation_contexts


THREE_RULES = (RuleName.CLAIMEVEN, RuleName.BASEINVERSE, RuleName.VERTICAL)
# White refutation-avoidance policies (see docs/victor-opening-and-app-readiness.md):
#   off               keep the Negamax-ranked move
#   first_unrefuted   skip any refuted move; first unrefuted wins (previous default)
#   certified_only    only a verified CL/BI/VE refutation can skip a move
#   positive_evidence leave the Negamax move only for an unrefuted move that
#                     reaches a restricted White threat cover
WHITE_POLICIES = ('off', 'first_unrefuted', 'certified_only', 'positive_evidence')


@dataclass(frozen=True)
class SolverBudget:
    # Exact search reaches the hard 24-cell ceiling; nodes/seconds bound the work.
    exact: SearchBudget = field(default_factory=lambda: SearchBudget(max_remaining=24))
    cover_nodes: int = 10_000
    white_contexts: int = 8
    strategic_children: int = 7
    fallback_depth: int = 4
    policy_audit: SearchBudget = field(default_factory=lambda: SearchBudget(
        nodes=20_000, seconds=0.25, max_remaining=10, table_entries=20_000))
    # Strategic rule universe for Black covers and White contexts (ablations).
    rules: tuple = ALL_RULES
    # Optional wall-clock limit for one whole analysis. Strategic work is skipped
    # once it passes; a legal move is always returned. None = node budgets only.
    deadline: float | None = None
    # Exact opening book (completed offline oracle searches) consulted first.
    # ``book_exclude`` drops entries whose ONLY sources are listed (held-out runs).
    opening_book: bool = True
    book_exclude: tuple = ()
    white_refutation: str = 'first_unrefuted'
    # Optional build-time native accelerator; None preserves the original solver.
    native: NativeBudget | None = None

    def __post_init__(self):
        object.__setattr__(self, 'rules', _checked_rules(self.rules))
        if self.native is not None and type(self.native) is not NativeBudget:
            raise ValueError('native must be NativeBudget or None')
        if type(self.exact) is not SearchBudget or type(self.policy_audit) is not SearchBudget:
            raise ValueError('search budgets must be SearchBudget')
        for name in ('cover_nodes', 'white_contexts', 'strategic_children'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f'{name} must be a nonnegative integer')
        if self.strategic_children > 7:
            raise ValueError('strategic_children must be <=7')
        if type(self.fallback_depth) is not int or not 1 <= self.fallback_depth <= 6:
            raise ValueError('fallback_depth must be integer 1..6')
        if self.white_refutation not in WHITE_POLICIES:
            raise ValueError(f'white_refutation must be one of {WHITE_POLICIES}')
        if type(self.opening_book) is not bool:
            raise ValueError('opening_book must be a bool')
        if type(self.book_exclude) is not tuple or not all(
                s in ('shallow', 'closure', 'benchmark') for s in self.book_exclude):
            raise ValueError('book_exclude must be a tuple of opening-book source names')
        if self.deadline is not None and (type(self.deadline) not in (int, float)
                                          or not isfinite(self.deadline) or self.deadline <= 0):
            raise ValueError('deadline must be a positive finite number of seconds, or None')


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
    # White moves checked for a Black reply reaching a Black cover:
    # 'certified' (CL/BI/VE theorem: the move cannot win), 'nine_rule'
    # (exploratory cover only) or 'unrefuted' (none found within budgets).
    white_refutations: tuple[tuple[int, str], ...] = ()
    deadline_reached: bool = False
    move_proof: MoveProof | None = None

    @property
    def justified_move(self):
        return self.move_kind in ('exact', 'opening_book', 'terminal_win', 'forced_defense',
                                  'strategic_nonloss', 'verified_policy_nonloss', 'search_nonloss')


def _certificate(position, budget):
    cover = search_covering_set(position, node_budget=budget.cover_nodes)
    if cover.witness is None:
        return None
    cert = certificate_from_witness(cover.witness)
    verdict = verify_certificate(cert)
    return cert if verdict.status is CertificateStatus.HYPOTHESES_VERIFIED else None


def _refutation(child, budget, expired):
    """Does some Black reply to this White move reach a Black cover?

    'certified' only for an independently verified CL/BI/VE certificate, which
    proves the White move cannot win. A nine-rule cover is exploratory evidence.
    """
    found = 'unrefuted'
    for b in child.legal():
        grandchild = child.drop(b)
        if grandchild.value is not None:
            continue  # Immediate Black wins are handled by the tactical filters.
        if expired():
            return 'unchecked'
        position = Position.from_board(grandchild.board, 0)
        if _certificate(position, budget):
            return 'certified'
        if expired():  # finer deadline granularity: one cover search per check
            return 'unchecked' if found == 'unrefuted' else found
        if found == 'unrefuted' and set(budget.rules) - set(THREE_RULES):
            cover = search_nine_rule_cover(position, node_budget=budget.cover_nodes,
                                           rules=budget.rules)
            if cover.witness is not None:
                found = 'nine_rule'
    return found


def _exact_choice(p, exact, budget):
    """An optimal move; equal-value ties go to the bounded Negamax ranking.

    Every candidate has the same proved W/D/L value, so the choice stays exact.
    In lost or drawn positions this prefers resilient moves against fallible
    opponents instead of the first column in centre order.
    """
    return preferred_move(p.board, p.player_to_move, exact.move_values, exact.value,
                          budget.fallback_depth)


def _book_result(p, budget, started):
    """A SolverResult from an exact opening-book entry, or None (normal policy).

    Entries are completed offline oracle searches; absent, unresolved, invalid
    or disabled lookups never influence the move.
    """
    if not budget.opening_book or p.terminal:
        return None
    hit = opening_book.lookup(p, exclude_sources=budget.book_exclude)
    if hit.status != 'exact':
        return None
    exact = ExactResult('opening_book', hit.value, hit.move_values, 0, 0, 0,
                        monotonic() - started)
    return SolverResult(p, _exact_choice(p, exact, budget), 'opening_book', hit.value, None,
                        'exact opening-book entry (completed independent oracle search); '
                        'Negamax ranks equal-value moves', exact)


def analyze_position(board, player_to_move, budget: SolverBudget = SolverBudget(), *,
                     exact: ExactResult | None = None):
    """Evaluate and select on a detached basic-valid standard-board snapshot.

    Historical reachability is not established without a game replay. Exact
    values apply to the mathematical position. Whole-board Black covers require
    White to move; restricted White contexts require Black to move. ``exact``
    may pass an already completed search of this same position.
    """
    if type(budget) is not SolverBudget:
        raise ValueError('budget must be SolverBudget')
    started = monotonic()
    p = Position.from_board(board, player_to_move)
    book = _book_result(p, budget, started)
    if book is not None:
        return book
    bits = Bits.from_position(p)
    limit = budget.exact
    if budget.deadline is not None and (limit.seconds is None or limit.seconds > budget.deadline):
        limit = replace(limit, seconds=budget.deadline)
    proof, scores = None, None
    if exact is None and budget.native is not None and not p.terminal:
        scores = NegamaxAgent(budget.fallback_depth).score_moves(
            SimpleNamespace(board=p.board, current_player=player_to_move))
        order = tuple(sorted(range(7), key=lambda c: (-scores.get(c, -float('inf')),
                                                    (3, 2, 4, 1, 5, 0, 6).index(c))))
        native_limit = budget.native
        if budget.deadline is not None:
            available = max(0.0, budget.deadline - (monotonic() - started))
            if native_limit.seconds is None or native_limit.seconds > available:
                native_limit = replace(native_limit, seconds=available)
        proof = prove_moves(p, native_limit, order)
        if proof.intervals:
            # Native bounds replace, rather than duplicate, the Python exact pass.
            # Only singleton intervals become exact move values; unknowns stay unknown.
            exact = ExactResult('unknown_native_partial', None, (), 0, 0, 0, 0.0)
            if proof.optimal_moves:
                move = next(c for c in order if c in proof.optimal_moves)
                if proof.status == 'all_moves':
                    exact = ExactResult('exact', proof.lower,
                        tuple((c, lo) for c, lo, _ in proof.intervals),
                        proof.nodes, proof.cache_hits, native_limit.table_entries, proof.elapsed)
                return SolverResult(p, move, 'exact', proof.lower, None,
                    'completed bounded proof of an optimal move (tactics and established CL bounds); '
                    'unresolved alternatives retain explicit bounds', exact, move_proof=proof)
    if exact is None:
        exact = solve_exact(p, limit)
    if p.terminal:
        return SolverResult(p, None, 'terminal', exact.value, None, 'game over', exact)
    if exact.status == 'exact':
        return SolverResult(p, _exact_choice(p, exact, budget), 'exact', exact.value, None,
                            'completed terminal-only search; Negamax ranks equal-value moves',
                            exact)
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
    if scores is None:
        scores = NegamaxAgent(budget.fallback_depth).score_moves(
            SimpleNamespace(board=p.board, current_player=player_to_move))
    if proof is not None and proof.intervals:
        safe = tuple(c for c in safe if c in proof.admissible_moves)
    safe = tuple(sorted(safe, key=lambda c: -scores[c]))
    # A score below -WIN_SCORE is a completed forced loss inside the horizon
    # (terminal leaves only); strategic preferences never choose such a move.
    candidates = tuple(c for c in safe if scores[c] > -WIN_SCORE)[:budget.strategic_children]
    move, kind = safe[0], 'forced_defense' if forced else 'heuristic'
    reason = ('only move avoiding immediate defeat' if forced else
              f'depth-{budget.fallback_depth} Negamax heuristic with deterministic center ties')
    black_cover, white_covers, cert, audit, child_cover = None, (), None, None, None
    bound, white_count, white_child_covers, refutations = None, 0, (), ()
    late = False

    def expired():
        nonlocal late
        late = late or (budget.deadline is not None
                        and monotonic() - started >= budget.deadline)
        return late

    if budget.cover_nodes and not expired():
        if player_to_move == 0:
            black_cover = search_nine_rule_cover(p, node_budget=budget.cover_nodes,
                                                 rules=budget.rules)
            if black_cover.witness:
                verdict = verify_nine_rule_witness(p, black_cover.witness)
                if verdict.status is not VerificationStatus.VERIFIED_UNCERTIFIED:
                    raise RuntimeError('producer/verifier disagreement')
                cert = None if expired() else _certificate(p, budget)
                if cert:
                    bound = 'Black >= draw (established CL/BI/VE theorem on this position)'
                elif not expired():
                    audit = NineRulePolicy(black_cover.witness).audit(budget.policy_audit)
                    if audit.status == 'verified_policy_nonloss':
                        bound = 'Black >= draw (complete adversarial replay of concrete policy)'
            if not forced and scores[safe[0]] < WIN_SCORE:
                move, kind, reason, white_child_covers, refutations = _white_choice(
                    bits, candidates, budget, expired, (move, kind, reason))
        else:
            contexts = white_evaluation_contexts(p)
            white_count = len(contexts)
            if budget.white_contexts:
                white_covers = search_white_covers(p, node_budget=budget.cover_nodes,
                                                  context_budget=budget.white_contexts,
                                                  rules=budget.rules)
            # A Black move to a verified three-rule cover is demonstrably
            # non-losing. Otherwise prefer a nine-rule child cover as research.
            children = tuple((c, Position.from_board(bits.drop(c).board, 0))
                             for c in candidates)
            for c, child in children:
                if expired():
                    break
                child_cert = _certificate(child, budget)
                if child_cert:
                    move, kind, cert = c, 'strategic_nonloss', child_cert
                    bound = 'Black >= draw (move reaches established CL/BI/VE theorem)'
                    reason = 'child certificate independently checked; sigma_R can continue'
                    break
            if cert is None:
                for c, child in children:
                    if expired():
                        break
                    cover = search_nine_rule_cover(child, node_budget=budget.cover_nodes,
                                                   rules=budget.rules)
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
    if proof is not None and proof.intervals and proof.lower >= 0:
        kind = 'search_nonloss'
        bound = 'mover >= draw (completed move proof; a win may remain unresolved)'
        reason = 'selection restricted to moves with a proved non-loss lower bound'
    if late:
        reason += '; analysis deadline reached, remaining strategic checks skipped'
    return SolverResult(p, move, kind, None, bound, reason, exact, black_cover,
                        white_covers, white_count, cert, audit, child_cover, white_child_covers,
                        refutations, late, proof)


def _acceptable(status, policy):
    return status == 'unrefuted' or (policy == 'certified_only' and status == 'nine_rule')


def white_choice_from_scan(scan, fallback, policy):
    """(column, label) from [(column, refutation status, White cover found)] in Negamax order.

    ``label`` is None when the fallback stands. A refutation proves at most that
    a move cannot WIN; it says nothing about whether an alternative avoids
    losing, so a move chosen here never carries an outcome claim.
    """
    if policy == 'off':
        return fallback, None
    first = None
    for c, status, cover in scan:
        if not _acceptable(status, policy):
            continue
        if cover:
            return c, 'exploratory_white_context'
        if first is None:
            first = c
    if policy != 'positive_evidence' and first is not None and first != fallback:
        return first, 'exploratory_unrefuted'
    return fallback, None


def _white_choice(bits, candidates, budget, expired, fallback):
    """White may leave the Negamax move after a refutation scan (policy-dependent).

    Scan candidates in Negamax order: does some Black reply reach a Black cover
    (a certified refutation proves the move cannot win)? Restricted White threat
    covers are searched for acceptable candidates. Every choice here is
    exploratory; nothing becomes an outcome claim.
    """
    move, kind, reason = fallback
    policy = budget.white_refutation
    if policy == 'off' or (policy == 'positive_evidence' and not budget.white_contexts):
        return move, kind, reason, (), ()
    scan, covers = [], {}
    for c in candidates:
        if expired():
            break
        child = bits.drop(c)
        status = _refutation(child, budget, expired)
        found = ()
        if (_acceptable(status, policy) and budget.white_contexts and
                white_evaluation_contexts(Position.from_board(child.board, 1))):
            found = search_white_covers(Position.from_board(child.board, 1),
                                        node_budget=budget.cover_nodes,
                                        context_budget=budget.white_contexts,
                                        rules=budget.rules)
        has_cover = any(cover.status is SearchStatus.FOUND for cover in found)
        scan.append((c, status, has_cover))
        if has_cover:
            covers[c] = found
            break  # the first acceptable move with a White cover decides
        if _acceptable(status, policy) and not budget.white_contexts:
            break  # without White contexts there is nothing further to prefer
    choice, label = white_choice_from_scan(scan, move, policy)
    refutations = tuple((c, status) for c, status, _ in scan)
    if label == 'exploratory_white_context':
        return (choice, label, 'unrefuted move reaches a restricted White threat cover; '
                'outcome remains conditional', covers[choice], refutations)
    if label == 'exploratory_unrefuted':
        return (choice, label, 'higher-ranked moves let Black reach a Black cover; no reply '
                'to this move reaches one within budgets (not an outcome claim)', (), refutations)
    return move, kind, reason, (), refutations


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
        if self.budget.native is not None and native_available():
            # Native mode re-proves each position, as the stateless public adapter
            # does. Original certificate/policy retention is unchanged when off.
            self.last_result = analyze_position(board, player_to_move, self.budget)
            return self.last_result
        p = Position.from_board(board, player_to_move)
        bits = Bits.from_position(p)
        continuation = None
        # Exact book entries, like exact search, take precedence over a retained plan.
        book = _book_result(p, self.budget, monotonic())
        if book is not None:
            self.cert = self.policy = self.expected = self.audit = None
            self.last_result = book
            return book
        if player_to_move == 1 and self.expected is not None:
            continuation = next((self.history + (c,) for c in self.expected.legal()
                                 if self.expected.drop(c) == bits), None)
        # Exact and immediate wins have priority over a retained response plan.
        exact = None
        if continuation is not None and not p.terminal:
            limit = self.budget.exact
            if self.budget.deadline is not None and (
                    limit.seconds is None or limit.seconds > self.budget.deadline):
                limit = replace(limit, seconds=self.budget.deadline)
            exact = solve_exact(p, limit)
            if exact.status == 'exact' or bits.winning(1):
                self.cert = self.policy = self.expected = self.audit = None
                self.last_result = SolverResult(p,
                    _exact_choice(p, exact, self.budget) if exact.status == 'exact'
                    else bits.winning(1)[0],
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
        # Reuse a search of this same board; never repeat it inside analysis.
        result = analyze_position(board, player_to_move, self.budget, exact=exact)
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
