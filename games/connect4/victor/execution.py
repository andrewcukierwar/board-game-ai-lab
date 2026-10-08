"""Replayable, conditional nine-rule Black policy derived from Allis ch.6–7.

The old sigma_R implementation is untouched. This policy has no universal
composition theorem. A bounded adversarial replay may establish non-loss for
one concrete starting board and policy, or return a counterexample/unknown.
"""
from dataclasses import dataclass

from .exact import Bits, Cutoff, SearchBudget, Work
from .geometry import Square
from .nine_rule_verification import verify_nine_rule_witness
from .verification import VerificationStatus
from .nine_rules import NineRuleWitness
from .rules import RuleName

CL, BI, VE = RuleName.CLAIMEVEN, RuleName.BASEINVERSE, RuleName.VERTICAL
AE, LI, HI = RuleName.AFTEREVEN, RuleName.LOWINVERSE, RuleName.HIGHINVERSE
BC, BE, SB = RuleName.BASECLAIM, RuleName.BEFORE, RuleName.SPECIALBEFORE


def bit(s):
    return 1 << (7 * s.column + s.row - 1)


def _compile(candidate):
    """Local source obligations; callers still must establish their context."""
    c = candidate
    if c.rule in (CL, BI, VE):
        return (Obligation(c.rule, c.squares),)
    if c.rule in (AE, BE, SB):
        return (tuple(Obligation(k.rule, k.squares) for k in c.components)
                + ((Obligation(BI, c.roles),) if c.rule == SB else ()))
    if c.rule == LI:
        return (Obligation(LI, tuple(s for k in c.components for s in k.squares)),)
    return (Obligation(c.rule, c.roles),)


@dataclass(frozen=True)
class Obligation:
    kind: RuleName
    squares: tuple[Square, ...]


@dataclass(frozen=True)
class RuleState:
    index: int  # identity in the ORIGINAL verified witness, never re-enumerated
    phase: str
    obligations: tuple[Obligation, ...]


@dataclass(frozen=True)
class PolicyState:
    board: Bits
    rules: tuple[RuleState, ...]
    pending: tuple[Square, ...] = ()


@dataclass(frozen=True)
class PolicyDecision:
    status: str
    column: int | None = None
    kind: str = ''
    state: PolicyState | None = None
    detail: str = ''


@dataclass(frozen=True)
class PolicyAudit:
    status: str
    nodes: int
    elapsed: float
    counterexample: tuple[int, ...] = ()
    detail: str = ''


class NineRulePolicy:
    """Immutable initial witness, residual obligations and terminal-first play.

    Accepts ONLY independently checked Black covers. Every public selection
    replays from the exact initial board and rejects divergent Black moves.
    Rule retirement requires actual blocking of all original target groups;
    timing rules can stay active until a terminal win instead of retiring early.
    """
    def __init__(self, witness: NineRuleWitness):
        if type(witness) is not NineRuleWitness:
            raise ValueError('expected a NineRuleWitness')
        try:
            position = witness.context.position
        except AttributeError as exc:
            raise ValueError('missing witness position') from exc
        verdict = verify_nine_rule_witness(position, witness)
        if verdict.status is not VerificationStatus.VERIFIED_UNCERTIFIED:
            raise ValueError('nine-rule witness failed independent verification')
        self.witness = witness
        self.initial = Bits.from_position(witness.context.position)
        states = []
        for i, e in enumerate(witness.evidence):
            states.append(RuleState(i, 'waiting', _compile(e.candidate)))
        self.start = PolicyState(self.initial, tuple(states))

    def _satisfied(self, p, obligation):
        squares, kind = obligation.squares, obligation.kind
        black = p.pieces[1]
        if kind == CL:
            return bool(black & bit(squares[1]))
        if kind in (VE, BI):
            return any(black & bit(s) for s in squares)
        return False

    def _prune(self, state):
        states = []
        for r in state.rules:
            if state.board.value is not None:
                states.append(RuleState(r.index, 'terminal', ()))
                continue
            e = self.witness.evidence[r.index]
            blocked = all(any(state.board.pieces[1] & bit(s) for s in g.squares)
                          for g in e.conditional_solved_groups)
            obligations = tuple(o for o in r.obligations if not self._satisfied(state.board, o))
            states.append(RuleState(r.index, 'retired' if blocked else r.phase,
                                    () if blocked else obligations))
        return PolicyState(state.board, tuple(states), state.pending)

    def _white(self, state, column):
        p = state.board
        s = Square(5 - p.heights[column], column)
        after = p.drop(column)
        if after.value is not None:
            return self._prune(PolicyState(after, state.rules))
        states, demands = [], []
        for r in state.rules:
            updated, triggered = [], False
            for o in r.obligations:
                k, sq = o.kind, o.squares
                if self._satisfied(p, o):
                    continue
                if k in (CL, VE, BI):
                    if s == sq[0] or (k == BI and s == sq[1]):
                        demands.append(sq[1] if s == sq[0] else sq[0])
                        triggered = True
                    updated.append(o)
                elif k == LI:
                    pairs = (sq[:2], sq[2:])
                    if s in (sq[0], sq[2]):
                        demands.append(next(pair[1] for pair in pairs if s == pair[0]))
                        updated.extend(Obligation(VE, pair) for pair in pairs)
                        triggered = True
                    else:
                        updated.append(o)
                elif k == HI:
                    triples = (sq[:3], sq[3:])
                    if s in (sq[0], sq[3]):
                        mine = next(t for t in triples if t[0] == s)
                        other = next(t for t in triples if t[0] != s)
                        demands.append(mine[1])
                        updated.append(Obligation(CL, other[1:]))
                        # §6.6: only an ORIGINALLY playable lower licenses this BI.
                        if self.initial.heights[other[0].column] == other[0].row - 1:
                            updated.append(Obligation(BI, (mine[2], other[0])))
                        triggered = True
                    else:
                        updated.append(o)
                elif k == BC:
                    first, second, third = sq
                    upper = Square(second.row_index - 1, second.column)
                    if s in sq:
                        demands.append(second if s == third else third)
                        updated.append(Obligation(CL, (second, upper)) if s == first else
                                       Obligation(BI, (first, upper)))
                        triggered = True
                    else:
                        updated.append(o)
            states.append(RuleState(r.index, 'activated' if triggered else r.phase, tuple(updated)))
        return self._prune(PolicyState(after, tuple(states), tuple(sorted(set(demands)))))

    def _decision(self, state):
        p = state.board
        if p.value is not None:
            return PolicyDecision('terminal', state=state)
        if p.turn != 1:
            return PolicyDecision('opponent_to_move', state=state)
        wins = p.winning(1)
        if wins:
            return PolicyDecision('selected', wins[0], 'terminal_win', state)
        if len(state.pending) > 1:
            return PolicyDecision('conflicting_obligations', state=state,
                                  detail=', '.join(s.name for s in state.pending))
        if state.pending:
            s = state.pending[0]
            if p.heights[s.column] != s.row - 1:
                return PolicyDecision('unplayable_obligation', state=state, detail=s.name)
            return PolicyDecision('selected', s.column, 'forced_response', state)
        forbidden = set()
        for r in state.rules:
            for o in r.obligations:
                if o.kind == CL:
                    forbidden.add(o.squares[0])
                elif o.kind == LI:
                    forbidden.update((o.squares[0], o.squares[2]))
                elif o.kind == HI:
                    forbidden.update((o.squares[0], o.squares[3]))
                elif o.kind == BC:
                    forbidden.update(o.squares)
        choices = [c for c in p.legal() if Square(5 - p.heights[c], c) not in forbidden]
        if not choices:
            return PolicyDecision('no_permitted_spare', state=state)
        # Prefer the established even-row convention. Mixed Before Verticals
        # also allow odd lower spares; these are exploratory until policy audit.
        choices.sort(key=lambda c: (p.heights[c] + 1) % 2)
        return PolicyDecision('selected', choices[0], 'conditional_spare', state)

    def _black(self, state, column):
        return self._prune(PolicyState(state.board.drop(column), state.rules))

    def select(self, continuation=()):
        if (type(continuation) is not tuple or len(continuation) > self.initial.remaining
                or any(type(c) is not int or not 0 <= c < 7 for c in continuation)):
            return PolicyDecision('invalid_continuation')
        state = self.start
        for c in continuation:
            if state.board.value is not None or c not in state.board.legal():
                return PolicyDecision('illegal_continuation', state=state)
            if state.board.turn == 0:
                state = self._white(state, c)
            else:
                decision = self._decision(state)
                if decision.column != c:
                    return PolicyDecision('policy_violated', state=state)
                state = self._black(state, c)
        return self._decision(state)

    def audit(self, budget: SearchBudget = SearchBudget(max_remaining=10)):
        """Every White move, fixed deterministic Black response, to terminals.

        A completed audit proves this policy non-losing on THIS board, not the
        nine-rule theorem or optimality. A failing White history is preserved.
        No depth frontier is accepted as a safe leaf. Bound includes cache hits.
        """
        if type(budget) is not SearchBudget:
            raise ValueError('budget must be SearchBudget')
        work, cache = Work(budget), set()
        if self.initial.remaining > budget.max_remaining:
            return PolicyAudit('unknown_remaining_cap', 0, 0.0)

        def visit(state, history):
            work.enter()
            p = state.board
            if p.value is not None:
                if p.value and p.turn == 1:  # Black to move after White won
                    return ('refuted', history, 'White wins')
                return None
            if state in cache:
                return None
            if p.turn == 0:
                for c in p.legal():
                    failure = visit(self._white(state, c), history + (c,))
                    if failure:
                        return failure
            else:
                decision = self._decision(state)
                if decision.column is None:
                    return ('unsupported_policy', history, decision.status)
                failure = visit(self._black(state, decision.column), history + (decision.column,))
                if failure:
                    return failure
            if len(cache) >= budget.table_entries:
                raise Cutoff('unknown_table_budget')
            cache.add(state)
            return None

        try:
            failure = visit(self.start, ())
            status, history, detail = failure or ('verified_policy_nonloss', (), '')
        except Cutoff as exc:
            status, history, detail = str(exc), (), ''
        from time import monotonic
        return PolicyAudit(status, work.nodes, monotonic() - work.start, history, detail)
