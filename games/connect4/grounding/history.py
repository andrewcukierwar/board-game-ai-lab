"""Immutable move records; built before committing a session mutation."""
from dataclasses import dataclass

from .analysis import outcome


@dataclass(frozen=True)
class AlphaZeroMoveEvidence:
    """Original search evidence for one AlphaZero move; the guard never alters visits."""
    model_id: str
    tactical_guard: bool
    simulations: int
    original_column: int  # unguarded maximum-visit PUCT choice
    root_visits: tuple[int, ...]
    guard_reason: str | None  # None when the guard is disabled

    @classmethod
    def from_decision(cls, model_id, decision):
        return cls(model_id, decision.guard is not None, sum(decision.visits), decision.original_action,
                   tuple(int(n) for n in decision.visits), decision.guard_reason)

    def to_dict(self):
        return {'model_id': self.model_id, 'simulations': self.simulations,
                'original_column': self.original_column, 'root_visits': list(self.root_visits),
                'tactical_guard': self.tactical_guard, 'guard_reason': self.guard_reason}


@dataclass(frozen=True)
class MoveRecord:
    move_number: int  # One stone = one move (ply), unlike thesis paired notation.
    revision_before: int
    revision: int
    player: int
    agent_type: str
    agent_depth: int | None
    column: int
    board_before: tuple
    board_after: tuple
    outcome_status: str
    winner: int | None
    search_stats: tuple[int, int] | None = None  # requested/executed F1 simulations
    alphazero: AlphaZeroMoveEvidence | None = None  # AlphaZero moves only

    def to_dict(self):
        return {'move_number': self.move_number, 'revision_before': self.revision_before,
                'revision': self.revision, 'player': self.player,
                'agent': {'type': self.agent_type, **({'depth': self.agent_depth}
                                                    if self.agent_depth is not None else {}),
                          **({'tactical_guard': self.alphazero.tactical_guard}
                             if self.alphazero is not None else {})},
                'column': self.column,
                'board_before': [list(row) for row in self.board_before],
                'board_after': [list(row) for row in self.board_after],
                'outcome': {'status': self.outcome_status, 'winner': self.winner},
                **({'search': {'requested_simulations': self.search_stats[0],
                               'executed_simulations': self.search_stats[1]}}
                   if self.search_stats is not None else {}),
                **({'alphazero': self.alphazero.to_dict()} if self.alphazero is not None else {})}


def record_move(before, after, player, config, column, revision, search_stats=None, alphazero=None):
    result = outcome(after)
    return MoveRecord(
        move_number=sum(v != ' ' for row in after for v in row),
        revision_before=revision, revision=revision + 1, player=player,
        agent_type=config['type'], agent_depth=config.get('depth'), column=column,
        board_before=tuple(tuple(row) for row in before),
        board_after=tuple(tuple(row) for row in after),
        outcome_status=result['status'], winner=result['winner'], search_stats=search_stats,
        alphazero=alphazero)
