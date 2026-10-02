"""Immutable move records; built before committing a session mutation."""
from dataclasses import dataclass

from .analysis import outcome


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

    def to_dict(self):
        return {'move_number': self.move_number, 'revision_before': self.revision_before,
                'revision': self.revision, 'player': self.player,
                'agent': {'type': self.agent_type, **({'depth': self.agent_depth}
                                                    if self.agent_depth is not None else {})},
                'column': self.column,
                'board_before': [list(row) for row in self.board_before],
                'board_after': [list(row) for row in self.board_after],
                'outcome': {'status': self.outcome_status, 'winner': self.winner}}


def record_move(before, after, player, config, column, revision):
    result = outcome(after)
    return MoveRecord(
        move_number=sum(v != ' ' for row in after for v in row),
        revision_before=revision, revision=revision + 1, player=player,
        agent_type=config['type'], agent_depth=config.get('depth'), column=column,
        board_before=tuple(tuple(row) for row in before),
        board_after=tuple(tuple(row) for row in after),
        outcome_status=result['status'], winner=result['winner'])
