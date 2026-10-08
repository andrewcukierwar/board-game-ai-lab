"""Exploratory implication checks; NO strategic certificate acceptance.

The strategy reference model enumerates all allowed spare moves. It does not
invent a deterministic Black policy or reuse the exact oracle's move evaluator.
"""
from dataclasses import asdict, dataclass
from random import Random

from games.connect4.victor import (
    Position, SearchStatus, VerificationStatus, search_covering_set, verify_coverage_witness,
)
from .exact_oracle import MAX_REMAINING, replay, solve, verify_replay
from . import reference_game as matrix


def generated_histories(*, seed=1988, attempts=64, plies=34):
    """Bounded no-win-biased legal replay sampler; not an exhaustive/uniform search."""
    if type(attempts) is not int or not 0 <= attempts <= 256:
        raise ValueError('attempts must be 0..256')
    if type(plies) is not int or plies not in (34, 36, 38):
        raise ValueError('plies must be 34, 36 or 38')
    rng = Random(seed)
    histories = []
    seen = set()
    for _ in range(attempts):
        p, moves = replay(()), []
        for _ in range(plies):
            choices = tuple(c for c in p.legal_columns if not p.drop(c).terminal)
            if not choices:
                break
            c = rng.choice(choices)
            p = p.drop(c)
            moves.append(c)
        if len(moves) == plies and p.board not in seen:
            seen.add(p.board)
            histories.append(tuple(moves))
    return tuple(histories)


@dataclass(frozen=True)
class ExecutionResult:
    status: str
    visited_positions: int
    white_edges: int
    forced_reply_edges: int
    spare_reply_edges: int
    proactive_pair_edges: int
    suffix: tuple[int, ...] = ()
    observation: str = ''


def examine_strategy(position, witness, *, max_remaining=8, position_budget=100_000):
    """All White moves AND all permitted Black replies for a fixed fragment set.

    BI: reply at the other endpoint. VE/CL: reply above White's lower. Already
    Black-blocked BI/VE instances are retired. Spare Black moves may take any
    legal cell except a still-active CL lower. §7.1 supplies that prohibition;
    §7.2 permits proactive BI/VE occupation. This set-valued completion is our
    reconstructed reference model, not an algorithm explicitly given by Allis.
    Budget counts uncached states, including terminals; cutoff is unknown.
    """
    if type(max_remaining) is not int or not 0 <= max_remaining <= MAX_REMAINING:
        raise ValueError('max_remaining must be integer 0..10')
    if type(position_budget) is not int or not 0 <= position_budget <= 1_000_000:
        raise ValueError('position_budget must be integer 0..1000000')
    if position.turn != 0 or position.terminal:
        raise ValueError('requires nonterminal White-to-move replay')
    declared = Position.from_board(position.board, position.turn)
    if verify_coverage_witness(declared, witness).status != VerificationStatus.VERIFIED_UNCERTIFIED:
        raise ValueError('requires independently verified coverage witness')
    rules = tuple((e.candidate.rule.value,
                   tuple((s.row_index, s.column) for s in e.candidate.squares))
                  for e in witness.evidence)
    visited = white = forced = spare = proactive = 0
    seen = set()

    def result(status, suffix=(), observation=''):
        return ExecutionResult(status, visited, white, forced, spare, proactive,
                               suffix, observation)

    if position.remaining > max_remaining:
        return result('unknown_remaining_cap')

    def active(board):
        return tuple((kind, cells) for kind, cells in rules
                     if not (board[cells[1][0]][cells[1][1]] == 'O' if kind == 'claimeven'
                             else any(board[r][c] == 'O' for r, c in cells)))

    def visit(board, turn, required, suffix):
        nonlocal visited, white, forced, spare, proactive
        key = board, turn, required
        if key in seen:
            return None
        if visited >= position_budget:
            return result('unknown_position_budget')
        visited += 1
        seen.add(key)
        won = matrix.winner(board)
        if won == 0:
            return result('potential_counterexample', suffix, 'White won in reference execution')
        columns = matrix.legal_moves(board)
        if won == 1 or not columns:
            return None
        pending = active(board)
        if turn == 0:
            if any(any(board[r][c] != ' ' for r, c in cells) for _, cells in pending):
                return result('obligation_failure', suffix, 'unresolved pair is not empty')
            for c in columns:
                square = matrix.landing(board, c)
                obligations = []
                for kind, cells in pending:
                    if square in cells:
                        if kind == 'baseinverse':
                            obligations.append(cells[1] if square == cells[0] else cells[0])
                        elif square == cells[0]:
                            obligations.append(cells[1])
                        else:
                            return result('obligation_failure', suffix + (c,), 'White reached upper first')
                if len(obligations) > 1:
                    return result('obligation_failure', suffix + (c,), 'conflicting responses')
                white += 1
                failed = visit(matrix.drop(board, 0, c), 1,
                               obligations[0] if obligations else None, suffix + (c,))
                if failed:
                    return failed
        else:
            if required is not None:
                c = required[1]
                if c not in columns or matrix.landing(board, c) != required:
                    return result('obligation_failure', suffix, 'mandatory response is not playable')
                choices = (c,)
            else:
                forbidden = {cells[0] for kind, cells in pending if kind == 'claimeven'}
                choices = tuple(c for c in columns if matrix.landing(board, c) not in forbidden)
                if not choices:
                    return result('obligation_failure', suffix, 'no legal spare move preserving CL')
            for c in choices:
                if required is not None:
                    forced += 1
                else:
                    spare += 1
                    square = matrix.landing(board, c)
                    proactive += any(kind != 'claimeven' and square in cells for kind, cells in pending)
                failed = visit(matrix.drop(board, 1, c), 0, None, suffix + (c,))
                if failed:
                    return failed
        return None

    failed = visit(position.board, 0, None, ())
    return failed or result('bounded_execution_checked_outcome_uncertified')


def compare_history(moves, *, cover_budget=1000, oracle_budget=100_000,
                    execution_budget=100_000, max_remaining=8):
    """Recompute reachability, exact outcome, coverage and optional execution."""
    p = replay(moves)
    verify_replay(p.board, p.turn, moves)
    if p.turn != 0 or p.terminal:
        raise ValueError('comparison requires nonterminal White-to-move history')
    exact = solve(p, max_remaining=max_remaining, position_budget=oracle_budget)
    declared = Position.from_board(p.board, p.turn)
    cover = search_covering_set(declared, node_budget=cover_budget)
    verification = None
    execution = None
    if cover.status == SearchStatus.FOUND:
        verification = verify_coverage_witness(declared, cover.witness).status.value
        if verification != VerificationStatus.VERIFIED_UNCERTIFIED.value:
            raise AssertionError('search emitted an invalid finite coverage witness')
        execution = asdict(examine_strategy(p, cover.witness, max_remaining=max_remaining,
                                           position_budget=execution_budget))
    suspicious = cover.status == SearchStatus.FOUND and exact.value_for(0) == 1
    return {
        'moves': list(moves), 'board': [''.join(row) for row in p.board], 'turn': p.turn,
        'remaining': p.remaining, 'oracle_status': exact.status,
        'white_value': exact.value_for(0), 'black_value': exact.value_for(1),
        'move_values_for_white': [list(pair) for pair in exact.move_values],
        'oracle_positions': exact.visited_positions, 'oracle_cache_hits': exact.cache_hits,
        'cover_status': cover.status.value, 'cover_nodes': cover.expanded_nodes,
        'cover_budget': cover_budget, 'verification': verification,
        'selected_rules': [] if cover.witness is None else [
            {'rule': e.candidate.rule.value, 'squares': [s.name for s in e.candidate.squares],
             'covered_groups': [[s.name for s in g.squares] for g in e.conditional_solved_groups]}
            for e in cover.witness.evidence],
        'execution': execution, 'potential_counterexample': suspicious,
        'interpretation': 'exploratory_implication_check_outcome_uncertified',
    }
