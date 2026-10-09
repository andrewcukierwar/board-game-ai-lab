"""Small reproducible solver experiment; never canonical benchmark/training data.

Run from repo root:
  PYTHONPATH=tests .venv/bin/python -m victor_validation.functional_experiment
Writes no files; redirect stdout to a new report artifact if desired.
"""
from collections import Counter
from dataclasses import replace
import json
import platform
from random import Random
from statistics import median
from time import monotonic

from games.connect4.victor import Position, SearchBudget, SolverBudget, select_move, solve_exact
from games.connect4.victor.coverage import search_covering_set
from games.connect4.victor.execution import NineRulePolicy
from games.connect4.victor.nine_rules import search_nine_rule_cover
from games.connect4.victor.rules import RuleName
from games.connect4.victor.cli import play_game
from .exact_oracle import has_four, replay, solve
from .reference_game import exhaustive_value
from .nine_rule_reference import DIAGRAMS


def sample(seed, remaining, count, *, quiet=False):
    """Reachable nonterminal positions, biased to avoid terminal random moves.

    This is an exposure sampler, not a distribution of naturally played games.
    Both turns are selected through odd/even remaining counts.
    """
    rng = Random(seed)
    for _ in range(count):
        while True:
            p, history = replay(()), ()
            while p.remaining > remaining:
                choices = [c for c in p.legal_columns if not p.drop(c).terminal]
                if not choices:
                    break
                c = rng.choice(choices)
                p, history = p.drop(c), history + (c,)
            immediate = any(has_four(bits | (1 << (7*c+h))) for bits in (p.white,p.black)
                            for c,h in enumerate(p.heights) if h<6)
            if p.remaining == remaining and (not quiet or not immediate):
                yield history, p
                break


def main():
    start = monotonic()
    budget = SearchBudget(nodes=100_000, seconds=0.25, max_remaining=22,
                          table_entries=100_000)
    rows, contradictions, policy_status, rule_counts = [], [], Counter(), Counter()
    three_rules = {RuleName.CLAIMEVEN, RuleName.BASEINVERSE, RuleName.VERTICAL}
    inventories = [(n,20,False) for n in (4,6,8,9,10,12,14,16)]
    inventories += [(n,6,True) for n in (10,11,14,18,22)]
    for remaining,count,quiet in inventories:
        for history, p in sample(1988 + remaining + 1000*quiet, remaining, count, quiet=quiet):
            position = Position.from_board(p.board, p.turn)
            exact = solve_exact(position, budget)
            oracle = solve(p,max_remaining=10,position_budget=1_000_000) if remaining<=10 else None
            if oracle and exact.status == 'exact':
                assert oracle.status == 'exact'
                assert exact.value == oracle.mover_value
                assert dict(exact.move_values) == dict(oracle.move_values)
            if remaining == 4:
                assert exact.value == exhaustive_value(p.board,p.turn)
            cover_start = monotonic()
            cover = search_nine_rule_cover(position,node_budget=10_000) if p.turn == 0 else None
            cover_seconds = monotonic() - cover_start
            audit = None
            if cover and cover.witness:
                rule_counts.update(e.candidate.rule.value for e in cover.witness.evidence)
                audit = NineRulePolicy(cover.witness).audit(replace(budget,max_remaining=12))
                policy_status[audit.status] += 1
                if audit.status == 'refuted':
                    contradictions.append(dict(moves=history, continuation=audit.counterexample,
                                               detail=audit.detail))
                if exact.status == 'exact':
                    assert exact.value <= 0
            # Use the actual decision API on ALL positions. Matching exact
            # decisions must be optimal under independently enumerated moves.
            decision = select_move(p.board,p.turn,SolverBudget(exact=budget,
                strategic_children=2,white_contexts=4,
                policy_audit=replace(budget,max_remaining=12)))
            assert decision.move in p.legal_columns
            if oracle and decision.exact_value is not None:
                assert dict(oracle.move_values)[decision.move] == oracle.mover_value
            rows.append(dict(moves=history,remaining=remaining,turn=p.turn,quiet=quiet,
                exact_status=exact.status,value=exact.value,nodes=exact.nodes,
                seconds=exact.elapsed,table_entries=exact.table_entries,hits=exact.cache_hits,
                cover=None if not cover else cover.status.value,cover_seconds=cover_seconds,
                composite_cover=bool(cover and cover.witness and any(
                    e.candidate.rule not in three_rules for e in cover.witness.evidence)),
                policy=None if audit is None else audit.status,
                move=decision.move,move_kind=decision.move_kind,
                justified=decision.justified_move,oracle_checked=oracle is not None))
    game_budget = SolverBudget(exact=SearchBudget(nodes=100_000,seconds=0.5,max_remaining=12),
                               strategic_children=3,white_contexts=4)
    games = []
    for i,(white,black) in enumerate((('victor','random'),('random','victor'),
            ('victor','negamax:4'),('negamax:4','victor'),
            ('victor','mcts:32'),('mcts:32','victor'),('victor','victor'))):
        games.append(play_game(white,black,seed=20261008+i,budget=game_budget))
    # A thesis starting position exercises actual retained composite responses.
    games.append(play_game('negamax:4','victor',moves=DIAGRAMS['6.10'],
                           seed=1988,budget=game_budget))
    summary = dict(positions=len(rows),exact=sum(r['exact_status']=='exact' for r in rows),
        oracle_checked=sum(r['oracle_checked'] for r in rows),oracle_mismatches=0,
        matrix_oracle_checked=20,
        black_context_positions=sum(r['turn']==0 for r in rows),
        covers=sum(r['cover']=='compatible_covering_set_found' for r in rows),
        composite_covers=sum(r['composite_cover'] for r in rows),
        justified_moves=sum(r['justified'] for r in rows),
        move_kinds=dict(Counter(r['move_kind'] for r in rows)),
        policy_statuses=dict(policy_status),selected_rules=dict(rule_counts),
        exact_statuses=dict(Counter(r['exact_status'] for r in rows)),
        exact_nodes_total=sum(r['nodes'] for r in rows),
        exact_seconds_total=sum(r['seconds'] for r in rows),
        exact_seconds_median=median(r['seconds'] for r in rows),
        exact_seconds_max=max(r['seconds'] for r in rows),
        exact_nodes_max=max(r['nodes'] for r in rows),
        table_entries_max=max(r['table_entries'] for r in rows),
        cover_seconds_total=sum(r['cover_seconds'] for r in rows),
        cover_seconds_median=median(r['cover_seconds'] for r in rows if r['cover'] is not None))
    print(json.dumps(dict(schema='victor-functional-experiment-v1',python=platform.python_version(),
        platform=platform.platform(),settings=dict(exact_nodes=100_000,exact_seconds=0.25,
            exact_remaining=22,positions_per_remaining=20,seed_formula='1988 + remaining + 1000*quiet',
            remaining=[4,6,8,9,10,12,14,16],game_seed=20261008,
            quiet_remaining=[10,11,14,18,22],quiet_positions_per_remaining=6,
            game_nodes=100_000,game_seconds=0.5,game_remaining=12,
            game_strategic_children=3,fallback_depth=4),
        summary=summary,positions=rows,games=games,contradictions=contradictions,
        total_seconds=monotonic()-start),indent=2))


if __name__ == '__main__':
    main()
