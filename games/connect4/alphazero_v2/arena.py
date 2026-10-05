"""Paired, side-swapped arena runner and model-free opponents (torch-free).

For every opening row the agent under test plays two games from the identical
position, owning X in game 0 and O in game 1; side swapping changes which agent
owns each color, not whose turn the board specifies. Every agent call gets its
own RNG derived from (namespace, opening id, game index, role) so results do
not depend on execution order. Decisions are validated through the engine;
any illegal move aborts the arena as a correctness failure.

Cooperative stopping: ``check()`` runs before every game and every move. A
stop raised between games ends the arena incomplete; a stop raised mid-game
records that game as abandoned (no outcome, never scored).
"""
import time

from ..agents.mcts_agent import MCTSAgent
from ..connect4 import Connect4
from .data import seeded_rng
from .oracle import engine_position
from .reference_negamax import ReferenceNegamaxAgent, VERSION as NEGAMAX_VERSION

ARENA_DOMAIN = "connect4-alphazero-v2-arena"


class StopEvaluation(Exception):
    """Raised by a check callable to stop an arena cooperatively."""


class ArenaCorrectnessError(RuntimeError):
    pass


class RandomOpponent:
    name = "random"

    def describe(self):
        return dict(name=self.name, rule="uniform over legal columns with the per-game RNG")

    def choose(self, game, rng):
        return rng.choice(sorted(game.get_valid_moves())), {}


class NegamaxOpponent:
    def __init__(self, depth):
        self.depth = depth
        self.name = f"negamax{depth}"

    def describe(self):
        return dict(name=self.name, version=NEGAMAX_VERSION, depth=self.depth, ties="seeded uniform")

    def choose(self, game, rng):
        agent = ReferenceNegamaxAgent(self.depth, rng=rng)
        move = agent.choose_move(game)
        return move, dict(root_scores={str(k): v for k, v in sorted(agent.last_scores.items())})


class GuardedUCTOpponent:
    """Existing standalone UCT with random rollouts; its root tactical guards are always on."""

    def __init__(self, simulations=800):
        self.simulations = simulations
        self.name = f"guarded_uct_{simulations}"

    def describe(self):
        return dict(name=self.name, simulations=self.simulations, implementation="agents.mcts_agent.MCTSAgent",
                    root_tactical_guards="always on: immediate win, then exclude moves allowing an immediate reply win")

    def choose(self, game, rng):
        return MCTSAgent(self.simulations, rng=rng).choose_move(game), {}


def agent_rng(namespace, opening_id, game_index, role, seed):
    return seeded_rng(f"{ARENA_DOMAIN}:{namespace}:{opening_id}:{game_index}:{role}", seed)


def play_game(agents, opening, rngs, *, check=None):
    """Play from ``opening`` with agents[0] owning X and agents[1] owning O.

    Returns a record; ``abandoned`` is True when check() stopped it mid-game.
    """
    game = engine_position(list(opening))
    moves, decisions = list(opening), []
    started = time.perf_counter()
    while not game.is_game_over():
        if check is not None:
            try:
                check()
            except StopEvaluation as stop:
                return dict(moves=moves, abandoned=True, stop_reason=str(stop), decisions=decisions,
                            seconds=time.perf_counter() - started)
        owner = game.current_player
        before = time.perf_counter()
        move, info = agents[owner].choose(Connect4(game.board, game.current_player), rngs[owner])
        elapsed = time.perf_counter() - before
        if type(move) is not int or move not in game.get_valid_moves() or not game.make_move(move):
            raise ArenaCorrectnessError(f"{agents[owner].name} returned illegal move {move!r} after {moves}")
        moves.append(move)
        decisions.append(dict(color=owner, move=move, seconds=elapsed, **info))
    return dict(moves=moves, abandoned=False, winner=game.check_winner(), decisions=decisions,
                seconds=time.perf_counter() - started)


PAIRED_GAMES = ((0, 0), (1, 1))  # (game index, agent color): agent owns X, then O


def paired_game(agent, opponent, row, game_index, *, namespace, seed, check=None, keep_decisions=True):
    """One game of an opening pair; its RNGs depend only on (namespace, row, game, role, seed).

    Independent of execution order, so a durable campaign can resume an arena at
    game granularity and obtain exactly the record ``run_paired_arena`` produces.
    """
    agent_color = dict(PAIRED_GAMES)[game_index]
    agents = (agent, opponent) if agent_color == 0 else (opponent, agent)
    roles = ("agent", "opponent") if agent_color == 0 else ("opponent", "agent")
    rngs = tuple(agent_rng(namespace, row["id"], game_index, role, seed) for role in roles)
    result = play_game(agents, row["moves"], rngs, check=check)
    record = dict(opening_id=row["id"], family=row["family"], stratum=row["stratum"],
                  prefix_length=len(row["moves"]), game_index=game_index, agent_color=agent_color,
                  agent=agent.name, opponent=opponent.name, **result)
    if not keep_decisions:
        record["decisions"] = [dict(color=d["color"], move=d["move"], seconds=d["seconds"])
                               for d in record["decisions"]]
    if not result["abandoned"]:
        winner = result["winner"]
        record["result"] = "draw" if winner == -1 else ("win" if winner == agent_color else "loss")
    return record


def run_paired_arena(agent, opponent, openings, *, namespace, seed, check=None, on_game=None,
                     keep_decisions=True):
    """Play every opening row twice (agent as X, then as O). Returns (records, status)."""
    records = []
    for row in openings:
        for game_index, _ in PAIRED_GAMES:
            if check is not None:
                try:
                    check()
                except StopEvaluation as stop:
                    return records, dict(complete=False, stop_reason=str(stop))
            record = paired_game(agent, opponent, row, game_index, namespace=namespace, seed=seed, check=check,
                                 keep_decisions=keep_decisions)
            records.append(record)
            if on_game is not None:
                on_game(record)
            if record["abandoned"]:
                return records, dict(complete=False, stop_reason=record["stop_reason"])
    return records, dict(complete=True, stop_reason=None)


def scored(records):
    """Completed (non-abandoned) game records only."""
    return [r for r in records if not r["abandoned"]]
