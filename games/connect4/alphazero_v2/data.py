"""Immutable v2 examples, completed-game labels, reflection and generation replay.

Torch-free. No tactical proofs, value anchoring or diagnostic fixtures: value
targets are only the actual completed-game outcome for the pre-move actor.
"""
from dataclasses import dataclass, replace
import hashlib
import json
import math
import random

from ..connect4 import Connect4
from .config import ENCODING, POLICY_TARGET, VALUE_TARGET


def pre_move_ply(game):
    """Zero-based pre-move occupancy: pieces already on the board."""
    return sum(cell != " " for row in game.board for cell in row)


def visit_target(visits):
    """Training policy target pi(a) = N(a) / sum N, independent of any temperature."""
    counts = tuple(visits)
    if (len(counts) != 7 or any(type(v) is not int or v < 0 for v in counts) or sum(counts) < 1):
        raise ValueError("Expected seven nonnegative integer visits with positive total")
    total = sum(counts)
    return tuple(v / total for v in counts)


def seeded_rng(domain, seed):
    """Domain-separated random.Random so streams never share state."""
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    return random.Random(int(hashlib.sha256(f"{domain}:{seed}".encode()).hexdigest(), 16))


def encode_board(game):
    """Torch-free current-player encoding, identical to neural_mcts.encode_current_player."""
    own = game.piece
    if game.current_player not in (0, 1) or own != ("X" if game.current_player == 0 else "O"):
        raise ValueError("current_player and piece disagree")
    return tuple(tuple(0 if c == " " else 1 if c == own else -1 for c in row) for row in game.board)


def outcome_for_actor(winner, actor):
    """+1 if the completed-game winner is the pre-move actor, 0 draw, -1 otherwise."""
    if type(winner) is not int or winner not in (-1, 0, 1):
        raise ValueError("Winner must be X=0, O=1 or draw=-1")
    if type(actor) is not int or actor not in (0, 1):
        raise ValueError("Actor must be X=0 or O=1")
    return 0 if winner == -1 else (1 if winner == actor else -1)


@dataclass(frozen=True)
class V2Example:
    """One searched pre-move position. ``outcome`` is None until the game completes.

    The policy target is always normalized root visits; ``action_temperature``
    and ``action`` record execution only and never alter the target.
    """
    observation: tuple[tuple[int, ...], ...]
    actor: int
    ply: int
    visits: tuple[int, ...]
    action: int
    action_temperature: float
    outcome: int | None = None

    def __post_init__(self):
        rows = tuple(tuple(row) for row in self.observation)
        if len(rows) != 6 or any(len(row) != 7 for row in rows) or any(
                type(c) is not int or c not in (-1, 0, 1) for row in rows for c in row):
            raise ValueError("Observation must be a canonical 6x7 board of -1/0/+1 ints")
        if type(self.actor) is not int or self.actor not in (0, 1):
            raise ValueError("Actor must be X=0 or O=1")
        own = sum(c == 1 for row in rows for c in row)
        other = sum(c == -1 for row in rows for c in row)
        if type(self.ply) is not int or self.ply != own + other or other - own != self.actor:
            raise ValueError("ply/actor disagree with the observation's piece counts")
        visits = tuple(self.visits)
        target = visit_target(visits)
        if any(target[a] > 0 for a in range(7) if rows[0][a] != 0):
            raise ValueError("Visit target assigns mass to a full column")
        if type(self.action) is not int or not 0 <= self.action < 7 or visits[self.action] < 1:
            raise ValueError("Action must be a visited legal column")
        if (isinstance(self.action_temperature, bool) or type(self.action_temperature) not in (int, float)
                or not math.isfinite(self.action_temperature) or self.action_temperature < 0):
            raise ValueError("action_temperature must be finite and nonnegative")
        if self.outcome is not None and (type(self.outcome) is not int or self.outcome not in (-1, 0, 1)):
            raise ValueError("Outcome must be -1, 0, +1 or pending None")
        object.__setattr__(self, "observation", rows)
        object.__setattr__(self, "visits", visits)
        object.__setattr__(self, "action_temperature", float(self.action_temperature))

    @property
    def policy_target(self):
        return visit_target(self.visits)

    def record(self):
        return dict(observation=self.observation, actor=self.actor, ply=self.ply, visits=self.visits,
                    action=self.action, action_temperature=self.action_temperature, outcome=self.outcome,
                    encoding=ENCODING, policy_target=POLICY_TARGET, value_target=VALUE_TARGET)


def capture_example(game, decision):
    """Capture the searched pre-move state; call BEFORE applying decision.action.move."""
    if game.is_game_over():
        raise ValueError("Training observation must be pre-move and nonterminal")
    if decision.search.root.game_state.__dict__ != game.__dict__ or decision.ply != pre_move_ply(game):
        raise ValueError("Decision does not belong to this pre-move state")
    return V2Example(encode_board(game), game.current_player, decision.ply, decision.search.visits,
                     decision.action.move, decision.action.temperature)


@dataclass(frozen=True)
class CompletedGame:
    generation: int
    index: int
    moves: tuple[int, ...]
    winner: int
    examples: tuple[V2Example, ...]


def finalize_game(generation, index, moves, pending):
    """Replay the full history from empty, verify every capture, label by actual outcome.

    Rejects unfinished games, moves after termination and any pre-move mismatch.
    """
    moves, pending = tuple(moves), tuple(pending)
    if not moves or len(moves) != len(pending):
        raise ValueError("Every played move needs exactly one pre-move example")
    game = Connect4()
    for move, example in zip(moves, pending):
        if game.is_game_over():
            raise ValueError("History continues after termination")
        if not isinstance(example, V2Example) or example.outcome is not None:
            raise ValueError("Expected pending V2Examples")
        if (example.observation != encode_board(game) or example.actor != game.current_player
                or example.ply != pre_move_ply(game) or example.action != move):
            raise ValueError("Example does not match the replayed pre-move state/action")
        if not game.make_move(move):
            raise ValueError("History replay rejected a move")
    if not game.is_game_over():
        raise ValueError("Outcomes require a completed game")
    winner = game.check_winner()
    labeled = tuple(replace(e, outcome=outcome_for_actor(winner, e.actor)) for e in pending)
    return CompletedGame(generation, index, moves, winner, labeled)


def reflect_example(example):
    """Horizontal mirror: reverse observation columns, visits and action; keep actor/ply/z."""
    if not isinstance(example, V2Example) or example.outcome is None:
        raise ValueError("Reflection requires a completed V2Example")
    return replace(example, observation=tuple(row[::-1] for row in example.observation),
                   visits=example.visits[::-1], action=6 - example.action)


class ReflectionAugmenter:
    """Independently reflect each sampled example with probability p (own RNG stream)."""
    domain = "connect4-alphazero-v2-horizontal-reflection"

    def __init__(self, probability=0.5, *, seed):
        if (isinstance(probability, bool) or type(probability) not in (int, float)
                or not math.isfinite(probability) or not 0 <= probability <= 1):
            raise ValueError("Reflection probability must be finite in [0,1]")
        self.probability = float(probability)
        self.rng = seeded_rng(self.domain, seed)
        self.reflected = self.unreflected = 0

    def batch(self, examples):
        flags = [self.rng.random() < self.probability for _ in examples]
        self.reflected += sum(flags)
        self.unreflected += len(flags) - sum(flags)
        return [reflect_example(e) if flag else e for e, flag in zip(examples, flags)], flags


class GenerationReplay:
    """Complete finalized generations only; whole oldest generations expire.

    Positions are addressed as (generation, game index, ply). Sampling is
    uniform over every position in the retained window, without replacement
    within a batch.
    """

    def __init__(self, window_generations=8, max_games=2048):
        for name, value in (("window_generations", window_generations), ("max_games", max_games)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.window_generations, self.max_games = window_generations, max_games
        self._generations = []  # [(generation id, tuple[CompletedGame])], oldest first
        self._index = ()

    def add_generation(self, generation, games):
        """Append one complete generation; return the generation ids evicted."""
        games = tuple(games)
        if type(generation) is not int or generation < 1:
            raise ValueError("Generation ids must be positive integers")
        if self._generations and generation <= self._generations[-1][0]:
            raise ValueError("Generation ids must strictly increase")
        if not games or any(not isinstance(g, CompletedGame) or g.generation != generation
                            or any(e.outcome is None for e in g.examples) or not g.examples for g in games):
            raise ValueError("Replay accepts only nonempty completed games of this generation")
        if [g.index for g in games] != list(range(len(games))):
            raise ValueError("Game indices must be 0..n-1 in collection order")
        retained = (self._generations + [(generation, games)])[-self.window_generations:]
        if sum(len(gen_games) for _, gen_games in retained) > self.max_games:
            raise ValueError("Replay window would exceed max_games")
        evicted = [g for g, _ in self._generations if g not in {r for r, _ in retained}]
        self._generations = retained
        self._index = tuple((gi, game.index, ply) for gi, (_, gen_games) in enumerate(self._generations)
                            for game in gen_games for ply in range(len(game.examples)))
        return tuple(evicted)

    @property
    def generations(self):
        return tuple(g for g, _ in self._generations)

    @property
    def games(self):
        return sum(len(games) for _, games in self._generations)

    def __len__(self):
        return len(self._index)

    def counts(self):
        return {g: dict(games=len(games), positions=sum(len(game.examples) for game in games))
                for g, games in self._generations}

    def address(self, position):
        gi, game_index, ply = self._index[position]
        return self._generations[gi][0], game_index, ply

    def example(self, position):
        gi, game_index, ply = self._index[position]
        return self._generations[gi][1][game_index].examples[ply]

    def sample(self, batch_size, rng):
        """Return (positions, examples): batch_size distinct uniform positions."""
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if batch_size > len(self):
            raise ValueError("Replay holds fewer positions than one batch")
        positions = rng.sample(range(len(self)), batch_size)
        return positions, [self.example(p) for p in positions]

    def iter_games(self):
        for _, games in self._generations:
            yield from games

    def digest(self):
        """Content identity over every retained game and example, in order."""
        digest = hashlib.sha256()
        for game in self.iter_games():
            digest.update(json.dumps(dict(generation=game.generation, index=game.index, moves=game.moves,
                                          winner=game.winner, examples=[e.record() for e in game.examples]),
                                     sort_keys=True, allow_nan=False).encode())
        return digest.hexdigest()
