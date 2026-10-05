"""Model-blind frozen evaluation packages: tactical, solved and opening banks (torch-free).

Each package kind has a *development* split (usable for champion selection
and debugging) and a *sealed* split (used once, after champion selection, by
``evaluation.final`` only). Splits are disjoint by reflection family; all
previously inspected fixtures, probes, blind suites and historical arena
openings (and their reflections) are excluded. No model is consulted: positions
come from seeded random / reference-Negamax play and labels come from exact
engine replay plus the torch-free oracle.

Usage (one-time freeze; outputs are committed with content hashes):

    python -m games.connect4.alphazero_v2.packages build --output-dir DIR
    python -m games.connect4.alphazero_v2.packages verify --directory DIR [--resolve]
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys
import time

from ..connect4 import Connect4
from .oracle import (HEIGHT, WIDTH, BitboardSolver, EXHAUSTIVE_VERSION, SOLVER_VERSION, SolverBudgetExceeded,
                     board_key, engine_position, exhaustive_action_values, family_key, legal_moves_scan,
                     line_directions, mirror_moves, safe_moves_scan, tactical_label, winning_moves_scan,
                     _grid, _opponent_threats)
from .reference_negamax import ReferenceNegamaxAgent

PACKAGE_FORMAT = "connect4-alphazero-v2-evaluation-package"
PACKAGE_VERSION = 1
FROZEN_DIRECTORY = Path(__file__).with_name("frozen")
REPO_ROOT = Path(__file__).resolve().parents[3]
BUILD_SEED = 4_303_002
SPLITS = ("sealed", "development")  # sealed quotas are filled first from the shuffled pool
TACTICAL_QUOTAS = dict(sealed=100, development=25)       # bases per (category, actor)
SOLVED_QUOTAS = dict(sealed=50, development=25)          # bases per (value, actor)
OPENING_PREFIX_BASES = dict(sealed=40, development=40)   # each base adds its mirror
EMPTY_BOARD_PAIRS = 20
TACTICAL_STAGES = (("early", 6, 13), ("middle", 14, 23), ("late", 24, 41))
SOLVED_STAGES = (("s12_19", 12, 19), ("s20_27", 20, 27), ("s28_41", 28, 41))
DIRECTIONS = ("horizontal", "vertical", "diagonal_up", "diagonal_down")
SOLVER_NODE_BUDGET = 3_000_000
OVERSAMPLE = 1.5                    # candidates collected per quota slot, so cells can be stratified
EXHAUSTIVE_MIN_PIECES = 24          # method-B cross-check is attempted at or beyond this occupancy
EXHAUSTIVE_STATE_BUDGET = 2_000_000

# Every previously inspected position source (Phase 4C/4D). Local experiment
# files are gitignored, so their content hashes and the derived family keys are
# committed in frozen/exclusions.json.
EXCLUSION_SOURCES = (
    ("dqn_diagnostic", "games/connect4/dqn/diagnostic_positions.json", "positions"),
    ("dqn_validation", "games/connect4/dqn/validation_positions.json", "positions"),
    ("dqn_replication", "games/connect4/dqn/replication_positions.json", "positions"),
    ("phase4d2c_blind", "experiment-output/phase4d2c-neural-symmetry-seed42-20261005/preflight/blind-positions.json",
     "positions"),
    ("phase4d2d_blind", "experiment-output/phase4d2d-neural-root-noise-preflight-20261005/blind-positions.json",
     "positions"),
    ("phase4d2f_exact_value_blind",
     "experiment-output/phase4d2f-neural-value-anchoring-preflight-20261005/exact-value-blind.json", "positions"),
)
# neural_self_play._BASE_POSITIONS (14 rows with mirrors) and neural_evaluation.OPENINGS,
# copied verbatim so building packages never imports the v1 torch modules.
ORIGINAL_PROBES = ([], [3], [3, 2, 4, 3, 2, 4, 1, 5], [3, 2, 4, 3, 2, 4, 1, 5, 1], [0, 1, 0, 1, 0, 2],
                   [0, 1, 0, 1, 2, 1, 2], [0, 1, 0, 1, 2, 1], [0, 1, 0, 1, 0], [0, 0, 0, 0, 0, 0])
HISTORICAL_OPENINGS = ((3, 2), (3, 4), (0, 3, 2), (6, 3, 4), (2, 4, 3, 2), (4, 2, 3, 4))


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def rows_sha256(rows):
    return hashlib.sha256(canonical_json(rows).encode()).hexdigest()


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stage_of(pieces, stages):
    for name, low, high in stages:
        if low <= pieces <= high:
            return name
    return None


# Exclusions -----------------------------------------------------------------------------

def collect_exclusions(root=REPO_ROOT):
    """Return (family keys, source records). Missing sources are an error, not a skip."""
    families, sources = set(), []
    for name, relative, field in EXCLUSION_SOURCES:
        path = Path(root) / relative
        if not path.is_file():
            raise FileNotFoundError(f"Exclusion source missing: {relative}")
        rows = json.loads(path.read_text())[field]
        keys = {family_key(list(row["moves"])) for row in rows}
        families |= keys
        sources.append(dict(name=name, path=relative, sha256=file_sha256(path), rows=len(rows), families=len(keys)))
    for name, histories in (("original_probes", ORIGINAL_PROBES), ("historical_openings", HISTORICAL_OPENINGS)):
        keys = set()
        for moves in histories:
            for variant in (list(moves), mirror_moves(moves)):
                keys.add(family_key(variant))
        families |= keys
        sources.append(dict(name=name, path=None, sha256=rows_sha256([list(m) for m in histories]),
                            rows=len(histories), families=len(keys)))
    return families, sources


# Model-blind position generator ------------------------------------------------------------

GENERATOR_STYLES = (("tactical", 0.4, (1, 2)), ("positional", 0.1, (2, 4)))


def sample_game(rng):
    """One seeded game in a seeded style: each move is uniform random with the style's
    probability, otherwise reference Negamax at one of the style's depths with seeded ties.
    "tactical" games are short and threat-rich; "positional" games are long and drawish."""
    _, random_probability, depths = rng.choice(GENERATOR_STYLES)
    game, moves = Connect4(), []
    agents = {depth: ReferenceNegamaxAgent(depth, rng=rng) for depth in depths}
    while not game.is_game_over():
        if rng.random() < random_probability:
            move = rng.choice(sorted(game.get_valid_moves()))
        else:
            move = agents[rng.choice(depths)].choose_move(game)
        if not game.make_move(move):
            raise RuntimeError("Generator produced an illegal move")
        moves.append(move)
    return moves


def is_nonterminal(moves):
    return not engine_position(moves).is_game_over()


def engine_tactics(moves):
    """Independent engine-successor labels: (winning columns, safe columns, opponent threat columns)."""
    game = engine_position(moves)
    mover = game.current_player

    def after(state, move):
        child = Connect4(state.board, state.current_player)
        if not child.make_move(move):
            raise RuntimeError("Engine rejected a legal move")
        return child
    legal = sorted(game.get_valid_moves())
    wins = [m for m in legal if after(game, m).check_winner() == mover]
    safe = []
    for move in legal:
        child = after(game, move)
        if move in wins or child.is_game_over():
            safe.append(move)
        elif not any(after(child, reply).check_winner() == 1 - mover for reply in child.get_valid_moves()):
            safe.append(move)
    # Opponent threats: columns where the opponent would complete four if it were to move now.
    swapped = Connect4(game.board, 1 - mover)
    threats = [m for m in legal if after(swapped, m).check_winner() == 1 - mover]
    return wins, safe, threats


def tactical_row(moves, category, action):
    """Fully labelled tactical row; both labelers must agree."""
    wins, safe, threats = engine_tactics(moves)
    if category == "unique_win":
        if wins != [action] or winning_moves_scan(moves) != [action]:
            raise ValueError("Tactical labelers disagree (win)")
        direction = sorted(line_directions(moves, action))[0]
    else:
        if wins or safe != [action] or safe_moves_scan(moves) != [action] or not threats:
            raise ValueError("Tactical labelers disagree (safe)")
        if sorted(threats) != sorted(_opponent_threats(moves)):
            raise ValueError("Threat labelers disagree")
        # A unique safe response is the single opponent threat column; label the blocked line.
        blocked = _opponent_directions(moves, action)
        direction = sorted(blocked)[0] if blocked else "indirect"
    return dict(moves=list(moves), actor=len(moves) % 2, ply=len(moves), category=category,
                expected_action=action, direction=direction, legal=sorted(legal_moves_scan(moves)),
                immediate_wins=wins, safe_actions=safe, opponent_threats=sorted(threats))


def _opponent_directions(moves, col):
    """Directions of the opponent four that a piece of the opponent at ``col`` would complete."""
    grid, heights = _grid(moves)
    row, player = heights[col], 1 - len(moves) % 2
    found = []
    for name, (dc, dr) in (("horizontal", (1, 0)), ("vertical", (0, 1)),
                           ("diagonal_up", (1, 1)), ("diagonal_down", (1, -1))):
        count = 1
        for sign in (1, -1):
            c, r = col + sign * dc, row + sign * dr
            while 0 <= c < WIDTH and 0 <= r < HEIGHT and grid[c][r] == player:
                count += 1
                c, r = c + sign * dc, r + sign * dr
        if count >= 4:
            found.append(name)
    return found


def mirror_tactical_row(row):
    mirrored = tactical_row(mirror_moves(row["moves"]), row["category"], 6 - row["expected_action"])
    if mirrored["direction"] not in (row["direction"], _flip_direction(row["direction"])):
        raise ValueError("Mirror changed the tactical direction")
    return mirrored


def _flip_direction(direction):
    return {"diagonal_up": "diagonal_down", "diagonal_down": "diagonal_up"}.get(direction, direction)


def solved_row(moves, solver):
    """Exact per-action outcomes (method A) with the state value cross-derived two ways."""
    actions = solver.action_values(moves)
    value = solver.value(moves)
    if value != max(actions.values()):
        raise ValueError("Solver state value disagrees with its action values")
    return dict(moves=list(moves), actor=len(moves) % 2, ply=len(moves), value=value,
                action_values={str(a): v for a, v in sorted(actions.items())},
                optimal_actions=sorted(a for a, v in actions.items() if v == value),
                losing_actions=sorted(a for a, v in actions.items() if v == -1),
                opponent_threat=bool(_opponent_threats(moves)),
                forced_block_loses=bool(_opponent_threats(moves)) and len(safe_moves_scan(moves)) == 1
                and actions[safe_moves_scan(moves)[0]] == -1,
                labels=dict(method_a=SOLVER_VERSION, method_b=None))


def solved_eligible(moves):
    """Nontrivial, decision-relevant solved candidates (value-specific rules applied after solving)."""
    return not winning_moves_scan(moves) and stage_of(len(moves), SOLVED_STAGES) is not None


def solved_value_acceptable(row):
    value, actions = row["value"], row["action_values"]
    if value == 1:
        return any(v != 1 for v in actions.values())          # some action fails to win
    if value == 0:
        return any(v == -1 for v in actions.values())         # some action loses
    # Loss: exclude losses already proved by a one-reply scan (every move allows an immediate win).
    return bool(safe_moves_scan(row["moves"]))


def cross_check_solved(row):
    """Method B (exhaustive) agreement where feasible; returns the method label or None."""
    if row["ply"] < EXHAUSTIVE_MIN_PIECES:
        return None
    try:
        values = exhaustive_action_values(row["moves"], EXHAUSTIVE_STATE_BUDGET)
    except SolverBudgetExceeded:
        return None
    if {str(a): v for a, v in sorted(values.items())} != row["action_values"]:
        raise ValueError(f"Independent solvers disagree on {row['moves']}")
    return EXHAUSTIVE_VERSION


def mirror_solved_row(row, solver):
    mirrored = solved_row(mirror_moves(row["moves"]), solver)
    expected = {str(6 - int(a)): v for a, v in row["action_values"].items()}
    if mirrored["action_values"] != dict(sorted(expected.items())) or mirrored["value"] != row["value"]:
        raise ValueError("Mirror changed exact values")
    mirrored["labels"] = dict(row["labels"])
    return mirrored


# Builders ---------------------------------------------------------------------------------

class Pool:
    """Seeded stream of candidate positions from generated games, deduplicated by family."""

    def __init__(self, seed, excluded):
        self.rng = random.Random(seed)
        self.seen = set(excluded)
        self.games = 0

    def candidates(self, per_game, accept):
        while True:
            moves = sample_game(self.rng)
            self.games += 1
            plies = list(range(len(moves)))
            self.rng.shuffle(plies)
            taken = 0
            for ply in plies:
                prefix = moves[:ply]
                if taken >= per_game:
                    break
                if board_key(prefix) == board_key(mirror_moves(prefix)):
                    continue  # self-symmetric boards cannot form a two-row mirror family
                key = family_key(prefix)
                if key in self.seen or not accept(prefix):
                    continue
                self.seen.add(key)
                taken += 1
                yield prefix


def build_tactical(excluded, seed, *, max_games=200_000, log=print):
    pool = Pool(seed, excluded)
    quotas = {(split, category, actor): TACTICAL_QUOTAS[split]
              for split in SPLITS for category in ("unique_win", "unique_safe") for actor in (0, 1)}
    # Direction x stage balance: per (category, actor) fill cells round-robin.
    buckets = defaultdict(list)
    chosen = defaultdict(list)
    examined = 0

    def accept(prefix):
        return len(prefix) >= 6 and is_nonterminal(prefix) and tactical_label(prefix)[0] is not None
    for prefix in pool.candidates(2, accept):
        examined += 1
        category, action = tactical_label(prefix)
        row = tactical_row(prefix, category, action)
        stage = stage_of(row["ply"], TACTICAL_STAGES)
        buckets[(category, row["actor"], row["direction"], stage)].append(row)
        if all(_cell_capacity(buckets, category, actor) >= OVERSAMPLE * sum(quotas[(s, category, actor)]
                                                                           for s in SPLITS)
               for category in ("unique_win", "unique_safe") for actor in (0, 1)):
            break
        if pool.games >= max_games:
            break
    for category in ("unique_win", "unique_safe"):
        for actor in (0, 1):
            cells = [(d, s) for d in DIRECTIONS + ("indirect",) for s, _, _ in TACTICAL_STAGES]
            for split in SPLITS:
                need = quotas[(split, category, actor)]
                while len(chosen[(split, category, actor)]) < need:
                    progressed = False
                    for direction, stage in cells:
                        bucket = buckets[(category, actor, direction, stage)]
                        if bucket and len(chosen[(split, category, actor)]) < need:
                            chosen[(split, category, actor)].append(bucket.pop(0))
                            progressed = True
                    if not progressed:
                        break
    log(f"tactical: games={pool.games} examined={examined}")
    return chosen, dict(games=pool.games, examined=examined)


def _cell_capacity(buckets, category, actor):
    return sum(len(v) for k, v in buckets.items() if k[0] == category and k[1] == actor)


def build_solved(excluded, seed, *, max_games=100_000, log=print, budget_seconds=None):
    pool = Pool(seed, excluded)
    solver = BitboardSolver(SOLVER_NODE_BUDGET)
    quotas = {(split, value, actor): SOLVED_QUOTAS[split]
              for split in SPLITS for value in (1, 0, -1) for actor in (0, 1)}
    buckets = defaultdict(list)
    chosen = defaultdict(list)
    examined = skipped_budget = rejected = 0
    started = time.monotonic()
    totals = {k: sum(quotas[(s, k[0], k[1])] for s in SPLITS) for k in {(v, a) for _, v, a in quotas}}
    for prefix in pool.candidates(1, lambda p: is_nonterminal(p) and solved_eligible(p)):
        examined += 1
        stage = stage_of(len(prefix), SOLVED_STAGES)
        # Skip work for cells already full.
        solver.table.clear()
        solver.nodes = 0
        try:
            row = solved_row(prefix, solver)
        except SolverBudgetExceeded:
            skipped_budget += 1
            continue
        if not solved_value_acceptable(row):
            rejected += 1
            continue
        key = (row["value"], row["actor"])
        if sum(len(buckets[(key[0], key[1], s)]) for s, _, _ in SOLVED_STAGES) >= OVERSAMPLE * totals[key]:
            continue  # enough candidates to stratify this cell
        buckets[(row["value"], row["actor"], stage)].append(row)
        if examined % 50 == 0:
            log(f"solved: examined={examined} games={pool.games} have=" + str(
                {k: sum(len(buckets[(k[0], k[1], s)]) for s, _, _ in SOLVED_STAGES) for k in totals}))
        if all(sum(len(buckets[(k[0], k[1], s)]) for s, _, _ in SOLVED_STAGES) >= OVERSAMPLE * totals[k]
               for k in totals):
            break
        if pool.games >= max_games or (budget_seconds and time.monotonic() - started > budget_seconds):
            break
    for (value, actor) in totals:
        for split in SPLITS:
            need = quotas[(split, value, actor)]
            while len(chosen[(split, value, actor)]) < need:
                progressed = False
                for stage, _, _ in SOLVED_STAGES:
                    bucket = buckets[(value, actor, stage)]
                    if bucket and len(chosen[(split, value, actor)]) < need:
                        chosen[(split, value, actor)].append(bucket.pop(0))
                        progressed = True
                if not progressed:
                    break
    log(f"solved: games={pool.games} examined={examined} skipped_budget={skipped_budget} rejected={rejected}")
    return chosen, dict(games=pool.games, examined=examined, skipped_budget=skipped_budget,
                        rejected_trivial=rejected), solver


def opening_acceptable(prefix):
    return (2 <= len(prefix) <= 8 and is_nonterminal(prefix) and not winning_moves_scan(prefix)
            and not _opponent_threats(prefix))


def build_openings(excluded, seed):
    rng = random.Random(seed)
    seen = set(excluded)
    chosen = {split: [] for split in SPLITS}
    lengths = {0: (2, 4, 6, 8), 1: (3, 5, 7)}
    attempts = 0
    for split in SPLITS:
        for parity in (0, 1):
            target = OPENING_PREFIX_BASES[split] // 2
            count = 0
            while count < target:
                attempts += 1
                length = lengths[parity][count % len(lengths[parity])]
                prefix = []
                for _ in range(length):
                    prefix.append(rng.choice(legal_moves_scan(prefix)))
                if board_key(prefix) == board_key(mirror_moves(prefix)) or not opening_acceptable(prefix):
                    continue
                key = family_key(prefix)
                if key in seen:
                    continue
                seen.add(key)
                chosen[split].append(prefix)
                count += 1
    return chosen, dict(attempts=attempts)


def opening_rows(prefixes, split):
    rows = []
    for pair in range(EMPTY_BOARD_PAIRS):
        rows.append(dict(id=f"{split}-empty-{pair:02d}", moves=[], stratum="empty", family="empty-board",
                         pair_seed=pair))
    for index, prefix in enumerate(prefixes):
        family = f"{split}-prefix-{index:02d}"
        for orientation, moves in (("base", prefix), ("mirror", mirror_moves(prefix))):
            rows.append(dict(id=f"{family}-{orientation}", moves=list(moves), stratum="prefix", family=family,
                             to_move=len(moves) % 2, pair_seed=len(rows)))
    return rows


def package_document(kind, split, rows, metadata):
    return dict(format=PACKAGE_FORMAT, format_version=PACKAGE_VERSION, kind=kind, split=split,
                rows_sha256=rows_sha256(rows), rows=rows, metadata=metadata)


def write_package(directory, kind, split, rows, metadata):
    path = Path(directory) / f"{kind}-{split}.json"
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.write_text(json.dumps(package_document(kind, split, rows, metadata), indent=1, sort_keys=True) + "\n")
    return path


def build_all(output_dir, *, seed=BUILD_SEED, log=print, solved_budget_seconds=None):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    excluded, sources = collect_exclusions()
    created = datetime.now(timezone.utc).isoformat()
    common = dict(created_utc=created, build_seed=seed, model_blind=True, training_feedback=False,
                  generator=dict(styles=[dict(name=n, random_probability=p, negamax_depths=list(d))
                                         for n, p, d in GENERATOR_STYLES],
                                 rule="per game a seeded style; each move uniform random with the style "
                                      "probability, else reference Negamax at a seeded style depth with seeded "
                                      "ties; self-symmetric boards excluded"),
                  dedupe="reflection family of (board, actor); transpositions share a board",
                  exclusions=dict(families=len(excluded), sources=sources),
                  split_rule="sealed quotas filled first from the shuffled pool, then development; disjoint families")
    (output / "exclusions.json").write_text(json.dumps(dict(
        sources=sources, family_keys=sorted(excluded), family_keys_sha256=rows_sha256(sorted(excluded))),
        indent=1, sort_keys=True) + "\n")
    written = {}

    # Openings first (cheap); their families are excluded from the position packages too.
    openings, opening_stats = build_openings(excluded, seed + 1)
    for split in SPLITS:
        rows = opening_rows(openings[split], split)
        written[f"openings-{split}"] = write_package(output, "openings", split, rows, dict(
            common, stats=opening_stats, empty_board_pairs=EMPTY_BOARD_PAIRS,
            prefix_bases=OPENING_PREFIX_BASES[split], prefix_lengths="2-8; even (X to move) and odd (O to move) "
            "bases balanced 20/20; each base plus mirror", requirements="nonterminal, no immediate win for "
            "either side, not self-symmetric", exact_values="not computed: beyond the bounded Python oracle"))
        for prefix in openings[split]:
            excluded.add(family_key(prefix))
    log("openings written")

    chosen, stats = build_tactical(excluded, seed + 2, log=log)
    for split in SPLITS:
        rows = []
        for category in ("unique_win", "unique_safe"):
            for actor in (0, 1):
                for index, base in enumerate(chosen[(split, category, actor)]):
                    family = f"{split}-{category}-{'xo'[actor]}-{index:03d}"
                    for orientation, row in (("base", base), ("mirror", mirror_tactical_row(base))):
                        rows.append(dict(row, id=f"{family}-{orientation}", family=family,
                                         stage=stage_of(row["ply"], TACTICAL_STAGES)))
                    excluded.add(family_key(base["moves"]))
        achieved = Counter((r["category"], r["actor"]) for r in rows if r["id"].endswith("base"))
        written[f"tactical-{split}"] = write_package(output, "tactical", split, rows, dict(
            common, stats=stats, quota_bases_per_category_actor=TACTICAL_QUOTAS[split],
            achieved_bases={f"{c}/{'xo'[a]}": n for (c, a), n in sorted(achieved.items())},
            stages=TACTICAL_STAGES, labels="unique immediate win or unique one-reply safe response; engine "
            "successor labels and an independent grid scan must agree on every row (both orientations)"))
    log("tactical written")

    chosen, stats, solver = build_solved(excluded, seed + 3, log=log, budget_seconds=solved_budget_seconds)
    for split in SPLITS:
        rows = []
        for value in (1, 0, -1):
            for actor in (0, 1):
                for index, base in enumerate(chosen[(split, value, actor)]):
                    family = f"{split}-{'LDW'[value + 1]}-{'xo'[actor]}-{index:03d}"
                    base = dict(base, labels=dict(base["labels"], method_b=cross_check_solved(base)))
                    solver.table.clear()
                    solver.nodes = 0
                    for orientation, row in (("base", base), ("mirror", mirror_solved_row(base, solver))):
                        rows.append(dict(row, id=f"{family}-{orientation}", family=family,
                                         stage=stage_of(row["ply"], SOLVED_STAGES)))
                    excluded.add(family_key(base["moves"]))
        achieved = Counter((r["value"], r["actor"]) for r in rows if r["id"].endswith("base"))
        written[f"solved-{split}"] = write_package(output, "solved", split, rows, dict(
            common, stats=stats, quota_bases_per_value_actor=SOLVED_QUOTAS[split],
            achieved_bases={f"{'LDW'[v + 1]}/{'xo'[a]}": n for (v, a), n in sorted(achieved.items())},
            stages=SOLVED_STAGES, method_a=SOLVER_VERSION, method_b=EXHAUSTIVE_VERSION,
            method_b_rule=f"attempted for every base with >= {EXHAUSTIVE_MIN_PIECES} pieces under a "
                          f"{EXHAUSTIVE_STATE_BUDGET}-state budget; mirrors re-solved by method A",
            selection="no immediate win for the mover; wins need a non-winning action; draws need a losing "
                      "action; losses must not be provable by a one-reply scan"))
    log("solved written")
    manifest = {name: dict(path=path.name, sha256=file_sha256(path)) for name, path in written.items()}
    manifest["exclusions"] = dict(path="exclusions.json", sha256=file_sha256(output / "exclusions.json"))
    (output / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return manifest


# Loading and verification -----------------------------------------------------------------

def load_package(path, *, expected_sha256=None):
    path = Path(path)
    if expected_sha256 is not None and file_sha256(path) != expected_sha256:
        raise ValueError(f"Package file hash differs: {path}")
    document = json.loads(path.read_text())
    if (document.get("format"), document.get("format_version")) != (PACKAGE_FORMAT, PACKAGE_VERSION):
        raise ValueError("Not an AlphaZero v2 evaluation package")
    if rows_sha256(document["rows"]) != document["rows_sha256"]:
        raise ValueError("Package rows do not match their content hash")
    return document


def verify_directory(directory, *, resolve=False, log=print):
    """Structural + label verification; ``resolve`` re-solves every solved row with method A."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    for name, entry in manifest.items():
        if file_sha256(directory / entry["path"]) != entry["sha256"]:
            raise ValueError(f"{name} hash mismatch")
    exclusions = set(json.loads((directory / "exclusions.json").read_text())["family_keys"])
    families_by_split = defaultdict(set)
    summary = {}
    solver = BitboardSolver(SOLVER_NODE_BUDGET * 4)
    for name, entry in sorted(manifest.items()):
        if name == "exclusions":
            continue
        document = load_package(directory / entry["path"])
        kind, split = document["kind"], document["split"]
        for row in document["rows"]:
            moves = row["moves"]
            if moves:
                engine_position(moves)
                if family_key(moves) in exclusions:
                    raise ValueError(f"Excluded family present: {row['id']}")
                families_by_split[split].add(family_key(moves))
            if kind == "tactical":
                expected = tactical_row(moves, row["category"], row["expected_action"])
                if any(expected[k] != row[k] for k in expected):
                    raise ValueError(f"Tactical label mismatch: {row['id']}")
            elif kind == "solved":
                if not is_nonterminal(moves) or row["value"] != max(row["action_values"].values()):
                    raise ValueError(f"Solved row inconsistent: {row['id']}")
                if resolve:
                    solver.table.clear()
                    solver.nodes = 0
                    actions = {str(a): v for a, v in sorted(solver.action_values(moves).items())}
                    if actions != row["action_values"]:
                        raise ValueError(f"Solved label mismatch: {row['id']}")
            elif kind == "openings" and moves and not opening_acceptable(moves):
                raise ValueError(f"Opening row invalid: {row['id']}")
        summary[name] = len(document["rows"])
        log(f"verified {name}: {len(document['rows'])} rows")
    overlap = families_by_split["sealed"] & families_by_split["development"]
    if overlap:
        raise ValueError(f"Development and sealed families overlap: {len(overlap)}")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--output-dir", required=True)
    build.add_argument("--seed", type=int, default=BUILD_SEED)
    verify = sub.add_parser("verify")
    verify.add_argument("--directory", default=str(FROZEN_DIRECTORY))
    verify.add_argument("--resolve", action="store_true")
    args = parser.parse_args(argv)
    log = lambda message: print(message, flush=True)  # noqa: E731
    if args.command == "build":
        print(json.dumps(build_all(args.output_dir, seed=args.seed, log=log), indent=1))
    else:
        print(json.dumps(verify_directory(args.directory, resolve=args.resolve, log=log), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
