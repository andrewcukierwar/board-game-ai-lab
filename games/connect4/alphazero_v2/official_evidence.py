"""The official evidence contract of the AlphaZero v2 campaign (torch-free and read-only, Phase 4D.3B.5).

One authoritative definition of every artifact that ``official_results`` relies on. The producer and every
acceptance check use it unchanged:

* ``declaration.json``: the frozen declaration, certified by its SHA-256 (the authorization token);
* ``packages/<name>.json``: the declared evaluation packages, certified by the declaration's hashes;
* ``state.json``: the COMPLETE record (lifecycle, counters, completion record and certified hashes);
* ``final/started.json``: the selections and descriptions fixed before any sealed inference;
* ``final/seed-<S>.json``: one sealed result per declared seed.

A sealed result has exactly one producer, ``derive_seed_result``. Every summary, gate and copied field in it is a
deterministic function of the result's raw evidence, the declaration, the packages and ``started.json``.
``seed_result_problems`` first validates the raw evidence against the declaration: exact row IDs, workloads and
ladder opponents, with every game replayed. It then requires the whole result to equal what ``derive_seed_result``
makes from that evidence. So evidence cannot be removed, truncated or retyped, and no summary can disagree with
its evidence, without the result being refused.

Every validator is total: it returns problems for any JSON value and never raises on malformed input. The one
optional diagnostic in any official artifact is listed in ``OPTIONAL_PATHS``. This module never trains, never runs a
model and never writes a file. It checks consistency, not authenticity: a coherent forgery by someone who can
rewrite every file is out of scope.
"""
import hashlib
import json
import math
from pathlib import Path

from . import statistics
from .arena import PAIRED_GAMES, scored
from .config import V2Config
from .launch_control import (COMPLETE, LAUNCH_CONTROL_VERSION, REJECTED_DECLARATION_TOKENS, RUNNING_FINAL,
                             SELECTION_COMPLETE, launch_control_declaration, sha256_file, state_sequence)
from .oracle import BitboardState, family_key
from .packages import load_package

# Declaration --------------------------------------------------------------------------------------

DECLARATION_FORMAT = "connect4-alphazero-v2-campaign-declaration"
DECLARATION_VERSION = 3
DECLARATION_KEYS = {"format", "format_version", "name", "kind", "seeds", "primary_seed", "config", "generations",
                    "budgets", "runtime", "evaluation", "champion", "final", "packages", "thresholds",
                    "retention", "phase4d2f_checkpoint", "notes",
                    "execution_source", "runtime_identity", "frozen_records", "launch_control"}
# Declaration items constrained only by their type: removing one leaves a different, still valid declaration
# (which the token then no longer authorizes). Free-text notes, and the frozen records bound by a non-research
# declaration (a research declaration binds exactly FROZEN_RECORD_NAMES).
DECLARATION_FREE_ITEMS = (("notes", "*"), ("frozen_records", "*"))
FROZEN_RECORD_NAMES = ("build-provenance.json", "exclusions.json", "manifest.json")
RETENTION_KEYS = {"resume_boundaries", "inference_snapshots", "archived_games"}
EXECUTION_SOURCE_KEYS = {"scheme", "sha256", "files", "groups"}
OFFICIAL_SEEDS = (42, 314159)  # the frozen research declaration's seeds
OPPONENT_NAMES = ("random", "negamax1", "negamax2", "negamax4", "guarded_uct_800")
LADDER_NAMES = OPPONENT_NAMES + ("initial_v2_512", "phase4d2f_512")
BUDGET_KEYS = {"per_run_training_seconds", "campaign_seconds", "evaluation_games_ceiling", "planned_evaluation_games"}
CHAMPION_KEYS = {"schedule", "arena_openings", "baseline_opponents", "baseline_rows", "gate", "tactical_package"}
FINAL_KEYS = {"ladder", "ladder_openings", "calibration_games", "calibration_constant", "one_time"}
EVALUATION_SETTING_KEYS = {"simulations", "tactical_seeds", "root_noise", "tactical_guard", "temperature", "ties"}
DEPLOYMENT = dict(root_noise=False, tactical_guard=False, temperature=0, ties="seeded uniform")
SEALED_PACKAGES = ("tactical-sealed", "solved-sealed", "openings-sealed")
REQUIRED_PACKAGES = SEALED_PACKAGES + ("openings-development", "solved-development")
PACKAGE_DIRECTORY = "packages"  # the campaign directory's copy of every declared package

# Evidence units ------------------------------------------------------------------------------------

DEVELOPMENT_GAME_KINDS = ("development_arena_game", "development_baseline_game")
SEALED_GAME_KINDS = ("final_ladder_game", "final_nn_only_game")
CALIBRATION_GAME_KINDS = ("calibration_game",)
GAME_KINDS = DEVELOPMENT_GAME_KINDS + SEALED_GAME_KINDS + CALIBRATION_GAME_KINDS
ROW_KINDS = ("development_tactics_row", "final_tactical_row", "final_solved_row")

# state.json ----------------------------------------------------------------------------------------

OWNER_KEYS = {"pid", "host", "started_utc"}
HISTORY_INFO_KEYS = {SELECTION_COMPLETE: {"selections"}, RUNNING_FINAL: {"selections", "started_sha256"}}
OUTCOME_KEYS = {"status", "final_results", "started_sha256", "completion", "note"}
COMPLETION_KEYS = {"endpoint", "elapsed_seconds", "training_seconds", "evaluation_games_started", "limits"}
COMPLETION_LIMITS = ("campaign_seconds", "per_run_training_seconds", "evaluation_games_ceiling")
COUNTER_KEYS = {"note", "time", "training", "evaluation"}
TIME_KEYS = {"elapsed_seconds", "phase_seconds", "training_seconds", "evaluation_games_started"}
TRAINING_KEYS = {"generations_attempted", "generations_completed", "selfplay_games_attempted",
                 "selfplay_games_completed", "plies", "optimizer_steps"}
EVALUATION_KEYS = {"by_kind", "development_games", "sealed_games", "calibration_games", "total_games_started",
                   "total_games_completed", "ceiling"}

# Selections, started.json and sealed results ------------------------------------------------------------

SELECTION_RULE = "development evidence only; sealed packages untouched"
SELECTION_KEYS = {"seed", "generation", "inference", "decisions", "rule", "declaration_sha256"}
INFERENCE_KEYS = {"path", "sha256", "bytes"}
STARTED_KEYS = {"selections", "descriptions", "runtime", "execution_sha256", "declaration_sha256"}
DESCRIPTION_KEYS = {"agent", "opponents"}
SEARCH_AGENT_KEYS = {"name", "kind", "simulations", "root_noise", "tactical_guard", "temperature", "ties"}
AGENT_KEYS = SEARCH_AGENT_KEYS | {"path", "sha256", "weights_sha256"}
OPPONENT_KEYS = {"random": {"name", "rule"}, "negamax": {"name", "version", "depth", "ties"},
                 "guarded_uct": {"name", "simulations", "implementation", "root_tactical_guards"},
                 "initial_v2_512": SEARCH_AGENT_KEYS | {"weights_sha256"},
                 "phase4d2f_512": SEARCH_AGENT_KEYS | {"path", "sha256"}}
SEED_RESULT_KEYS = {"seed", "completeness", "selection", "agent", "overlap_families", "tactical", "tactical_nn_only",
                    "solved", "value", "ladder", "nn_only_vs_random", "calibration", "evidence"}
EVIDENCE_KEYS = {"tactical", "solved", "calibration"}
SEARCH_EVIDENCE_KEYS = {"choices", "visits", "root_values"}
TACTICAL_EVIDENCE_KEYS = SEARCH_EVIDENCE_KEYS | {"nn_only_choice"}
SOLVED_EVIDENCE_KEYS = SEARCH_EVIDENCE_KEYS | {"raw_value"}
CALIBRATION_RECORD_KEYS = {"game", "ply", "actor", "value", "outcome"}
ARENA_KEYS = {"summary", "records", "gate"}
LADDER_ENTRY_KEYS = ARENA_KEYS | {"opponent"}
ARENA_RECORD_KEYS = {"opening_id", "family", "stratum", "prefix_length", "game_index", "agent_color", "agent",
                     "opponent", "moves", "abandoned", "winner", "decisions", "seconds", "result"}
DECISION_KEYS = {"color", "move", "seconds"}
BOARD_CELLS, MIN_DECISIVE_PLIES, COLUMNS = 42, 7, 7

# The explicitly optional diagnostics: removing one is accepted (when present, its type is still checked).
# "*" matches any key. The declaration, packages, started.json and sealed results have no optional field.
OPTIONAL_PATHS = {"state.json": (("counters", "time", "phase_seconds", "*"),)}


def is_optional(artifact, path):
    return any(len(pattern) == len(path) and all(p in ("*", k) for p, k in zip(pattern, path))
               for pattern in OPTIONAL_PATHS.get(artifact, ()))


def champion_name(seed):
    return f"seed{seed}-champion"


def nn_only_name(seed):
    return f"seed{seed}-champion-nn-only"


# Strict JSON and value domains ---------------------------------------------------------------------

class MalformedRecord(ValueError):
    """A record acceptance relies on is not strictly well formed (duplicate keys, NaN/Infinity, bad seeds)."""


def strict_json(data):
    """Parse JSON for acceptance. Duplicate object keys and non-finite numbers are refused, never resolved."""
    def pairs(items):
        keys = [key for key, _ in items]
        if len(set(keys)) != len(keys):
            raise MalformedRecord(f"duplicate JSON object keys {sorted({k for k in keys if keys.count(k) > 1})}")
        return dict(items)

    def constant(name):
        raise MalformedRecord(f"non-finite JSON number {name}")
    return json.loads(data, object_pairs_hook=pairs, parse_constant=constant)


def read_strict(path):
    """(document, problems) of a file parsed strictly; a problem instead of an exception."""
    try:
        return strict_json(Path(path).read_bytes()), []
    except (OSError, ValueError) as error:  # MalformedRecord and JSONDecodeError are ValueErrors
        return None, [f"{Path(path).name} is not strict JSON: {error}"]


def normalized(value):
    """The value as JSON stores it (tuples become lists, keys become strings)."""
    return json.loads(json.dumps(value))


def is_count(value):
    return type(value) is int and value >= 0


def is_positive(value):
    return type(value) is int and value >= 1


def is_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def is_seconds(value):
    return is_number(value) and value >= 0


def is_unit_value(value):
    return is_number(value) and -1 <= value <= 1


def is_sha256(value):
    return isinstance(value, str) and len(value) == 64 and set(value) <= set("0123456789abcdef")


def is_distinct_list(value, item):
    try:
        return isinstance(value, list) and bool(value) and all(map(item, value)) and len(set(value)) == len(value)
    except TypeError:  # unhashable items
        return False


def differences(actual, expected, path, limit=10):
    """Type-strict deep comparison: where ``actual`` differs from ``expected`` (empty means identical).

    ``True == 1`` and ``1 == 1.0`` in Python; neither is equal here. Dict key order is irrelevant; list order is not.
    """
    found = []

    def walk(a, e, where):
        if len(found) >= limit:
            return
        if type(a) is not type(e):
            found.append(f"{where} is {type(a).__name__}, expected {type(e).__name__}")
        elif isinstance(e, dict):
            if set(a) != set(e):
                found.append(f"{where} keys differ by {sorted(set(a) ^ set(e), key=str)}")
            else:
                for key in e:
                    walk(a[key], e[key], f"{where}.{key}")
        elif isinstance(e, list):
            if len(a) != len(e):
                found.append(f"{where} has {len(a)} items, expected {len(e)}")
            else:
                for index, (x, y) in enumerate(zip(a, e)):
                    walk(x, y, f"{where}[{index}]")
        elif a != e:
            found.append(f"{where} is {a!r}, expected {e!r}")
    walk(actual, expected, path)
    return found


def same(actual, expected):
    return not differences(actual, expected, "", limit=1)


def keys_problems(name, value, keys):
    if not isinstance(value, dict):
        return [f"{name} is missing or not an object"]
    if set(value) != set(keys):
        return [f"{name} must have exactly keys {sorted(keys)}; differs by {sorted(set(value) ^ set(keys))}"]
    return []


def seed_mapping_problems(name, value, seeds, valid=None, domain="valid"):
    """A required per-seed mapping: exactly the declared seeds (no missing, extra or non-canonical key)."""
    if not isinstance(value, dict):
        return [f"{name} is missing or not a per-seed mapping"]
    if set(value) != seeds:
        return [f"{name} must cover exactly seeds {sorted(seeds)}; has {sorted(value)}"]
    if valid is None:
        return []
    return [f"{name}[{seed}] is not {domain}: {value[seed]!r}" for seed in sorted(value) if not valid(value[seed])]


def safe_sha256_file(path):
    try:
        return sha256_file(path)
    except OSError:
        return None


def capped(problems, limit=12):
    return problems if len(problems) <= limit else problems[:limit] + [f"... and {len(problems) - limit} more"]


# Declaration contract ----------------------------------------------------------------------------------

def required_seed_keys(declaration):
    """The exact key set of every per-seed mapping: the declared seeds as canonical decimal strings."""
    seeds = declaration.get("seeds") if isinstance(declaration, dict) else None
    if (not isinstance(seeds, list) or not seeds or any(type(s) is not int for s in seeds)
            or len(set(seeds)) != len(seeds)):
        raise MalformedRecord(f"declared seeds are malformed: {seeds!r}")
    if declaration.get("kind") == "research" and seeds != list(OFFICIAL_SEEDS):
        raise MalformedRecord(f"a research declaration must declare exactly seeds {list(OFFICIAL_SEEDS)}")
    return {str(seed) for seed in seeds}


def planned_evaluation_games(declaration):
    champion, final = declaration["champion"], declaration["final"]
    per_check = 2 * champion["arena_openings"] + 2 * len(champion["baseline_rows"]) * len(champion["baseline_opponents"])
    per_seed = (len(champion["schedule"]) * per_check
                + 2 * final["ladder_openings"] * len(final["ladder"])
                + 2 * final["ladder_openings"]  # NN Only versus Random
                + final["calibration_games"])
    return per_seed * len(declaration["seeds"])


def is_package_name(name):
    return isinstance(name, str) and bool(name) and name.replace("-", "").isalnum()


def removable_declaration_item(declaration, path):
    """Whether removing the item at ``path`` leaves a declaration the contract still accepts."""
    if not any(len(p) == len(path) and all(a in ("*", b) for a, b in zip(p, path)) for p in DECLARATION_FREE_ITEMS):
        return False
    return path[0] != "frozen_records" or declaration.get("kind") != "research"


def files_digest(files):
    """The execution-source digest scheme: SHA-256 of the sorted JSON map of relative path -> file SHA-256."""
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def execution_source_problems(source):
    """The declared execution source is internally consistent (its partition is checked again at launch)."""
    if keys_problems("execution_source", source, EXECUTION_SOURCE_KEYS) or not isinstance(source["scheme"], str):
        return ["execution_source must be exactly {scheme, sha256, files, groups}"]
    files, groups = source["files"], source["groups"]
    if not isinstance(files, dict) or not files or not all(isinstance(k, str) and is_sha256(v)
                                                           for k, v in files.items()):
        return ["execution_source files must map relative paths to SHA-256 digests"]
    if source["sha256"] != files_digest(files):
        return ["execution_source digest differs from its files"]
    members = []
    if not isinstance(groups, dict) or not groups:
        return ["execution_source groups must partition its files"]
    for name, group in sorted(groups.items()):
        if (keys_problems(name, group, {"files", "sha256"}) or not isinstance(group["files"], list)
                or not all(isinstance(f, str) and f in files for f in group["files"])
                or group["files"] != sorted(set(group["files"]))
                or group["sha256"] != files_digest({f: files[f] for f in group["files"]})):
            return [f"execution_source group {name!r} is not a digest of its sorted member files"]
        members += group["files"]
    if sorted(members) != sorted(files):
        return ["execution_source groups must partition its files"]
    return []


def runtime_problems(declaration):
    """The declared runtime settings and the runtime identity they must agree with (names checked at launch)."""
    runtime, identity = declaration["runtime"], declaration["runtime_identity"]
    if keys_problems("runtime", runtime, {"threads", "deterministic_algorithms"}) \
            or not is_positive(runtime["threads"]) or type(runtime["deterministic_algorithms"]) is not bool:
        return ["runtime must be exactly {threads: positive integer, deterministic_algorithms: boolean}"]
    if not isinstance(identity, dict) or not isinstance(identity.get("thread_environment"), dict):
        return ["runtime_identity must be an object with a thread_environment"]
    environment = identity["thread_environment"]
    if any(value is None for value in identity.values()) or any(value is None for value in environment.values()):
        return ["declared runtime identity has unavailable fields"]
    threads = runtime["threads"]
    if (not same(identity.get("intra_op_threads"), threads) or not same(identity.get("inter_op_threads"), threads)
            or not same(identity.get("deterministic_algorithms"), runtime["deterministic_algorithms"])
            or not environment or not all(same(value, str(threads)) for value in environment.values())):
        return ["declared runtime identity disagrees with the declared runtime settings"]
    return []


def declaration_problems(declaration):
    """Every way a declaration violates the evidence contract: every check that needs neither torch nor files.

    Launch (``campaign.validate_declaration``) runs this, then checks the runtime field names, the execution-source
    partition and the bound files. Acceptance runs this on the certified declaration.
    """
    problems = keys_problems("declaration", declaration, DECLARATION_KEYS)
    if problems:
        return problems
    d = declaration
    if (d["format"], d["format_version"]) != (DECLARATION_FORMAT, DECLARATION_VERSION) \
            or type(d["format_version"]) is not int:
        problems.append(f"not a format-{DECLARATION_VERSION} v2 campaign declaration")
    if not isinstance(d["kind"], str) or not isinstance(d["name"], str):
        problems.append("declaration kind and name must be strings")
    if not isinstance(d["notes"], list) or not all(isinstance(note, str) for note in d["notes"]):
        problems.append("declaration notes must be a list of strings")
    if not same(d["launch_control"], normalized(launch_control_declaration())):
        problems.append("declared launch-control semantics differ from this launcher")
    retention = d["retention"]
    if keys_problems("retention", retention, RETENTION_KEYS) or not is_positive(retention["resume_boundaries"]) \
            or not all(isinstance(retention[k], str) for k in ("inference_snapshots", "archived_games")):
        problems.append("retention must be exactly {resume_boundaries, inference_snapshots, archived_games}")
    records = d["frozen_records"]
    if not isinstance(records, dict) or not all(
            not keys_problems(name, entry, {"path", "sha256"}) and isinstance(entry["path"], str)
            and is_sha256(entry["sha256"]) for name, entry in records.items()):
        problems.append("frozen_records must map names to exactly {path, sha256}")
    elif d["kind"] == "research" and sorted(records) != sorted(FROZEN_RECORD_NAMES):
        problems.append(f"a research declaration must bind {list(FROZEN_RECORD_NAMES)}")
    problems += execution_source_problems(d["execution_source"])
    problems += runtime_problems(d)
    try:
        seeds = required_seed_keys(d)
        if type(d["primary_seed"]) is not int or str(d["primary_seed"]) not in seeds:
            problems.append("the primary seed must be one of the declared seeds")
    except MalformedRecord as error:
        problems.append(str(error))
    generations, config = d["generations"], d["config"]
    if not is_positive(generations):
        problems.append("generations must be a positive integer")
    try:
        if not isinstance(config, dict):
            raise ValueError("config is not an object")
        V2Config.from_dict(dict(config, seed=0))
        if "seed" in config or not same(config["max_generations"], generations):
            raise ValueError("config must omit the seed and run exactly the declared generations")
    except (ValueError, TypeError) as error:  # V2Config's own validation, refusing any malformed field
        problems.append(f"config is not a valid V2Config: {error}")
    budgets = d["budgets"]
    budget_problems = keys_problems("budgets", budgets, BUDGET_KEYS)
    if not budget_problems:
        for name in ("per_run_training_seconds", "campaign_seconds"):
            if not is_seconds(budgets[name]) or budgets[name] <= 0:
                budget_problems.append(f"budget {name} must be positive finite seconds")
        if not is_positive(budgets["evaluation_games_ceiling"]):
            budget_problems.append("budget evaluation_games_ceiling must be a positive integer")
        if not is_count(budgets["planned_evaluation_games"]):
            budget_problems.append("budget planned_evaluation_games must be a non-negative integer")
    problems += budget_problems
    champion = d["champion"]
    champion_problems = keys_problems("champion", champion, CHAMPION_KEYS)
    if not champion_problems:
        schedule = champion["schedule"]
        if (not isinstance(schedule, list) or not all(type(g) is int for g in schedule)
                or sorted(set(schedule)) != schedule
                or not all(1 <= g <= (generations if is_positive(generations) else 0) for g in schedule)):
            champion_problems.append("champion schedule must be increasing generations within the campaign")
        if not is_positive(champion["arena_openings"]):
            champion_problems.append("champion arena_openings must be a positive integer")
        if not is_distinct_list(champion["baseline_opponents"], lambda n: n in OPPONENT_NAMES):
            champion_problems.append(f"champion baseline_opponents must be distinct names from {list(OPPONENT_NAMES)}")
        if not is_distinct_list(champion["baseline_rows"], lambda r: isinstance(r, str)):
            champion_problems.append("champion baseline_rows must be distinct row IDs")
        if not same(champion["gate"], normalized({k: v for k, v in statistics.CHAMPION_GATE.items()
                                                  if k != "schedule"})):
            champion_problems.append("declared champion gate differs from the executed gate")
        if not is_package_name(champion["tactical_package"]):
            champion_problems.append("champion tactical_package must name a package")
    problems += champion_problems
    final = d["final"]
    final_problems = keys_problems("final", final, FINAL_KEYS)
    if not final_problems:
        if not is_distinct_list(final["ladder"], lambda n: n in LADDER_NAMES):
            final_problems.append(f"final ladder must be distinct opponents from {list(LADDER_NAMES)}")
        for name in ("ladder_openings", "calibration_games"):
            if not is_positive(final[name]):
                final_problems.append(f"final {name} must be a positive integer")
        if final["calibration_constant"] is not None and not is_number(final["calibration_constant"]):
            final_problems.append("final calibration_constant must be null or a finite number")
        if final["one_time"] is not True:
            final_problems.append("the final evaluation must be one_time")
    problems += final_problems
    evaluation = d["evaluation"]
    evaluation_problems = keys_problems("evaluation", evaluation, EVALUATION_SETTING_KEYS)
    if not evaluation_problems:
        if not is_positive(evaluation["simulations"]):
            evaluation_problems.append("evaluation simulations must be a positive integer")
        if not is_distinct_list(evaluation["tactical_seeds"], lambda s: type(s) is int):
            evaluation_problems.append("evaluation tactical_seeds must be distinct integers")
        evaluation_problems += [f"evaluation {k} must be {v!r} (deployment conventions)"
                                for k, v in DEPLOYMENT.items() if not same(evaluation[k], v)]
    problems += evaluation_problems
    packages = d["packages"]
    if not isinstance(packages, dict):
        problems.append("packages is not an object")
    else:
        required = set(REQUIRED_PACKAGES) | ({champion["tactical_package"]} if not champion_problems else set())
        if required - set(packages):
            problems.append(f"packages must include {sorted(required - set(packages))}")
        for name, entry in sorted(packages.items()):
            if not is_package_name(name) or keys_problems(name, entry, {"path", "sha256"}) \
                    or not isinstance(entry["path"], str) or not is_sha256(entry["sha256"]):
                problems.append(f"package {name!r} must be a plain name bound to exactly {{path, sha256}}")
    if not same(d["thresholds"], normalized(statistics.ACCEPTANCE_THRESHOLDS)):
        problems.append("declared thresholds differ from the executed thresholds")
    checkpoint = d["phase4d2f_checkpoint"]
    if keys_problems("phase4d2f_checkpoint", checkpoint, {"path", "sha256"}) or not isinstance(checkpoint["path"], str) \
            or not is_sha256(checkpoint["sha256"]):
        problems.append("phase4d2f_checkpoint needs exactly a path and a SHA-256")
    if not problems:
        planned = planned_evaluation_games(d)
        if budgets["planned_evaluation_games"] != planned:
            problems.append(f"planned evaluation games must be {planned}")
        elif planned > budgets["evaluation_games_ceiling"]:
            problems.append("planned evaluation games exceed the ceiling")
    return problems


def package_problems(declaration, documents):
    """The declared packages (``name -> loaded document``) against what the declaration draws from them."""
    problems = []
    for name, document in sorted(documents.items()):
        rows = document.get("rows")
        if f"{document.get('kind')}-{document.get('split')}" != name:
            problems.append(f"package {name} holds {document.get('kind')}-{document.get('split')}")
        if not isinstance(rows, list) or not all(isinstance(r, dict) and isinstance(r.get("id"), str) for r in rows) \
                or len({r["id"] for r in rows}) != len(rows):
            problems.append(f"package {name} rows need distinct string IDs")
    if problems:
        return problems
    champion, final = declaration["champion"], declaration["final"]
    if len(documents["openings-sealed"]["rows"]) < final["ladder_openings"]:
        problems.append("openings-sealed has fewer rows than ladder_openings")
    development = documents["openings-development"]["rows"]
    if len(development) < champion["arena_openings"]:
        problems.append("openings-development has fewer rows than arena_openings")
    if set(champion["baseline_rows"]) - {row["id"] for row in development}:
        problems.append("champion baseline_rows are not all openings-development rows")
    return problems


class OfficialContext:
    """The certified declaration and its packages: everything acceptance derives expectations from."""

    def __init__(self, declaration, token, packages):
        self.declaration, self.token, self.packages = declaration, token, packages
        self.seeds = required_seed_keys(declaration)
        self._families, self._positions = None, {}

    def rows(self, name):
        return self.packages[name]

    def sealed_openings(self):
        return self.packages["openings-sealed"][:self.declaration["final"]["ladder_openings"]]

    def position(self, row):
        """A package row's position (bitboard rules; package rows are verified legal and nonterminal)."""
        if row["id"] not in self._positions:
            self._positions[row["id"]] = BitboardState.from_history(row["moves"])
        return self._positions[row["id"]]

    def legal(self, row):
        return {column for column in range(COLUMNS) if self.position(row).can_play(column)}

    def sealed_families(self):
        if self._families is None:
            self._families = {family_key(row["moves"])
                              for row in self.rows("tactical-sealed") + self.rows("solved-sealed")}
        return self._families


def load_context(directory, token):
    """(context, problems): the campaign directory's certified declaration and package copies."""
    directory = Path(directory)
    if not is_sha256(token) or token in REJECTED_DECLARATION_TOKENS:
        return None, [f"declaration {token!r} is malformed or REJECTED / NOT AUTHORIZED"]
    if safe_sha256_file(directory / "declaration.json") != token:
        return None, ["declaration.json differs from the declaration the campaign ran under"]
    declaration, problems = read_strict(directory / "declaration.json")
    if problems:
        return None, problems
    problems = declaration_problems(declaration)
    if problems:
        return None, ["declaration.json: " + p for p in problems]
    documents = {}
    for name, entry in sorted(declaration["packages"].items()):
        path = directory / PACKAGE_DIRECTORY / f"{name}.json"
        if safe_sha256_file(path) != entry["sha256"]:
            problems.append(f"{PACKAGE_DIRECTORY}/{name}.json differs from the package the declaration binds")
            continue
        try:
            documents[name] = load_package(path, expected_sha256=entry["sha256"])
        except (ValueError, KeyError, TypeError, AttributeError) as error:
            problems.append(f"{PACKAGE_DIRECTORY}/{name}.json is not a valid package: {error}")
    if not problems:
        problems = package_problems(declaration, documents)
    if problems:
        return None, problems
    return OfficialContext(declaration, token, {name: document["rows"] for name, document in documents.items()}), []


def expected_units(context):
    """Exactly the evidence units the declaration and packages require, per unit kind (all seeds)."""
    champion, final = context.declaration["champion"], context.declaration["final"]
    checks = len(champion["schedule"])
    per_seed = dict(
        development_arena_game=checks * 2 * champion["arena_openings"],
        development_baseline_game=checks * 2 * len(champion["baseline_rows"]) * len(champion["baseline_opponents"]),
        # Every candidate's rows, plus generation 0 (the first check's incumbent). Later incumbents are earlier
        # candidates whose rows the owning process reuses.
        development_tactics_row=(checks + 1 if checks else 0) * len(context.rows(champion["tactical_package"])),
        final_tactical_row=len(context.rows("tactical-sealed")),
        final_solved_row=len(context.rows("solved-sealed")),
        final_ladder_game=len(final["ladder"]) * 2 * final["ladder_openings"],
        final_nn_only_game=2 * final["ladder_openings"],
        calibration_game=final["calibration_games"])
    return {kind: count * len(context.seeds) for kind, count in per_seed.items()}


# Selections and started.json ---------------------------------------------------------------------------

def selection_problems(context, name, selection, seed):
    """One seed's development selection; every copy (history, started.json, results) must pass this."""
    problems = keys_problems(name, selection, SELECTION_KEYS)
    if problems:
        return problems
    if type(selection["seed"]) is not int or str(selection["seed"]) != seed:
        problems.append(f"{name} names seed {selection['seed']!r}")
    if selection["declaration_sha256"] != context.token:
        problems.append(f"{name} is not a selection under {context.token}")
    if selection["rule"] != SELECTION_RULE:
        problems.append(f"{name} rule is not the development-only selection rule")
    schedule, decisions = context.declaration["champion"]["schedule"], selection["decisions"]
    generation = selection["generation"]
    if (not isinstance(decisions, dict) or set(decisions) != {str(g) for g in schedule}
            or not all(type(v) is bool for v in decisions.values())):
        problems.append(f"{name} decisions must map exactly the scheduled generations {schedule} to booleans")
    elif not same(generation, max([g for g in schedule if decisions[str(g)]], default=0)):
        problems.append(f"{name} generation {generation!r} is not the latest promoted generation")
    inference = selection["inference"]
    if keys_problems(f"{name} inference", inference, INFERENCE_KEYS) or not is_sha256(inference["sha256"]) \
            or not is_positive(inference["bytes"]) or not isinstance(inference["path"], str) \
            or type(generation) is not int or not inference["path"].startswith(f"generations/generation-{generation:04d}/"):
        problems.append(f"{name} inference must be exactly {{path, sha256, bytes}} inside the selected generation")
    return problems


def selections_problems(context, name, selections):
    problems = seed_mapping_problems(name, selections, context.seeds)
    if problems:
        return problems
    return [p for seed in sorted(context.seeds) for p in selection_problems(context, f"{name}[{seed}]",
                                                                             selections[seed], seed)]


def search_agent_problems(context, name, description, keys):
    problems = keys_problems(name, description, keys)
    if problems:
        return problems
    evaluation = context.declaration["evaluation"]
    if not isinstance(description["kind"], str):
        problems.append(f"{name} kind is not a string")
    if not same(description["simulations"], evaluation["simulations"]):
        problems.append(f"{name} simulations differ from the declared simulations")
    problems += [f"{name} {k} is not the declared {evaluation[k]!r}" for k in DEPLOYMENT
                 if not same(description[k], evaluation[k])]
    return problems


def opponent_problems(context, name, description):
    """A ladder opponent's description: its kind's exact keys, its own name and its declaration-derived values."""
    label = f"opponent {name}"
    kind = "negamax" if name.startswith("negamax") else "guarded_uct" if name.startswith("guarded_uct_") else name
    if kind in ("initial_v2_512", "phase4d2f_512"):
        problems = search_agent_problems(context, label, description, OPPONENT_KEYS[kind])
    else:
        problems = keys_problems(label, description, OPPONENT_KEYS[kind])
    if problems:
        return problems
    if description["name"] != name:
        problems.append(f"{label} describes {description['name']!r}")
    texts = dict(random=("rule",), negamax=("version",), guarded_uct=("implementation", "root_tactical_guards"))
    problems += [f"{label} {k} is not a string" for k in texts.get(kind, ()) if not isinstance(description[k], str)]
    if kind == "negamax" and (not same(description["depth"], int(name[len("negamax"):]))
                              or description["ties"] != DEPLOYMENT["ties"]):
        problems.append(f"{label} depth or ties differ from its name")
    if kind == "guarded_uct" and not same(description["simulations"], int(name[len("guarded_uct_"):])):
        problems.append(f"{label} simulations differ from its name")
    if kind == "initial_v2_512" and not is_sha256(description["weights_sha256"]):
        problems.append(f"{label} weights_sha256 is not a SHA-256")
    if kind == "phase4d2f_512" and not same({k: description[k] for k in ("path", "sha256")},
                                            context.declaration["phase4d2f_checkpoint"]):
        problems.append(f"{label} is not the declared Phase 4D.2f checkpoint")
    return problems


def description_problems(context, seed, description, selection):
    name = f"final/started.json descriptions[{seed}]"
    problems = keys_problems(name, description, DESCRIPTION_KEYS)
    if problems:
        return problems
    agent = description["agent"]
    agent_problems = search_agent_problems(context, f"{name} agent", agent, AGENT_KEYS)
    if not agent_problems:
        if agent["name"] != champion_name(seed):
            agent_problems.append(f"{name} agent is named {agent['name']!r}, not {champion_name(seed)!r}")
        if not is_sha256(agent["weights_sha256"]):
            agent_problems.append(f"{name} agent weights_sha256 is not a SHA-256")
        if (agent["path"], agent["sha256"]) != (selection["inference"]["path"], selection["inference"]["sha256"]):
            agent_problems.append(f"{name} agent is not the selected checkpoint")
    problems += agent_problems
    opponents, ladder = description["opponents"], context.declaration["final"]["ladder"]
    if not isinstance(opponents, dict) or set(opponents) != set(ladder):
        problems.append(f"{name} opponents must be exactly the declared ladder {ladder}")
    else:
        problems += [p for opponent in ladder for p in opponent_problems(context, opponent, opponents[opponent])]
    return problems


def started_problems(context, started, selections):
    """final/started.json against the declaration and the selections recorded in state.json."""
    problems = keys_problems("final/started.json", started, STARTED_KEYS)
    if problems:
        return problems
    declaration = context.declaration
    if started["declaration_sha256"] != context.token:
        problems.append("final/started.json names another declaration")
    if started["execution_sha256"] != declaration["execution_source"]["sha256"]:
        problems.append("final/started.json execution digest differs from the declared execution source")
    if not same(started["runtime"], declaration["runtime_identity"]):
        problems.append("final/started.json runtime differs from the declared runtime identity")
    selection_problems_found = selections_problems(context, "final/started.json selections", started["selections"])
    problems += selection_problems_found
    if not selection_problems_found and not same(started["selections"], selections):
        problems.append("final/started.json selections differ from the selections in state.json")
    descriptions = started["descriptions"]
    description_mapping = seed_mapping_problems("final/started.json descriptions", descriptions, context.seeds)
    problems += description_mapping
    if not description_mapping and not selection_problems_found:
        problems += [p for seed in sorted(context.seeds)
                     for p in description_problems(context, seed, descriptions[seed], started["selections"][seed])]
    return problems


# Sealed results ----------------------------------------------------------------------------------------

def with_overlap_sensitivity(rows, overlap, metric):
    """Prespecified sensitivity: drop every package family seen in training (either orientation)."""
    kept = [r for r in rows if family_key(r["moves"]) not in overlap]
    return dict(complete_set=metric(rows), overlap_excluded=metric(kept) if kept else None,
                overlap_excluded_rows=len(rows) - len(kept))


def derive_seed_result(context, seed, started, raw):
    """The one producer of a sealed result for ``seed``.

    ``raw``: ``overlap_families`` (sorted family keys seen in training), ``tactical`` and ``solved`` (row ID ->
    row evidence), ``calibration`` (records), ``ladder`` (opponent -> game records) and ``nn_only_vs_random``
    (game records). Every other field is derived here, from ``raw``, the declaration, the packages and
    ``started``. The owning process and acceptance call this same function.
    """
    final, thresholds = context.declaration["final"], context.declaration["thresholds"]
    tactics, solved = context.rows("tactical-sealed"), context.rows("solved-sealed")
    openings = context.sealed_openings()
    overlap, tactical, solved_evidence = set(raw["overlap_families"]), raw["tactical"], raw["solved"]

    def arena(name, records):
        summary = statistics.arena_summary(scored(records), planned_games=2 * len(openings))
        rule = thresholds["arena"].get(name)
        return dict(summary=summary, records=records, gate=statistics.arena_gate(summary, rule) if rule else None)
    choices = {k: v["choices"] for k, v in tactical.items()}
    visits = {k: v["visits"] for k, v in tactical.items()}
    nn_choices = {k: [v["nn_only_choice"]] for k, v in tactical.items()}
    solved_choices = {k: v["choices"] for k, v in solved_evidence.items()}
    values = {k: v["raw_value"] for k, v in solved_evidence.items()}
    description = started["descriptions"][str(seed)]
    return dict(
        seed=seed, completeness="complete", selection=started["selections"][str(seed)], agent=description["agent"],
        overlap_families=sorted(overlap),
        tactical=with_overlap_sensitivity(tactics, overlap,
                                          lambda rows: statistics.tactical_metrics(rows, choices, visits)),
        tactical_nn_only=with_overlap_sensitivity(tactics, overlap,
                                                  lambda rows: statistics.tactical_metrics(rows, nn_choices)),
        solved=with_overlap_sensitivity(solved, overlap,
                                        lambda rows: statistics.solved_decision_metrics(rows, solved_choices)),
        value=with_overlap_sensitivity(solved, overlap, lambda rows: statistics.value_metrics(rows, values)),
        ladder={name: dict(arena(name, raw["ladder"][name]), opponent=description["opponents"][name])
                for name in final["ladder"]},
        nn_only_vs_random=arena("nn_only_vs_random", raw["nn_only_vs_random"]),
        calibration=statistics.calibration_summary(raw["calibration"], constant=final["calibration_constant"]),
        evidence=dict(tactical=tactical, solved=solved_evidence, calibration=raw["calibration"]))


def search_evidence_problems(context, name, evidence, row, keys):
    """One sealed row's search evidence: one entry per declared tactical seed, consistent with the position."""
    problems = keys_problems(name, evidence, keys)
    if problems:
        return problems
    evaluation = context.declaration["evaluation"]
    count, simulations = len(evaluation["tactical_seeds"]), evaluation["simulations"]
    legal = context.legal(row)
    choices, visits, roots = evidence["choices"], evidence["visits"], evidence["root_values"]
    if not isinstance(choices, list) or len(choices) != count or not all(type(c) is int and c in legal for c in choices):
        problems.append(f"{name} choices must be {count} legal actions (one per declared seed)")
    if (not isinstance(visits, list) or len(visits) != count
            or not all(isinstance(v, list) and len(v) == COLUMNS and all(map(is_count, v)) for v in visits)):
        problems.append(f"{name} visits must be {count} lists of {COLUMNS} visit counts")
    elif any(sum(v) != simulations or any(v[a] for a in range(COLUMNS) if a not in legal) for v in visits):
        problems.append(f"{name} visits must sum to {simulations} root simulations over legal actions only")
    elif not problems and any(v[c] != max(v) for v, c in zip(visits, choices)):
        problems.append(f"{name} choices are not maximum-visit actions")
    if not isinstance(roots, list) or len(roots) != count or not all(map(is_unit_value, roots)):
        problems.append(f"{name} root_values must be {count} values in [-1, 1]")
    elif not all(same(v, roots[0]) for v in roots):
        problems.append(f"{name} root_values differ between search seeds (the root network value is seed-free)")
    if keys is TACTICAL_EVIDENCE_KEYS:
        nn_choice = evidence["nn_only_choice"]
        if type(nn_choice) is not int or nn_choice not in legal:
            problems.append(f"{name} nn_only_choice must be a legal action")
    elif not is_unit_value(evidence["raw_value"]) or not (isinstance(roots, list) and roots
                                                           and same(evidence["raw_value"], roots[0])):
        problems.append(f"{name} raw_value must equal the search's root network value")
    return problems


def row_evidence_problems(context, name, evidence, package, keys):
    if not isinstance(evidence, dict):
        return [f"{name} is missing or not a mapping of row IDs"]
    rows = context.rows(package)
    ids = {row["id"] for row in rows}
    if set(evidence) != ids:
        return [f"{name} must cover exactly the {len(ids)} rows of {package}; missing "
                f"{sorted(ids - set(evidence))[:5]}, extra {sorted(set(evidence) - ids)[:5]}"]
    return [p for row in rows for p in search_evidence_problems(context, f"{name}[{row['id']}]", evidence[row["id"]],
                                                                row, keys)]


def calibration_problems(context, name, records):
    """The held-out calibration workload: every declared game, every ply, outcomes consistent with the game."""
    games = context.declaration["final"]["calibration_games"]
    if not isinstance(records, list) or not records:
        return [f"{name} is missing or not a list of records"]
    problems = []
    for index, record in enumerate(records):
        record_problems = keys_problems(f"{name}[{index}]", record, CALIBRATION_RECORD_KEYS)
        if not record_problems and not (is_count(record["game"]) and is_count(record["ply"])
                                        and type(record["actor"]) is int and is_unit_value(record["value"])
                                        and type(record["outcome"]) is int and record["outcome"] in (-1, 0, 1)):
            record_problems.append(f"{name}[{index}] has a value outside its domain")
        problems += record_problems
    if problems:
        return capped(problems)
    blocks = []
    for record in records:
        if not blocks or blocks[-1][0]["game"] != record["game"]:
            blocks.append([])
        blocks[-1].append(record)
    order = [block[0]["game"] for block in blocks]
    if order != list(range(games)):
        return [f"{name} must hold games 0..{games - 1} once each, in order; has {order[:10]}"]
    for block in blocks:
        label, first = f"{name} game {block[0]['game']}", block[0]["outcome"]
        if [r["ply"] for r in block] != list(range(len(block))) or any(r["actor"] != r["ply"] % 2 for r in block):
            problems.append(f"{label} must record every ply from 0 with the mover's actor")
        elif any(r["outcome"] != first * (-1) ** r["ply"] for r in block):
            problems.append(f"{label} outcomes are not one actor-relative game outcome")
        elif first == 0 and len(block) != BOARD_CELLS:
            problems.append(f"{label} is a draw with {len(block)} plies; a drawn game fills the board")
        elif first != 0 and not (MIN_DECISIVE_PLIES <= len(block) <= BOARD_CELLS and block[-1]["outcome"] == 1):
            problems.append(f"{label} is a decisive game whose last mover did not win after {len(block)} plies")
    return capped(problems)


def replay(state, moves):
    """(movers, winner) of the moves played from ``state``; winner None if illegal, post-terminal or unfinished.

    The oracle's bitboard rules, independent of the engine that played the game: a move completing four wins,
    a full board without four is a draw (-1).
    """
    movers, winner = [], None
    for move in moves:
        if winner is not None or not 0 <= move < COLUMNS or not state.can_play(move):
            return movers, None
        mover = state.moves % 2
        if state.is_winning_move(move):
            winner = mover
        state = state.played(move)
        movers.append(mover)
    if winner is None and state.moves == BOARD_CELLS:
        winner = -1
    return movers, winner


def arena_records_problems(context, name, records, agent, opponent):
    """Exactly the paired games of the declared sealed openings, each replayed from its opening."""
    expected = [(row, game_index, color) for row in context.sealed_openings() for game_index, color in PAIRED_GAMES]
    if not isinstance(records, list) or len(records) != len(expected):
        return [f"{name} must hold exactly the {len(expected)} paired games of the declared sealed openings"]
    problems = []
    for index, (record, (row, game_index, color)) in enumerate(zip(records, expected)):
        label = f"{name}[{index}]"
        record_problems = keys_problems(label, record, ARENA_RECORD_KEYS)
        if record_problems:
            problems += record_problems
            continue
        prefix = len(row["moves"])
        identity = dict(opening_id=row["id"], family=row["family"], stratum=row["stratum"], prefix_length=prefix,
                        game_index=game_index, agent_color=color, agent=agent, opponent=opponent, abandoned=False)
        if not same({k: record[k] for k in identity}, identity):
            problems.append(f"{label} is not the paired game {row['id']}#{game_index} of {agent} versus {opponent}")
            continue
        moves, decisions = record["moves"], record["decisions"]
        if not isinstance(moves, list) or not all(type(m) is int for m in moves) or moves[:prefix] != row["moves"]:
            problems.append(f"{label} moves do not extend the opening")
            continue
        movers, winner = replay(context.position(row), moves[prefix:])
        if winner is None:
            problems.append(f"{label} moves are not a legal, finished game")
            continue
        result = "draw" if winner == -1 else ("win" if winner == color else "loss")
        if not same(record["winner"], winner) or not same(record["result"], result):
            problems.append(f"{label} winner or result differs from the replayed game")
        if (not isinstance(decisions, list) or len(decisions) != len(movers)
                or any(keys_problems("decision", d, DECISION_KEYS) for d in decisions)):
            problems.append(f"{label} decisions must be one {{color, move, seconds}} per played move")
        elif not all(same(d["color"], mover) and same(d["move"], move) and is_seconds(d["seconds"])
                     for d, mover, move in zip(decisions, movers, moves[prefix:])):
            problems.append(f"{label} decisions differ from the replayed moves")
        if not is_seconds(record["seconds"]):
            problems.append(f"{label} seconds is not finite non-negative seconds")
    return capped(problems)


def overlap_problems(context, name, overlap):
    if not isinstance(overlap, list) or not all(isinstance(f, str) for f in overlap) or overlap != sorted(set(overlap)):
        return [f"{name} must be sorted, distinct family keys"]
    unknown = [f for f in overlap if f not in context.sealed_families()]
    return [f"{name} names families outside the sealed tactical and solved packages: {unknown[:3]}"] if unknown else []


def raw_evidence(context, seed, result):
    """(raw, problems): the result's raw evidence, if it is structurally complete for the declaration."""
    name = f"final/seed-{seed}.json"
    evidence, ladder, nn_only = result["evidence"], result["ladder"], result["nn_only_vs_random"]
    problems = keys_problems(f"{name} evidence", evidence, EVIDENCE_KEYS)
    if not problems:
        problems += row_evidence_problems(context, f"{name} evidence.tactical", evidence["tactical"],
                                          "tactical-sealed", TACTICAL_EVIDENCE_KEYS)
        problems += row_evidence_problems(context, f"{name} evidence.solved", evidence["solved"],
                                          "solved-sealed", SOLVED_EVIDENCE_KEYS)
        problems += calibration_problems(context, f"{name} evidence.calibration", evidence["calibration"])
    problems += overlap_problems(context, f"{name} overlap_families", result["overlap_families"])
    declared = context.declaration["final"]["ladder"]
    if not isinstance(ladder, dict) or set(ladder) != set(declared):
        problems.append(f"{name} ladder must contain exactly the declared opponents {declared}; has "
                        f"{sorted(ladder) if isinstance(ladder, dict) else type(ladder).__name__}")
    else:
        for opponent in declared:
            entry_problems = keys_problems(f"{name} ladder.{opponent}", ladder[opponent], LADDER_ENTRY_KEYS)
            problems += entry_problems or arena_records_problems(
                context, f"{name} ladder.{opponent}.records", ladder[opponent]["records"], champion_name(seed),
                opponent)
    nn_problems = keys_problems(f"{name} nn_only_vs_random", nn_only, ARENA_KEYS)
    problems += nn_problems or arena_records_problems(context, f"{name} nn_only_vs_random.records",
                                                      nn_only["records"], nn_only_name(seed), "random")
    if problems:
        return None, problems
    return dict(overlap_families=result["overlap_families"], tactical=evidence["tactical"], solved=evidence["solved"],
                calibration=evidence["calibration"], ladder={k: ladder[k]["records"] for k in declared},
                nn_only_vs_random=nn_only["records"]), []


def seed_result_problems(context, started, seed, result):
    """Every way ``result`` is not the complete sealed result for ``seed`` (a canonical key) under ``context``.

    ``started`` must already satisfy ``started_problems``. Empty means the result is official evidence.
    """
    name = f"final/seed-{seed}.json"
    problems = keys_problems(name, result, SEED_RESULT_KEYS)
    if problems:
        return problems
    raw, problems = raw_evidence(context, seed, result)
    if problems:
        return capped(problems)
    try:
        expected = normalized(derive_seed_result(context, int(seed), started, raw))
    except Exception as error:  # noqa: BLE001 - validated evidence always derives; anything else refuses
        return [f"{name} evidence cannot be summarized: {type(error).__name__}: {error}"]
    return [f"{name} is not what its evidence derives: {d}" for d in differences(result, expected, "result")]


# The COMPLETE record ------------------------------------------------------------------------------------

def completion_record(time_account, budgets):
    """The COMPLETE record's authoritative endpoint: the account measured at the completion barrier."""
    return dict(endpoint="completion barrier: one monotonic reading in the owning process; limits inclusive",
                elapsed_seconds=time_account["elapsed_seconds"],
                training_seconds=dict(time_account["training_seconds"]),
                evaluation_games_started=time_account["evaluation_games_started"],
                limits={k: budgets[k] for k in COMPLETION_LIMITS})


def certified_evidence_problems(directory, context, final_results, started_sha256, selections, accepted=None):
    """Every way the files a COMPLETE record certifies differ from the contract.

    ``selections`` are those recorded in state.json. Validated results are stored in ``accepted`` (seed -> result),
    so a caller returns exactly the documents that were validated.
    """
    directory = Path(directory)
    problems = selections_problems(context, "recorded selections", selections)
    started = None
    started_path = directory / "final" / "started.json"
    if not is_sha256(started_sha256) or safe_sha256_file(started_path) != started_sha256:
        problems.append("final/started.json differs from the hash recorded when the sealed phase started")
    else:
        started, parse_problems = read_strict(started_path)
        problems += parse_problems or started_problems(context, started, selections)
    problems += seed_mapping_problems("outcome.final_results", final_results, context.seeds, is_sha256, "a SHA-256")
    if problems:
        return problems
    for seed in sorted(context.seeds):
        path = directory / "final" / f"seed-{seed}.json"
        if safe_sha256_file(path) != final_results[seed]:
            problems.append(f"{path} differs from the hash certified by COMPLETE")
            continue
        result, parse_problems = read_strict(path)
        seed_problems = parse_problems or seed_result_problems(context, started, seed, result)
        problems += seed_problems
        if not seed_problems and accepted is not None:
            accepted[seed] = result
    return problems


def counter_problems(context, counters):
    """The counter account of a COMPLETE campaign: exact structure, seed coverage, domains, totals and the
    declaration-derived unit counts. Limits are checked against the completion record and at the barrier."""
    problems = keys_problems("counters", counters, COUNTER_KEYS)
    if problems:
        return problems
    declaration, seeds = context.declaration, context.seeds
    if not isinstance(counters["note"], str):
        problems.append("counters.note is not a description")
    time_account, training, evaluation = counters["time"], counters["training"], counters["evaluation"]
    time_problems = keys_problems("counters.time", time_account, TIME_KEYS)
    if not time_problems:
        if not is_seconds(time_account["elapsed_seconds"]):
            time_problems.append("counters.time.elapsed_seconds is not finite non-negative seconds")
        time_problems += seed_mapping_problems("counters.time.training_seconds", time_account["training_seconds"],
                                               seeds, is_seconds, "finite non-negative seconds")
        phases = time_account["phase_seconds"]  # optional diagnostics: typed, but no required phase names
        if not isinstance(phases, dict) or not all(is_seconds(v) for v in phases.values()):
            time_problems.append("counters.time.phase_seconds is not a mapping of phase names to seconds")
        if not is_count(time_account["evaluation_games_started"]):
            time_problems.append("counters.time.evaluation_games_started is not a non-negative integer")
    problems += time_problems
    training_problems = seed_mapping_problems("counters.training", training, seeds)
    if not training_problems:
        generations = declaration["generations"]
        games = generations * declaration["config"]["games_per_generation"]
        for seed in sorted(seeds):
            name, entry = f"counters.training[{seed}]", training[seed]
            entry_problems = keys_problems(name, entry, TRAINING_KEYS)
            if not entry_problems and not all(is_count(v) for v in entry.values()):
                entry_problems.append(f"{name} has values that are not non-negative integers")
            if not entry_problems and not (entry["generations_attempted"] == entry["generations_completed"]
                                           == generations):
                entry_problems.append(f"seed {seed} completed {entry['generations_completed']} of {generations} "
                                      f"generations ({entry['generations_attempted']} attempted)")
            if not entry_problems and not (entry["selfplay_games_attempted"] == entry["selfplay_games_completed"]
                                           == games):
                entry_problems.append(f"seed {seed} completed {entry['selfplay_games_completed']} of {games} "
                                      f"self-play games ({entry['selfplay_games_attempted']} attempted)")
            training_problems += entry_problems
    problems += training_problems
    evaluation_problems = keys_problems("counters.evaluation", evaluation, EVALUATION_KEYS)
    if not evaluation_problems:
        by_kind = evaluation["by_kind"]
        evaluation_problems = keys_problems("counters.evaluation.by_kind", by_kind, GAME_KINDS + ROW_KINDS)
        if not evaluation_problems:
            expected = expected_units(context)
            for kind, units in sorted(by_kind.items()):
                if keys_problems(kind, units, ("started", "completed")) or not all(map(is_count, units.values())):
                    evaluation_problems.append(f"counters.evaluation.by_kind[{kind}] is malformed")
                elif not units["started"] == units["completed"] == expected[kind]:
                    evaluation_problems.append(f"{kind}: {units['started']} evidence units started and "
                                               f"{units['completed']} completed; the declaration requires "
                                               f"{expected[kind]}")
    if not evaluation_problems:
        def total(kinds, key):
            return sum(by_kind[kind][key] for kind in kinds)
        ceiling = declaration["budgets"]["evaluation_games_ceiling"]
        totals = dict(development_games=total(DEVELOPMENT_GAME_KINDS, "completed"),
                      sealed_games=total(SEALED_GAME_KINDS, "completed"),
                      calibration_games=total(CALIBRATION_GAME_KINDS, "completed"),
                      total_games_started=total(GAME_KINDS, "started"),
                      total_games_completed=total(GAME_KINDS, "completed"), ceiling=ceiling)
        for key, value in totals.items():
            if not same(evaluation[key], value):
                evaluation_problems.append(f"counters.evaluation.{key} is {evaluation[key]!r}; the units give {value}")
        if totals["total_games_started"] > ceiling:
            evaluation_problems.append("evaluation games started exceed the declared ceiling")
        if not time_problems and not same(time_account["evaluation_games_started"], totals["total_games_started"]):
            evaluation_problems.append("counters.time.evaluation_games_started differs from the games started")
    return problems + evaluation_problems


def completion_problems(context, completion, counters):
    """The completion record: exact structure and seed coverage; its endpoint, every seed's training seconds
    and the games started within the declared limits (inclusive); and equal to the account it certifies."""
    budgets, seeds = context.declaration["budgets"], context.seeds
    problems = keys_problems("outcome.completion", completion, COMPLETION_KEYS)
    if problems:
        return problems
    if not isinstance(completion["endpoint"], str):
        problems.append("outcome.completion.endpoint is not a description")
    if not same(completion["limits"], {k: budgets[k] for k in COMPLETION_LIMITS}):
        problems.append("completion record limits differ from the declaration")
    if not is_seconds(completion["elapsed_seconds"]):
        problems.append("outcome.completion.elapsed_seconds is not finite non-negative seconds")
    elif not completion["elapsed_seconds"] <= budgets["campaign_seconds"]:
        problems.append("completion endpoint is past the campaign wall-clock budget")
    training = completion["training_seconds"]
    training_problems = seed_mapping_problems("outcome.completion.training_seconds", training, seeds, is_seconds,
                                              "finite non-negative seconds")
    if not training_problems:
        training_problems = [f"seed {seed} training seconds exceed the per-seed budget at completion"
                             for seed in sorted(seeds) if not training[seed] <= budgets["per_run_training_seconds"]]
    problems += training_problems
    if not is_count(completion["evaluation_games_started"]):
        problems.append("outcome.completion.evaluation_games_started is not a non-negative integer")
    elif not completion["evaluation_games_started"] <= budgets["evaluation_games_ceiling"]:
        problems.append("evaluation games started exceed the ceiling at completion")
    # The same quantities are stored in the account the record certifies: both copies must agree exactly.
    time_account = counters.get("time") if isinstance(counters, dict) else None
    for key in ("elapsed_seconds", "training_seconds", "evaluation_games_started"):
        if not isinstance(time_account, dict) or key not in time_account or not same(time_account[key],
                                                                                      completion[key]):
            problems.append(f"outcome.completion.{key} differs from counters.time.{key}")
    return problems


def envelope_problems(document):
    """The state document around the record (``CampaignStateFile.validate`` has checked its keys and sequence)."""
    problems = []
    if document["launch_control"] != LAUNCH_CONTROL_VERSION or document["state"] != COMPLETE:
        problems.append("state.json is not a COMPLETE record of this launch-control version")
    owner = document["owner"]
    if keys_problems("owner", owner, OWNER_KEYS) or not is_count(owner["pid"]) or not isinstance(owner["host"], str) \
            or not isinstance(owner["started_utc"], str):
        problems.append("state.json owner must be exactly {pid, host, started_utc}")
    for index, entry in enumerate(document["history"]):
        keys = {"state", "utc"} | HISTORY_INFO_KEYS.get(entry["state"], set())
        if keys_problems(f"history[{index}]", entry, keys) or not isinstance(entry["utc"], str):
            problems.append(f"history[{index}] ({entry['state']}) must be exactly {sorted(keys)} with a utc time")
    outcome = document["outcome"]
    outcome_problems = keys_problems("outcome", outcome, OUTCOME_KEYS)
    if not outcome_problems and (outcome["status"] != COMPLETE or not isinstance(outcome["note"], str)):
        outcome_problems.append("outcome status or note is malformed")
    return problems + outcome_problems


def history_problems(context, document):
    """The selections and sealed-phase start recorded in the transition history (each state appears once)."""
    if not same(document["seeds"], context.declaration["seeds"]) \
            or [e["state"] for e in document["history"]] != list(state_sequence(context.declaration["seeds"])):
        return ["state seeds or history differ from the declaration's state sequence"]
    entries = {entry["state"]: entry for entry in document["history"]}
    problems = []
    for state in (SELECTION_COMPLETE, RUNNING_FINAL):
        problems += selections_problems(context, f"history {state} selections", entries[state]["selections"])
    if not problems and not same(entries[SELECTION_COMPLETE]["selections"], entries[RUNNING_FINAL]["selections"]):
        problems.append("selections changed between DEVELOPMENT_SELECTION_COMPLETE and RUNNING_FINAL_EVALUATION")
    if not is_sha256(entries[RUNNING_FINAL]["started_sha256"]) \
            or entries[RUNNING_FINAL]["started_sha256"] != document["outcome"]["started_sha256"]:
        problems.append("started.json hash differs from the one recorded on entering the sealed phase")
    return problems


def complete_record_problems(directory, document, accepted=None):
    """Every way a COMPLETE state document (strictly parsed and passing ``CampaignStateFile.validate``) fails the
    contract; empty means its certified results (stored in ``accepted``) are official evidence."""
    problems = envelope_problems(document)
    if problems:
        return problems
    context, problems = load_context(directory, document["declaration_sha256"])
    if problems:
        return problems
    problems = history_problems(context, document)
    if problems:
        return problems
    outcome, counters = document["outcome"], document["counters"]
    selections = next(h["selections"] for h in document["history"] if h["state"] == RUNNING_FINAL)
    return (certified_evidence_problems(directory, context, outcome["final_results"], outcome["started_sha256"],
                                        selections, accepted)
            + counter_problems(context, counters)
            + completion_problems(context, outcome["completion"], counters))
