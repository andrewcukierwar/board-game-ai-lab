"""Independent CL/BI/VE strategic certificate verification (Phase 6B.4A).

Standard library only. This module deliberately imports NOTHING from the engine,
grounding or the rest of Victor. It re-derives board legality, the 69 groups,
rule prerequisites, disjointness, coverage and replay from raw cells, so a
defect in the producer (enumeration, search, ``Position`` validation or the
coverage-witness verifier) cannot be silently shared with this boundary.

Success means only that every finite hypothesis H1-H4 of theorem
``cl-bi-ve-black-nonloss-v1`` holds for the exact bound board. The theorem's
universal strategy argument is a reviewed paper proof (docs/phase6b3 and
docs/phase6b4a), not a proof-assistant result. A verified result therefore
carries the implied ``Black value >= 0`` bound as PROVISIONAL research evidence.
It is never a draw/win claim, an exact value, an optimal move or a public
outcome field.

Squares are named a1..g6 (row 1 at the bottom). Boards are top-first 6x7
matrices of ``' '``, ``'X'`` (White, player 0) and ``'O'`` (Black, player 1).
"""
import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Literal

ROWS, COLUMNS = 6, 7
EMPTY, WHITE, BLACK = ' ', 'X', 'O'

SCHEMA_ID = 'victor-strategic-certificate-v1'
THEOREM_ID = 'cl-bi-ve-black-nonloss-v1'
RULESET_ID = 'connect4-standard-7x6-v1'
RULE_MODEL_ID = 'allis1988-cl-bi-ve-v1'
COMPATIBILITY_MODEL_ID = 'allis1988-s7.4-cl-bi-ve-disjoint-v1'
VERIFIER_ID = 'victor-independent-certificate-verifier-6b4a-v1'
STRATEGY_FORMULATION_ID = 'sigma-r-permissive-spare-v1'

CLAIMEVEN, BASEINVERSE, VERTICAL = 'claimeven', 'baseinverse', 'vertical'
SUPPORTED_RULES = frozenset((CLAIMEVEN, BASEINVERSE, VERTICAL))
# Real Allis rules outside this theorem: unsupported (not malformed) input.
UNSUPPORTED_RULES = frozenset(('aftereven', 'lowinverse', 'highinverse', 'baseclaim',
                               'before', 'specialbefore'))
# Pigeonhole: 22 pairwise disjoint two-square instances would need 44 squares.
MAX_RULES = ROWS * COLUMNS // 2
DEFAULT_WORK_BUDGET = 100_000  # Exceeds the work of every schema-bounded input.

THEOREM_SPEC = MappingProxyType({
    THEOREM_ID: MappingProxyType({
        'statement': (
            'Standard 7x6 Connect 4, White (X) moves first. For a board B with White to '
            'move, if H1-H4 hold for a rule collection R, Black has a strategy under which '
            'White never completes a four, so Black value >= 0 (at least a draw).'),
        'hypotheses': MappingProxyType({
            'H1': 'B obeys gravity, has equal White and Black counts (White to move), '
                  'is not full and contains no four for either player.',
            'H2': 'Each r in R is a Claimeven (two empty vertically adjacent squares, '
                  'upper even), Vertical (same, upper odd) or Baseinverse (two distinct '
                  'current landing squares).',
            'H3': 'The complete square sets of the instances in R are pairwise disjoint.',
            'H4': 'Every group with no Black stone contains a Claimeven upper square or '
                  'both squares of a Baseinverse or Vertical in R.',
        }),
        'strategy_formulation': STRATEGY_FORMULATION_ID,
        'strategy': (
            'Forced reply: White on an active CL/VE lower -> Black takes the square above; '
            'White on an active BI square -> Black takes the other. Otherwise Black may play '
            'ANY landing square that is not the lower square of an active Claimeven. An '
            'even-row landing square always qualifies (L3) and is a valid refinement.'),
        'not_assumed': ('historical reachability', 'Zugzwang control', 'absence of odd threats',
                        'strategy execution traces'),
        'bound': 'black_value_at_least_draw',
        'proof': 'docs/phase6b3-victor-independent-audit.md section 2, as clarified in '
                 'docs/phase6b4a-victor-certificates.md section 5',
        'review_status': 'paper proof; independent second review or formalisation pending',
    }),
})


# --------------------------------------------------------------- geometry

def _name(square: tuple[int, int]) -> str:
    return f'{"abcdefg"[square[0]]}{square[1]}'


def _parse_square(value: object) -> tuple[int, int] | None:
    """Exact ``str`` 'a1'..'g6' -> (column 0..6, row 1..6); anything else -> None."""
    if type(value) is not str or len(value) != 2:
        return None
    column, row = 'abcdefg'.find(value[0]), '123456'.find(value[1])
    return None if column < 0 or row < 0 else (column, row + 1)


def _derive_groups() -> tuple[tuple[tuple[int, int], ...], ...]:
    groups = set()
    for column in range(COLUMNS):
        for row in range(1, ROWS + 1):
            for dc, dr in ((1, 0), (0, 1), (1, 1), (1, -1)):
                cells = tuple((column + i * dc, row + i * dr) for i in range(4))
                if all(0 <= c < COLUMNS and 1 <= r <= ROWS for c, r in cells):
                    groups.add(tuple(sorted(cells)))
    return tuple(sorted(groups))


GROUPS = _derive_groups()
GROUP_NAMES = tuple('-'.join(_name(s) for s in g) for g in GROUPS)
_GROUP_INDEX = MappingProxyType({n: i for i, n in enumerate(GROUP_NAMES)})
if len(GROUPS) != 69:  # pragma: no cover - import-time geometry self-check
    raise RuntimeError('standard board must have 69 groups')


def canonical_group_name(square_names) -> str:
    """Name a group by its four square names, column-major; raises if not a group."""
    squares = [_parse_square(s) for s in square_names]
    if len(squares) != 4 or None in squares:
        raise ValueError('a group needs four square names a1..g6')
    group = '-'.join(_name(s) for s in sorted(squares))
    if group not in _GROUP_INDEX:
        raise ValueError('squares do not form one of the 69 groups')
    return group


# ------------------------------------------------------------ data shapes

@dataclass(frozen=True, slots=True)
class RuleInstance:
    """UNTRUSTED claimed instance; CL/VE ``(lower, upper)``, BI by ascending column."""

    kind: str
    squares: tuple[str, str]

    @property
    def label(self) -> str:
        return f'{self.kind}:{"-".join(self.squares)}'


@dataclass(frozen=True, slots=True)
class StrategicCertificate:
    """UNTRUSTED producer claim. It has no status, outcome or acceptance fields.

    ``slots`` prevents attaching forged attributes. Every field is re-checked;
    version identifiers must equal this verifier's supported values exactly.
    """

    board: tuple[tuple[str, ...], ...]
    rules: tuple[RuleInstance, ...]
    board_digest: str
    player_to_move: int = 0
    defender: int = 1
    replay: tuple[int, ...] | None = None
    assignments: tuple[tuple[str, int], ...] | None = None
    schema: str = SCHEMA_ID
    theorem: str = THEOREM_ID
    ruleset: str = RULESET_ID
    rule_model: str = RULE_MODEL_ID
    compatibility_model: str = COMPATIBILITY_MODEL_ID


class CertificateStatus(str, Enum):
    HYPOTHESES_VERIFIED = 'theorem_hypotheses_verified'
    REJECTED = 'rejected'
    UNSUPPORTED = 'unsupported_version_or_context'
    UNKNOWN = 'unknown_resource_cutoff'


class ReplayStatus(str, Enum):
    NOT_SUPPLIED = 'not_supplied'
    VERIFIED = 'verified_legal_history'
    REJECTED = 'rejected'
    NOT_EVALUATED = 'not_evaluated'


@dataclass(frozen=True, slots=True)
class Finding:
    """``code`` is stable (e.g. ``H3.overlap``); ``detail`` is human-readable."""

    code: str
    detail: str


@dataclass(frozen=True, slots=True)
class RecomputedEvidence:
    """Everything below is derived by this verifier from the bound board and rules."""

    board_digest: str
    white_stones: int
    black_stones: int
    empty_squares: int
    landing_squares: tuple[str, ...]
    group_count: int
    target_groups: tuple[str, ...]
    blocked_group_count: int
    rules: tuple[str, ...]
    rule_coverage: tuple[tuple[str, ...], ...]
    assignment: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class CertificateVerification:
    """A recomputed verdict. Never deserialize one as authority: re-run the verifier."""

    status: CertificateStatus
    findings: tuple[Finding, ...]
    hypotheses: tuple[tuple[str, str], ...]
    replay_status: ReplayStatus
    evidence: RecomputedEvidence | None
    certificate_digest: str | None
    theorem_id: str = field(default=THEOREM_ID, init=False)
    strategy_formulation: str = field(default=STRATEGY_FORMULATION_ID, init=False)
    ruleset_id: str = field(default=RULESET_ID, init=False)
    rule_model_id: str = field(default=RULE_MODEL_ID, init=False)
    compatibility_model_id: str = field(default=COMPATIBILITY_MODEL_ID, init=False)
    schema_id: str = field(default=SCHEMA_ID, init=False)
    verifier_id: str = field(default=VERIFIER_ID, init=False)
    game_theoretic_certification: Literal['not_certified_pending_independent_review'] = field(
        default='not_certified_pending_independent_review', init=False)

    @property
    def provisional_bound(self) -> Literal['black_value_at_least_draw'] | None:
        """Research-only implication of the theorem; gated from public outcome fields."""
        return ('black_value_at_least_draw'
                if self.status is CertificateStatus.HYPOTHESES_VERIFIED else None)

    @property
    def position_provenance(self) -> str:
        if self.status is not CertificateStatus.HYPOTHESES_VERIFIED:
            return 'not_established'
        if self.replay_status is ReplayStatus.VERIFIED:
            return 'verified_legal_history'
        return 'mathematical_position_only'

    @property
    def history_backed(self) -> bool:
        """Required before any proof-backed claim about an ACTUAL game is displayed."""
        return (self.status is CertificateStatus.HYPOTHESES_VERIFIED
                and self.replay_status is ReplayStatus.VERIFIED)


# ------------------------------------------------------------- producer aids

def _encode_board(board: tuple[tuple[str, ...], ...]) -> str:
    return '/'.join(''.join('.' if c == EMPTY else c for c in row) for row in board)


def _digest(board: tuple[tuple[str, ...], ...], player_to_move: int) -> str:
    text = f'{RULESET_ID};to_move={player_to_move};board={_encode_board(board)}'
    return 'sha256:' + hashlib.sha256(text.encode('ascii')).hexdigest()


def board_digest(board, player_to_move: int = 0) -> str:
    """Stable board identity. Shape/cells are checked; legality is NOT asserted."""
    frozen = _freeze_board(board)
    if frozen is None or type(player_to_move) is not int or player_to_move not in (0, 1):
        raise ValueError('board must be a 6x7 list/tuple matrix of " ", "X", "O"')
    return _digest(frozen, player_to_move)


def draft_certificate(board, rules, *, replay=None, assignments=None) -> StrategicCertificate:
    """Producer convenience: freeze inputs and attach the board digest. Still UNTRUSTED."""
    frozen = _freeze_board(board)
    if frozen is None:
        raise ValueError('board must be a 6x7 list/tuple matrix of " ", "X", "O"')
    return StrategicCertificate(
        board=frozen, rules=tuple(rules), board_digest=_digest(frozen, 0),
        replay=None if replay is None else tuple(replay),
        assignments=None if assignments is None else tuple(tuple(a) for a in assignments))


_FIELDS = ('schema', 'theorem', 'ruleset', 'rule_model', 'compatibility_model', 'board',
           'player_to_move', 'defender', 'board_digest', 'rules', 'replay', 'assignments')


def certificate_to_mapping(certificate: StrategicCertificate) -> dict:
    """Bounded shape adapter; JSON-ready only for well-formed scalar inputs.

    This producer aid neither serializes nor formats leaves. Malformed scalar
    claims remain data for ``verify_certificate_mapping`` to reject safely.
    """
    try:
        if type(certificate) is not StrategicCertificate:
            raise _reject('schema.type', 'expected exact StrategicCertificate')
        data = {name: getattr(certificate, name) for name in _FIELDS}
        board, rules, replay, assignments = _copy_containers(data)
    except _Stop as stop:
        finding = stop.findings[0]
        raise ValueError(f'{finding.code}: {finding.detail}') from None
    except AttributeError:
        raise ValueError('schema.malformed: incomplete certificate record') from None
    data['board'] = [list(row) for row in board]
    data['rules'] = [{'kind': kind, 'squares': list(squares)} for kind, squares in rules]
    data['replay'] = None if replay is None else list(replay)
    data['assignments'] = None if assignments is None else [list(a) for a in assignments]
    return data


class _Stop(Exception):
    def __init__(self, status: CertificateStatus, findings: list[Finding]):
        super().__init__(status)
        self.status, self.findings = status, tuple(findings)


def _reject(code: str, detail: str) -> _Stop:
    return _Stop(CertificateStatus.REJECTED, [Finding(code, detail)])


def _is_sequence(value):
    # Tuple membership on types can invoke a hostile metaclass __eq__.
    return type(value) is list or type(value) is tuple


def _sequence(value, limit, code, detail, *, exact=False):
    """Bounded indexed copy of exact built-ins, even if a list grows while copied."""
    if not _is_sequence(value):
        raise _reject(code, detail)
    size = len(value)
    wrong_size = size != limit if exact else size > limit
    if wrong_size:
        raise _reject(code, detail)
    try:
        return tuple(value[i] for i in range(size))
    except IndexError:
        raise _reject('schema.input_changed', 'sequence shrank during snapshot') from None


def _mapping_fields(data, allowed, required):
    """Reject sizes/keys before lookup, hashing, sorting or formatting caller keys."""
    if type(data) is not dict:
        raise _reject('schema.mapping_type', 'expected an exact dict')
    size = len(data)
    if size > len(allowed):
        raise _reject('schema.unknown_field', 'too many mapping fields')
    # Exact dict iteration calls no key hooks. A fixed iteration count bounds
    # copying; concurrent size changes are rejected without formatting objects.
    items, iterator = [], iter(data.items())
    try:
        for _ in range(size):
            key, value = next(iterator)
            if type(key) is not str or len(key) > max(map(len, allowed)) or key not in allowed:
                raise _reject('schema.unknown_field', 'unrecognized mapping field')
            items.append((key, value))
    except (RuntimeError, StopIteration):
        raise _reject('schema.input_changed', 'mapping changed during snapshot') from None
    copied = dict(items)  # Only bounded exact strings are hashed here.
    if any(key not in copied for key in required):
        raise _reject('schema.missing_field', 'required mapping field missing')
    return copied


def _copy_containers(data, *, mapping_rules=False):
    """Fixed-depth, size-bounded copy; never recurse into a scalar or unknown object."""
    board = tuple(_sequence(row, COLUMNS, 'H1.shape', 'board rows need exactly 7 cells',
                            exact=True)
                  for row in _sequence(data['board'], ROWS, 'H1.shape',
                                       'board needs exactly 6 rows', exact=True))
    if not _is_sequence(data['rules']):
        raise _reject('schema.rules', 'rules need an exact list or tuple')
    rules = []
    for index, rule in enumerate(_sequence(data['rules'], MAX_RULES, 'H3.cardinality',
                                          'rules need at most 21 instances')):
        if mapping_rules:
            entry = _mapping_fields(rule, ('kind', 'squares'), ('kind', 'squares'))
            kind, squares = entry['kind'], entry['squares']
        else:
            if type(rule) is not RuleInstance:
                raise _reject('schema.rule_type', f'rule {index} needs an exact RuleInstance')
            kind, squares = rule.kind, rule.squares
        rules.append((kind, _sequence(squares, 2, 'rules.square',
                                      f'rule {index} needs exactly two squares', exact=True)))
    replay, assignments = data.get('replay'), data.get('assignments')
    if replay is not None:
        if not _is_sequence(replay):
            raise _reject('replay.type', 'replay needs an exact list or tuple')
        replay = _sequence(replay, ROWS * COLUMNS, 'replay.too_long',
                           'replay needs at most 42 moves')
    if assignments is not None:
        if not _is_sequence(assignments):
            raise _reject('H4.assignment_type', 'assignments need an exact list or tuple')
        assignments = tuple(_sequence(entry, 2, 'H4.assignment_entry',
                                      f'assignment {index} needs exactly two fields', exact=True)
                            for index, entry in enumerate(_sequence(
                                assignments, len(GROUPS), 'H4.assignment_cardinality',
                                'assignments need at most 69 entries')))
    return board, tuple(rules), replay, assignments


def _snapshot_data(data, *, mapping_rules=False):
    """Copy then validate ALL leaves before constructing records or hashing data.

    Strings and integers have domain-derived size/range bounds too. Snapshotting
    a concurrently edited graph is not atomic; checks and digest use only this
    private immutable result, never the caller graph again.
    """
    board, rules, replay, assignments = _copy_containers(data, mapping_rules=mapping_rules)
    for name in ('schema', 'theorem', 'ruleset', 'rule_model', 'compatibility_model'):
        value = data[name]
        if type(value) is not str or len(value) > 128:
            raise _reject('schema.version_type',
                          f'{name} needs an exact str of at most 128 characters')
    for name, code in (('defender', 'schema.defender'),
                       ('player_to_move', 'schema.player_to_move')):
        value = data[name]
        if type(value) is not int or value not in (0, 1):
            raise _reject(code, f'{name} needs an exact int 0 or 1')
    value = data['board_digest']
    if type(value) is not str or len(value) > 71:
        raise _reject('binding.board_digest',
                      'board_digest needs an exact str of at most 71 characters')
    if any(type(cell) is not str or cell not in (EMPTY, WHITE, BLACK)
           for row in board for cell in row):
        raise _reject('H1.shape', 'board cells need exact strings " ", "X", "O"')
    for index, (kind, squares) in enumerate(rules):
        if type(kind) is not str:
            raise _reject('rules.kind_type', f'rule {index} kind needs an exact str')
        if len(kind) > 32:
            raise _reject('rules.unknown_kind', f'rule {index} kind exceeds 32 characters')
        if any(type(square) is not str or len(square) != 2 for square in squares):
            raise _reject('rules.square', f'rule {index} needs two exact two-character strings')
    if replay is not None:
        for ply, column in enumerate(replay):
            if type(column) is not int or not 0 <= column < COLUMNS:
                raise _reject('replay.column', f'ply {ply} needs an exact int column 0..6')
    if assignments is not None:
        for index, (group, rule_index) in enumerate(assignments):
            if type(group) is not str or len(group) > 11 or type(rule_index) is not int:
                raise _reject('H4.assignment_entry',
                              f'assignment {index} needs a bounded str and exact int')
            if not 0 <= rule_index < MAX_RULES:
                raise _reject('H4.assignment_rule_index', f'assignment {index} needs a rule index 0..20')
    return StrategicCertificate(
        **{name: data[name] for name in _FIELDS
           if name not in ('board', 'rules', 'replay', 'assignments')},
        board=board, rules=tuple(RuleInstance(kind, squares) for kind, squares in rules),
        replay=replay, assignments=assignments)


def certificate_from_mapping(data: object) -> StrategicCertificate:
    """Bounded strict parse; only replay/assignments are optional.

    Raises ``ValueError`` with a stable finding code, without caller callbacks.
    Exact dicts and list/tuple arrays are supported; no authority fields are.
    """
    try:
        required = tuple(k for k in _FIELDS if k not in ('replay', 'assignments'))
        copied = _mapping_fields(data, _FIELDS, required)
        return _snapshot_data(copied, mapping_rules=True)
    except _Stop as stop:
        finding = stop.findings[0]
        raise ValueError(f'{finding.code}: {finding.detail}') from None


def _snapshot_certificate(certificate):
    if type(certificate) is not StrategicCertificate:
        raise _reject('schema.type', 'certificate must be an exact StrategicCertificate')
    data = {name: getattr(certificate, name) for name in _FIELDS}
    return _snapshot_data(data)


# ---------------------------------------------------------------- verifier

class _Cutoff(Exception):
    pass


class _Budget:
    def __init__(self, limit: int):
        self.limit, self.used = limit, 0

    def charge(self, units: int) -> None:
        self.used += units
        if self.used > self.limit:
            raise _Cutoff


def _freeze_board(board) -> tuple[tuple[str, ...], ...] | None:
    """Exact list/tuple 6x7 of exact ``str`` cells; no subclasses, bools or ints."""
    try:
        rows = tuple(_sequence(row, COLUMNS, 'H1.shape', 'invalid board row', exact=True)
                     for row in _sequence(board, ROWS, 'H1.shape', 'invalid board', exact=True))
    except _Stop:
        return None
    if any(type(c) is not str or c not in (EMPTY, WHITE, BLACK) for row in rows for c in row):
        return None
    return rows


def _cell(board, square: tuple[int, int]) -> str:
    column, row = square
    return board[ROWS - row][column]


def _landing(board, column: int) -> tuple[int, int] | None:
    for row in range(1, ROWS + 1):
        if _cell(board, (column, row)) == EMPTY:
            return column, row
    return None


def _winners(board) -> list[str]:
    return [p for p in (WHITE, BLACK) if any(all(_cell(board, s) == p for s in g) for g in GROUPS)]


def _versions(cert) -> list[Finding]:
    expected = (('schema', SCHEMA_ID), ('theorem', THEOREM_ID), ('ruleset', RULESET_ID),
                ('rule_model', RULE_MODEL_ID), ('compatibility_model', COMPATIBILITY_MODEL_ID))
    findings = []
    for name, value in expected:
        claimed = getattr(cert, name)
        if type(claimed) is not str:
            raise _reject('schema.version_type', f'{name} must be a string identifier')
        if claimed != value:
            findings.append(Finding(f'version.{name}', f'unsupported {name} {claimed!r}; '
                                                        f'this verifier supports {value!r}'))
    return findings


def _snapshot_rules(raw) -> tuple[list[tuple], list[Finding], list[Finding]]:
    """Read every claimed instance exactly once; never trust its constructor."""
    if not _is_sequence(raw):
        raise _reject('schema.rules', 'rules must be a list or tuple')
    if len(raw) > MAX_RULES:
        raise _reject('H3.cardinality', f'{len(raw)} two-square instances cannot be pairwise '
                                        f'disjoint on 42 squares (max {MAX_RULES})')
    rules, rejected, unsupported, seen = [], [], [], set()
    for index, rule in enumerate(raw):
        if type(rule) is not RuleInstance:
            raise _reject('schema.rule_type', f'rule {index} is not a RuleInstance')
        kind, squares = rule.kind, rule.squares
        if type(kind) is not str:
            rejected.append(Finding('rules.kind_type', f'rule {index} kind must be an exact str'))
            continue
        if kind in UNSUPPORTED_RULES:
            unsupported.append(Finding('rules.unsupported_kind',
                                       f'rule {index}: {kind} is outside {THEOREM_ID}'))
            continue
        if kind not in SUPPORTED_RULES:
            rejected.append(Finding('rules.unknown_kind',
                                    f'rule {index}: unrecognized kind {kind!r}'))
            continue
        if not _is_sequence(squares) or len(squares) != 2:
            rejected.append(Finding('rules.square', f'rule {index} needs exactly two squares'))
            continue
        parsed = tuple(_parse_square(s) for s in squares)
        if None in parsed:
            rejected.append(Finding('rules.square',
                                    f'rule {index}: invalid square name'))
            continue
        key = (kind, frozenset(parsed))
        if key in seen:
            rejected.append(Finding('rules.duplicate', f'rule {index} duplicates {kind} '
                                    f'{"-".join(_name(s) for s in sorted(parsed))}'))
            continue
        seen.add(key)
        rules.append((kind, parsed[0], parsed[1]))
    return rules, rejected, unsupported


def _label(rule: tuple) -> str:
    return f'{rule[0]}:{_name(rule[1])}-{_name(rule[2])}'


def _h2(board, rules: list[tuple]) -> list[Finding]:
    findings = []
    landings = {s for c in range(COLUMNS) if (s := _landing(board, c)) is not None}
    for kind, a, b in rules:
        label = f'{kind}:{_name(a)}-{_name(b)}'
        if a == b:
            findings.append(Finding('H2.distinct', f'{label} repeats one square'))
            continue
        occupied = [_name(s) for s in (a, b) if _cell(board, s) != EMPTY]
        if occupied:
            findings.append(Finding('H2.occupied', f'{label} uses occupied {occupied}'))
        if kind == BASEINVERSE:
            if a[0] == b[0]:
                findings.append(Finding('H2.same_column', f'{label} squares share a column'))
            elif a[0] > b[0]:
                findings.append(Finding('rules.noncanonical_order',
                                        f'{label} must list squares by ascending column'))
            unplayable = [_name(s) for s in (a, b) if s not in landings]
            if unplayable and not occupied:
                findings.append(Finding('H2.not_playable',
                                        f'{label}: {unplayable} not a current landing square'))
            continue
        if a[0] != b[0] or b[1] != a[1] + 1:
            findings.append(Finding('H2.adjacency', f'{label} must be (lower, upper) directly '
                                                    'above each other'))
            continue
        if (b[1] % 2 == 0) != (kind == CLAIMEVEN):
            findings.append(Finding('H2.parity', f'{label}: upper square {_name(b)} must be '
                                    f'{"even" if kind == CLAIMEVEN else "odd"}'))
    return findings


def _h3(rules: list[tuple]) -> list[Finding]:
    findings = []
    for i, first in enumerate(rules):
        for second in rules[i + 1:]:
            shared = {first[1], first[2]} & {second[1], second[2]}
            if shared:
                findings.append(Finding('H3.overlap', f'{_label(first)} and {_label(second)} '
                                        f'share {sorted(_name(s) for s in shared)}'))
    return findings


def _coverage_squares(rule: tuple) -> frozenset:
    return frozenset((rule[2],)) if rule[0] == CLAIMEVEN else frozenset(rule[1:])


def _h4(board, rules: list[tuple], raw_assignments, budget: _Budget):
    targets = tuple(i for i, g in enumerate(GROUPS) if all(_cell(board, s) != BLACK for s in g))
    budget.charge(len(targets) * max(1, len(rules)))
    needs = [_coverage_squares(r) for r in rules]
    covered_by = {t: tuple(j for j, need in enumerate(needs) if need <= set(GROUPS[t]))
                  for t in targets}
    findings = [Finding('H4.uncovered', f'White group {GROUP_NAMES[t]} is not covered')
                for t in targets if not covered_by[t]]
    if raw_assignments is not None:
        findings += _assignments(raw_assignments, targets, covered_by, len(rules), budget)
    return targets, covered_by, findings


def _assignments(raw, targets, covered_by, rule_count, budget) -> list[Finding]:
    """Optional producer assignments must match recomputed coverage EXACTLY."""
    if not _is_sequence(raw):
        return [Finding('H4.assignment_type', 'assignments must be a list or tuple')]
    if len(raw) > len(GROUPS):
        return [Finding('H4.assignment_cardinality', f'{len(raw)} assignments exceed 69 groups')]
    budget.charge(len(raw))
    findings, assigned, target_set = [], set(), set(targets)
    for entry in raw:
        if (not _is_sequence(entry) or len(entry) != 2 or type(entry[0]) is not str
                or type(entry[1]) is not int):
            findings.append(Finding('H4.assignment_entry',
                                    'assignment needs an exact group string and int index'))
            continue
        group, index = entry
        if group not in _GROUP_INDEX:
            findings.append(Finding('H4.assignment_group', f'{group!r} is not a canonical group'))
            continue
        g = _GROUP_INDEX[group]
        if g not in target_set:
            findings.append(Finding('H4.assignment_non_target', f'{group} contains a Black stone'))
        elif g in assigned:
            findings.append(Finding('H4.assignment_duplicate', f'{group} is assigned twice'))
        elif not 0 <= index < rule_count:
            findings.append(Finding('H4.assignment_rule_index', f'{group} -> missing rule {index}'))
        elif index not in covered_by[g]:
            findings.append(Finding('H4.assignment_incorrect',
                                    f'rule {index} does not cover {group}'))
        assigned.add(g)
    missing = [GROUP_NAMES[t] for t in targets if t not in assigned]
    if missing:
        findings.append(Finding('H4.assignment_incomplete', f'unassigned targets {missing}'))
    return findings


def _replay(board, raw, budget: _Budget) -> list[Finding]:
    """Replay from empty with gravity, alternation and terminal stopping; bind exactly."""
    if not _is_sequence(raw):
        return [Finding('replay.type', 'replay must be a list or tuple of columns')]
    if len(raw) > ROWS * COLUMNS:
        return [Finding('replay.too_long', f'{len(raw)} moves exceed 42 squares')]
    budget.charge(len(raw) * 16)
    grid = [[EMPTY] * COLUMNS for _ in range(ROWS)]
    over = False
    for ply, column in enumerate(raw):
        if type(column) is not int or not 0 <= column < COLUMNS:
            return [Finding('replay.column', f'ply {ply}: expected an exact int column 0..6')]
        if over:
            return [Finding('replay.move_after_terminal', f'ply {ply} follows a completed four')]
        square = _landing(grid, column)
        if square is None:
            return [Finding('replay.full_column', f'ply {ply}: column {column} is full')]
        stone = WHITE if ply % 2 == 0 else BLACK
        grid[ROWS - square[1]][column] = stone
        over = any(square in g and all(_cell(grid, s) == stone for s in g) for g in GROUPS)
    frozen = tuple(tuple(row) for row in grid)
    if frozen != board:
        return [Finding('replay.board_mismatch', 'replayed board differs from the certified board')]
    if over or len(raw) % 2:
        return [Finding('replay.terminal_or_turn',
                        'replay must end nonterminal with White to move')]
    return []


def _plain(value: object) -> object:
    if _is_sequence(value):
        return [_plain(v) for v in value]
    if type(value) is int or type(value) is str or value is None:
        return value
    raise _reject('schema.scalar', 'unsupported digest scalar')


def _certificate_digest(cert, board, rules) -> str:
    payload = {
        'schema': cert.schema, 'theorem': cert.theorem, 'ruleset': cert.ruleset,
        'rule_model': cert.rule_model, 'compatibility_model': cert.compatibility_model,
        'board': _encode_board(board), 'player_to_move': cert.player_to_move,
        'defender': cert.defender, 'rules': [_label(r) for r in rules],
        'replay': _plain(cert.replay), 'assignments': _plain(cert.assignments),
    }
    text = json.dumps(payload, sort_keys=True, separators=(',', ':'))
    return 'sha256:' + hashlib.sha256(text.encode('utf-8')).hexdigest()


def verify_certificate(certificate: object, *,
                       work_budget: int = DEFAULT_WORK_BUDGET) -> CertificateVerification:
    """Independently verify H1-H4 (and any replay) of an UNTRUSTED certificate.

    Bounded structural/type snapshotting precedes all work charges and hashing.
    Subsequent stages run in order: versions, context, board digest
    binding, the remaining H1 predicates, rule parsing, H2, H3, H4 (with any
    assignments), then replay. The first stage with findings
    decides the status, reporting all of that stage's findings. Replay structure
    is checked at ingress; its game predicates run only after H1-H4 pass. A
    well-formed bad replay rejects while recording that H1-H4 held.
    """
    if type(work_budget) is not int or work_budget < 0:
        raise ValueError('work_budget must be a nonnegative integer')
    budget = _Budget(work_budget)
    states = {h: 'not_evaluated' for h in ('H1', 'H2', 'H3', 'H4')}
    replay_status = ReplayStatus.NOT_EVALUATED
    digest = None

    def result(status, findings=(), evidence=None):
        return CertificateVerification(status, tuple(findings), tuple(states.items()),
                                       replay_status, evidence, digest)

    try:
        certificate = _snapshot_certificate(certificate)
        unsupported = _versions(certificate)
        if unsupported:
            raise _Stop(CertificateStatus.UNSUPPORTED, unsupported)
        defender, to_move = certificate.defender, certificate.player_to_move
        if type(defender) is not int or defender not in (0, 1):
            raise _reject('schema.defender', 'defender must be the exact int 1 (Black)')
        if defender != 1:
            raise _Stop(CertificateStatus.UNSUPPORTED, [Finding(
                'context.defender', 'White strategic contexts are outside this theorem')])
        if type(to_move) is not int or to_move not in (0, 1):
            raise _reject('schema.player_to_move', 'player_to_move must be the exact int 0 or 1')

        # H1 on a private snapshot; the caller's containers are never consulted again.
        budget.charge(len(GROUPS))
        board = certificate.board
        # Bind the exact cells and side to move before interpreting them any further.
        computed = _digest(board, to_move)
        if type(certificate.board_digest) is not str or certificate.board_digest != computed:
            raise _reject('binding.board_digest', 'board_digest does not bind this exact board '
                                                  'and side to move')
        h1 = [Finding('H1.gravity', f'column {"abcdefg"[c]} has a stone above an empty square')
              for c in range(COLUMNS)
              if any(_cell(board, (c, r)) == EMPTY and _cell(board, (c, r + 1)) != EMPTY
                     for r in range(1, ROWS))]
        whites = sum(row.count(WHITE) for row in board)
        blacks = sum(row.count(BLACK) for row in board)
        if whites - blacks not in (0, 1):
            h1.append(Finding('H1.counts', f'{whites} White and {blacks} Black stones'))
        if h1:
            states['H1'] = 'failed'
            raise _Stop(CertificateStatus.REJECTED, h1)
        if whites - blacks != to_move:
            states['H1'] = 'failed'
            raise _reject('H1.turn',
                          f'player_to_move {to_move} contradicts counts {whites}/{blacks}')
        if to_move == 1:
            states['H1'] = 'failed'
            raise _Stop(CertificateStatus.UNSUPPORTED, [Finding(
                'context.black_to_move', 'theorem requires White to move')])
        winners = _winners(board)
        if winners:
            states['H1'] = 'failed'
            code = 'H1.contradictory_winners' if len(winners) == 2 else 'H1.terminal_win'
            raise _reject(code, f'board already contains a four for {winners}')
        if whites + blacks == ROWS * COLUMNS:
            states['H1'] = 'failed'
            raise _reject('H1.terminal_full', 'board is full')
        states['H1'] = 'verified'

        rules, rejected, unsupported = _snapshot_rules(certificate.rules)
        if rejected:
            states['H2'] = 'failed'
            raise _Stop(CertificateStatus.REJECTED, rejected)
        if unsupported:
            raise _Stop(CertificateStatus.UNSUPPORTED, unsupported)
        digest = _certificate_digest(certificate, board, rules)
        budget.charge(len(rules))
        h2 = _h2(board, rules)
        if h2:
            states['H2'] = 'failed'
            raise _Stop(CertificateStatus.REJECTED, h2)
        states['H2'] = 'verified'
        budget.charge(len(rules) * len(rules))
        h3 = _h3(rules)
        if h3:
            states['H3'] = 'failed'
            raise _Stop(CertificateStatus.REJECTED, h3)
        states['H3'] = 'verified'
        targets, covered_by, h4 = _h4(board, rules, certificate.assignments, budget)
        if h4:
            states['H4'] = 'failed'
            raise _Stop(CertificateStatus.REJECTED, h4)
        states['H4'] = 'verified'

        replay_status = ReplayStatus.NOT_SUPPLIED
        if certificate.replay is not None:
            replay_findings = _replay(board, certificate.replay, budget)
            if replay_findings:
                replay_status = ReplayStatus.REJECTED
                raise _Stop(CertificateStatus.REJECTED, replay_findings)
            replay_status = ReplayStatus.VERIFIED
    except _Stop as stop:
        if stop.findings[0].code == 'H1.shape':
            states['H1'] = 'failed'
        if stop.findings[0].code.startswith('replay.'):
            replay_status = ReplayStatus.REJECTED
        return result(stop.status, stop.findings)
    except _Cutoff:
        return result(CertificateStatus.UNKNOWN, (Finding(
            'resource.work_budget', f'work budget {work_budget} exhausted; no verdict'),))
    except (TypeError, ValueError, AttributeError, IndexError, KeyError):
        return result(CertificateStatus.REJECTED, (Finding(
            'schema.malformed', 'malformed certificate structure'),))

    evidence = RecomputedEvidence(
        board_digest=computed, white_stones=whites, black_stones=blacks,
        empty_squares=ROWS * COLUMNS - whites - blacks,
        landing_squares=tuple(_name(s) for c in range(COLUMNS)
                              if (s := _landing(board, c)) is not None),
        group_count=len(GROUPS), target_groups=tuple(GROUP_NAMES[t] for t in targets),
        blocked_group_count=len(GROUPS) - len(targets), rules=tuple(_label(r) for r in rules),
        rule_coverage=tuple(tuple(GROUP_NAMES[t] for t in targets if j in covered_by[t])
                            for j in range(len(rules))),
        assignment=tuple((GROUP_NAMES[t], covered_by[t][0]) for t in targets))
    return result(CertificateStatus.HYPOTHESES_VERIFIED, (), evidence)


def verify_certificate_mapping(data: object, *,
                               work_budget: int = DEFAULT_WORK_BUDGET) -> CertificateVerification:
    """Parse untrusted JSON-like input strictly, then verify; parse errors reject."""
    if type(work_budget) is not int or work_budget < 0:
        raise ValueError('work_budget must be a nonnegative integer')
    try:
        certificate = certificate_from_mapping(data)
    except (TypeError, ValueError) as exc:
        code, _, detail = str(exc).partition(': ')
        code = code if code.startswith(('schema.', 'H1.', 'H3.', 'H4.', 'rules.',
                                        'replay.', 'binding.')) else 'schema.malformed'
        return CertificateVerification(
            CertificateStatus.REJECTED, (Finding(code, detail or str(exc)),),
            tuple((h, 'not_evaluated') for h in ('H1', 'H2', 'H3', 'H4')),
            ReplayStatus.NOT_EVALUATED, None, None)
    return verify_certificate(certificate, work_budget=work_budget)
