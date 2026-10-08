"""Phase 6B.4A independent strategic certificate verifier: adversarial tests.

Every verdict below comes from ``games.connect4.victor.certificate``, which
re-derives H1-H4 from raw cells. Fixtures are built from the standalone 6B.3
audit model (no production code), and production is used only as an untrusted
producer or as a sabotage target. A verified status is a finite-hypothesis
result, never a game value; see docs/phase6b4a-victor-certificates.md.
"""
import ast
from dataclasses import fields, replace
import json
from pathlib import Path
from random import Random
import subprocess
import sys

import pytest

import games.connect4.victor as victor
from games.connect4.victor import certificate as C
from games.connect4.victor.certificate import (
    CertificateStatus as S, ReplayStatus, RuleInstance, StrategicCertificate,
    board_digest, certificate_from_mapping, certificate_to_mapping, draft_certificate,
    verify_certificate, verify_certificate_mapping,
)
from games.connect4.victor.certificate_producer import certificate_from_witness
from test_victor_coverage import THESIS_MOVES, play
from test_victor_independent_audit import CLAIMODD, MIXED, OVERLAP, UNREACHABLE, moves, sq
from victor_validation import independent_audit as audit
from victor_validation.spare_policies import explore

ROOT = Path(__file__).resolve().parents[1]
KIND = {'CL': 'claimeven', 'BI': 'baseinverse', 'VE': 'vertical',
        'CLODD': 'claimeven', 'BIFLOAT': 'baseinverse'}  # Relaxed shapes keep a real kind.


def name(square):
    return f'{"abcdefg"[square[0]]}{square[1] + 1}'


def instances(raw_rules):
    """Audit ``(kind, (c, h), (c, h))`` -> certificate rules (BI by ascending column)."""
    out = []
    for kind, a, b in raw_rules:
        if KIND[kind] == 'baseinverse':
            a, b = sorted((a, b))
        out.append(RuleInstance(KIND[kind], (name(a), name(b))))
    return tuple(out)


def board_of(history):
    return tuple(tuple(row) for row in audit.replay(moves(history)).matrix())


def cert_for(history, raw_rules, **kw):
    return draft_certificate(board_of(history), instances(raw_rules), **kw)


def matrix_cert(rows, rules=(), **kw):
    return draft_certificate(tuple(tuple(r) for r in rows), rules, **kw)


def codes(result):
    return tuple(f.code for f in result.findings)


def verdict(result):
    return result.status, codes(result), result.hypotheses, result.replay_status, result.evidence


# Valid fixtures found with the 6B.3 audit generator (seed 4401) and pinned here.
CL_ONLY = ('5522554442434415065233263300006123666011', (('CL', sq('b5'), sq('b6')),))
BI_ONLY = ('6033510011652441443630252611022350644625', (('BI', sq('d6'), sq('f6')),))
VE_ONLY = ('262244444400006615110066115515222563',
           (('VE', sq('d2'), sq('d3')), ('VE', sq('d4'), sq('d5'))))
MIXED3 = ('511322464155553320540623416612',
          (('BI', sq('b6'), sq('c6')), ('CL', sq('d5'), sq('d6')),
           ('CL', sq('e5'), sq('e6')), ('VE', sq('a4'), sq('a5'))))
ALL_BLOCKED = ('4433414466332200334566116122110622005505', ())  # Every group has a Black stone.
# Baseinverse d6-e5 although column e lands on e4 (6B.3 "floating BI").
FLOATING_BI = ('5550660011542250004435113333',
               (('CL', sq('c3'), sq('c4')), ('CL', sq('b5'), sq('b6')),
                ('BIFLOAT', sq('d6'), sq('e5')), ('CL', sq('g3'), sq('g4'))))
VALID = {'cl_only': CL_ONLY, 'bi_only': BI_ONLY, 've_only': VE_ONLY, 'mixed_6b3': MIXED,
         'mixed_all_three': MIXED3, 'all_groups_blocked': ALL_BLOCKED}
EXPECTED_KINDS = {'cl_only': {'CL'}, 'bi_only': {'BI'}, 've_only': {'VE'},
                  'mixed_6b3': {'CL', 'BI', 'VE'}, 'mixed_all_three': {'CL', 'BI', 'VE'},
                  'all_groups_blocked': set()}
FULL_DRAW = ('OOXXOOX', 'OOXXOOX', 'XXOOXXO', 'OOXXOOX', 'XXOOXXO', 'XXOOXXO')


# ------------------------------------------------------------ trust boundary

def test_verifier_module_imports_only_the_standard_library():
    tree = ast.parse((ROOT / 'games/connect4/victor/certificate.py').read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split('.')[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, 'relative import crosses the trust boundary'
            imported.add(node.module.split('.')[0])
    assert imported <= {'hashlib', 'json', 'dataclasses', 'enum', 'types', 'typing'}


def test_verifier_runs_with_engine_grounding_and_victor_blocked():
    path = ROOT / 'games/connect4/victor/certificate.py'
    board = [list(r) for r in board_of(MIXED[0])]
    rules = [(r.kind, r.squares) for r in instances(MIXED[1])]
    code = f'''
import importlib.abc, importlib.util, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('games', 'api', 'numpy', 'torch', 'victor_validation'):
            raise AssertionError('forbidden import: ' + fullname)
sys.meta_path.insert(0, Block())
spec = importlib.util.spec_from_file_location('isolated_certificate', {str(path)!r})
C = importlib.util.module_from_spec(spec)
sys.modules['isolated_certificate'] = C
spec.loader.exec_module(C)
rules = tuple(C.RuleInstance(k, tuple(s)) for k, s in {rules!r})
result = C.verify_certificate(C.draft_certificate({board!r}, rules))
assert result.status is C.CertificateStatus.HYPOTHESES_VERIFIED, result
assert len(C.GROUPS) == 69 and not any(m.split('.')[0] == 'games' for m in sys.modules)
'''
    run = subprocess.run([sys.executable, '-I', '-c', code], capture_output=True, text=True,
                         timeout=30)
    assert run.returncode == 0, run.stderr


def battery():
    """Valid and invalid certificates built WITHOUT production code."""
    certs = [cert_for(h, r) for h, r in VALID.values()]
    certs += [cert_for(OVERLAP[0], OVERLAP[1]), cert_for(CLAIMODD[0], CLAIMODD[1]),
              cert_for(FLOATING_BI[0], FLOATING_BI[1]), cert_for(MIXED3[0], MIXED3[1][1:]),
              cert_for(MIXED[0], MIXED[1], replay=moves(MIXED[0])),
              cert_for(MIXED[0], MIXED[1], replay=moves(MIXED[0])[:-2]),
              matrix_cert(UNREACHABLE, instances(audit.find_cover(
                  audit.Board.from_matrix(UNREACHABLE))[1])),
              matrix_cert(FULL_DRAW)]
    return certs


SABOTAGE = (
    ('games.connect4.grounding.analysis', ('validate_position', 'outcome', '_lines')),
    ('games.connect4.victor.position', ('validate_position', 'outcome')),
    ('games.connect4.victor.rules', ('enumerate_candidates', 'enumerate_claimevens',
                                     'enumerate_baseinverses', 'enumerate_verticals',
                                     '_vertical_pairs')),
    ('games.connect4.victor.compatibility', ('compatible', 'required_constraints',
                                             'checked_candidate')),
    ('games.connect4.victor.coverage', ('search_covering_set', 'black_evaluation_context',
                                        'analyze_candidates', 'compatible')),
    ('games.connect4.victor.evidence', ('analyze_candidates', 'enumerate_candidates')),
    ('games.connect4.victor.verification', ('verify_coverage_witness', '_candidate',
                                            '_opponent_groups')),
    ('games.connect4.victor', ('verify_coverage_witness', 'search_covering_set',
                               'enumerate_candidates', 'analyze_candidates', 'compatible',
                               'black_evaluation_context')),
)


def test_sabotaged_production_cannot_change_any_verdict(monkeypatch):
    witness = search_witness(THESIS_MOVES)
    certs = battery() + [certificate_from_witness(witness, replay=tuple(THESIS_MOVES))]
    before = [verdict(verify_certificate(c)) for c in certs]
    assert {v[0] for v in before} == {S.HYPOTHESES_VERIFIED, S.REJECTED}

    def boom(*args, **kwargs):
        raise AssertionError('certificate verifier called production code')

    import importlib
    for module, names in SABOTAGE:
        mod = importlib.import_module(module)
        for attr in names:
            monkeypatch.setattr(mod, attr, boom)
    monkeypatch.setattr(victor.Position, 'from_board', classmethod(boom))
    monkeypatch.setattr(victor.Position, '__post_init__', boom)
    monkeypatch.setattr('games.connect4.grounding.analysis.GROUPS', ())
    monkeypatch.setattr('games.connect4.victor.geometry.ALL_GROUPS', ())
    after = [verdict(verify_certificate(c)) for c in certs]
    assert after == before


def search_witness(history):
    result = victor.search_covering_set(play(history))
    assert result.status == victor.SearchStatus.FOUND
    return result.witness


# ------------------------------------------------------------ valid certificates

@pytest.mark.parametrize('label', sorted(VALID))
def test_valid_certificates_verify_all_hypotheses(label):
    history, raw = VALID[label]
    board = audit.replay(moves(history))
    assert audit.check_hypotheses(board, raw) == ()  # Independent 6B.3 agreement.
    result = verify_certificate(cert_for(history, raw))
    assert result.status is S.HYPOTHESES_VERIFIED and result.findings == ()
    assert result.hypotheses == tuple((h, 'verified') for h in ('H1', 'H2', 'H3', 'H4'))
    assert result.provisional_bound == 'black_value_at_least_draw'
    assert result.game_theoretic_certification == 'not_certified_pending_independent_review'
    assert result.replay_status is ReplayStatus.NOT_SUPPLIED
    assert result.position_provenance == 'mathematical_position_only' and not result.history_backed
    ev = result.evidence
    assert ev.group_count == 69 and ev.empty_squares == board.empty_count
    assert len(ev.target_groups) == len(audit.targets(board))
    assert ev.blocked_group_count == 69 - len(ev.target_groups)
    assert ev.rules == tuple(r.label for r in instances(raw))
    assert {g for g, _ in ev.assignment} == set(ev.target_groups)
    assert {r[0] for r in raw} == EXPECTED_KINDS[label]


def test_empty_rule_set_verifies_only_when_every_group_is_blocked():
    result = verify_certificate(cert_for(*ALL_BLOCKED))
    assert result.status is S.HYPOTHESES_VERIFIED
    assert result.evidence.target_groups == () and result.evidence.blocked_group_count == 69
    empty = verify_certificate(cert_for(MIXED3[0], ()))
    assert empty.status is S.REJECTED and set(codes(empty)) == {'H4.uncovered'}


def test_production_witness_formats_into_a_verifiable_certificate_with_replay():
    # Diagram 6.1 reconstruction: 34 empties, far beyond exact-solver range.
    cert = certificate_from_witness(search_witness(THESIS_MOVES), replay=tuple(THESIS_MOVES))
    result = verify_certificate(cert)
    assert result.status is S.HYPOTHESES_VERIFIED
    assert result.replay_status is ReplayStatus.VERIFIED and result.history_backed
    assert result.position_provenance == 'verified_legal_history'
    assert result.evidence.empty_squares == 34 and len(cert.assignments) == 40


def test_result_binds_every_version_and_a_stable_certificate_digest():
    cert = cert_for(*MIXED3)
    first, second = verify_certificate(cert), verify_certificate(cert_for(*MIXED3))
    assert first == second and first.certificate_digest.startswith('sha256:')
    assert (first.theorem_id, first.ruleset_id, first.rule_model_id, first.compatibility_model_id,
            first.schema_id, first.verifier_id, first.strategy_formulation) == (
        C.THEOREM_ID, C.RULESET_ID, C.RULE_MODEL_ID, C.COMPATIBILITY_MODEL_ID, C.SCHEMA_ID,
        C.VERIFIER_ID, C.STRATEGY_FORMULATION_ID)
    assert first.evidence.board_digest == cert.board_digest == board_digest(cert.board)
    reordered = replace(cert, rules=cert.rules[::-1])
    assert verify_certificate(reordered).certificate_digest != first.certificate_digest
    spec = C.THEOREM_SPEC[C.THEOREM_ID]
    assert set(C.THEOREM_SPEC) == {'cl-bi-ve-black-nonloss-v1'}
    assert set(spec['hypotheses']) == {'H1', 'H2', 'H3', 'H4'}
    assert spec['strategy_formulation'] == C.STRATEGY_FORMULATION_ID
    assert spec['bound'] == 'black_value_at_least_draw'
    assert 'historical reachability' in spec['not_assumed']


# --------------------------------------------------------------------- H1

def H1(rows, to_move=0):
    frozen = tuple(tuple(r) for r in rows)
    return StrategicCertificate(frozen, (), board_digest(frozen, to_move), player_to_move=to_move)


EMPTY_ROWS = ('       ',) * 6


def rows_with(bottom, second='       '):
    return ('       ',) * 4 + (second, bottom)


@pytest.mark.parametrize('rows,to_move,status,code', [
    (rows_with('XXO    '), 0, S.REJECTED, 'H1.turn'),             # White moved twice.
    (rows_with('XO     '), 1, S.REJECTED, 'H1.turn'),             # Equal counts, Black claimed.
    (rows_with('XXO    '), 1, S.UNSUPPORTED, 'context.black_to_move'),
    (rows_with('XXXO   '), 0, S.REJECTED, 'H1.counts'),
    (rows_with('OO     '), 0, S.REJECTED, 'H1.counts'),
    (rows_with('O      ', ' X     '), 0, S.REJECTED, 'H1.gravity'),
    (('       ',) * 2 + ('X      ', 'XO     ', 'XO     ', 'XOO    '), 0, S.REJECTED,
     'H1.terminal_win'),  # White a1-a4 with equal counts.
    (('       ',) * 2 + (' O     ', 'XO     ', 'XO     ', 'XOX    '), 0, S.REJECTED,
     'H1.terminal_win'),  # Black b1-b4 only.
    (FULL_DRAW, 0, S.REJECTED, 'H1.terminal_full'),
])
def test_h1_board_rejections(rows, to_move, status, code):
    result = verify_certificate(H1(rows, to_move))
    assert result.status is status and codes(result)[0] == code
    assert dict(result.hypotheses)['H1'] == 'failed'
    assert result.provisional_bound is None and result.evidence is None


def test_contradictory_winners_are_named():
    rows = ('       ',) * 2 + ('XO     ',) * 4
    result = verify_certificate(H1(rows))
    assert result.status is S.REJECTED and codes(result) == ('H1.contradictory_winners',)


class Cell(str):
    """A str subclass could override equality; cells must be EXACT str."""


@pytest.mark.parametrize('board', [
    ((' ',) * 7,) * 5 + ((' ',) * 6 + (Cell(' '),),),
    [[' '] * 7] * 5 + [[' '] * 6 + [Cell('X')]],
    EMPTY_ROWS[:5], EMPTY_ROWS + ('       ',), tuple(r + ' ' for r in EMPTY_ROWS),
    'not a board', None, ((' ',) * 7,) * 5 + ((' ',) * 6 + (0,),),
    ((' ',) * 7,) * 5 + ((' ',) * 6 + (True,),), ((' ',) * 7,) * 5 + ((' ',) * 6 + ('x',),),
    ((' ',) * 7,) * 5 + ((' ',) * 6 + ('XX',),), ((' ',) * 7,) * 5 + (' ' * 7,),
])
def test_malformed_boards_are_rejected_not_coerced(board):
    cert = StrategicCertificate(board, (), 'sha256:' + '0' * 64)
    result = verify_certificate(cert)
    assert result.status is S.REJECTED and codes(result) == ('H1.shape',)


@pytest.mark.parametrize('field_name,value,status,code', [
    ('player_to_move', True, S.REJECTED, 'schema.player_to_move'),
    ('player_to_move', 2, S.REJECTED, 'schema.player_to_move'),
    ('defender', True, S.REJECTED, 'schema.defender'),
    ('defender', 0, S.UNSUPPORTED, 'context.defender'),
    ('defender', '1', S.REJECTED, 'schema.defender'),
    ('theorem', 'cl-bi-ve-black-nonloss-v2', S.UNSUPPORTED, 'version.theorem'),
    ('theorem', 'cl-bi-ve-li-hi-black-nonloss-v1', S.UNSUPPORTED, 'version.theorem'),
    ('theorem', '', S.UNSUPPORTED, 'version.theorem'),
    ('theorem', None, S.REJECTED, 'schema.version_type'),
    ('ruleset', 'connect4-popout-7x6-v1', S.UNSUPPORTED, 'version.ruleset'),
    ('rule_model', 'allis1988-all-nine-rules-v1', S.UNSUPPORTED, 'version.rule_model'),
    ('compatibility_model', 'pairwise-any-v1', S.UNSUPPORTED, 'version.compatibility_model'),
    ('schema', 'draft-6a', S.UNSUPPORTED, 'version.schema'),
])
def test_identity_versions_and_context_fail_closed(field_name, value, status, code):
    result = verify_certificate(replace(cert_for(*MIXED3), **{field_name: value}))
    assert result.status is status and codes(result) == (code,)
    assert result.provisional_bound is None


def test_board_digest_binds_exact_cells_and_mutation_cannot_reuse_it():
    cert = cert_for(*MIXED3)
    board = [list(r) for r in cert.board]
    board[5][1], board[5][2] = board[5][2], board[5][1]  # Swap colours of b1 and c1.
    assert board != [list(r) for r in cert.board]
    moved = replace(cert, board=tuple(tuple(r) for r in board))
    assert codes(verify_certificate(moved)) == ('binding.board_digest',)
    assert codes(verify_certificate(replace(cert, board_digest=board_digest(cert.board, 1)))) == (
        'binding.board_digest',)
    assert codes(verify_certificate(replace(cert, board_digest=None))) == ('binding.board_digest',)
    # A producer that recomputes the digest is simply re-checked from scratch.
    rebound = replace(moved, board_digest=board_digest(moved.board))
    assert verify_certificate(rebound).status in (S.HYPOTHESES_VERIFIED, S.REJECTED)


def test_caller_containers_mutated_after_verification_do_not_alter_the_result():
    board = [list(r) for r in board_of(MIXED3[0])]
    cert = StrategicCertificate(board, instances(MIXED3[1]), board_digest(board))
    result = verify_certificate(cert)
    assert result.status is S.HYPOTHESES_VERIFIED
    board[5][1], board[5][2] = board[5][2], board[5][1]  # Caller mutates its list later...
    assert result.evidence.board_digest == cert.board_digest  # ...the verdict is a snapshot,
    assert codes(verify_certificate(cert)) == ('binding.board_digest',)  # and re-check fails.


# --------------------------------------------------------------------- H2

def with_rules(base, rules):
    return replace(cert_for(base[0], base[1]), rules=tuple(rules))


def R(kind, a, b):
    return RuleInstance(kind, (a, b))


@pytest.mark.parametrize('rules,code', [
    ([R('claimeven', 'd6', 'd5')], 'H2.adjacency'),       # Reversed (upper, lower) roles.
    ([R('claimeven', 'd4', 'd6')], 'H2.adjacency'),
    ([R('claimeven', 'd5', 'e6')], 'H2.adjacency'),
    ([R('vertical', 'd5', 'd6')], 'H2.parity'),            # Vertical needs an ODD upper.
    ([R('claimeven', 'a4', 'a5')], 'H2.parity'),           # "Claimodd".
    ([R('claimeven', 'a1', 'a2')], 'H2.occupied'),
    ([R('baseinverse', 'b6', 'b5')], 'H2.occupied'),
    ([R('baseinverse', 'd5', 'e6')], 'H2.not_playable'),   # e6 is above e5's landing square.
    ([R('baseinverse', 'c6', 'b6')], 'rules.noncanonical_order'),
    ([R('vertical', 'a4', 'a4')], 'H2.distinct'),
])
def test_h2_prerequisite_violations(rules, code):
    result = verify_certificate(with_rules(MIXED3, rules))
    assert result.status is S.REJECTED and code in codes(result)
    assert set(c.split('.')[0] for c in codes(result)) <= {'H2', 'rules'}
    assert dict(result.hypotheses) == {'H1': 'verified', 'H2': 'failed',
                                       'H3': 'not_evaluated', 'H4': 'not_evaluated'}


@pytest.mark.parametrize('rule,status,code', [
    (R('claimodd', 'f2', 'f3'), S.REJECTED, 'rules.unknown_kind'),
    (R('lowinverse', 'f2', 'f3'), S.UNSUPPORTED, 'rules.unsupported_kind'),
    (R('aftereven', 'f2', 'f3'), S.UNSUPPORTED, 'rules.unsupported_kind'),
    (R(victor.RuleName.CLAIMEVEN, 'd5', 'd6'), S.REJECTED, 'rules.kind_type'),  # str subclass
    (R('claimeven', 'h1', 'h2'), S.REJECTED, 'rules.square'),
    (R('claimeven', 'a6', 'a7'), S.REJECTED, 'rules.square'),
    (R('claimeven', 'a0', 'a1'), S.REJECTED, 'rules.square'),
    (R('claimeven', 'D5', 'D6'), S.REJECTED, 'rules.square'),
    (R('claimeven', 'd55', 'd6'), S.REJECTED, 'rules.square'),
    (R('claimeven', (3, 5), (3, 6)), S.REJECTED, 'rules.square'),
    (RuleInstance('claimeven', ('d5',)), S.REJECTED, 'rules.square'),
    (RuleInstance('claimeven', ('d4', 'd5', 'd6')), S.REJECTED, 'rules.square'),
])
def test_unrecognised_rules_and_coordinates(rule, status, code):
    result = verify_certificate(with_rules(MIXED3, instances(MIXED3[1]) + (rule,)))
    assert result.status is status and codes(result) == (code,)


def test_rule_objects_must_be_exact_types():
    class Forged(RuleInstance):
        __slots__ = ()
    rules = instances(MIXED3[1])
    forged = Forged(rules[0].kind, rules[0].squares)
    assert codes(verify_certificate(with_rules(MIXED3, (forged,) + rules[1:]))) == (
        'schema.rule_type',)
    assert codes(verify_certificate(replace(cert_for(*MIXED3), rules='claimeven:d5-d6'))) == (
        'schema.rules',)


# --------------------------------------------------------------------- H3

def test_documented_overlap_counterexample_fails_only_h3():
    result = verify_certificate(cert_for(*OVERLAP))
    assert result.status is S.REJECTED and codes(result) == ('H3.overlap',)
    assert 'e3' in result.findings[0].detail
    assert dict(result.hypotheses) == {'H1': 'verified', 'H2': 'verified', 'H3': 'failed',
                                       'H4': 'not_evaluated'}
    assert audit.white_can_force_win(audit.replay(moves(OVERLAP[0])))[0] is True


@pytest.mark.parametrize('extra,code', [
    ('duplicate', 'rules.duplicate'),
    ('bi_on_cl_lower', 'H3.overlap'),  # Allis 5.3 shape: one White move, two demands.
    ('cl_on_ve', 'H3.overlap'),
])
def test_overlaps_and_duplicates(extra, code):
    rules = instances(MIXED3[1])
    added = {'duplicate': rules[1], 'bi_on_cl_lower': R('baseinverse', 'd5', 'g5'),
             'cl_on_ve': R('claimeven', 'a3', 'a4')}[extra]
    landings = audit.replay(moves(MIXED3[0])).landings()
    assert sq('d5') in landings and sq('g5') in landings and sq('a3') in landings
    result = verify_certificate(with_rules(MIXED3, rules + (added,)))
    assert result.status is S.REJECTED and codes(result) == (code,)


def test_pigeonhole_rejects_more_than_21_instances_before_any_work():
    rules = tuple(R('claimeven', f'{c}{r}', f'{c}{r + 1}') for c in 'abcdefg' for r in (1, 3, 5))
    rules += (R('vertical', 'a2', 'a3'),)
    result = verify_certificate(matrix_cert(EMPTY_ROWS, rules))
    assert codes(result) == ('H3.cardinality',)


# --------------------------------------------------------------------- H4

@pytest.mark.parametrize('base', [MIXED, MIXED3, VE_ONLY, CL_ONLY, BI_ONLY])
def test_dropping_a_rule_is_judged_exactly_like_the_audit_model(base):
    board, rejected = audit.replay(moves(base[0])), 0
    for drop in range(len(base[1])):
        raw = base[1][:drop] + base[1][drop + 1:]
        result = verify_certificate(cert_for(base[0], raw))
        if not audit.check_hypotheses(board, raw):  # A redundant rule: still a valid cover.
            assert result.status is S.HYPOTHESES_VERIFIED
            continue
        rejected += 1
        assert result.status is S.REJECTED and set(codes(result)) == {'H4.uncovered'}
        assert dict(result.hypotheses)['H3'] == 'verified'
        kind, a, b = base[1][drop]
        need = {name(b)} if kind == 'CL' else {name(a), name(b)}
        for finding in result.findings:  # Every uncovered group needed the dropped rule.
            assert need <= set(finding.detail.split()[2].split('-'))
    assert rejected >= 1


def test_mixed_cover_contains_redundant_rules_that_may_be_dropped():
    redundant = [d for d in range(4) if verify_certificate(cert_for(
        MIXED3[0], MIXED3[1][:d] + MIXED3[1][d + 1:])).status is S.HYPOTHESES_VERIFIED]
    assert redundant == [0, 2]


def test_all_claimeven_empty_board_fails_coverage_of_odd_rows():
    rules = tuple(R('claimeven', f'{c}{r}', f'{c}{r + 1}') for c in 'abcdefg' for r in (1, 3, 5))
    result = verify_certificate(matrix_cert(EMPTY_ROWS, rules))
    assert result.status is S.REJECTED and set(codes(result)) == {'H4.uncovered'}
    uncovered = {f.detail.split()[2] for f in result.findings}
    assert uncovered == {f'{"abcdefg"[c]}{r}-{"abcdefg"[c + 1]}{r}-{"abcdefg"[c + 2]}{r}-'
                         f'{"abcdefg"[c + 3]}{r}' for c in range(4) for r in (1, 3, 5)}


def assignments_for(base):
    return verify_certificate(cert_for(*base)).evidence.assignment


@pytest.mark.parametrize('mutation,code', [
    ('drop', 'H4.assignment_incomplete'),
    ('duplicate', 'H4.assignment_duplicate'),
    ('wrong_rule', 'H4.assignment_incorrect'),
    ('missing_rule', 'H4.assignment_rule_index'),
    ('bool_index', 'H4.assignment_entry'),
    ('blocked_group', 'H4.assignment_non_target'),
    ('bogus_group', 'H4.assignment_group'),
    ('not_a_list', 'H4.assignment_type'),
])
def test_supplied_assignments_must_match_recomputed_coverage(mutation, code):
    good = assignments_for(MIXED3)
    assert verify_certificate(cert_for(*MIXED3, assignments=good)).status is S.HYPOTHESES_VERIFIED
    target_names = {g for g, _ in good}
    blocked = next(g for g in C.GROUP_NAMES if g not in target_names)
    group, index = good[0]
    wrong = next(j for j in range(4) if j != index and (group, j) not in good)
    bad = {
        'drop': good[1:], 'duplicate': good + good[:1], 'wrong_rule': ((group, wrong),) + good[1:],
        'missing_rule': ((group, 9),) + good[1:], 'bool_index': ((group, True),) + good[1:],
        'blocked_group': good + ((blocked, 0),), 'bogus_group': good + (('a1-a2-a3-a5', 0),),
        'not_a_list': 'all groups covered',
    }[mutation]
    result = verify_certificate(replace(cert_for(*MIXED3), assignments=bad))
    assert result.status is S.REJECTED and code in codes(result)
    assert all(c.startswith('H4.assignment') for c in codes(result))


def test_any_correct_alternative_assignment_is_accepted():
    good = dict(assignments_for(MIXED3))
    ev = verify_certificate(cert_for(*MIXED3)).evidence
    alternative = tuple((g, max(j for j, cov in enumerate(ev.rule_coverage) if g in cov))
                        for g in good)
    assert alternative != tuple(good.items())
    assert verify_certificate(cert_for(*MIXED3, assignments=alternative)).status is \
        S.HYPOTHESES_VERIFIED


# ---------------------------------------- relaxed-hypothesis counterexamples

def test_claimodd_counterexample_is_rejected_under_both_representations():
    as_claimeven = verify_certificate(cert_for(*CLAIMODD))
    assert codes(as_claimeven) == ('H2.parity', 'H2.parity')
    # The same squares ARE valid Verticals, which however cover only groups with both.
    as_vertical = verify_certificate(cert_for(CLAIMODD[0], tuple(('VE', a, b)
                                                                 for _, a, b in CLAIMODD[1])))
    assert as_vertical.status is S.REJECTED and set(codes(as_vertical)) == {'H4.uncovered'}
    assert audit.white_can_force_win(audit.replay(moves(CLAIMODD[0])))[0] is True


def test_floating_baseinverse_counterexample_fails_only_h2_playability():
    result = verify_certificate(cert_for(*FLOATING_BI))
    assert result.status is S.REJECTED and codes(result) == ('H2.not_playable',)
    assert "['e5']" in result.findings[0].detail
    assert audit.white_can_force_win(audit.replay(moves(FLOATING_BI[0])))[0] is True


# ------------------------------------------------------------- forged inputs

FORGED_KEYS = ('status', 'accepted', 'outcome', 'value', 'proven', 'safe', 'verified',
               'zugzwang_controlled', 'outcome_certification', 'target_groups', 'compatible')


def test_mapping_round_trip_is_strict_and_json_stable():
    cert = cert_for(*MIXED3, replay=moves(MIXED3[0]), assignments=assignments_for(MIXED3))
    data = json.loads(json.dumps(certificate_to_mapping(cert)))
    assert certificate_from_mapping(data) == cert
    assert verify_certificate_mapping(data) == verify_certificate(cert)


@pytest.mark.parametrize('key', FORGED_KEYS)
def test_forged_acceptance_or_outcome_fields_are_rejected(key):
    data = certificate_to_mapping(cert_for(*OVERLAP))
    data[key] = True
    result = verify_certificate_mapping(data)
    assert result.status is S.REJECTED and codes(result) == ('schema.unknown_field',)
    assert result.provisional_bound is None


@pytest.mark.parametrize('key', ['theorem', 'ruleset', 'player_to_move', 'defender', 'board'])
def test_mapping_never_defaults_identity_fields(key):
    data = certificate_to_mapping(cert_for(*MIXED3))
    del data[key]
    assert codes(verify_certificate_mapping(data)) == ('schema.missing_field',)


def test_certificates_cannot_carry_extra_authority():
    cert = cert_for(*MIXED3)
    with pytest.raises((AttributeError, TypeError)):
        object.__setattr__(cert, 'status', 'accepted')

    class Accepted(StrategicCertificate):
        __slots__ = ()
    forged = Accepted(*(getattr(cert, f.name) for f in fields(StrategicCertificate)))
    assert codes(verify_certificate(forged)) == ('schema.type',)
    assert codes(verify_certificate(certificate_to_mapping(cert))) == ('schema.type',)
    assert codes(verify_certificate(search_witness(THESIS_MOVES))) == ('schema.type',)


def test_forged_witness_claims_are_not_carried_into_a_certificate():
    witness = search_witness(THESIS_MOVES)
    honest = certificate_from_witness(witness, include_assignments=False)
    lying = replace(witness, evidence=tuple(replace(e, conditional_solved_groups=())
                                            for e in witness.evidence))
    object.__setattr__(lying, 'outcome_certification', 'certified')
    assert certificate_from_witness(lying, include_assignments=False) == honest
    # A witness whose rules are wrong yields a certificate that is rejected on H4.
    short = replace(witness, evidence=witness.evidence[1:], assignments=())
    assert set(codes(verify_certificate(certificate_from_witness(short)))) <= {
        'H4.uncovered', 'H4.assignment_incomplete'}


# --------------------------------------------------------------------- replay

@pytest.mark.parametrize('replay,code', [
    (lambda h: h[:-2], 'replay.board_mismatch'),
    (lambda h: h[:-1], 'replay.board_mismatch'),
    (lambda h: h[:-2] + h[-1:] + h[-2:-1], 'replay.board_mismatch'),
    (lambda h: h[:5] + (7,) + h[6:], 'replay.column'),
    (lambda h: h[:5] + (True,) + h[6:], 'replay.column'),
    (lambda h: (0,) * 7 + h, 'replay.full_column'),
    (lambda h: (0, 1, 0, 1, 0, 1, 0, 1) + h, 'replay.move_after_terminal'),
    (lambda h: h + (0,) * 40, 'replay.too_long'),
    (lambda h: ''.join(map(str, h)), 'replay.type'),
    (lambda h: list(h)[:-2] + [None, 3], 'replay.column'),
])
def test_incorrect_replays_reject_but_hypotheses_are_reported_separately(replay, code):
    history = moves(MIXED3[0])
    result = verify_certificate(replace(cert_for(*MIXED3), replay=replay(history)))
    assert result.status is S.REJECTED and codes(result) == (code,)
    assert result.replay_status is ReplayStatus.REJECTED
    # Unsafe replay structure/scalars reject at ingress, before mathematical work.
    state = ('not_evaluated' if code in ('replay.column', 'replay.too_long', 'replay.type')
             else 'verified')
    assert dict(result.hypotheses) == dict.fromkeys(('H1', 'H2', 'H3', 'H4'), state)
    assert not result.history_backed and result.provisional_bound is None


def test_correct_replay_binds_the_exact_board_and_turn():
    history = moves(MIXED3[0])
    result = verify_certificate(cert_for(*MIXED3, replay=history))
    assert result.replay_status is ReplayStatus.VERIFIED and result.history_backed
    # A replay is existential evidence: any legal transposition reaching the board counts.
    target, transpositions = audit.replay(history).matrix(), 0
    for i in range(len(history)):
        for j in range(i + 2, len(history), 2):
            swapped = list(history)
            swapped[i], swapped[j] = swapped[j], swapped[i]
            try:
                same = swapped != list(history) and audit.replay(swapped).matrix() == target
            except ValueError:
                continue
            if same:
                assert verify_certificate(cert_for(*MIXED3, replay=swapped)).history_backed
                transpositions += 1
    assert transpositions > 0


def test_unreachable_basic_valid_board_is_a_mathematical_position_only():
    board = audit.Board.from_matrix(UNREACHABLE)
    assert board.turn == 0 and audit.reachable(board) is False
    raw = audit.find_cover(board)[1]
    result = verify_certificate(matrix_cert(UNREACHABLE, instances(raw)))
    assert result.status is S.HYPOTHESES_VERIFIED
    assert result.position_provenance == 'mathematical_position_only' and not result.history_backed
    # No history exists, and none is invented: a supplied guess is rejected.
    guess = verify_certificate(matrix_cert(UNREACHABLE, instances(raw), replay=moves(MIXED[0])))
    assert codes(guess) == ('replay.board_mismatch',)


# ------------------------------------------------------------- symmetry

def mirror_rules(raw):
    flip = lambda s: (6 - s[0], s[1])
    return tuple((k, flip(a), flip(b)) for k, a, b in raw)


@pytest.mark.parametrize('label', sorted(VALID))
def test_mirrored_certificates_verify_with_distinct_identity(label):
    history, raw = VALID[label]
    mirrored_history = ''.join(str(6 - c) for c in moves(history))
    original = verify_certificate(cert_for(history, raw, replay=moves(history)))
    mirrored = verify_certificate(cert_for(mirrored_history, mirror_rules(raw),
                                           replay=moves(mirrored_history)))
    assert original.status is mirrored.status is S.HYPOTHESES_VERIFIED
    assert mirrored.history_backed
    assert len(mirrored.evidence.target_groups) == len(original.evidence.target_groups)
    if board_of(history) != board_of(mirrored_history):
        assert mirrored.evidence.board_digest != original.evidence.board_digest


def test_unmirrored_rules_on_a_mirrored_board_are_rejected():
    mirrored = ''.join(str(6 - c) for c in moves(MIXED3[0]))
    result = verify_certificate(cert_for(mirrored, MIXED3[1]))
    assert result.status is S.REJECTED and codes(result)[0].split('.')[0] in ('H2', 'H3', 'H4')


# ------------------------------------------------------------- resources

def test_work_budget_cutoff_is_unknown_never_acceptance():
    valid, invalid = cert_for(*MIXED3), cert_for(*OVERLAP)
    assert verify_certificate(valid, work_budget=0).status is S.UNKNOWN
    assert codes(verify_certificate(valid, work_budget=0)) == ('resource.work_budget',)
    for budget in range(0, 400, 7):
        assert verify_certificate(valid, work_budget=budget).status in (
            S.UNKNOWN, S.HYPOTHESES_VERIFIED)
        assert verify_certificate(invalid, work_budget=budget).status in (S.UNKNOWN, S.REJECTED)
    assert verify_certificate(valid, work_budget=0).provisional_bound is None
    for bad in (-1, True, 1.5, None):
        with pytest.raises(ValueError):
            verify_certificate(valid, work_budget=bad)


def test_default_budget_covers_the_largest_schema_bounded_input():
    rules = tuple(R('claimeven', f'{c}{r}', f'{c}{r + 1}') for c in 'abcdefg' for r in (1, 3, 5))
    big = matrix_cert(EMPTY_ROWS, rules, assignments=tuple((g, 0) for g in C.GROUP_NAMES))
    assert verify_certificate(big).status is S.REJECTED
    thesis = certificate_from_witness(search_witness(THESIS_MOVES), replay=tuple(THESIS_MOVES))
    assert verify_certificate(thesis).status is S.HYPOTHESES_VERIFIED


# ----------------------------------------------- status and public separation

def test_statuses_never_claim_outcomes_or_reuse_the_coverage_status():
    values = {s.value for s in S}
    assert values == {'theorem_hypotheses_verified', 'rejected',
                      'unsupported_version_or_context', 'unknown_resource_cutoff'}
    assert victor.VerificationStatus.VERIFIED_UNCERTIFIED.value not in values
    result = verify_certificate(cert_for(*MIXED3))
    for forbidden in ('value', 'outcome', 'winner', 'draw', 'best_move', 'solved',
                      'outcome_certification'):
        assert not hasattr(result, forbidden)
    # The old coverage path is unchanged and still uncertified.
    witness = search_witness(THESIS_MOVES)
    old = victor.verify_coverage_witness(play(THESIS_MOVES), witness)
    assert old.status is victor.VerificationStatus.VERIFIED_UNCERTIFIED
    assert old.outcome_certification == 'uncertified'


def test_certificate_boundary_is_not_wired_into_public_surfaces():
    assert not {'verify_certificate', 'StrategicCertificate'} & set(victor.__all__)
    for folder in ('api', 'games/connect4/agents'):
        for path in (ROOT / folder).rglob('*.py'):
            text = path.read_text()
            assert 'victor.certificate' not in text and 'certificate_producer' not in text, path


# ----------------------------------- differential check vs the 6B.3 audit model

def mutate(rng, board, rules):
    pool = [r for r in audit.enumerate_rules(board, claimodd=True, floating_bi=True)
            if r not in rules]
    out = []
    for i in range(len(rules)):
        out.append(rules[:i] + rules[i + 1:])
    if pool:
        out.append(rules + (rng.choice(pool),))
        if rules:
            i = rng.randrange(len(rules))
            out.append(rules[:i] + (rng.choice(pool),) + rules[i + 1:])
    return out


def test_verifier_agrees_with_independent_audit_checker_on_bounded_sample():
    rng = Random(6404)
    compared = accepted = rejected = 0
    stage_seen = set()
    for empties in (4, 6, 8, 10, 12, 14) * 150:
        history = audit.sample_history(rng, 42 - empties, follow_up=rng.choice((0.7, 0.95)))
        if history is None:
            continue
        board = audit.replay(history)
        if board.turn != 0:
            continue
        text = ''.join(map(str, history))
        for cover in audit.enumerate_covers(board, limit=6, budget=20_000):
            for rules in (cover, *mutate(rng, board, cover)):
                theirs = audit.check_hypotheses(board, rules)
                mine = verify_certificate(cert_for(text, rules))
                compared += 1
                if not theirs:
                    assert mine.status is S.HYPOTHESES_VERIFIED, (text, rules, mine.findings)
                    accepted += 1
                else:
                    assert mine.status is S.REJECTED, (text, rules, theirs)
                    stage = codes(mine)[0].split('.')[0]
                    assert stage in {t.split(':')[0] for t in theirs}, (text, rules, mine, theirs)
                    stage_seen.add(stage)
                    rejected += 1
    assert (compared, accepted, rejected) == (741, 219, 522)  # Pinned seed; bounded sample.
    assert stage_seen == {'H2', 'H3', 'H4'}


def test_production_cover_existence_matches_certificate_verification():
    rng = Random(6405)
    found = 0
    for empties in (6, 8, 10, 12, 14) * 10:
        history = audit.sample_history(rng, 42 - empties, follow_up=0.8)
        if history is None or audit.replay(history).turn != 0:
            continue
        result = victor.search_covering_set(play(list(history)))
        mine = audit.find_cover(audit.replay(history))[0]
        if result.status == victor.SearchStatus.FOUND:
            assert mine == 'found'
            cert = certificate_from_witness(result.witness, replay=history)
            verdict_ = verify_certificate(cert)
            assert verdict_.status is S.HYPOTHESES_VERIFIED and verdict_.history_backed
            found += 1
        else:
            assert result.status == victor.SearchStatus.EXHAUSTIVE_NO_COVER and mine == 'none'
    assert found >= 5


# --------------------------------- strategy wording clarification (section 5)

# CL c5-c6 and CL d3-d4: H1-H4 hold. Black playing the ACTIVE CL lower c5 as a
# "spare" (forbidden by sigma_R) lets White take c6 and win.
CL_LOWER_SPARE = ('550066550544410241221101261344663650',
                  (('CL', sq('c5'), sq('c6')), ('CL', sq('d3'), sq('d4'))))


def test_spare_restriction_is_load_bearing_but_odd_row_spares_are_safe():
    board = audit.replay(moves(CL_LOWER_SPARE[0]))
    rules = CL_LOWER_SPARE[1]
    assert verify_certificate(cert_for(*CL_LOWER_SPARE)).status is S.HYPOTHESES_VERIFIED
    assert audit.white_can_force_win(board)[0] is False  # The theorem's bound holds.
    status, detail, stats = explore(board, rules, 'unrestricted')
    assert status == 'violated' and detail[0] == 'White wins at c6'
    assert stats['invariant_breaks'] > 0
    for policy in ('permissive', 'even_row'):
        assert explore(board, rules, policy)[0] == 'held'


@pytest.mark.parametrize('fixture', [MIXED, MIXED3, VE_ONLY, CL_LOWER_SPARE])
def test_permissive_and_even_row_spare_policies_both_hold(fixture):
    board = audit.replay(moves(fixture[0]))
    permissive = explore(board, fixture[1], 'permissive')
    even = explore(board, fixture[1], 'even_row')
    assert permissive[:2] == even[:2] == ('held', None)
    assert permissive[2]['invariant_breaks'] == even[2]['invariant_breaks'] == 0


def test_permissive_policy_really_exercises_odd_row_spares_on_sampled_covers():
    rng = Random(5501)
    covers = odd = 0
    for empties in (6, 8, 10, 12) * 80:
        history = audit.sample_history(rng, 42 - empties, follow_up=rng.choice((0.5, 0.9)))
        if history is None or audit.replay(history).turn != 0:
            continue
        board = audit.replay(history)
        for rules in audit.enumerate_covers(board, limit=2, budget=20_000):
            permissive = explore(board, rules, 'permissive')
            assert permissive[0] == 'held', (history, rules, permissive[1])
            assert explore(board, rules, 'even_row')[0] == 'held'
            covers += 1
            odd += permissive[2]['odd_row_spares'] > 0
    assert covers >= 20 and odd >= 10
