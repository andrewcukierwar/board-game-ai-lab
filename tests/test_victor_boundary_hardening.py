"""Bounded ingress and complete snapshot regressions; no new game model."""
from dataclasses import replace

import pytest

from games.connect4.victor import certificate as C, strategy as S
from test_victor_assurance_audit import EARLY_HISTORY, certificate


class Hostile:
    """No callback on an unsupported Python input is permissible."""
    def boom(self, *args, **kwargs):
        raise AssertionError('untrusted callback executed')

    __repr__ = __str__ = __format__ = __iter__ = __len__ = __getitem__ = boom
    __eq__ = __hash__ = boom


class HostileMeta(type):
    __eq__ = __hash__ = __repr__ = Hostile.boom


class HostileMetaObject(Hostile, metaclass=HostileMeta):
    pass


class HostileStr(str):
    __str__ = __repr__ = __format__ = __eq__ = __hash__ = Hostile.boom


class HostileInt(int):
    __str__ = __repr__ = __format__ = __eq__ = __hash__ = Hostile.boom


class HostileList(list):
    __iter__ = __len__ = __getitem__ = __repr__ = Hostile.boom


class HostileTuple(tuple):
    __iter__ = __len__ = __getitem__ = __repr__ = Hostile.boom


class HostileDict(dict):
    __iter__ = __len__ = __getitem__ = __repr__ = Hostile.boom


def mutable_certificate():
    original = certificate()
    assigned = C.verify_certificate(original).evidence.assignment
    return replace(original, board=[list(row) for row in original.board],
                   rules=[C.RuleInstance(r.kind, list(r.squares)) for r in original.rules],
                   replay=list(EARLY_HISTORY), assignments=[list(a) for a in assigned])


def test_complete_private_snapshot_survives_post_digest_mutations(monkeypatch):
    mutable = mutable_certificate()
    expected = C.verify_certificate(mutable)
    expected_move = S.select_black_move(S.StrategyRequest(mutable, (0,)))
    mutable2 = mutable_certificate()
    digest = C._certificate_digest
    calls = []

    def interleave(snapshot, board, rules):
        assert snapshot is not mutable
        assert type(snapshot.board) is type(snapshot.rules) is tuple
        assert all(type(row) is tuple for row in snapshot.board)
        assert all(type(rule.squares) is tuple for rule in snapshot.rules)
        assert type(snapshot.replay) is type(snapshot.assignments) is tuple
        assert all(type(entry) is tuple for entry in snapshot.assignments)
        bound = digest(snapshot, board, rules)
        mutable.board[5][0] = 'X'
        mutable.rules[0].squares[:] = ['g6', 'g5']
        mutable.rules[:] = []
        mutable.replay[:] = [6]
        mutable.assignments[0][:] = ['forged', 100]
        mutable.assignments[:] = []
        calls.append(bound)
        return bound

    monkeypatch.setattr(C, '_certificate_digest', interleave)
    assert C.verify_certificate(mutable) == expected
    # Strategy maintains its separate copy and re-verifies it independently.
    mutable = mutable2
    assert S.select_black_move(S.StrategyRequest(mutable, (0,))) == expected_move
    assert calls == [expected.certificate_digest] * 2


@pytest.mark.parametrize('field', ['replay', 'assignments'])
def test_invalid_snapshot_cannot_be_repaired_by_post_digest_edit(monkeypatch, field):
    original = certificate()
    valid = (list(EARLY_HISTORY) if field == 'replay'
             else list(C.verify_certificate(original).evidence.assignment))
    mutable = [6] if field == 'replay' else []
    claim = replace(original, **{field: mutable})
    expected = C.verify_certificate(claim)
    digest = C._certificate_digest

    def interleave(*args):
        bound = digest(*args)
        mutable[:] = valid
        return bound

    monkeypatch.setattr(C, '_certificate_digest', interleave)
    assert C.verify_certificate(claim) == expected
    assert expected.status is C.CertificateStatus.REJECTED
    assert C.verify_certificate(claim).history_backed


@pytest.mark.parametrize('entrypoint', ['verifier', 'mapping', 'strategy'])
@pytest.mark.parametrize('case', [
    'board', 'row', 'rules', 'squares', 'replay', 'assignments', 'assignment',
])
def test_oversized_shapes_reject_before_hashing_even_with_zero_budget(monkeypatch, entrypoint, case):
    cert = certificate()
    data = C.certificate_to_mapping(cert)
    changes = {
        'board': ('board', [Hostile()] * 1000),
        'row': ('board', [list(row) for row in cert.board[:-1]] + [[Hostile()] * 1000]),
        'rules': ('rules', [Hostile()] * 1000),
        'squares': ('rules', [C.RuleInstance('claimeven', [Hostile()] * 1000)]),
        'replay': ('replay', [Hostile()] * 1000),
        'assignments': ('assignments', [Hostile()] * 1000),
        'assignment': ('assignments', [[Hostile()] * 1000]),
    }
    field, value = changes[case]
    claim = replace(cert, **{field: value})
    data[field] = [{'kind': 'claimeven', 'squares': [Hostile()] * 1000}] if case == 'squares' else value

    def no_hash(*args):
        raise AssertionError('malformed ingress reached hashing')

    monkeypatch.setattr(C, '_certificate_digest', no_hash)
    monkeypatch.setattr(C, '_digest', no_hash)
    for budget in (0, 69):
        if entrypoint == 'mapping':
            result = C.verify_certificate_mapping(data, work_budget=budget)
        elif entrypoint == 'strategy':
            result = S.select_black_move(S.StrategyRequest(claim, (0,)),
                                         certificate_work_budget=budget)
            assert result.status is S.StrategyStatus.INVALID_CERTIFICATE
            assert result.column is result.certificate_digest is None
            continue
        else:
            result = C.verify_certificate(claim, work_budget=budget)
        assert result.status is C.CertificateStatus.REJECTED
        assert result.certificate_digest is result.provisional_bound is None


@pytest.mark.parametrize('entrypoint', ['verifier', 'mapping', 'strategy'])
@pytest.mark.parametrize('case', [
    'recursive_replay', 'recursive_assignments', 'replay_object', 'assignment_object',
    'rule_object', 'square_object', 'kind_subclass', 'square_subclass', 'cell_subclass',
    'replay_bool', 'replay_int_subclass', 'index_bool', 'index_int_subclass',
    'version_subclass', 'board_list_subclass', 'replay_tuple_subclass',
    'huge_version', 'huge_kind', 'huge_group', 'huge_index',
    'board_meta', 'rules_meta', 'squares_meta', 'replay_meta', 'assignments_meta', 'entry_meta',
])
def test_malformed_leaves_and_recursion_never_invoke_callbacks(entrypoint, case):
    cert = certificate()
    recursive = []
    recursive.append(recursive)
    changes = {
        'board_meta': ('board', HostileMetaObject()),
        'rules_meta': ('rules', HostileMetaObject()),
        'squares_meta': ('rules', [C.RuleInstance('claimeven', HostileMetaObject())]),
        'replay_meta': ('replay', HostileMetaObject()),
        'assignments_meta': ('assignments', HostileMetaObject()),
        'entry_meta': ('assignments', [HostileMetaObject()]),
        'recursive_replay': ('replay', recursive),
        'recursive_assignments': ('assignments', [[recursive, 0]]),
        'replay_object': ('replay', [Hostile()]),
        'assignment_object': ('assignments', [[Hostile(), 0]]),
        'rule_object': ('rules', [Hostile()]),
        'square_object': ('rules', [C.RuleInstance('claimeven', [Hostile(), 'a2'])]),
        'kind_subclass': ('rules', [C.RuleInstance(HostileStr('claimeven'), ('a1', 'a2'))]),
        'square_subclass': ('rules', [C.RuleInstance('claimeven', (HostileStr('a1'), 'a2'))]),
        'cell_subclass': ('board', [list(row) for row in cert.board[:-1]] +
                          [[HostileStr(' ')] + list(cert.board[-1][1:])]),
        'replay_bool': ('replay', [True]),
        'replay_int_subclass': ('replay', [HostileInt(0)]),
        'index_bool': ('assignments', [[C.GROUP_NAMES[0], True]]),
        'index_int_subclass': ('assignments', [[C.GROUP_NAMES[0], HostileInt(0)]]),
        'version_subclass': ('theorem', HostileStr(C.THEOREM_ID)),
        'board_list_subclass': ('board', HostileList(cert.board)),
        'replay_tuple_subclass': ('replay', HostileTuple(EARLY_HISTORY)),
        'huge_version': ('theorem', 'x' * 1000),
        'huge_kind': ('rules', [C.RuleInstance('x' * 1000, ('a1', 'a2'))]),
        'huge_group': ('assignments', [['x' * 1000, 0]]),
        'huge_index': ('assignments', [[C.GROUP_NAMES[0], 1 << 10000]]),
    }
    field, value = changes[case]
    claim = replace(cert, **{field: value})
    if entrypoint == 'mapping':
        data = C.certificate_to_mapping(cert)
        if field == 'rules' and case not in ('rule_object', 'rules_meta'):
            data[field] = [{'kind': r.kind, 'squares': r.squares} for r in value]
        else:
            data[field] = value
        result = C.verify_certificate_mapping(data, work_budget=0)
    elif entrypoint == 'strategy':
        result = S.select_black_move(S.StrategyRequest(claim, (0,)), certificate_work_budget=0)
        assert result.status is S.StrategyStatus.INVALID_CERTIFICATE
        assert result.column is None
        return
    else:
        result = C.verify_certificate(claim, work_budget=0)
    assert result.status is C.CertificateStatus.REJECTED
    assert result.certificate_digest is result.provisional_bound is None


@pytest.mark.parametrize('where', ['outer', 'rule'])
def test_mapping_keys_checked_without_hash_equality_or_repr(where):
    class Key(Hostile):
        armed = False

        def __hash__(self):
            if self.armed:
                return self.boom()
            return 123

    key = Key()
    data = C.certificate_to_mapping(certificate())
    target = data if where == 'outer' else data['rules'][0]
    del target['schema' if where == 'outer' else 'kind']
    target[key] = 'forged'
    key.armed = True
    result = C.verify_certificate_mapping(data)
    assert result.status is C.CertificateStatus.REJECTED
    assert result.findings[0].code == 'schema.unknown_field'


@pytest.mark.parametrize('value', [HostileDict(), Hostile()])
def test_mapping_container_type_checked_before_hooks(value):
    assert C.verify_certificate_mapping(value).status is C.CertificateStatus.REJECTED


def test_mapping_tuple_arrays_and_optional_omissions_still_work():
    cert = mutable_certificate()
    data = C.certificate_to_mapping(cert)
    data['board'] = tuple(tuple(row) for row in data['board'])
    data['rules'] = tuple(dict(kind=r['kind'], squares=tuple(r['squares'])) for r in data['rules'])
    data['replay'] = tuple(data['replay'])
    data['assignments'] = tuple(tuple(a) for a in data['assignments'])
    assert C.verify_certificate_mapping(data) == C.verify_certificate(cert)
    del data['replay'], data['assignments']
    result = C.verify_certificate_mapping(data)
    assert result.status is C.CertificateStatus.HYPOTHESES_VERIFIED
    assert result.position_provenance == 'mathematical_position_only'
    assert C.verify_certificate_mapping(data, work_budget=0).status is C.CertificateStatus.UNKNOWN


def test_shape_adapter_keeps_malformed_leaves_as_data_but_bounds_containers():
    leaf = Hostile()
    data = C.certificate_to_mapping(replace(certificate(), replay=[leaf]))
    assert data['replay'][0] is leaf
    assert C.verify_certificate_mapping(data).status is C.CertificateStatus.REJECTED
    for cert in (replace(certificate(), replay=Hostile()),
                 replace(certificate(), replay=[0] * 43),
                 replace(certificate(), rules=[Hostile()])):
        with pytest.raises(ValueError):
            C.certificate_to_mapping(cert)


@pytest.mark.parametrize('budget', [True, -1, HostileInt(0), Hostile()],
                         ids=['bool', 'negative', 'int-subclass', 'object'])
def test_mapping_invalid_budget_is_a_programming_error_even_for_malformed_data(budget):
    with pytest.raises(ValueError, match='work_budget'):
        C.verify_certificate_mapping(Hostile(), work_budget=budget)


@pytest.mark.parametrize('entrypoint', ['verifier', 'strategy'])
@pytest.mark.parametrize('edit', ['grow', 'shrink'])
def test_list_resize_during_snapshot_is_bounded_and_consistent(monkeypatch, entrypoint, edit):
    mutable = mutable_certificate()
    expected = (C.verify_certificate(mutable) if entrypoint == 'verifier'
                else S.select_black_move(S.StrategyRequest(mutable, (0,))))
    module = C if entrypoint == 'verifier' else S
    edited = []

    def interleave_range(*args):
        # Simulate a thread switch after the replay's length was captured, before
        # indexed copying. No verification predicate or supplied leaf has hooks.
        if args == (len(EARLY_HISTORY),) and not edited:
            if edit == 'grow':
                mutable.replay.extend([Hostile()] * 1000)
            else:
                mutable.replay.clear()
            edited.append(True)
        return range(*args)

    monkeypatch.setattr(module, 'range', interleave_range, raising=False)
    result = (C.verify_certificate(mutable) if entrypoint == 'verifier'
              else S.select_black_move(S.StrategyRequest(mutable, (0,))))
    assert edited == [True]
    if edit == 'grow':
        assert result == expected
    elif entrypoint == 'verifier':
        assert result.status is C.CertificateStatus.REJECTED
        assert result.findings[0].code == 'schema.input_changed'
        assert result.certificate_digest is None
    else:
        assert result.status is S.StrategyStatus.INVALID_CERTIFICATE
        assert result.column is result.certificate_digest is None


def test_dict_resize_during_mapping_snapshot_is_structured_rejection(monkeypatch):
    data = C.certificate_to_mapping(certificate())
    edited = []

    def interleave_next(iterator):
        if not edited:
            del data['schema']
            edited.append(True)
        return next(iterator)

    monkeypatch.setattr(C, 'next', interleave_next, raising=False)
    result = C.verify_certificate_mapping(data)
    assert result.status is C.CertificateStatus.REJECTED
    assert result.findings[0].code == 'schema.input_changed'
    assert result.certificate_digest is None


def test_incomplete_exact_records_reject_with_safe_diagnostics():
    incomplete = C.StrategicCertificate.__new__(C.StrategicCertificate)
    incomplete_rule = C.RuleInstance.__new__(C.RuleInstance)
    for cert in (incomplete, replace(certificate(), rules=[incomplete_rule])):
        result = C.verify_certificate(cert)
        assert result.status is C.CertificateStatus.REJECTED
        assert result.findings[0].code == 'schema.malformed'
        selected = S.select_black_move(S.StrategyRequest(cert, (0,)))
        assert selected.status is S.StrategyStatus.INVALID_CERTIFICATE
        assert selected.column is None
        with pytest.raises(ValueError, match='schema.malformed'):
            C.certificate_to_mapping(cert)
