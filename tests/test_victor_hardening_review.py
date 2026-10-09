"""Phase 6B.5B independent review probes of the 6B.5A boundary hardening."""
import threading
import sys
from dataclasses import replace

import pytest

from games.connect4.victor import certificate as C, strategy as S
from test_victor_assurance_audit import EARLY_HISTORY, certificate


@pytest.mark.parametrize('missing', ['certificate', 'continuation', 'both'])
def test_incomplete_strategy_request_is_a_structured_result(missing):
    # R1: an exact record built without __init__ previously raised AttributeError.
    request = object.__new__(S.StrategyRequest)
    if missing != 'certificate' and missing != 'both':
        object.__setattr__(request, 'certificate', certificate())
    if missing != 'continuation' and missing != 'both':
        object.__setattr__(request, 'continuation', (0,))
    result = S.select_black_move(request)
    assert result.status is S.StrategyStatus.UNSUPPORTED
    assert result.findings[0].code == 'request.type'
    assert result.column is result.certificate_digest is None


@pytest.mark.parametrize('case', ['group', 'version', 'kind', 'square'])
def test_bounded_non_ascii_and_surrogate_strings_never_escape(case):
    cert = certificate()
    assigned = C.verify_certificate(cert).evidence.assignment
    claim = {
        'group': replace(cert, assignments=(('\ud800', 0),) + assigned),
        'version': replace(cert, theorem='\udfff' * 128),
        'kind': replace(cert, rules=cert.rules + (C.RuleInstance('\ud800' * 32, ('g5', 'g6')),)),
        'square': replace(cert, rules=cert.rules + (C.RuleInstance('claimeven', ('\ud800a', 'g6')),)),
    }[case]
    result = C.verify_certificate(claim)
    assert result.status in (C.CertificateStatus.REJECTED, C.CertificateStatus.UNSUPPORTED)
    assert result.provisional_bound is None
    # A digest may exist only after rule parsing; it must then be ASCII-safe JSON.
    assert result.certificate_digest is None or result.certificate_digest.isascii()


@pytest.mark.parametrize('field', ['replay', 'assignments'])
def test_threaded_mutation_yields_only_whole_snapshot_verdicts(field):
    """Real thread switches, not scheduling hooks. Coverage of interleavings is
    probabilistic; the invariant asserted for every observed result is not."""
    base = certificate()
    if field == 'replay':
        valid = tuple(EARLY_HISTORY)
        invalid = valid[:-1] + ((valid[-1] + 1) % 7,)
        slot = len(valid) - 1
    else:
        valid = C.verify_certificate(base).evidence.assignment
        group, index = valid[0]
        invalid = ((group, (index + 1) % len(base.rules)),) + valid[1:]
        slot = 0
    good = C.verify_certificate(replace(base, **{field: valid}))
    bad = C.verify_certificate(replace(base, **{field: invalid}))
    assert good.status is C.CertificateStatus.HYPOTHESES_VERIFIED
    assert bad.status is C.CertificateStatus.REJECTED and bad.certificate_digest
    live = list(valid)
    claim = replace(base, **{field: live})
    stop = threading.Event()

    def flip():
        while not stop.is_set():
            live[slot] = invalid[slot]
            for _ in range(200):
                pass
            live[slot] = valid[slot]
            for _ in range(200):
                pass

    interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    thread = threading.Thread(target=flip)
    thread.start()
    try:
        for i in range(600):
            assert C.verify_certificate(claim) in (good, bad)
            if i % 10 == 0:
                selected = S.select_black_move(S.StrategyRequest(claim, (0,)))
                assert selected.status in (S.StrategyStatus.MOVE_SELECTED,
                                           S.StrategyStatus.INVALID_CERTIFICATE)
                if selected.status is S.StrategyStatus.MOVE_SELECTED:
                    assert selected.certificate_digest == good.certificate_digest
    finally:
        stop.set()
        thread.join()
        sys.setswitchinterval(interval)
