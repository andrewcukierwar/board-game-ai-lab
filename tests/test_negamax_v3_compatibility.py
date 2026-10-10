"""Only explicitly enumerated loader/metadata edits may preserve old manifests."""
from pathlib import Path
import pytest
from scripts.negamax_v3_compatibility import approved_revision


def test_approved_revision_retains_all_timed_and_gate_code():
    archive=Path('docs/search-negamax-v3/tooling-compatibility/benchmark-original.py.txt')
    old=(archive if archive.exists() else Path('scripts/benchmark_negamax_v3.py')).read_bytes()
    revised=approved_revision('scripts/benchmark_negamax_v3.py',old)
    # Entire approved output is text substitutions outside measured/gate bodies.
    for name in ('decision','memory_run','run','analyze','audit','summarize'):
        import ast
        def body(source):
            tree=ast.parse(source)
            return ast.dump(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name),include_attributes=False)
        assert body(old)==body(revised)


def test_unknown_role_rejected():
    with pytest.raises(ValueError,match='Unapproved'):
        approved_revision('games/connect4/agents/negamax_agent.py',b'')


@pytest.mark.parametrize('tamper',[False,True])
def test_hash_certificate_cannot_authorize_changed_timing_or_gates(monkeypatch,tmp_path,tamper):
    import json
    from scripts import negamax_v3_compatibility as compat
    archive=Path('docs/search-negamax-v3/tooling-compatibility/benchmark-original.py.txt')
    old=(archive if archive.exists() else Path('scripts/benchmark_negamax_v3.py')).read_bytes()
    new=approved_revision('scripts/benchmark_negamax_v3.py',old)
    if tamper:
        assert new.count(b'geo >= 1.05')==1
        new=new.replace(b'geo >= 1.05',b'geo >= 1.00')
    monkeypatch.chdir(tmp_path)
    current=Path('scripts/benchmark_negamax_v3.py')
    current.parent.mkdir()
    current.write_bytes(new)
    compat.ROOT.mkdir(parents=True)
    (compat.ROOT/'original.txt').write_bytes(old)
    certificate=dict(files={str(current):dict(old_sha256=compat.digest(old),new_sha256=compat.digest(new),archive='original.txt')},helpers={})
    (compat.ROOT/'certificate.json').write_text(json.dumps(certificate))
    if tamper:
        with pytest.raises(AssertionError,match='Measured code'):
            compat.verify_source_hash(str(current),compat.digest(old))
    else:
        compat.verify_source_hash(str(current),compat.digest(old))
        # An unregistered revision or altered archive fails closed too.
        with pytest.raises(AssertionError,match='Unknown original'):
            compat.verify_source_hash(str(current),'wrong')
        (compat.ROOT/'original.txt').write_bytes(b'wrong')
        with pytest.raises(AssertionError,match='Corrupt archived'):
            compat.verify_source_hash(str(current),compat.digest(old))
