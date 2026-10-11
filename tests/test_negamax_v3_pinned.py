"""Shallow/squashed Git history must never substitute a mutable engine."""
import subprocess
from pathlib import Path
import pytest
from scripts import negamax_v3_pinned as pinned


def test_exact_git_baseline_and_missing_object_fallback(monkeypatch):
    expected=Path(pinned.COPY).read_bytes()
    assert pinned.baseline_bytes()==expected
    def absent(*args,**kwargs):
        raise subprocess.CalledProcessError(128,args[0])
    monkeypatch.setattr(pinned.subprocess,'check_output',absent)
    assert pinned.baseline_bytes()==expected


def test_bad_available_git_bytes_fail_closed(monkeypatch):
    monkeypatch.setattr(pinned.subprocess,'check_output',lambda *a,**kw:b'wrong')
    with pytest.raises(ValueError,match='immutable'):
        pinned.baseline_bytes()


def test_bad_fallback_bytes_fail_closed(monkeypatch,tmp_path):
    def absent(*args,**kwargs): raise OSError('missing git')
    monkeypatch.setattr(pinned.subprocess,'check_output',absent)
    bad=tmp_path/'bad.py'
    bad.write_text('wrong')
    monkeypatch.setattr(pinned,'COPY',str(bad))
    with pytest.raises(ValueError,match='immutable'):
        pinned.baseline_bytes()
