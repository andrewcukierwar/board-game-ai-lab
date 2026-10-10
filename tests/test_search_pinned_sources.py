"""Phase baselines must load identically with or without their Git commits.

CI checks out one commit, and a squash merge can make the phase commits
unreachable. Regression: 391 search tests failed there with ``git show`` errors.
"""
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from games.connect4.agents.negamax_agent import NegamaxAgent
from scripts import pinned_git_source as pinned
from scripts.benchmark_negamax_incremental import load_baseline
from scripts.benchmark_public_agents import position
from scripts.negamax_iterative_variants import load_variant as load_iterative
from scripts.negamax_tt_variants import load_variant as load_tt

PINS = [(commit, path) for commit, paths in pinned.PINNED.items() for path in paths]
EVIDENCE = Path('docs/search-negamax-v2')


@pytest.fixture
def no_phase_commits(monkeypatch):
    """Behave like a depth-1 checkout: every ``git show`` of a phase commit fails."""
    def missing(command, **kwargs):
        raise subprocess.CalledProcessError(128, command)
    monkeypatch.setattr(pinned.subprocess, 'check_output', missing)


@pytest.mark.parametrize('commit,path', PINS)
def test_committed_copy_matches_pin_and_any_available_git_object(commit, path):
    copy = Path(pinned.committed_copy(commit, path)).read_bytes()
    assert hashlib.sha256(copy).hexdigest() == pinned.PINNED[commit][path]
    present = subprocess.run(['git', 'cat-file', '-e', f'{commit}:{path}'],
                             capture_output=True).returncode == 0
    if present:  # Full clones additionally prove the copy is the Git object.
        assert subprocess.check_output(['git', 'show', f'{commit}:{path}']) == copy
    assert pinned.pinned_source(commit, path) == copy


@pytest.mark.parametrize('commit,path', PINS)
def test_fallback_returns_identical_bytes_without_git_objects(commit, path, no_phase_commits):
    data = pinned.pinned_source(commit, path)
    assert hashlib.sha256(data).hexdigest() == pinned.PINNED[commit][path]


def test_pins_are_the_digests_recorded_by_each_phase_manifest():
    incremental = json.loads((EVIDENCE / 'incremental-evaluation/manifest.json').read_text())
    tt = json.loads((EVIDENCE / 'transposition-table/manifest.json').read_text())
    iterative = json.loads((EVIDENCE / 'iterative-deepening/manifest.json').read_text())
    for manifest in (incremental, tt, iterative):
        recorded = pinned.PINNED[manifest['baseline_commit']]
        assert recorded[pinned.AGENT_PATH] == manifest['baseline_sha256']
    deeper = pinned.EVIDENCE + 'deeper-search/'
    assert {deeper + name: value for name, value in incremental['fixture_source_hashes'].items()
            }.items() <= pinned.PINNED[incremental['baseline_commit']].items()
    assert tt['fixture_source_sha256'] == pinned.PINNED[tt['baseline_commit']][
        pinned.EVIDENCE + 'incremental-evaluation/manifest.json']
    assert iterative['fixture_source_sha256'] == pinned.PINNED[iterative['baseline_commit']][
        pinned.EVIDENCE + 'transposition-table/manifest.json']


def test_changed_copy_and_unpinned_paths_fail_closed(tmp_path, monkeypatch, no_phase_commits):
    commit = '2e79b15ecdb1967345a1e66593201f9803c89757'
    with pytest.raises(ValueError, match='No pinned digest'):
        pinned.pinned_source(commit, 'games/connect4/connect4.py')
    with pytest.raises(ValueError, match='No pinned digest'):
        pinned.pinned_source('0' * 40, pinned.AGENT_PATH)
    edited = tmp_path / 'direct-source.py'
    edited.write_bytes(Path(pinned.AGENT_COPIES[commit]).read_bytes() + b'\n# edited\n')
    monkeypatch.setitem(pinned.AGENT_COPIES, commit, str(edited))
    with pytest.raises(ValueError, match='does not match its pinned SHA-256'):
        pinned.pinned_source(commit, pinned.AGENT_PATH)


@pytest.mark.parametrize('history,depth', [([], 4), ([3, 2, 4, 3], 5), ([1, 4, 6, 0, 6], 4)])
def test_every_baseline_loader_runs_without_phase_commits(history, depth, no_phase_commits):
    game = position(history)
    expected = NegamaxAgent(depth).score_moves(game)
    agents = [load_baseline().NegamaxAgent, load_tt('baseline').NegamaxAgent,
              load_tt('packed-entry').NegamaxAgent, load_iterative('direct').NegamaxAgent,
              load_iterative('combined').ExperimentalAgent]
    for agent in agents:
        assert list(agent(depth).score_moves(game).items()) == list(expected.items())
