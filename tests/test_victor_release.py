"""Focused release regressions for overlapping requests and failure recovery."""
from scripts.validate_victor_release import controlled_failures
import json
from pathlib import Path


def test_research_overlap_failure_retry_and_disabled_session():
    result = controlled_failures()
    assert all(result[key] for key in (
        'busy_retry', 'separate_mcts', 'health', 'failed_state_unchanged',
        'reservation_released', 'lost_response_reconciled', 'disabled_session_safe'))


def test_release_fixture_is_an_unchanged_small_selection_of_frozen_truth():
    fixture = json.loads(Path('tests/fixtures/victor_release.json').read_text())
    sources = ('docs/victor-astra/dev.json', 'docs/victor-second-pass/heldout.json')
    frozen = {}
    for source in sources:
        for row in json.loads(Path(source).read_text())['positions']:
            frozen[tuple(row['history'])] = row['oracle']['move_values']
    assert len(fixture['exact']) == 6 and len(fixture['timed']) == 20
    for row in fixture['exact'] + fixture['timed']:
        assert row['move_values'] == frozen[tuple(row['history'])]
    assert {max(row['move_values'].values()) for row in fixture['exact']} == {-1, 0, 1}
