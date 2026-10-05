"""Phase 4D.3B.1 launch-control primitives (torch-free): journal durability and torn records,
transition validation, time leases, crash charging, game reservations and the single-writer lock.
Fake clocks only; nothing sleeps for budget durations."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import textwrap
import time

import pytest

from games.connect4.alphazero_v2 import launch_control as L

ROOT = Path(__file__).resolve().parents[1]
LIMITS = dict(per_run_training_seconds=100.0, campaign_seconds=1000.0, evaluation_games_ceiling=3)


class Clock:
    def __init__(self, now=0.0):
        self.now = now

    def __call__(self):
        return self.now


def open_journal(directory, **state):
    return L.Journal(Path(directory) / 'journal.jsonl', L.CampaignState(**state))


def new_campaign(directory):
    journal = open_journal(directory)
    journal.append(dict(event='campaign_created', declaration_sha256='d' * 64))
    return journal


def start_attempt(journal, scope=42, clock=None, limits=LIMITS):
    number = len(journal.state.attempts) + 1
    journal.append(dict(event='attempt_start', attempt=number, scope=str(scope)))
    budget = L.Budget(journal, limits, scope=scope, attempt=number, clock=clock or Clock())
    budget.lease(force=True)
    return budget


def reopen(directory, seeds=(42,)):
    """Simulate a restarted process: fold the journal from disk and recover open attempts."""
    journal = open_journal(directory)
    journal.repair()
    L.recover_open_attempts(journal, seeds)
    return journal


# Journal --------------------------------------------------------------------------------------

def test_journal_round_trips_with_checksums_and_sequence(tmp_path):
    journal = new_campaign(tmp_path)
    start_attempt(journal)
    again = open_journal(tmp_path)
    assert again.records == journal.records and again.torn_tail is None
    lines = (tmp_path / 'journal.jsonl').read_text().splitlines()
    assert [json.loads(line)['seq'] for line in lines] == list(range(len(lines)))


def test_torn_trailing_record_is_discarded_preserved_and_recorded(tmp_path):
    journal = new_campaign(tmp_path)
    start_attempt(journal)
    valid = (tmp_path / 'journal.jsonl').read_bytes()
    torn = b'{"seq": 3, "record": {"event": "unit_begin", "unit": "x"'  # crash mid-write: no newline
    (tmp_path / 'journal.jsonl').write_bytes(valid + torn)
    reopened = open_journal(tmp_path)
    assert reopened.torn_tail == torn and len(reopened.records) == len(journal.records)
    with pytest.raises(RuntimeError, match='torn'):
        reopened.append(dict(event='lease', attempt=1, scope='42', total_seconds=1, training_seconds=0))
    record = reopened.repair()
    assert record['event'] == 'torn_tail_discarded' and record['bytes'] == len(torn)
    assert (tmp_path / record['saved_as']).read_bytes() == torn
    assert (tmp_path / 'journal.jsonl').read_bytes().startswith(valid)
    assert open_journal(tmp_path).torn_tail is None


def test_complete_but_unterminated_record_is_torn(tmp_path):
    """A record whose newline never reached disk was never acknowledged by fsync, so it never authorized work."""
    journal = new_campaign(tmp_path)
    start_attempt(journal)
    path = tmp_path / 'journal.jsonl'
    data = path.read_bytes()
    last = data.rstrip(b'\n').rsplit(b'\n', 1)[-1]
    path.write_bytes(data[:-1])  # drop only the final newline
    reopened = open_journal(tmp_path)
    assert reopened.torn_tail == last and len(reopened.records) == len(journal.records) - 1


def test_terminated_record_with_bad_checksum_at_the_tail_is_torn(tmp_path):
    journal = new_campaign(tmp_path)
    start_attempt(journal)
    path = tmp_path / 'journal.jsonl'
    lines = path.read_bytes().splitlines(keepends=True)
    last = json.loads(lines[-1])
    last['record']['total_seconds'] = 1e9  # garbage written in place of a fsynced record
    path.write_bytes(b''.join(lines[:-1]) + json.dumps(last).encode() + b'\n')
    assert open_journal(tmp_path).torn_tail is not None


def test_damage_before_a_valid_record_is_refused(tmp_path):
    journal = new_campaign(tmp_path)
    start_attempt(journal)
    path = tmp_path / 'journal.jsonl'
    lines = path.read_bytes().splitlines(keepends=True)
    tampered = json.loads(lines[1])
    tampered['record']['scope'] = '7'  # edited without a valid checksum
    path.write_bytes(lines[0] + json.dumps(tampered).encode() + b'\n' + b''.join(lines[2:]))
    with pytest.raises(L.JournalCorrupt, match='precede'):
        open_journal(tmp_path)
    path.write_bytes(lines[0] + b''.join(lines[2:]))  # a deleted record breaks the sequence
    with pytest.raises(L.JournalCorrupt, match='sequence'):
        open_journal(tmp_path)


def test_invalid_transitions_are_refused_before_they_become_durable(tmp_path):
    journal = new_campaign(tmp_path)
    start_attempt(journal)
    size = (tmp_path / 'journal.jsonl').stat().st_size
    commit = dict(event='generation_committed', seed=42, attempt=1, files={}, summary=None)
    with pytest.raises(L.JournalCorrupt, match='out of order'):
        journal.append(dict(commit, generation=1))  # generation 0 first
    journal.append(dict(commit, generation=0))
    with pytest.raises(L.JournalCorrupt, match='out of order'):
        journal.append(dict(commit, generation=0))  # never committed twice
    with pytest.raises(L.JournalCorrupt, match='uncommitted'):
        journal.append(dict(event='champion_decision', seed=42, generation=1, previous=0, promote=True))
    with pytest.raises(L.JournalCorrupt, match='not open'):
        journal.append(dict(event='unit_complete', unit='u', attempt=1, evidence={}))
    unit = dict(unit='u', kind='development_arena_game', seed=42, attempt=1)
    journal.append(dict(unit, event='unit_begin'))
    journal.append(dict(unit, event='unit_complete', evidence={'result': 'win'}))
    with pytest.raises(L.JournalCorrupt, match='already complete'):
        journal.append(dict(unit, event='unit_begin'))  # a completed unit is never rerun
    with pytest.raises(L.JournalCorrupt, match='Unknown'):
        journal.append(dict(event='invented'))
    assert open_journal(tmp_path).records == journal.records  # refused records never reached disk
    assert (tmp_path / 'journal.jsonl').stat().st_size > size


def test_duplicate_champion_decision_and_selection_are_refused(tmp_path):
    journal = open_journal(tmp_path, schedule=(1,))
    journal.append(dict(event='campaign_created', declaration_sha256='d' * 64))
    start_attempt(journal)
    for generation in (0, 1):
        if generation:
            journal.append(dict(event='generation_begin', seed=42, generation=1, attempt=1))
        journal.append(dict(event='generation_committed', seed=42, generation=generation, attempt=1, files={},
                            summary=None))
    decision = dict(event='champion_decision', seed=42, generation=1, previous=0, promote=True)
    journal.append(decision)
    with pytest.raises(L.JournalCorrupt, match='duplicate'):
        journal.append(decision)
    with pytest.raises(L.JournalCorrupt, match='differs from the development champion'):
        journal.append(dict(event='seed_selected', seed=42, generation=0))
    journal.append(dict(event='seed_selected', seed=42, generation=1))
    with pytest.raises(L.JournalCorrupt, match='selected twice'):
        journal.append(dict(event='seed_selected', seed=42, generation=1))


# Time leases and crash charging -----------------------------------------------------------------

def test_lease_is_durable_before_time_is_used_and_renews_ahead_of_expiry(tmp_path):
    clock = Clock()
    journal = new_campaign(tmp_path)
    budget = start_attempt(journal, clock=clock)
    assert journal.state.attempt_charge(1) == (L.LEASE_SECONDS, 0.0)
    clock.now = L.LEASE_SECONDS - L.RENEW_BELOW_SECONDS  # exactly RENEW_BELOW remaining: no renewal yet
    budget.check('evaluation')
    assert journal.state.attempt_charge(1)[0] == L.LEASE_SECONDS
    clock.now += 1
    budget.check('evaluation')
    assert journal.state.attempt_charge(1)[0] == clock.now + L.LEASE_SECONDS


def test_hard_crash_is_charged_through_its_last_lease_and_never_resets(tmp_path):
    clock = Clock()
    journal = new_campaign(tmp_path)
    budget = start_attempt(journal, clock=clock)
    budget.begin_training()
    clock.now = 40.0
    budget.check('training')
    # Hard crash: no attempt_end. A restarted process charges the whole durable lease.
    journal = reopen(tmp_path)
    recovered = journal.state.attempts[1]['recovered']
    assert recovered['total_seconds'] == L.LEASE_SECONDS >= 40.0
    assert recovered['training_seconds'] == L.LEASE_SECONDS
    assert journal.state.charged_seconds() == L.LEASE_SECONDS
    # Repeated crashes accumulate; a new attempt starts from the cumulative charge.
    for _ in range(3):
        start_attempt(journal, clock=Clock())
        journal = reopen(tmp_path)
    assert journal.state.charged_seconds() == 4 * L.LEASE_SECONDS
    second = start_attempt(journal, clock=Clock(), limits=dict(LIMITS, campaign_seconds=10_000.0))
    assert second.total_seconds() == 4 * L.LEASE_SECONDS
    with pytest.raises(L.BudgetExhausted, match='per-run'):
        second.check('training')  # 300 s of charged training already exceeds the 100 s per-run limit


def test_clean_end_charges_measured_time_and_downtime_is_not_charged(tmp_path):
    journal = new_campaign(tmp_path)
    clock = Clock(5_000.0)
    budget = start_attempt(journal, clock=clock)
    budget.begin_training()
    clock.now += 30
    budget.end_training()
    clock.now += 70
    budget.settle('stopped: test')
    assert journal.state.attempt_charge(1) == (100.0, 30.0)
    # The next invocation starts after a long outage; only its own active time is added.
    later = Clock(1_000_000.0)
    second = start_attempt(reopen(tmp_path), clock=later)
    later.now += 25
    assert second.total_seconds() == 125.0 and second.training_seconds() == 30.0


def test_settled_training_time_survives_a_crash_after_training(tmp_path):
    clock = Clock()
    journal = new_campaign(tmp_path)
    budget = start_attempt(journal, clock=clock)
    budget.begin_training()
    clock.now = 12.0
    budget.end_training()  # settled lease: exactly 12 s of training
    clock.now = 50.0
    journal = reopen(tmp_path)
    assert journal.state.attempt_charge(1) == (L.LEASE_SECONDS, 12.0)


def test_start_checks_refuse_at_limits_and_completion_rechecks_overruns(tmp_path):
    clock = Clock()
    budget = start_attempt(new_campaign(tmp_path), clock=clock)
    budget.begin_training()
    clock.now = 99.0
    budget.check('training')                       # the last update may start
    clock.now = 101.0                              # ... and overrun the per-run limit
    assert 'per-run' in budget.completion_violation(training=True)
    assert budget.completion_violation() is None   # the campaign limit still holds
    with pytest.raises(L.BudgetExhausted, match='per-run'):
        budget.check('training')
    budget.end_training()
    clock.now = 1000.0
    assert budget.completion_violation() is None   # finishing exactly at the limit is within it
    with pytest.raises(L.BudgetExhausted, match='campaign'):
        budget.check('evaluation')                 # nothing new may start at the limit
    clock.now = 1000.5
    assert 'campaign' in budget.completion_violation()


def test_stop_request_is_cooperative_and_not_budget_exhaustion(tmp_path):
    budget = start_attempt(new_campaign(tmp_path))
    budget.request_stop('signal 15')
    with pytest.raises(L.CampaignStop) as stopped:
        budget.check('evaluation')
    assert not isinstance(stopped.value, L.BudgetExhausted) and budget.last_stop is stopped.value


# Evaluation-game reservations -------------------------------------------------------------------

def begin_game(journal, attempt, unit):
    journal.append(dict(event='unit_begin', unit=unit, kind='final_ladder_game', seed=42, attempt=attempt))


def test_game_reservation_is_charged_before_play_and_survives_a_crash(tmp_path):
    journal = new_campaign(tmp_path)
    budget = start_attempt(journal, scope='final')
    for index in range(2):
        budget.require_game_slot()
        begin_game(journal, 1, f'g{index}')
        journal.append(dict(event='unit_complete', unit=f'g{index}', attempt=1, evidence={'result': 'win'}))
    budget.require_game_slot()
    begin_game(journal, 1, 'g2')  # in flight when the process dies
    journal = reopen(tmp_path)
    assert journal.state.games_charged == 3
    abandoned = journal.state.units['g2']['abandoned']
    assert len(abandoned) == 1 and 'hard interruption' in abandoned[0]['reason']
    consumption = journal.state.consumption()
    assert consumption['evaluation_games_charged'] == 3 and consumption['evaluation_games_completed'] == 2
    second = start_attempt(journal, scope='final')
    with pytest.raises(L.BudgetExhausted, match='ceiling'):
        second.require_game_slot()  # the abandoned game consumed the last slot
    second.check('evaluation')       # non-game work is unaffected by an exactly reached ceiling


def test_unit_lifecycle_counts_attempts_completions_and_restarts(tmp_path):
    journal = new_campaign(tmp_path)
    start_attempt(journal, scope='final')
    begin_game(journal, 1, 'g0')
    journal.append(dict(event='unit_abandoned', unit='g0', attempt=1, reason='stopped: signal 2'))
    journal.append(dict(event='attempt_end', attempt=1, scope='final', status='stopped', total_seconds=1,
                        training_seconds=0))
    start_attempt(journal, scope='final')
    begin_game(journal, 2, 'g0')
    journal.append(dict(event='unit_complete', unit='g0', attempt=2, evidence={'result': 'draw'}))
    units = journal.state.consumption()['units']['final_ladder_game']
    assert units == dict(attempted=2, completed=1, abandoned=1, restarted_units=1)
    assert journal.state.evidence('g0') == {'result': 'draw'}


def test_attempt_cannot_end_with_open_units(tmp_path):
    journal = new_campaign(tmp_path)
    start_attempt(journal, scope='final')
    begin_game(journal, 1, 'g0')
    with pytest.raises(L.JournalCorrupt, match='open units'):
        journal.append(dict(event='attempt_end', attempt=1, scope='final', status='x', total_seconds=0,
                            training_seconds=0))


# Views ------------------------------------------------------------------------------------------

def test_views_are_written_once_and_must_match(tmp_path):
    path = tmp_path / 'view.json'
    digest = L.publish_view(path, dict(a=1))
    assert L.publish_view(path, dict(a=1)) == digest
    with pytest.raises(L.InconsistentCampaign):
        L.publish_view(path, dict(a=2))
    with pytest.raises(FileExistsError):
        L.write_once(path, b'x')


# Single writer ----------------------------------------------------------------------------------

def test_second_writer_is_refused_in_process(tmp_path):
    lock = L.CampaignLock(tmp_path)
    with pytest.raises(L.CampaignLocked):
        L.CampaignLock(tmp_path)
    lock.release()
    L.CampaignLock(tmp_path).release()


def test_lock_held_by_another_process_is_refused_and_released_on_death(tmp_path):
    script = textwrap.dedent(f'''
        import sys, time
        from games.connect4.alphazero_v2.launch_control import CampaignLock
        lock = CampaignLock({str(tmp_path)!r})
        print("locked", flush=True)
        time.sleep(120)
    ''')
    process = subprocess.Popen([sys.executable, '-c', script], cwd=ROOT, stdout=subprocess.PIPE, text=True,
                               env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
    try:
        assert process.stdout.readline().strip() == 'locked'
        with pytest.raises(L.CampaignLocked):
            L.CampaignLock(tmp_path)
    finally:
        process.send_signal(signal.SIGKILL)  # hard death releases the flock
        process.wait(timeout=30)
    deadline = time.time() + 10
    while True:
        try:
            L.CampaignLock(tmp_path).release()
            break
        except L.CampaignLocked:
            assert time.time() < deadline
            time.sleep(0.05)


def test_rejected_token_is_listed():
    assert L.REJECTED_DECLARATION_TOKENS == (
        '2741314399741c1b20510b8a2beb42e0938b0aeafc9dd0c487201e57208867d8',)
