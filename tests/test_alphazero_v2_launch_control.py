"""Phase 4D.3B.2 fail-closed launch-control primitives (torch-free): the atomic state file and its
transition rules, terminal states, the single-owner lock and the in-process monotonic budget.
Fake clocks only; nothing sleeps for budget durations."""
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import time

import pytest

from games.connect4.alphazero_v2 import launch_control as L

ROOT = Path(__file__).resolve().parents[1]
LIMITS = dict(per_run_training_seconds=100.0, campaign_seconds=1000.0, evaluation_games_ceiling=3)
SEEDS = (42, 314159)
OLD_TOKENS = ('2741314399741c1b20510b8a2beb42e0938b0aeafc9dd0c487201e57208867d8',
              '8a8a52b100c7e32aa5e3e5b05469578084c8e12f37750ba81fee071990ff5e39')
OUTCOME = dict(status='INCOMPLETE', reason='test')


class Clock:
    def __init__(self, now=0.0):
        self.now = now

    def __call__(self):
        return self.now


def created(directory):
    return L.CampaignStateFile.create(directory, declaration_sha256='d' * 64, seeds=SEEDS)


def advance_to_complete(state):
    for name in L.state_sequence(SEEDS)[1:-1]:
        state.transition(name)
    state.transition(L.COMPLETE, outcome=dict(status='COMPLETE'))


# State file and transitions -----------------------------------------------------------------------

def test_state_sequence_is_the_simplified_lifecycle():
    assert L.state_sequence(SEEDS) == ('CREATED', 'RUNNING_SEED_42', 'RUNNING_SEED_314159',
                                       'DEVELOPMENT_SELECTION_COMPLETE', 'RUNNING_FINAL_EVALUATION', 'COMPLETE')
    assert L.TERMINAL_STATES == ('COMPLETE', 'INCOMPLETE')


def test_transitions_are_durable_and_only_follow_the_line(tmp_path):
    state = created(tmp_path)
    with pytest.raises(L.InvalidTransition):
        state.transition('RUNNING_SEED_314159')  # skipping a seed
    with pytest.raises(L.InvalidTransition):
        state.transition('COMPLETE', outcome=dict(status='COMPLETE'))
    state.transition('RUNNING_SEED_42', counters=dict(x=1))
    reread = L.CampaignStateFile.load(tmp_path)
    assert reread.state == 'RUNNING_SEED_42' and reread.document['counters'] == dict(x=1)
    assert [h['state'] for h in reread.document['history']] == ['CREATED', 'RUNNING_SEED_42']
    with pytest.raises(L.InvalidTransition):
        state.transition('RUNNING_SEED_42')  # no repeat
    with pytest.raises(L.InvalidTransition):
        state.transition('INCOMPLETE')  # a terminal state needs an outcome
    assert sorted(p.name for p in tmp_path.iterdir()) == ['state.json']  # no temporary leftovers


@pytest.mark.parametrize('terminal', ['COMPLETE', 'INCOMPLETE'])
def test_terminal_states_never_reopen(tmp_path, terminal):
    state = created(tmp_path)
    if terminal == 'COMPLETE':
        advance_to_complete(state)
    else:
        state.transition('RUNNING_SEED_42')
        state.transition(L.INCOMPLETE, outcome=OUTCOME)
    before = state.path.read_bytes()
    for target in L.state_sequence(SEEDS) + (L.INCOMPLETE,):
        with pytest.raises(L.InvalidTransition, match='terminal'):
            state.transition(target, outcome=OUTCOME if target in L.TERMINAL_STATES else None)
    with pytest.raises(L.InvalidTransition, match='terminal'):
        state.record_counters(dict(more=1))
    assert state.path.read_bytes() == before
    reread = L.CampaignStateFile.load(tmp_path)
    assert reread.terminal and reread.state == terminal


def test_state_validation_rejects_forged_or_damaged_documents(tmp_path):
    state = created(tmp_path)
    state.transition('RUNNING_SEED_42')
    good = json.loads(state.path.read_text())
    forged = [
        dict(good, state='RUNNING_FINAL_EVALUATION'),  # history does not end at the state
        dict(good, history=good['history'][:1] + [dict(state='RUNNING_SEED_314159')], state='RUNNING_SEED_314159'),
        dict(good, outcome=dict(status='COMPLETE')),     # outcome without a terminal state
        dict(good, format='other'),
        {k: v for k, v in good.items() if k != 'owner'},
    ]
    for document in forged:
        state.path.write_text(json.dumps(document))
        with pytest.raises(L.StateCorrupt):
            L.CampaignStateFile.load(tmp_path)
    state.path.write_bytes(b'{"format": "connect4-alphazero-v2-official')  # torn bytes
    with pytest.raises(L.StateCorrupt):
        L.CampaignStateFile.load(tmp_path)


def test_state_file_is_created_once(tmp_path):
    created(tmp_path)
    with pytest.raises(FileExistsError):
        created(tmp_path)


def test_atomic_replace_leaves_old_or_new_bytes(tmp_path, monkeypatch):
    path = tmp_path / 'state.json'
    path.write_bytes(b'old')

    def crash(*args):
        raise OSError('crash before rename')
    monkeypatch.setattr(L.os, 'replace', crash)
    with pytest.raises(OSError):
        L.atomic_replace(path, b'new')
    assert path.read_bytes() == b'old'
    monkeypatch.undo()
    L.atomic_replace(path, b'new')
    assert path.read_bytes() == b'new'


def test_write_once_never_overwrites(tmp_path):
    L.publish_view(tmp_path / 'a.json', dict(x=1))
    with pytest.raises(FileExistsError):
        L.publish_view(tmp_path / 'a.json', dict(x=1))


# Budget (monotonic, in-process only) --------------------------------------------------------------

def test_start_checks_refuse_at_limits_and_completion_rechecks_overruns():
    clock = Clock()
    budget = L.Budget(LIMITS, clock=clock)
    budget.check('evaluation')
    clock.now = 999.9
    assert budget.completion_violation() is None
    clock.now = 1000.0
    with pytest.raises(L.BudgetExhausted, match='campaign wall-clock'):
        budget.check('evaluation')  # starting at the limit is refused (>=)
    assert budget.completion_violation() is None  # finishing exactly at the limit is accepted
    clock.now = 1000.1
    assert isinstance(budget.completion_violation(), L.BudgetExhausted)


def test_training_budget_is_per_seed_collection_and_optimization_only():
    clock = Clock()
    budget = L.Budget(LIMITS, clock=clock)
    budget.begin_training(42)
    clock.now = 60.0
    budget.end_training()
    clock.now = 500.0  # evaluation time does not count against the training budget
    budget.check('training', 42)
    budget.begin_training(42)
    clock.now = 540.0
    with pytest.raises(L.BudgetExhausted, match='seed 42 collection'):
        budget.check('training', 42)
    budget.check('training', 314159)  # the other seed has its own budget
    clock.now = 541.0
    budget.end_training()
    assert budget.training_seconds(42) == 101.0
    assert isinstance(budget.completion_violation(seeds=[42]), L.BudgetExhausted)
    assert budget.completion_violation(seeds=[314159]) is None


def test_stop_request_ends_work_cooperatively():
    budget = L.Budget(LIMITS, clock=Clock())
    budget.request_stop('signal 15')
    with pytest.raises(L.CampaignStop, match='signal 15') as raised:
        budget.check('evaluation')
    assert not isinstance(raised.value, L.BudgetExhausted)
    assert isinstance(budget.completion_violation(), L.CampaignStop)


def test_game_ceiling_is_reserved_before_play():
    budget = L.Budget(LIMITS, clock=Clock())
    for _ in range(3):
        budget.start_game()
    with pytest.raises(L.BudgetExhausted, match='ceiling'):
        budget.start_game()
    assert budget.games_started == 3


def test_phase_timings_accumulate():
    clock = Clock()
    budget = L.Budget(LIMITS, clock=clock)
    for _ in range(2):
        with budget.phase('final'):
            clock.now += 5
    assert budget.snapshot([42])['phase_seconds'] == dict(final=10)


def test_no_lease_or_journal_accounting_exists():
    """Official time is measured in the owning process only; nothing is reconstructed across processes."""
    for name in ('LEASE_SECONDS', 'RENEW_BELOW_SECONDS', 'Journal', 'CampaignState', 'recover_open_attempts',
                 'close_open_work', 'read_state'):
        assert not hasattr(L, name), name
    assert not any(hasattr(L.Budget, name) for name in ('lease', 'settle', '_write_lease'))
    for module in ('launch_control.py', 'campaign.py'):
        names = code_identifiers(ROOT / 'games/connect4/alphazero_v2' / module)
        assert not [n for n in names if 'journal' in n.lower() or 'lease' in n.lower().replace('release', '')]


def code_identifiers(path):
    import tokenize
    with open(path, 'rb') as stream:
        return {t.string for t in tokenize.tokenize(stream.readline) if t.type == tokenize.NAME}


# Single owner ----------------------------------------------------------------------------------------

def test_second_owner_is_refused_in_process(tmp_path):
    first = L.CampaignLock(tmp_path)
    with pytest.raises(L.CampaignLocked):
        L.CampaignLock(tmp_path)
    first.release()
    L.CampaignLock(tmp_path).release()


def test_lock_held_by_another_process_is_refused_and_released_on_death(tmp_path):
    script = textwrap.dedent(f'''
        import sys, time
        sys.path.insert(0, {str(ROOT)!r})
        from games.connect4.alphazero_v2.launch_control import CampaignLock
        CampaignLock({str(tmp_path)!r})
        print("locked", flush=True)
        time.sleep(60)
    ''')
    process = subprocess.Popen([sys.executable, '-c', script], stdout=subprocess.PIPE, text=True)
    try:
        assert process.stdout.readline().strip() == 'locked'
        with pytest.raises(L.CampaignLocked):
            L.CampaignLock(tmp_path)
    finally:
        process.kill()
        process.wait()
    deadline = time.time() + 10
    while True:
        try:
            L.CampaignLock(tmp_path).release()
            break
        except L.CampaignLocked:
            assert time.time() < deadline
            time.sleep(0.05)


def test_both_previous_tokens_are_rejected():
    assert L.REJECTED_DECLARATION_TOKENS == OLD_TOKENS
