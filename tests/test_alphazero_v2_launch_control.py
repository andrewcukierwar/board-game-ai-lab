"""Phase 4D.3B.2/4D.3B.3 fail-closed launch-control primitives (torch-free): the atomic state file and its
transition rules, immutable terminal states, the single-owner lock, the in-process monotonic budget and
its completion barrier. Fake clocks only; nothing sleeps for budget durations."""
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
              '8a8a52b100c7e32aa5e3e5b05469578084c8e12f37750ba81fee071990ff5e39',
              'ed641aea2fef86a29000df55d1e46d9a32bf48dc3ec8f5dbcc0f8449668d0bbb',
              '34b4d899639d59d1bd4408ecc505b7eb20cb4d9c2fdc12095402c246887ac390')
OUTCOME = dict(status='INCOMPLETE', reason='test')
COMPLETE_OUTCOME = dict(status='COMPLETE')


class Clock:
    def __init__(self, now=0.0):
        self.now = now

    def __call__(self):
        return self.now


def created(directory):
    return L.CampaignStateFile.create(directory, declaration_sha256='d' * 64, seeds=SEEDS)


def advance_to_final(state):
    for name in L.state_sequence(SEEDS)[1:-1]:
        state.transition(name)


def advance_to_complete(state):
    advance_to_final(state)
    state.terminate(L.COMPLETE, outcome=COMPLETE_OUTCOME)


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
        state.transition('INCOMPLETE')  # terminal states are entered only through terminate()
    with pytest.raises(L.InvalidTransition):
        state.terminate(L.COMPLETE, outcome=COMPLETE_OUTCOME)  # COMPLETE only from RUNNING_FINAL_EVALUATION
    with pytest.raises(L.InvalidTransition):
        state.terminate(L.INCOMPLETE, outcome=COMPLETE_OUTCOME)  # the outcome must name the state
    assert sorted(p.name for p in tmp_path.iterdir()) == ['state.json']  # no temporary leftovers


@pytest.mark.parametrize('terminal', ['COMPLETE', 'INCOMPLETE'])
def test_terminal_states_never_reopen(tmp_path, terminal):
    state = created(tmp_path)
    stale = None
    if terminal == 'COMPLETE':
        advance_to_final(state)
        stale = L.CampaignStateFile.load(tmp_path)  # an older copy of the non-terminal state (the F2 shape)
        state.terminate(L.COMPLETE, outcome=COMPLETE_OUTCOME)
    else:
        state.transition('RUNNING_SEED_42')
        stale = L.CampaignStateFile.load(tmp_path)
        state.terminate(L.INCOMPLETE, outcome=OUTCOME)
    before = state.path.read_bytes()
    for target in L.state_sequence(SEEDS):
        with pytest.raises(L.InvalidTransition, match='terminal'):
            state.transition(target)
    for target, outcome in ((L.COMPLETE, COMPLETE_OUTCOME), (L.INCOMPLETE, OUTCOME)):
        with pytest.raises(L.InvalidTransition, match='terminal'):
            state.terminate(target, outcome=outcome)
    with pytest.raises(L.InvalidTransition, match='terminal'):
        state.record_counters(dict(more=1))
    assert state.mark_incomplete(reason='late error') == terminal  # conditional: a no-op on a terminal state
    # A stale non-terminal copy re-reads before deciding: it can never overwrite the visible terminal state.
    assert stale.state not in L.TERMINAL_STATES
    assert stale.mark_incomplete(reason='stale error handler') == terminal
    for write in (lambda: stale.record_counters(dict(more=1)),
                  lambda: stale.terminate(L.INCOMPLETE, outcome=OUTCOME),
                  lambda: stale.transition(L.state_sequence(SEEDS)[2])):
        with pytest.raises((L.InvalidTransition, L.StateCorrupt)):
            write()
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
    assert sorted(p.name for p in tmp_path.iterdir()) == ['state.json']  # the temporary file is removed
    monkeypatch.undo()
    L.atomic_replace(path, b'new')
    assert path.read_bytes() == b'new'


def failing_directory_fsync(monkeypatch, error=OSError):
    """Inject one failure at the directory fsync that follows a state.json rename (the F2 boundary)."""
    real, calls = L.fsync_directory, []

    def fsync_directory(path):
        calls.append(path)
        if len(calls) == 1:
            raise error('injected directory fsync failure after rename')
        return real(path)
    monkeypatch.setattr(L, 'fsync_directory', fsync_directory)
    return calls


@pytest.mark.parametrize('error', [OSError, KeyboardInterrupt])
def test_failure_after_complete_is_visible_is_adopted_and_never_downgraded(tmp_path, monkeypatch, error):
    """F2 at the state-file level: the owner adopts the visible COMPLETE; its error path cannot replace it."""
    state = created(tmp_path)
    advance_to_final(state)
    failing_directory_fsync(monkeypatch, error)
    with pytest.raises(error):
        state.terminate(L.COMPLETE, outcome=COMPLETE_OUTCOME)
    monkeypatch.undo()
    visible = state.path.read_bytes()
    assert L.CampaignStateFile.load(tmp_path).state == 'COMPLETE'
    assert state.state == 'COMPLETE' and state.data == visible  # the owner agrees with every reader
    assert state.mark_incomplete(reason='error handler') == 'COMPLETE'
    assert state.path.read_bytes() == visible
    assert [h['state'] for h in json.loads(visible)['history']][-2:] == ['RUNNING_FINAL_EVALUATION', 'COMPLETE']


def test_failure_before_complete_is_visible_ends_incomplete(tmp_path, monkeypatch):
    state = created(tmp_path)
    advance_to_final(state)

    def no_rename(*args):
        raise OSError('injected rename failure')
    monkeypatch.setattr(L.os, 'replace', no_rename)
    with pytest.raises(OSError):
        state.terminate(L.COMPLETE, outcome=COMPLETE_OUTCOME)
    monkeypatch.undo()
    assert state.state == 'RUNNING_FINAL_EVALUATION' == L.CampaignStateFile.load(tmp_path).state
    assert state.mark_incomplete(reason='rename failed') == 'INCOMPLETE'
    history = [h['state'] for h in L.CampaignStateFile.load(tmp_path).document['history']]
    assert history[-2:] == ['RUNNING_FINAL_EVALUATION', 'INCOMPLETE'] and 'COMPLETE' not in history
    assert sorted(p.name for p in tmp_path.iterdir()) == ['state.json']


def test_incomplete_after_a_partial_running_write_keeps_the_visible_history(tmp_path, monkeypatch):
    """INCOMPLETE is built from the visible document, never from a stale copy that would hide a transition."""
    state = created(tmp_path)
    failing_directory_fsync(monkeypatch)
    with pytest.raises(OSError):
        state.transition('RUNNING_SEED_42')
    monkeypatch.undo()
    assert state.state == 'RUNNING_SEED_42'
    state.mark_incomplete(reason='error')
    history = [h['state'] for h in L.CampaignStateFile.load(tmp_path).document['history']]
    assert history == ['CREATED', 'RUNNING_SEED_42', 'INCOMPLETE']


def test_state_is_never_blindly_replaced(tmp_path):
    state = created(tmp_path)
    state.transition('RUNNING_SEED_42')
    foreign = json.loads(state.path.read_text())
    foreign['owner']['pid'] = -1
    state.path.write_bytes(L.view_bytes(foreign))  # not the bytes this owner wrote
    before = state.path.read_bytes()
    for write in (lambda: state.transition('RUNNING_SEED_314159'), lambda: state.record_counters(dict(x=1)),
                  lambda: state.terminate(L.INCOMPLETE, outcome=OUTCOME)):
        with pytest.raises(L.StateCorrupt, match='refusing to replace'):
            write()
    state.path.write_bytes(b'{"torn')
    with pytest.raises(L.StateCorrupt):
        state.mark_incomplete(reason='unreadable state is never overwritten')
    assert state.path.read_bytes() == b'{"torn'
    state.path.write_bytes(before)
    assert L.CampaignStateFile.load(tmp_path).document == foreign


def test_terminal_outcome_must_name_its_state(tmp_path):
    state = created(tmp_path)
    advance_to_complete(state)
    document = json.loads(state.path.read_text())
    state.path.write_bytes(L.view_bytes(dict(document, outcome=dict(status='INCOMPLETE'))))
    with pytest.raises(L.StateCorrupt, match='status'):
        L.CampaignStateFile.load(tmp_path)


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


@pytest.mark.parametrize('at,eligible', [(999.9, True), (1000.0, True), (1000.1, False)])
def test_completion_barrier_deadline_is_inclusive(at, eligible):
    """Same boundary as every acceptance check: elapsed at the endpoint <= campaign_seconds is eligible."""
    clock = Clock()
    budget = L.Budget(LIMITS, clock=clock)
    clock.now = at
    if eligible:
        assert budget.cross_completion_barrier([42])['elapsed_seconds'] == at
    else:
        with pytest.raises(L.BudgetExhausted, match='at the completion barrier'):
            budget.cross_completion_barrier([42])


def test_completion_barrier_latches_one_endpoint():
    clock = Clock()
    budget = L.Budget(LIMITS, clock=clock)
    budget.begin_training(42)
    clock.now = 100.0
    budget.end_training()
    clock.now = 999.0
    account = budget.cross_completion_barrier([42])
    assert account['elapsed_seconds'] == 999.0 and account['training_seconds'] == {'42': 100.0}
    clock.now = 5000.0  # time spent certifying never changes the recorded endpoint
    assert budget.elapsed() == 999.0 and budget.snapshot([42]) == account
    assert budget.completion_violation(seeds=[42]) is None
    budget.request_stop('signal 15')  # after the latch: recorded, never a stop
    assert budget.stop_reason is None and budget.late_stop_requests == ['signal 15']
    with pytest.raises(RuntimeError, match='once'):
        budget.cross_completion_barrier([42])


@pytest.mark.parametrize('limit', ['stop', 'training', 'games'])
def test_completion_barrier_refuses_a_stop_or_any_exceeded_limit(limit):
    clock = Clock()
    budget = L.Budget(LIMITS, clock=clock)
    if limit == 'stop':
        budget.request_stop('signal 15')  # handled before the latch
    elif limit == 'training':
        budget.begin_training(42)
        clock.now = 100.5
        budget.end_training()
    else:
        budget.games_started = LIMITS['evaluation_games_ceiling'] + 1
    with pytest.raises(L.CampaignStop):
        budget.cross_completion_barrier([42])
    assert budget.barrier_time is not None  # latched: a refused barrier is never retried


def test_completion_barrier_latches_before_reading_the_stop_flag():
    """A stop handled while the endpoint is being read (the latest possible moment) is still seen."""
    budget = L.Budget(LIMITS, clock=Clock())

    def clock_with_signal():
        budget.request_stop('signal 15')  # a handler running inside the latch statement, before the store
        return 1.0
    budget.clock = clock_with_signal
    with pytest.raises(L.CampaignStop, match='signal 15'):
        budget.cross_completion_barrier([42])


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


def test_all_four_previous_tokens_are_rejected():
    assert L.REJECTED_DECLARATION_TOKENS == OLD_TOKENS
