"""Phase 4D.3B/4D.3B.2 torch-side preflight: evaluation agents, diagnostics, profiling helpers and the
fail-closed official campaign launcher (declaration binding, single owner, no resume, deadlines and budgets).
Every run uses tiny synthetic settings in temporary directories; no research training, learned
checkpoint or strength claim is produced."""
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import textwrap
import time

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from games.connect4.connect4 import Connect4
from games.connect4.alphazero_v2 import campaign as C
from games.connect4.alphazero_v2 import diagnostics as D
from games.connect4.alphazero_v2 import evaluation as E
from games.connect4.alphazero_v2 import packages as P
from games.connect4.alphazero_v2 import profiling as PR
from games.connect4.alphazero_v2 import provenance
from games.connect4.alphazero_v2.arena import RandomOpponent, run_paired_arena
from games.connect4.alphazero_v2.config import V2Config
from games.connect4.alphazero_v2.data import V2Example, encode_board, finalize_game, pre_move_ply
from games.connect4.alphazero_v2.generation import GenerationRunner, initial_model, load_resume_boundary
from games.connect4.alphazero_v2 import launch_control as L
from games.connect4.alphazero_v2.network import V2Inference, weights_sha256
from games.connect4.alphazero_v2.oracle import engine_position
from games.connect4.alphazero_v2.provenance import THREAD_ENVIRONMENT, required_thread_environment

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True, scope='module')
def bounded_torch_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny_config(**changes):
    values = dict(self_play_simulations=3, games_per_generation=2, replay_generations=2,
                  replay_max_games=4, batch_size=8, max_generations=6)
    values.update(changes)
    return V2Config(**values)


def history_example(moves, ply):
    game = engine_position(moves[:ply])
    visits = [0] * 7
    visits[moves[ply]] = 2
    return V2Example(encode_board(game), game.current_player, pre_move_ply(game), tuple(visits), moves[ply], 1.0)


# Evaluation ----------------------------------------------------------------------------------

def test_untrained_model_is_the_runner_initialization():
    assert weights_sha256(E.untrained_model(42)) == weights_sha256(GenerationRunner(tiny_config(seed=42)).model)
    assert weights_sha256(initial_model(7)) != weights_sha256(initial_model(42))


def test_evaluation_between_generations_never_changes_the_training_trajectory(tmp_path):
    plain, evaluated = GenerationRunner(tiny_config()), GenerationRunner(tiny_config())
    plain.run_generation()
    evaluated.run_generation()
    before = E.process_rng_fingerprint()
    inference = V2Inference(evaluated.model)
    rows = [dict(id='r0', moves=[3, 3]), dict(id='r1', moves=[0, 1, 0, 1, 0, 2])]
    first = E.search_rows(inference, rows, simulations=4, seeds=(0, 1))
    assert E.search_rows(inference, rows, simulations=4, seeds=(0, 1)) == first
    E.raw_values(inference, rows)
    E.calibration_games(evaluated.model, evaluated.config, games=1, seed=3)
    openings = [dict(id='e', moves=[], stratum='empty', family='empty-board')]
    run_paired_arena(E.V2SearchArenaAgent(inference, 4), RandomOpponent(), openings, namespace='x', seed=1)
    after = E.process_rng_fingerprint()
    assert before[0] == after[0] and before[1] == after[1] and torch.equal(before[2], after[2])
    plain.run_generation()
    evaluated.run_generation()
    assert evaluated.state_sha256() == plain.state_sha256()


def test_v2_arena_agents_use_deployment_conventions():
    inference = V2Inference(initial_model(3))
    agent = E.V2SearchArenaAgent(inference, 8)
    description = agent.describe()
    assert (description['root_noise'], description['tactical_guard'], description['temperature']) == (False, False, 0)
    game = engine_position([3, 3])
    move, info = agent.choose(game, random.Random(0))
    assert info['visits'][move] == max(info['visits']) and sum(info['visits']) == 8
    prediction = inference.predict(game)
    nn_move, _ = E.V2NNOnlyAgent(inference).choose(game, random.Random(0))
    assert prediction.policy[nn_move] == max(prediction.policy)


def test_phase4d2f_adapter_is_hash_pinned(tmp_path):
    bogus = tmp_path / 'candidate.pt'
    bogus.write_bytes(b'not a checkpoint')
    with pytest.raises(ValueError, match='hash'):
        E.V1SearchArenaAgent(bogus, E.PHASE4D2F_CHECKPOINT['sha256'])
    path = ROOT / E.PHASE4D2F_CHECKPOINT['path']
    if not path.exists():
        pytest.skip('retained Phase 4D.2f checkpoint not present locally')
    agent = E.V1SearchArenaAgent(path, E.PHASE4D2F_CHECKPOINT['sha256'], simulations=8)
    move, info = agent.choose(engine_position([3]), random.Random(0))
    assert move in engine_position([3]).get_valid_moves() and sum(info['visits']) == 8
    assert agent.describe()['tactical_guard'] is False


def test_calibration_records_raw_pre_search_values_and_outcomes():
    model = initial_model(5)
    records = E.calibration_games(model, tiny_config(), games=2, seed=9)
    assert records == E.calibration_games(model, tiny_config(), games=2, seed=9)
    games = {r['game'] for r in records}
    assert games == {0, 1}
    for record in records:
        assert -1 <= record['value'] <= 1 and record['outcome'] in (-1, 0, 1)
    inference = V2Inference(model)
    first = [r for r in records if r['game'] == 0][0]
    assert first['ply'] == 0 and first['value'] == pytest.approx(inference.predict(Connect4()).value)


# Diagnostics ---------------------------------------------------------------------------------------

def test_contradictions_count_proven_wins_that_were_not_converted():
    moves = [0, 1, 0, 1, 0, 1, 6, 1]  # X ignores its win at ply 6; O then wins
    game = finalize_game(1, 0, moves, [history_example(moves, ply) for ply in range(len(moves))])
    result = D.game_diagnostics([game])
    assert result['contradictions'] == 1 and result['contradictions_by_actor_value_ply']['x/win/ply06+'] == dict(
        contradictions=1, proven=1)
    assert result['draw_games'] == 0 and result['game_length']['max'] == 8


def test_generation_summary_has_deterministic_diagnostics_and_separate_resources():
    runner = GenerationRunner(tiny_config())
    summary = runner.run_generation()
    diagnostics = summary['diagnostics']
    assert sum(diagnostics['sampling']['samples_by_age'].values()) == summary['sampled_positions']
    search = diagnostics['search']
    assert search['decisions'] == summary['new_positions'] and 1 <= search['max_search_depth'] <= 3
    assert 0 <= search['terminal_leaf_fraction'] <= 1 and 0 < search['root_visit_coverage'] <= 1
    assert diagnostics['games']['unique_positions'] <= summary['new_positions']
    assert 'resources' in summary and 'resources' not in runner.history[-1]
    assert runner.history[-1]['diagnostics'] == diagnostics


# Profiling helpers -------------------------------------------------------------------------------

def test_synthetic_full_scale_replay_is_legal_and_resumable(tmp_path):
    config = tiny_config(games_per_generation=3, replay_generations=2, replay_max_games=6)
    for kind in ('typical', 'worst_case'):
        runner = PR.full_scale_runner(kind, config)
        assert runner.replay.games == 6 and runner.trainer.steps == 1
        if kind == 'worst_case':
            assert len(runner.replay) == 6 * 42
        (tmp_path / kind).mkdir()
        artifacts = runner.save_boundary(tmp_path / kind)
        loaded = load_resume_boundary(artifacts['resume'], restore_global_rng=False)
        assert loaded.state_sha256() == runner.state_sha256()
    positions = PR.varied_positions(20)
    assert len(positions) == 20 and not any(engine_position(m).is_game_over() for m in positions)


def test_quantiles_are_nearest_rank_including_minimum():
    values = list(range(1, 101))
    assert D.quantiles(values, (0.0, 0.5, 0.95, 1.0)) == dict(p0=1, p50=50, p95=95, p100=100)


# Phase 4D.3B.2 official campaign: shared fixtures ---------------------------------------------------

OLD_TOKENS = {'a663454': '2741314399741c1b20510b8a2beb42e0938b0aeafc9dd0c487201e57208867d8',   # Phase 4D.3B
              'c302c18': '8a8a52b100c7e32aa5e3e5b05469578084c8e12f37750ba81fee071990ff5e39',   # Phase 4D.3B.1
              'ff1ded6': 'ed641aea2fef86a29000df55d1e46d9a32bf48dc3ec8f5dbcc0f8449668d0bbb'}   # Phase 4D.3B.2
FROZEN_DECLARATION = ROOT / 'games/connect4/alphazero_v2/frozen/campaign-declaration.json'


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


@pytest.fixture(scope='module')
def tiny_packages(tmp_path_factory):
    directory = tmp_path_factory.mktemp('campaign-packages') / 'packages'
    saved = {name: getattr(P, name) for name in ('EXCLUSION_SOURCES', 'TACTICAL_QUOTAS', 'SOLVED_QUOTAS',
                                                 'SOLVED_STAGES', 'OPENING_PREFIX_BASES', 'EMPTY_BOARD_PAIRS',
                                                 'OVERSAMPLE')}
    P.EXCLUSION_SOURCES = P.EXCLUSION_SOURCES[:3]
    P.TACTICAL_QUOTAS, P.SOLVED_QUOTAS = dict(sealed=1, development=1), dict(sealed=1, development=1)
    P.SOLVED_STAGES = (('late', 24, 41),)
    P.OPENING_PREFIX_BASES, P.EMPTY_BOARD_PAIRS, P.OVERSAMPLE = dict(sealed=2, development=2), 1, 1
    try:
        P.build_all(directory, seed=7, log=lambda message: None)
    finally:
        for name, value in saved.items():
            setattr(P, name, value)
    (directory / 'checkpoint.bin').write_bytes(b'stand-in for the retained 4D.2f checkpoint')
    return directory


def pinned_env(**extra):
    return dict(os.environ, PYTHONDONTWRITEBYTECODE='1', **required_thread_environment(1), **extra)


@pytest.fixture(scope='module')
def launch_runtime():
    """The configured runtime identity of a fresh, pinned interpreter (what a real launch binds)."""
    script = 'import json\nfrom games.connect4.alphazero_v2.provenance import configure_deterministic_runtime\n' \
             'print(json.dumps(configure_deterministic_runtime(1)))'
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, timeout=120, cwd=ROOT,
                            env=pinned_env())
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def declaration_document(package_dir, runtime, *, seeds=(42,), games=2, schedule=(1, 2), ceiling=None,
                         **budget_overrides):
    package_dir = Path(package_dir)
    manifest = json.loads((package_dir / 'manifest.json').read_text())
    config = V2Config(self_play_simulations=2, games_per_generation=games, replay_generations=2,
                      replay_max_games=2 * games, batch_size=8, max_generations=2).to_dict()
    checkpoint = package_dir / 'checkpoint.bin'
    declaration = C.build_declaration(
        {k: v for k, v in manifest.items() if k != 'exclusions'}, name='tiny-test', kind='test', config=config,
        seeds=seeds, primary_seed=seeds[0],
        budgets=dict(dict(per_run_training_seconds=3600, campaign_seconds=7200, evaluation_games_ceiling=10_000,
                          planned_evaluation_games=0), **budget_overrides),
        evaluation=dict(simulations=2, tactical_seeds=[0], root_noise=False, tactical_guard=False, temperature=0,
                        ties='seeded uniform'),
        champion=dict(schedule=list(schedule), arena_openings=1, baseline_opponents=['random'],
                      baseline_rows=C.baseline_rows(empty_pairs=1, prefix_families=0),
                      gate={k: v for k, v in C.statistics.CHAMPION_GATE.items() if k != 'schedule'},
                      tactical_package='tactical-development'),
        final=dict(ladder=['random', 'negamax1'], ladder_openings=1, calibration_games=1, calibration_constant=None,
                   one_time=True),
        runtime_identity=runtime, frozen_records=C.frozen_record_entries(package_dir, ('manifest.json',
                                                                                     'exclusions.json')),
        phase4d2f_checkpoint=dict(path=str(checkpoint), sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest()))
    if ceiling == 'planned':
        declaration['budgets']['evaluation_games_ceiling'] = declaration['budgets']['planned_evaluation_games']
    return declaration


def write_declaration(package_dir, declaration, name):
    path = Path(package_dir) / name
    path.write_text(json.dumps(declaration, indent=1, sort_keys=True))
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


# In-process launches: a fixed, fully pinned stand-in identity (the pytest interpreter cannot
# re-pin inter-op threads); generation boundaries still use the real identity internally.

@pytest.fixture
def fake_runtime(monkeypatch):
    for name in THREAD_ENVIRONMENT:
        monkeypatch.setenv(name, '1')
    identity = dict(C.runtime_identity(), intra_op_threads=1, inter_op_threads=1, deterministic_algorithms=True,
                    deterministic_warn_only=False, thread_environment=required_thread_environment(1))
    current = dict(identity=identity)
    monkeypatch.setattr(C, 'runtime_identity', lambda: json.loads(json.dumps(current['identity'])))
    monkeypatch.setattr(C.Campaign, 'configure_runtime', lambda self: C.runtime_identity())
    return current


def in_process(package_dir, fake, name, **options):
    return write_declaration(package_dir, declaration_document(package_dir, fake['identity'], **options), name)


def launch(directory, declaration, token, **options):
    return C.Campaign(directory, declaration, token, **options)


def run_campaign(directory, declaration, token, extra_check=None, **options):
    return launch(directory, declaration, token, **options).run(extra_check=extra_check)


def state_of(directory):
    return json.loads((Path(directory) / 'state.json').read_text())


def stop_at(phase, generation=None, kind=None, after=0):
    """extra_check that requests a cooperative stop at a phase (optionally a generation or unit kind)."""
    state = dict(seen=0, fired=False)

    def check(current, context):
        if state['fired'] or current != phase:
            return
        if generation is not None and context.get('generation') != generation:
            return
        if kind is not None and context.get('kind') != kind:
            return
        state['seen'] += 1
        if state['seen'] > after:
            state['fired'] = True
            raise C.CampaignStop(f'test stop at {phase}')
    return check


def tree(directory, exclude=('state.json', 'campaign.lock')):
    """Every file under a campaign directory with its hash (to prove nothing new ran)."""
    directory = Path(directory)
    return {p.relative_to(directory).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(directory.rglob('*')) if p.is_file() and p.name not in exclude}


def assert_not_official(directory):
    with pytest.raises(C.NotOfficialEvidence):
        C.official_results(directory)


# Declaration and tokens ---------------------------------------------------------------------------------

def test_full_declaration_plans_exactly_the_reviewed_evaluation_games():
    declaration = C.build_declaration({})
    assert declaration['budgets'] == dict(per_run_training_seconds=28800, campaign_seconds=86400,
                                          evaluation_games_ceiling=6400, planned_evaluation_games=6272)
    assert declaration['seeds'] == [42, 314159] and declaration['primary_seed'] == 42
    assert declaration['generations'] == 20 and declaration['champion']['schedule'] == [5, 10, 15, 20]
    assert V2Config.from_dict(dict(declaration['config'], seed=42)) == V2Config(seed=42)
    assert len(declaration['champion']['baseline_rows']) == 20
    assert declaration['format_version'] == 3 and declaration['launch_control'] == C.launch_control_declaration()
    assert declaration['runtime'] == dict(threads=1, deterministic_algorithms=True)  # no resume flags
    assert declaration['launch_control']['rejected_declaration_tokens'] == sorted(OLD_TOKENS.values())
    assert 'Forbidden' in declaration['launch_control']['resume']
    assert not any('lease' in key for key in declaration['launch_control'])


def test_declaration_validation_rejects_inconsistent_bindings(tiny_packages, launch_runtime):
    declaration = declaration_document(tiny_packages, launch_runtime)
    assert C.validate_declaration(declaration, tiny_packages)
    source = declaration['execution_source']
    tampered_files = dict(source['files'], **{'games/connect4/alphazero_v2/search.py': '0' * 64})
    changes = [
        dict(budgets=dict(declaration['budgets'], planned_evaluation_games=1)), dict(generations=3),
        dict(seeds=[42, 42]), dict(format_version=2), dict(champion=dict(declaration['champion'], schedule=[2, 1])),
        dict(packages=dict(declaration['packages'], **{'tactical-sealed': dict(
            declaration['packages']['tactical-sealed'], sha256='0' * 64)})),
        dict(champion=dict(declaration['champion'], gate=dict(declaration['champion']['gate'], score=0.5))),
        dict(thresholds=dict(declaration['thresholds'], solved=dict(optimal_preserving=0.5, avoidable_loss_max=0.5))),
        dict(launch_control=dict(declaration['launch_control'], resume='allowed')),
        dict(launch_control=dict(declaration['launch_control'], lease_seconds=300.0)),
        dict(launch_control=dict(declaration['launch_control'], rejected_declaration_tokens=[])),
        dict(runtime=dict(declaration['runtime'], strict_resume_runtime=True)),
        dict(execution_source=dict(source, files=tampered_files)),  # digest no longer matches its files
        dict(runtime_identity={k: v for k, v in launch_runtime.items() if k != 'cpu_model'}),
        dict(runtime_identity=dict(launch_runtime, cpu_model=None)),
        dict(runtime_identity=dict(launch_runtime, inter_op_threads=4)),
        dict(frozen_records=dict(declaration['frozen_records'], **{'manifest.json': dict(path='manifest.json',
                                                                                         sha256='0' * 64)})),
        dict(kind='research'),  # a research declaration must bind all three frozen records
    ]
    for change in changes:
        with pytest.raises(ValueError):
            C.validate_declaration(dict(declaration, **change), tiny_packages)


@pytest.mark.parametrize('commit', sorted(OLD_TOKENS))
def test_both_old_tokens_and_their_declarations_never_authorize(tmp_path, tiny_packages, launch_runtime, commit):
    token = OLD_TOKENS[commit]
    path, _ = write_declaration(tiny_packages, declaration_document(tiny_packages, launch_runtime), 'ok.json')
    with pytest.raises(C.LaunchRefused, match='REJECTED'):
        C.Campaign(tmp_path / 'c', path, token)
    old = subprocess.run(['git', '-C', str(ROOT), 'show', f'{commit}:{FROZEN_DECLARATION.relative_to(ROOT)}'],
                         capture_output=True)
    if old.returncode != 0:
        pytest.skip('original declaration bytes unavailable from git')
    assert hashlib.sha256(old.stdout).hexdigest() == token
    copy = tmp_path / 'campaign-declaration.json'
    copy.write_bytes(old.stdout)
    for loader in (lambda: C.load_declaration(copy), lambda: C.Campaign(tmp_path / 'c', copy, token)):
        with pytest.raises(C.LaunchRefused, match='REJECTED / NOT AUTHORIZED'):
            loader()
    report = C.preflight(copy)
    assert report['ok'] is False and 'REJECTED' in report['gates'][0]['detail']
    # The CLI refuses it in a fresh interpreter (exit 2) even with a pinned runtime; no directory is created.
    result = subprocess.run([sys.executable, '-m', 'games.connect4.alphazero_v2.campaign', 'launch', '--declaration',
                             str(copy), '--campaign-dir', str(tmp_path / 'cli'), '--authorize', token],
                            capture_output=True, text=True, timeout=120, cwd=ROOT, env=pinned_env())
    assert result.returncode == C.EXIT_REFUSED and 'REJECTED' in result.stdout
    assert not (tmp_path / 'c').exists() and not (tmp_path / 'cli').exists()


def test_the_new_token_is_required(tmp_path, tiny_packages, fake_runtime):
    path, token = in_process(tiny_packages, fake_runtime, 'token.json')
    _, other = in_process(tiny_packages, fake_runtime, 'token-other.json', games=3)
    for wrong in (other, '0' * 64, token.upper()):
        with pytest.raises(PermissionError, match='must equal the declaration SHA-256'):
            launch(tmp_path / 'c', path, wrong)
    assert not (tmp_path / 'c').exists()
    launch(tmp_path / 'c', path, token).close()
    assert state_of(tmp_path / 'c')['declaration_sha256'] == token


def test_old_multi_command_workflow_no_longer_exists():
    for command in (['run', '--seed', '42'], ['final-evaluate']):
        with pytest.raises(SystemExit) as raised:
            C.main(command + ['--declaration', 'd', '--campaign-dir', 'c', '--authorize', 'x'])
        assert raised.value.code == 2


def test_frozen_declaration_is_the_new_format_3_declaration():
    """The repository's frozen declaration is the re-frozen one: valid, bound, and not a rejected token."""
    token = hashlib.sha256(FROZEN_DECLARATION.read_bytes()).hexdigest()
    assert token not in OLD_TOKENS.values()
    declaration, digest = C.load_declaration(FROZEN_DECLARATION)
    assert digest == token and declaration['format_version'] == 3 and declaration['kind'] == 'research'
    assert declaration['execution_source'] == C.execution_source_declaration()  # matches this tree's sources
    assert declaration['launch_control'] == C.launch_control_declaration()
    assert sorted(declaration['frozen_records']) == sorted(C.FROZEN_RECORD_NAMES)
    assert declaration['budgets'] == C.build_declaration({})['budgets']
    assert declaration['config'] == C.build_declaration({})['config'] and declaration['seeds'] == [42, 314159]
    assert {k: v['sha256'] for k, v in declaration['packages'].items()} == {
        k: v['sha256'] for k, v in json.loads((FROZEN_DECLARATION.parent / 'manifest.json').read_text()).items()
        if k != 'exclusions'}
    assert C.unavailable_runtime_fields(declaration['runtime_identity']) == []


def test_every_declaration_field_is_bound_by_the_token(tmp_path, tiny_packages, launch_runtime):
    declaration = declaration_document(tiny_packages, launch_runtime)
    _, token = write_declaration(tiny_packages, declaration, 'base.json')
    for key in sorted(C.DECLARATION_KEYS):
        changed = json.loads(json.dumps(declaration))
        changed[key] = {'mutated': key} if not isinstance(changed[key], str) else changed[key] + '-mutated'
        path, other = write_declaration(tiny_packages, changed, f'mutated-{key}.json')
        assert other != token, key
        with pytest.raises((PermissionError, ValueError, KeyError, TypeError, AttributeError)):
            C.Campaign(tmp_path / f'c-{key}', path, token)  # the old token never authorizes the new bytes
        assert not (tmp_path / f'c-{key}').exists()


def test_launch_rejects_changed_packages_frozen_records_and_checkpoint(tmp_path, tiny_packages, fake_runtime):
    copy = tmp_path / 'packages'
    shutil.copytree(tiny_packages, copy)
    path, token = in_process(copy, fake_runtime, 'bound.json')
    launch(tmp_path / 'ok', path, token).close()
    for name, match in (('tactical-sealed.json', 'Package file hash'), ('exclusions.json', 'Frozen record'),
                        ('checkpoint.bin', 'checkpoint hash differs')):
        target = copy / name
        original = target.read_bytes()
        target.write_bytes(original + b' ')
        try:
            with pytest.raises((ValueError, C.LaunchRefused), match=match):
                launch(tmp_path / f'c-{name}', path, token)
            assert not (tmp_path / f'c-{name}').exists()
        finally:
            target.write_bytes(original)


# Source and runtime binding at launch -------------------------------------------------------------------

def tampered_identity(name):
    identity = provenance.execution_source_identity()
    files = dict(identity['files'], **{name: '0' * 64})
    return dict(identity, files=files, sha256=hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest())


@pytest.mark.parametrize('name,group', [('games/connect4/alphazero_v2/search.py', 'training'),
                                        ('games/connect4/connect4.py', 'training'),
                                        ('games/connect4/alphazero_v2/oracle.py', 'training'),
                                        ('games/connect4/alphazero_v2/evaluation.py', 'evaluation_and_launch'),
                                        ('games/connect4/alphazero_v2/statistics.py', 'evaluation_and_launch'),
                                        ('games/connect4/alphazero_v2/launch_control.py', 'evaluation_and_launch')])
def test_changed_execution_source_refuses_launch(tmp_path, tiny_packages, fake_runtime, monkeypatch, name, group):
    path, token = in_process(tiny_packages, fake_runtime, 'source.json')
    monkeypatch.setattr(C, 'execution_source_identity', lambda: tampered_identity(name))
    with pytest.raises(C.LaunchRefused, match=f'{group} source differs.*{name}'):
        launch(tmp_path / 'c', path, token)
    assert not (tmp_path / 'c').exists()  # refused before any campaign state exists


def test_source_groups_partition_the_closure():
    files = provenance.execution_source_files()
    groups = provenance.source_groups(provenance.execution_source_identity()['files'])
    assert sorted(groups['training']['files'] + groups['evaluation_and_launch']['files']) == list(files)
    assert set(provenance.TRAINING_SOURCE_FILES) <= set(files)
    assert 'games/connect4/alphazero_v2/launch_control.py' in groups['evaluation_and_launch']['files']


def test_training_path_loads_only_training_group_modules(tmp_path):
    script = f"""
import json, sys, torch
from pathlib import Path
torch.set_num_threads(1)
from games.connect4.alphazero_v2.config import V2Config
from games.connect4.alphazero_v2.generation import GenerationRunner, load_resume_boundary
from games.connect4.alphazero_v2.provenance import REPO_ROOT, TRAINING_SOURCE_FILES
runner = GenerationRunner(V2Config(self_play_simulations=2, games_per_generation=2, replay_generations=1,
                                   replay_max_games=2, batch_size=8, max_generations=2))
runner.run_generation({str(tmp_path)!r})
load_resume_boundary({str(tmp_path / 'generation-0001.resume.pt')!r}).run_generation()
loaded = sorted(Path(m.__file__).resolve().relative_to(REPO_ROOT).as_posix() for m in list(sys.modules.values())
                if getattr(m, '__file__', None) and Path(m.__file__).is_absolute()
                and Path(m.__file__).resolve().is_relative_to(REPO_ROOT))
print(json.dumps(sorted(set(loaded) - set(TRAINING_SOURCE_FILES))))
"""
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, timeout=180, cwd=ROOT,
                            env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1]) == []


COPIED_LAUNCH = '''
import json, sys
from games.connect4.alphazero_v2 import campaign as C
from games.connect4.alphazero_v2.provenance import execution_source_files
declaration, token, directory = sys.argv[1:4]
try:
    C.Campaign(directory, declaration, token).close()
    modules = [f[:-3].replace("/", ".").removesuffix(".__init__") for f in execution_source_files()]
    assert all(m in sys.modules for m in modules), "closure not loaded at launch"
    print(json.dumps(dict(accepted=True, preflight=C.preflight(declaration)["ok"])))
except C.LaunchRefused as error:
    print(json.dumps(dict(accepted=False, error=str(error))))
'''


def test_fresh_interpreter_refuses_a_one_byte_source_edit(tmp_path, tiny_packages, launch_runtime):
    """A copied repository tree: an edited execution file is refused by a fresh interpreter;
    documentation, tests and non-closure modules do not change executable identity."""
    tree_root = tmp_path / 'tree'
    for relative in provenance.execution_source_files():
        (tree_root / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, tree_root / relative)
    shutil.copytree(tiny_packages, tree_root / 'packages')
    path, token = write_declaration(tree_root / 'packages', declaration_document(tree_root / 'packages', launch_runtime),
                                    'declaration.json')

    def attempt(name):
        result = subprocess.run([sys.executable, '-c', COPIED_LAUNCH, str(path), token, str(tmp_path / name)],
                                capture_output=True, text=True, timeout=180, cwd=tree_root, env=pinned_env())
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout.strip().splitlines()[-1])
    assert attempt('clean') == dict(accepted=True, preflight=True)
    (tree_root / 'docs').mkdir()
    (tree_root / 'docs/notes.md').write_text('documentation only\n')
    (tree_root / 'tests').mkdir()
    (tree_root / 'tests/test_extra.py').write_text('def test_nothing():\n    pass\n')
    (tree_root / 'games/connect4/agents/negamax_agent.py').write_text('# legacy, outside the execution closure\n')
    assert attempt('docs-only') == dict(accepted=True, preflight=True)
    for relative, group in (('games/connect4/alphazero_v2/search.py', 'training'),
                            ('games/connect4/alphazero_v2/evaluation.py', 'evaluation_and_launch')):
        original = (tree_root / relative).read_bytes()
        (tree_root / relative).write_bytes(original + b'#')
        try:
            outcome = attempt(f'edited-{group}')
            assert outcome['accepted'] is False and f'{group} source differs' in outcome['error'] \
                and relative in outcome['error']
            assert not (tmp_path / f'edited-{group}').exists()
        finally:
            (tree_root / relative).write_bytes(original)


@pytest.mark.parametrize('key', C.RUNTIME_IDENTITY_KEYS)
def test_every_runtime_identity_field_refuses_launch(tmp_path, tiny_packages, fake_runtime, key):
    path, token = in_process(tiny_packages, fake_runtime, 'runtime.json')
    fake_runtime['identity'] = dict(fake_runtime['identity'], **{key: 'changed'})
    with pytest.raises(C.LaunchRefused, match=f'runtime differs from the declaration.*{key}'):
        launch(tmp_path / 'c', path, token)
    assert not (tmp_path / 'c').exists()


def test_unidentified_runtime_is_refused(tmp_path, tiny_packages, fake_runtime):
    path, token = in_process(tiny_packages, fake_runtime, 'unidentified.json')
    fake_runtime['identity'] = dict(fake_runtime['identity'], cpu_model=None)
    with pytest.raises(C.LaunchRefused, match='unavailable.*cpu_model'):
        launch(tmp_path / 'c', path, token)


def test_cpu_identity_probe_does_not_need_a_subprocess(monkeypatch):
    if sys.platform != 'darwin':
        pytest.skip('sysctlbyname probe is macOS-specific')
    assert provenance._sysctl_string('machdep.cpu.brand_string')
    provenance._cpu_model.cache_clear()
    monkeypatch.setattr(provenance.subprocess, 'run', lambda *a, **k: (_ for _ in ()).throw(OSError('blocked')))
    try:
        assert provenance._cpu_model()
    finally:
        provenance._cpu_model.cache_clear()


def test_preflight_is_a_launch_gate(tmp_path, tiny_packages, launch_runtime):
    path, _ = write_declaration(tiny_packages, declaration_document(tiny_packages, launch_runtime), 'gate.json')
    command = [sys.executable, '-m', 'games.connect4.alphazero_v2.campaign', 'preflight', '--declaration', str(path)]
    passed = subprocess.run(command, capture_output=True, text=True, timeout=180, cwd=ROOT, env=pinned_env())
    report = json.loads(passed.stdout)
    assert passed.returncode == 0 and report['ok'], report
    assert {g['name'] for g in report['gates']} == {'declaration', 'thread_environment', 'runtime_identity',
                                                    'execution_source', 'phase4d2f_checkpoint'}
    runtime = report['runtime']
    assert (runtime['inter_op_threads'], runtime['deterministic_algorithms'], runtime['cpu_model'] is not None) == (
        1, True, True)  # preflight configures the runtime it reports
    env = pinned_env()
    env.pop('OMP_NUM_THREADS')
    failed = subprocess.run(command, capture_output=True, text=True, timeout=180, cwd=ROOT, env=env)
    report = json.loads(failed.stdout)
    assert failed.returncode == C.EXIT_REFUSED and not report['ok']
    assert {g['name'] for g in report['gates'] if not g['ok']} >= {'thread_environment', 'runtime_identity'}
    other = dict(launch_runtime, cpu_model='Apple M4')
    path, _ = write_declaration(tiny_packages, declaration_document(tiny_packages, other), 'm4.json')
    report = json.loads(subprocess.run(command[:-1] + [str(path)], capture_output=True, text=True, timeout=180,
                                       cwd=ROOT, env=pinned_env()).stdout)
    assert not report['ok'] and 'cpu_model' in [g for g in report['gates'] if g['name'] == 'runtime_identity'][0][
        'detail']


# Identity drift after launch => INCOMPLETE ----------------------------------------------------------------

def test_source_drift_mid_campaign_is_incomplete_and_never_continues(tmp_path, tiny_packages, fake_runtime,
                                                                     monkeypatch):
    path, token = in_process(tiny_packages, fake_runtime, 'drift.json')
    real = C.execution_source_identity
    drift = dict(on=False)

    def trigger(phase, context):
        if context.get('generation') == 2:
            drift['on'] = True
    monkeypatch.setattr(C, 'execution_source_identity',
                        lambda: tampered_identity('games/connect4/alphazero_v2/data.py') if drift['on'] else real())
    status = run_campaign(tmp_path / 'c', path, token, extra_check=trigger)
    assert status.startswith('incomplete: IdentityDrift') and 'training source differs' in status \
        and 'data.py' in status
    state = state_of(tmp_path / 'c')
    assert state['state'] == 'INCOMPLETE' and state['outcome']['during'] == 'RUNNING_SEED_42'
    assert not (tmp_path / 'c/runs/seed-42/generations/generation-0002').exists()  # its work is never accepted
    drift['on'] = False  # even with the declared identity restored, the declaration never continues
    with pytest.raises(C.TerminalCampaign, match='already INCOMPLETE'):
        launch(tmp_path / 'c', path, token)
    assert state_of(tmp_path / 'c') == state


@pytest.mark.parametrize('key', ['deterministic_algorithms', 'intra_op_threads', 'torch', 'cpu_model',
                                 'thread_environment'])
def test_runtime_drift_mid_campaign_is_incomplete(tmp_path, tiny_packages, fake_runtime, key):
    path, token = in_process(tiny_packages, fake_runtime, 'runtime-drift.json')
    original = fake_runtime['identity']

    def trigger(phase, context):
        if phase == 'training' and context.get('generation') == 1:
            fake_runtime['identity'] = dict(original, **{key: 'drifted'})
    status = run_campaign(tmp_path / 'c', path, token, extra_check=trigger)
    assert status.startswith('incomplete: IdentityDrift') and key in status
    assert state_of(tmp_path / 'c')['state'] == 'INCOMPLETE'
    assert not (tmp_path / 'c/runs/seed-42/generations/generation-0001').exists()


def test_runtime_drift_during_sealed_evaluation_leaves_no_acceptable_evidence(tmp_path, tiny_packages,
                                                                             fake_runtime):
    """R1: evidence computed under a drifted runtime is never accepted, now or by any later invocation."""
    path, token = in_process(tiny_packages, fake_runtime, 'sealed-drift.json')
    original = fake_runtime['identity']

    def drift(point, campaign):
        if point == 'after_unit_begin:calibration_game':
            fake_runtime['identity'] = dict(original, deterministic_algorithms=False)
    status = run_campaign(tmp_path / 'c', path, token, fault=drift)
    assert status.startswith('incomplete: IdentityDrift') and 'deterministic_algorithms' in status
    state = state_of(tmp_path / 'c')
    assert state['state'] == 'INCOMPLETE' and state['outcome']['during'] == 'RUNNING_FINAL_EVALUATION'
    counts = state['counters']['evaluation']['by_kind']['calibration_game']
    assert counts == dict(started=1, completed=0)  # computed, never accepted
    assert not (tmp_path / 'c/final/seed-42.json').exists()
    assert_not_official(tmp_path / 'c')
    fake_runtime['identity'] = original
    before = tree(tmp_path / 'c', exclude=('campaign.lock',))
    with pytest.raises(C.TerminalCampaign):
        launch(tmp_path / 'c', path, token)
    assert tree(tmp_path / 'c', exclude=('campaign.lock',)) == before


# Normal completion ------------------------------------------------------------------------------------

def test_normal_campaign_completes_without_any_resume_path(tmp_path, tiny_packages, fake_runtime, monkeypatch):
    from games.connect4.alphazero_v2 import generation as G
    real, validated = G.load_resume_boundary, []

    def forbidden(path, *args, **kwargs):
        # Writing a diagnostic boundary validates its unpublished temporary file by reloading it;
        # loading any published boundary would be a continuation, which the official launcher never does.
        if not Path(path).name.endswith('.tmp'):
            raise AssertionError('the official launcher must never load a published resume boundary')
        validated.append(path)
        return real(path, *args, **kwargs)
    monkeypatch.setattr(G, 'load_resume_boundary', forbidden)
    path, token = in_process(tiny_packages, fake_runtime, 'normal.json')
    assert run_campaign(tmp_path / 'c', path, token) == 'completed'
    assert len(validated) == 3  # generations 0, 1 and 2 were written, never read back as a continuation
    state = state_of(tmp_path / 'c')
    assert state['state'] == 'COMPLETE'
    assert [h['state'] for h in state['history']] == ['CREATED', 'RUNNING_SEED_42', 'DEVELOPMENT_SELECTION_COMPLETE',
                                                      'RUNNING_FINAL_EVALUATION', 'COMPLETE']
    official = C.official_results(tmp_path / 'c')
    assert sorted(official['results']) == ['42'] and official['declaration_sha256'] == token
    assert sorted(p.name for p in (tmp_path / 'c').iterdir()) == ['campaign.lock', 'declaration.json', 'final',
                                                                  'runs', 'source', 'state.json']
    # A COMPLETE campaign never runs or transitions again.
    with pytest.raises(C.TerminalCampaign, match='already COMPLETE'):
        launch(tmp_path / 'c', path, token)
    assert state_of(tmp_path / 'c') == state


def test_official_launcher_has_no_resume_code_path():
    names = set()
    import tokenize
    with open(ROOT / 'games/connect4/alphazero_v2/campaign.py', 'rb') as stream:
        names = {t.string for t in tokenize.tokenize(stream.readline) if t.type == tokenize.NAME}
    assert not names & {'load_resume_boundary', 'save_resume_boundary', 'recover_open_attempts', 'Journal',
                        'repair', 'lease', 'final_evaluate', 'run_seed'}


def test_counters_record_every_unit_of_live_work(tmp_path, tiny_packages, fake_runtime):
    path, token = in_process(tiny_packages, fake_runtime, 'counters.json', seeds=(42, 7))
    assert run_campaign(tmp_path / 'c', path, token) == 'completed'
    declaration, state = json.loads(path.read_text()), state_of(tmp_path / 'c')
    counters = state['counters']
    for seed in ('42', '7'):
        training = counters['training'][seed]
        summaries = [json.loads((tmp_path / f'c/runs/seed-{seed}/generations/generation-{g:04d}/summary.json')
                                .read_text())['summary'] for g in (1, 2)]
        assert training['generations_attempted'] == training['generations_completed'] == 2
        assert training['selfplay_games_attempted'] == training['selfplay_games_completed'] == 4
        assert training['plies'] == sum(s['new_positions'] for s in summaries)
        assert training['optimizer_steps'] == sum(s['updates'] for s in summaries)
    evaluation = counters['evaluation']
    assert all(v['started'] == v['completed'] for v in evaluation['by_kind'].values())
    assert evaluation['total_games_started'] == evaluation['total_games_completed'] \
        == declaration['budgets']['planned_evaluation_games']
    assert evaluation['development_games'] + evaluation['sealed_games'] + evaluation['calibration_games'] \
        == evaluation['total_games_completed']
    assert evaluation['calibration_games'] == 2 and evaluation['sealed_games'] == 2 * (4 + 2)
    phases = counters['time']['phase_seconds']
    assert {'setup', 'seed-42:training', 'seed-7:development', 'final'} <= set(phases)
    assert counters['time']['elapsed_seconds'] <= declaration['budgets']['campaign_seconds']


def test_selections_are_fixed_before_any_sealed_inference(tmp_path, tiny_packages, fake_runtime):
    path, token = in_process(tiny_packages, fake_runtime, 'fixed.json', seeds=(42, 7))
    seen = {}

    def observe(point, campaign):
        if point == 'after_unit_begin:final_tactical_row' and not seen:
            seen['state'] = state_of(campaign.directory)
            seen['started'] = (campaign.directory / 'final/started.json').exists()
    assert run_campaign(tmp_path / 'c', path, token, fault=observe) == 'completed'
    assert seen['state']['state'] == 'RUNNING_FINAL_EVALUATION' and seen['started']
    fixed = seen['state']['history'][-1]['selections']
    for seed in ('42', '7'):
        assert fixed[seed] == json.loads((tmp_path / f'c/runs/seed-{seed}/selection.json').read_text())


# Deadlines and limits (fake clock) => INCOMPLETE -------------------------------------------------------

def test_deadline_during_development_is_incomplete(tmp_path, tiny_packages, fake_runtime):
    clock = Clock()
    path, token = in_process(tiny_packages, fake_runtime, 'dev-deadline.json', campaign_seconds=1000)

    def expire(point, campaign):
        if point == 'before_unit_complete:development_arena_game':
            clock.now = 1000.5
    status = run_campaign(tmp_path / 'c', path, token, clock=clock, fault=expire)
    assert status == 'incomplete: campaign wall-clock budget exceeded before the work was accepted'
    state = state_of(tmp_path / 'c')
    assert state['state'] == 'INCOMPLETE' and state['outcome']['during'] == 'RUNNING_SEED_42'
    assert state['counters']['evaluation']['by_kind']['development_arena_game'] == dict(started=1, completed=0)
    assert not list((tmp_path / 'c/runs/seed-42').glob('champion-check-*.json'))


def test_no_unit_starts_at_the_deadline(tmp_path, tiny_packages, fake_runtime):
    clock = Clock()
    path, token = in_process(tiny_packages, fake_runtime, 'start-deadline.json', campaign_seconds=1000)

    def expire(point, campaign):
        if point == 'after_generation_saved' and campaign.budget.phase_seconds.get('seed-42:training') is not None:
            clock.now = 1000.0  # exactly at the limit: nothing new may start
    status = run_campaign(tmp_path / 'c', path, token, clock=clock, fault=expire)
    assert status == 'incomplete: campaign wall-clock budget exhausted'
    state = state_of(tmp_path / 'c')
    assert state['counters']['evaluation']['total_games_started'] == 0
    assert not (tmp_path / 'c/runs/seed-42/generations/generation-0001/summary.json').exists()


def test_training_overrun_is_never_accepted(tmp_path, tiny_packages, fake_runtime, monkeypatch):
    clock = Clock()
    path, token = in_process(tiny_packages, fake_runtime, 'training-cap.json', per_run_training_seconds=100)
    from games.connect4.alphazero_v2 import generation as G
    real = G.GenerationRunner.run_generation

    def slow_last_update(self, *args, **kwargs):
        summary = real(self, *args, **kwargs)
        clock.now += 101.0  # the final optimizer update ran past the per-seed cap
        return summary
    monkeypatch.setattr(G.GenerationRunner, 'run_generation', slow_last_update)
    status = run_campaign(tmp_path / 'c', path, token, clock=clock)
    assert status == ('incomplete: seed 42 collection+optimization budget exceeded before the generation was '
                      'accepted')
    assert not (tmp_path / 'c/runs/seed-42/generations/generation-0001').exists()
    assert state_of(tmp_path / 'c')['counters']['training']['42']['generations_completed'] == 0


def test_last_calibration_overrun_is_incomplete(tmp_path, tiny_packages, fake_runtime, monkeypatch):
    """B3 regression: the last calibration game overruns the campaign cap after its final check."""
    clock = Clock()
    path, token = in_process(tiny_packages, fake_runtime, 'calibration.json', campaign_seconds=1000)
    import games.connect4.alphazero_v2.evaluation as E2
    real, seen = E2.calibration_games, {}

    def overrunning(*args, check=None, **kwargs):
        seen['check'] = check is not None
        records = real(*args, check=check, **kwargs)
        clock.now = 1000.5
        return records
    monkeypatch.setattr(E2, 'calibration_games', overrunning)
    status = run_campaign(tmp_path / 'c', path, token, clock=clock)
    assert seen['check'] is True
    assert status == 'incomplete: campaign wall-clock budget exceeded before the work was accepted'
    assert not (tmp_path / 'c/final/seed-42.json').exists()
    assert state_of(tmp_path / 'c')['outcome']['during'] == 'RUNNING_FINAL_EVALUATION'
    assert_not_official(tmp_path / 'c')


@pytest.mark.parametrize('at,eligible', [(9.9, True), (10.0, True), (10.1, False)],
                         ids=['just-under', 'exactly-at', 'just-over'])
def test_completion_barrier_deadline_boundary(tmp_path, tiny_packages, fake_runtime, at, eligible):
    """The endpoint is the barrier's one reading; limits are inclusive (<=), like every acceptance check."""
    clock = Clock()
    path, token = in_process(tiny_packages, fake_runtime, 'boundary.json', campaign_seconds=10)

    def late(point, campaign):
        if point == 'after_unit_begin:calibration_game':
            clock.now = 9.0
        if point == 'before_completion_barrier':  # every result is published and validated already
            clock.now = at
    status = run_campaign(tmp_path / 'c', path, token, clock=clock, fault=late)
    state = state_of(tmp_path / 'c')
    assert state['counters']['time']['elapsed_seconds'] == at
    if eligible:
        assert status == 'completed' and state['state'] == 'COMPLETE'
        assert state['outcome']['completion']['elapsed_seconds'] == at
        assert C.official_results(tmp_path / 'c')['completion']['elapsed_seconds'] == at
    else:
        assert status == 'incomplete: campaign wall-clock budget exceeded at the completion barrier'
        assert state['state'] == 'INCOMPLETE' and state['outcome']['during'] == 'RUNNING_FINAL_EVALUATION'
        assert (tmp_path / 'c/final/seed-42.json').exists()  # forensic artifact only
        assert_not_official(tmp_path / 'c')


def test_exact_game_ceiling_completes_and_one_less_is_incomplete(tmp_path, tiny_packages, fake_runtime):
    path, token = in_process(tiny_packages, fake_runtime, 'ceiling.json', ceiling='planned')
    assert run_campaign(tmp_path / 'exact', path, token) == 'completed'
    planned = json.loads(path.read_text())['budgets']['planned_evaluation_games']
    assert state_of(tmp_path / 'exact')['counters']['evaluation']['total_games_started'] == planned
    # A declaration cannot plan more games than its ceiling; one slot taken by anything else is fatal.

    def take_a_slot(point, campaign):
        if point == 'after_transition:RUNNING_FINAL_EVALUATION':
            campaign.budget.games_started += 1
    status = run_campaign(tmp_path / 'short', path, token, fault=take_a_slot)
    assert status == 'incomplete: evaluation-game ceiling reached'
    state = state_of(tmp_path / 'short')
    assert state['state'] == 'INCOMPLETE' and state['counters']['evaluation']['total_games_completed'] == planned - 1


def test_cooperative_stop_during_sealed_evaluation_is_incomplete(tmp_path, tiny_packages, fake_runtime):
    path, token = in_process(tiny_packages, fake_runtime, 'sealed-stop.json')
    status = run_campaign(tmp_path / 'c', path, token, extra_check=stop_at('evaluation', kind='final_ladder_game'))
    assert status == 'incomplete: test stop at evaluation'
    state = state_of(tmp_path / 'c')
    assert state['state'] == 'INCOMPLETE' and state['outcome']['during'] == 'RUNNING_FINAL_EVALUATION'
    assert_not_official(tmp_path / 'c')
    with pytest.raises(C.TerminalCampaign):
        launch(tmp_path / 'c', path, token)


# Ownership and existing directories ------------------------------------------------------------------------

def test_second_process_cannot_advance_a_live_campaign(tmp_path, tiny_packages, fake_runtime):
    path, token = in_process(tiny_packages, fake_runtime, 'lock.json')
    first = launch(tmp_path / 'c', path, token)
    try:
        with pytest.raises(C.CampaignLocked):
            launch(tmp_path / 'c', path, token)
        assert C.campaign_status(tmp_path / 'c')['state'] == 'CREATED'  # read-only status needs no lock
    finally:
        first.close()
    # The owner ended without a terminal state: the campaign is now INCOMPLETE, never continued.
    with pytest.raises(C.InterruptedCampaign, match=L.INTERRUPTED_MESSAGE):
        launch(tmp_path / 'c', path, token)
    assert state_of(tmp_path / 'c')['state'] == 'INCOMPLETE'


def test_existing_directories_are_never_adopted(tmp_path, tiny_packages, fake_runtime):
    path, token = in_process(tiny_packages, fake_runtime, 'bound-a.json')
    other, other_token = in_process(tiny_packages, fake_runtime, 'bound-b.json', games=3)
    (tmp_path / 'empty').mkdir()
    (tmp_path / 'unrelated').mkdir()
    (tmp_path / 'unrelated/data.txt').write_text('not a campaign')
    for name in ('empty', 'unrelated'):
        with pytest.raises(FileExistsError, match='new campaign directory'):
            launch(tmp_path / name, path, token)
    launch(tmp_path / 'c', path, token).close()
    before = tree(tmp_path / 'c', exclude=('campaign.lock',))
    with pytest.raises(FileExistsError, match='different declaration'):
        launch(tmp_path / 'c', other, other_token)
    assert tree(tmp_path / 'c', exclude=('campaign.lock',)) == before  # another declaration never touches it
    (tmp_path / 'c/state.json').write_text('{"torn')
    with pytest.raises(C.TerminalCampaign, match='cannot be resumed'):
        launch(tmp_path / 'c', path, token)
    assert C.campaign_status(tmp_path / 'c')['state'] is None
    with pytest.raises(L.StateCorrupt):
        C.official_results(tmp_path / 'c')


def test_forged_complete_state_without_matching_results_is_not_official(tmp_path, tiny_packages, fake_runtime):
    path, token = in_process(tiny_packages, fake_runtime, 'forged.json')
    assert run_campaign(tmp_path / 'c', path, token) == 'completed'
    result = tmp_path / 'c/final/seed-42.json'
    result.write_text(result.read_text().replace('"complete"', '"tampered"'))
    with pytest.raises(C.NotOfficialEvidence, match='differs from the hash certified'):
        C.official_results(tmp_path / 'c')


# Phase 4D.3B.3 terminal commit: completion barrier, signals and publication failures ------------------------
# F1 and F2 of the Phase 4D.3B.2 final launch review. Fault injection inside the real atomic publication only.

def writes_complete(source):
    return b'"state": "COMPLETE"' in Path(source).read_bytes()


@pytest.fixture
def saved_handlers():
    saved = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    yield
    for s, handler in saved.items():
        signal.signal(s, handler)


@pytest.fixture
def recorded_campaigns(monkeypatch):
    """Every Campaign the CLI constructs (to inspect its budget after ``main`` returns)."""
    made = []

    class Recording(C.Campaign):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            made.append(self)
    monkeypatch.setattr(C, 'Campaign', Recording)
    return made


def inject_into_terminal_publication(patch, directory, where, action):
    """Run ``action`` at one point inside the atomic publication of the COMPLETE document."""
    real_fsync, real_replace, real_dir_fsync = L.full_fsync, os.replace, L.fsync_directory
    temporary = directory / f'.state.json.{os.getpid()}.tmp'
    if where == 'temporary-fsync':
        def full_fsync(descriptor):
            real_fsync(descriptor)
            if temporary.exists() and writes_complete(temporary):
                action()
        patch.setattr(L, 'full_fsync', full_fsync)
    elif where in ('rename', 'after-rename'):
        def replace(source, target):
            complete = Path(target).name == 'state.json' and writes_complete(source)
            if complete and where == 'rename':
                action()
            real_replace(source, target)
            if complete and where == 'after-rename':
                action()
        patch.setattr(L.os, 'replace', replace)
    elif where == 'directory-fsync':
        fired = []

        def fsync_directory(path):
            if (Path(path) == directory and not fired and (directory / 'state.json').exists()
                    and state_of(directory)['state'] == 'COMPLETE'):
                fired.append(path)
                action()
            return real_dir_fsync(path)
        patch.setattr(L, 'fsync_directory', fsync_directory)
    else:
        raise ValueError(where)


@pytest.mark.parametrize('where', ['temporary-fsync', 'rename', 'directory-fsync'])
def test_signal_during_terminal_commit_cannot_certify_an_accepted_stop(tmp_path, tiny_packages, fake_runtime,
                                                                       recorded_campaigns, saved_handlers, capsys,
                                                                       where):
    """F1 (signal). The review's probe: a real SIGTERM, received by the CLI's installed handler, while COMPLETE
    is being published. Before 4D.3B.3 the handler accepted the stop (stop_reason set while the state was
    non-terminal) and COMPLETE was still certified. Now the barrier precedes the publication: the signal is
    post-completion, recorded, and can neither stop nor interrupt the commit."""
    path, token = in_process(tiny_packages, fake_runtime, f'signal-{where}.json')
    directory, seen = tmp_path / 'c', {}

    def deliver():
        seen['state'] = state_of(directory)['state']
        os.kill(os.getpid(), signal.SIGTERM)
        time.sleep(0.01)  # let the interpreter run the installed handler here, inside the publication
        seen['stop_reason'] = recorded_campaigns[0].budget.stop_reason
    with pytest.MonkeyPatch.context() as patch:
        inject_into_terminal_publication(patch, directory, where, deliver)
        code = C.main(['launch', '--declaration', str(path), '--campaign-dir', str(directory), '--authorize', token])
    budget = recorded_campaigns[0].budget
    assert seen['state'] == ('COMPLETE' if where == 'directory-fsync' else 'RUNNING_FINAL_EVALUATION')
    state = state_of(directory)
    assert state['state'] == 'COMPLETE' and code == C.EXIT_COMPLETED
    # A stop accepted before terminal commitment can never be certified COMPLETE:
    assert budget.stop_reason is None and seen['stop_reason'] is None
    assert budget.late_stop_requests == [f'signal {int(signal.SIGTERM)}']
    assert 'after the completion barrier' in capsys.readouterr().err
    assert C.official_results(directory)['completion'] == state['outcome']['completion']


def test_signal_immediately_before_the_completion_barrier_is_incomplete(tmp_path, tiny_packages, fake_runtime,
                                                                        saved_handlers):
    path, token = in_process(tiny_packages, fake_runtime, 'signal-before.json')
    campaign = launch(tmp_path / 'c', path, token, fault=lambda point, campaign: (
        signal.raise_signal(signal.SIGTERM) if point == 'before_completion_barrier' else None))
    C.install_stop_handlers(campaign)
    status = campaign.run()
    assert status == f'incomplete: signal {int(signal.SIGTERM)}'
    state = state_of(tmp_path / 'c')
    assert state['state'] == 'INCOMPLETE' and state['outcome']['during'] == 'RUNNING_FINAL_EVALUATION'
    assert 'COMPLETE' not in [h['state'] for h in state['history']]
    assert sorted(p.name for p in (tmp_path / 'c/final').glob('seed-*.json')) == ['seed-42.json']  # forensic
    assert_not_official(tmp_path / 'c')


def test_signal_after_complete_changes_nothing(tmp_path, tiny_packages, fake_runtime, saved_handlers):
    path, token = in_process(tiny_packages, fake_runtime, 'signal-after.json')
    seen = {}

    def after(point, campaign):
        if point == 'after_complete':
            seen['bytes'] = (tmp_path / 'c/state.json').read_bytes()
            signal.raise_signal(signal.SIGTERM)
    campaign = launch(tmp_path / 'c', path, token, fault=after)
    C.install_stop_handlers(campaign)
    assert campaign.run() == 'completed'
    signal.raise_signal(signal.SIGINT)  # handlers are still installed after the owner finished
    assert (tmp_path / 'c/state.json').read_bytes() == seen['bytes']
    assert campaign.budget.stop_reason is None and len(campaign.budget.late_stop_requests) == 2
    assert campaign.post_commit_errors == []
    C.official_results(tmp_path / 'c')


def test_clock_after_eligibility_never_changes_the_recorded_completion_time(tmp_path, tiny_packages, fake_runtime):
    """F1 (deadline). The review's probe: 9.9 s at the end of the work, 10.1 s at the terminal rename. The
    barrier's reading is the authoritative endpoint; certification time after it is not campaign work. The
    endpoint is bound in the COMPLETE record and acceptance re-checks it against the declared budget."""
    clock = Clock()
    path, token = in_process(tiny_packages, fake_runtime, 'post-eligibility.json', campaign_seconds=10)

    def late(point, campaign):
        if point == 'before_campaign_complete':
            clock.now = 9.9
        if point == 'after_completion_barrier':
            clock.now = 10.05
    directory = tmp_path / 'c'
    with pytest.MonkeyPatch.context() as patch:
        inject_into_terminal_publication(patch, directory, 'rename', lambda: setattr(clock, 'now', 10.1))
        status = run_campaign(directory, path, token, clock=clock, fault=late)
    assert status == 'completed' and clock.now == 10.1
    # A COMPLETE record whose endpoint is past the budget is never accepted, even with valid result files.
    original = (directory / 'state.json').read_bytes()
    late_endpoint = json.loads(original)
    late_endpoint['counters']['time']['elapsed_seconds'] = 10.1
    late_endpoint['outcome'].setdefault('completion', {})['elapsed_seconds'] = 10.1
    (directory / 'state.json').write_bytes(L.view_bytes(late_endpoint))
    with pytest.raises(C.NotOfficialEvidence, match='past the campaign wall-clock budget'):
        C.official_results(directory)
    (directory / 'state.json').write_bytes(original)
    state = state_of(directory)
    completion = state['outcome']['completion']
    assert completion['elapsed_seconds'] == state['counters']['time']['elapsed_seconds'] == 9.9
    assert completion['limits']['campaign_seconds'] == 10
    assert C.official_results(directory)['completion']['elapsed_seconds'] == 9.9


@pytest.mark.parametrize('where', ['temporary-fsync', 'rename'])
def test_publication_failure_before_complete_is_visible_is_incomplete(tmp_path, tiny_packages, fake_runtime,
                                                                       where):
    path, token = in_process(tiny_packages, fake_runtime, f'fail-before-{where}.json')
    directory = tmp_path / 'c'

    def fail():
        raise OSError(f'injected {where} failure')
    with pytest.MonkeyPatch.context() as patch:
        inject_into_terminal_publication(patch, directory, where, fail)
        with pytest.raises(OSError, match='injected'):
            run_campaign(directory, path, token)
    state = state_of(directory)
    assert state['state'] == 'INCOMPLETE' and 'OSError' in state['outcome']['reason']
    assert [h['state'] for h in state['history']][-2:] == ['RUNNING_FINAL_EVALUATION', 'INCOMPLETE']
    assert not list(directory.glob('.state.json.*'))  # no stray COMPLETE document left behind
    assert_not_official(directory)


@pytest.mark.parametrize('where,error', [('after-rename', OSError), ('after-rename', KeyboardInterrupt),
                                         ('directory-fsync', OSError), ('after_complete', RuntimeError)])
def test_failure_after_complete_is_visible_never_downgrades_it(tmp_path, tiny_packages, fake_runtime, where, error):
    """F2. The review's probe is ``directory-fsync``: an OSError at the directory fsync right after the terminal
    rename, while readers already see (and accept) COMPLETE. Before 4D.3B.3 the error handler then replaced it
    with INCOMPLETE rebuilt from a stale document. Now the owner adopts the visible COMPLETE and reports it."""
    path, token = in_process(tiny_packages, fake_runtime, f'fail-after-{where}-{error.__name__}.json')
    directory, seen = tmp_path / 'c', {}

    def fail():
        seen['visible'] = (directory / 'state.json').read_bytes()
        seen['accepted'] = sorted(C.official_results(directory)['results'])
        raise error(f'injected failure {where}')
    campaign = launch(directory, path, token, fault=lambda point, campaign: (
        fail() if point == where else None))
    with pytest.MonkeyPatch.context() as patch:
        if where != 'after_complete':
            inject_into_terminal_publication(patch, directory, where, fail)
        status = campaign.run()
    assert status == 'completed'
    assert seen['accepted'] == ['42'] and (directory / 'state.json').read_bytes() == seen['visible']
    state = state_of(directory)
    history = [h['state'] for h in state['history']]
    assert state['state'] == 'COMPLETE' and history[-2:] == ['RUNNING_FINAL_EVALUATION', 'COMPLETE']
    assert history.count('COMPLETE') == 1 and 'INCOMPLETE' not in history
    assert len(campaign.post_commit_errors) >= 1 and 'injected failure' in campaign.post_commit_errors[0]
    assert C.official_results(directory)['results'].keys() == {'42'}
    with pytest.raises(C.TerminalCampaign, match='already COMPLETE'):
        launch(directory, path, token)
    assert (directory / 'state.json').read_bytes() == seen['visible']


def test_official_results_validates_the_complete_record(tmp_path, tiny_packages, fake_runtime):
    """A COMPLETE string is not enough: the token, every certified file, the counts and the completion
    endpoint must all validate against the declaration."""
    path, token = in_process(tiny_packages, fake_runtime, 'acceptance.json', campaign_seconds=1000)
    directory = tmp_path / 'c'
    assert run_campaign(directory, path, token) == 'completed'
    state_path = directory / 'state.json'
    original = state_path.read_bytes()
    assert C.official_results(directory)['declaration_sha256'] == token

    def forged(change):
        document = json.loads(original)
        change(document)
        return document

    def at(document, *keys):
        for key in keys[:-1]:
            document = document[key]
        return document, keys[-1]

    def setter(*keys, value):
        def change(document):
            target, key = at(document, *keys)
            target[key] = value
        return change

    def remover(*keys):
        def change(document):
            target, key = at(document, *keys)
            del target[key]
        return change
    budgets = json.loads(path.read_text())['budgets']
    over = budgets['campaign_seconds'] + 1

    def endpoint_over(document):
        document['counters']['time']['elapsed_seconds'] = over
        document['outcome']['completion']['elapsed_seconds'] = over

    def training_over(document):
        for part in (document['counters']['time'], document['outcome']['completion']):
            part['training_seconds']['42'] = budgets['per_run_training_seconds'] + 1

    def games_over(document):
        ceiling = budgets['evaluation_games_ceiling'] + 1
        document['counters']['evaluation']['total_games_started'] = ceiling
        for part in (document['counters']['time'], document['outcome']['completion']):
            part['evaluation_games_started'] = ceiling
    forgeries = {
        'no completion record': remover('outcome', 'completion'),
        'endpoint past the budget': endpoint_over,
        'endpoint disagrees with its account': setter('outcome', 'completion', 'elapsed_seconds', value=0.0),
        'training past the per-seed budget': training_over,
        'games past the ceiling': games_over,
        'limits differ from the declaration': setter('outcome', 'completion', 'limits', 'campaign_seconds',
                                                     value=over),
        'unfinished evidence unit': setter('counters', 'evaluation', 'by_kind', 'calibration_game', 'completed',
                                           value=0),
        'missing generation': setter('counters', 'training', '42', 'generations_completed', value=1),
        'started.json hash': setter('outcome', 'started_sha256', value='0' * 64),
        'missing seed result': setter('outcome', 'final_results', value={}),
        'other declaration': setter('declaration_sha256', value='0' * 64),
        'rejected token': setter('declaration_sha256', value=OLD_TOKENS['ff1ded6']),
    }
    for name, change in forgeries.items():
        state_path.write_bytes(L.view_bytes(forged(change)))
        with pytest.raises(C.NotOfficialEvidence):
            C.official_results(directory)
            pytest.fail(f'forged COMPLETE accepted: {name}')
    state_path.write_bytes(original)
    assert C.official_results(directory)['declaration_sha256'] == token


# Hard interruption (real fresh interpreters): never resumable --------------------------------------------

CRASH_SCRIPT = '''
import json, os, sys
from pathlib import Path
from games.connect4.alphazero_v2 import campaign as C
declaration, token, directory, fault = sys.argv[1], sys.argv[2], Path(sys.argv[3]), json.loads(sys.argv[4])
count = [0]

def crash(point, campaign):
    if fault and point == fault[0]:
        count[0] += 1
        if count[0] == fault[1]:
            os._exit(91)  # hard interruption: no terminal state, no cleanup
print(json.dumps(C.Campaign(directory, declaration, token, fault=crash).run()))
'''

REFERENCE_SCRIPT = '''
import json, sys
from games.connect4.alphazero_v2.config import V2Config
from games.connect4.alphazero_v2.generation import GenerationRunner
from games.connect4.alphazero_v2.provenance import configure_deterministic_runtime
configure_deterministic_runtime(1)
declaration = json.load(open(sys.argv[1]))
hashes = {}
for seed in declaration["seeds"]:
    runner = GenerationRunner(V2Config.from_dict(dict(declaration["config"], seed=seed)))
    hashes[str(seed)] = [runner.state_sha256()]
    for _ in range(declaration["generations"]):
        runner.run_generation()
        hashes[str(seed)].append(runner.state_sha256())
print(json.dumps(hashes))
'''

CRASH_POINTS = [
    ('after_transition:RUNNING_SEED_42', 1),
    ('progress:game', 1),                               # a self-play game completed, before any counter snapshot
    ('progress:update', 1),
    ('after_generation_saved', 2),                      # a valid generation-1 resume boundary exists
    ('after_unit_begin:development_arena_game', 1),
    ('after_transition:RUNNING_SEED_7', 1),             # between seeds
    ('after_transition:DEVELOPMENT_SELECTION_COMPLETE', 1),
    ('after_transition:RUNNING_FINAL_EVALUATION', 1),   # selections fixed, no sealed inference yet
    ('after_unit_begin:final_ladder_game', 1),          # sealed interruption
    ('before_unit_complete:calibration_game', 2),
    ('before_campaign_complete', 1),                    # every result file written, COMPLETE not yet written
]


def strip_timing(value):
    if isinstance(value, dict):
        return {k: strip_timing(v) for k, v in value.items() if k not in ('seconds', 'utc', 'resources', 'files')}
    if isinstance(value, list):
        return [strip_timing(v) for v in value]
    return value


def fingerprint(directory):
    """Every deterministic campaign result (timings, and artifact files that embed Git provenance, excluded)."""
    directory = Path(directory)
    files = sorted(directory.glob('runs/seed-*/generations/generation-*/summary.json')) \
        + sorted(directory.glob('runs/seed-*/*.json')) + sorted(directory.glob('final/seed-*.json'))
    return {p.relative_to(directory).as_posix(): strip_timing(json.loads(p.read_text())) for p in files}


def cli_launch(path, token, directory):
    return subprocess.run([sys.executable, '-m', 'games.connect4.alphazero_v2.campaign', 'launch', '--declaration',
                           str(path), '--campaign-dir', str(directory), '--authorize', token],
                          capture_output=True, text=True, timeout=900, cwd=ROOT, env=pinned_env())


@pytest.fixture(scope='module')
def crash_matrix(tmp_path_factory, tiny_packages, launch_runtime):
    """Uninterrupted tiny two-seed campaigns and one hard crash + later invocations per point, in parallel."""
    from concurrent.futures import ThreadPoolExecutor
    root = tmp_path_factory.mktemp('crash-matrix')
    path, token = write_declaration(tiny_packages, declaration_document(
        tiny_packages, launch_runtime, seeds=(42, 7), schedule=(1, 2)), 'crash.json')

    def crash_run(directory, fault):
        return subprocess.run([sys.executable, '-c', CRASH_SCRIPT, str(path), token, str(directory),
                               json.dumps(fault)], capture_output=True, text=True, timeout=900, cwd=ROOT,
                              env=pinned_env())

    def scenario(index_point):
        index, point = index_point
        directory = root / f'crash-{index:02d}'
        first = crash_run(directory, list(point))
        after_crash = dict(state=state_of(directory), tree=tree(directory)) if first.returncode == 91 else None
        second = cli_launch(path, token, directory) if first.returncode == 91 else None
        after_second = dict(state=state_of(directory), tree=tree(directory)) if second else None
        third = cli_launch(path, token, directory) if second else None
        return point, dict(first=first, after_crash=after_crash, second=second, after_second=after_second,
                           third=third, directory=directory)
    with ThreadPoolExecutor(max_workers=max(2, (os.cpu_count() or 2) // 2)) as pool:
        plain = pool.submit(cli_launch, path, token, root / 'plain')
        again = pool.submit(crash_run, root / 'again', None)
        reference = pool.submit(subprocess.run, [sys.executable, '-c', REFERENCE_SCRIPT, str(path)],
                                capture_output=True, text=True, timeout=900, cwd=ROOT, env=pinned_env())
        crashes = dict(pool.map(scenario, enumerate(CRASH_POINTS)))
    return dict(root=root, path=path, token=token, plain=plain.result(), again=again.result(),
                reference=reference.result(), crashes=crashes)


def test_uninterrupted_tiny_campaign_completes_identically_to_the_expected_result(crash_matrix):
    root, plain = crash_matrix['root'], crash_matrix['plain']
    assert plain.returncode == C.EXIT_COMPLETED, plain.stdout[-2000:] + plain.stderr[-3000:]
    assert plain.stdout.strip().splitlines()[-1] == 'completed'
    assert crash_matrix['again'].returncode == 0, crash_matrix['again'].stderr[-3000:]
    assert state_of(root / 'plain')['state'] == state_of(root / 'again')['state'] == 'COMPLETE'
    # Two independent uninterrupted campaigns produce identical results.
    assert fingerprint(root / 'plain') == fingerprint(root / 'again') and fingerprint(root / 'plain')
    # Evaluation interleaving never changes training: every generation equals a standalone trajectory.
    assert crash_matrix['reference'].returncode == 0, crash_matrix['reference'].stderr[-3000:]
    expected = json.loads(crash_matrix['reference'].stdout.strip().splitlines()[-1])
    for seed, hashes in expected.items():
        for generation, digest in enumerate(hashes):
            summary = root / f'plain/runs/seed-{seed}/generations/generation-{generation:04d}'
            if generation:
                assert json.loads((summary / 'summary.json').read_text())['state_sha256'] == digest
    official = C.official_results(root / 'plain')
    assert sorted(official['results']) == ['42', '7']
    status = subprocess.run([sys.executable, '-m', 'games.connect4.alphazero_v2.campaign', 'status',
                             '--campaign-dir', str(root / 'plain')], capture_output=True, text=True, timeout=120,
                            cwd=ROOT, env=pinned_env())
    assert status.returncode == 0 and json.loads(status.stdout)['state'] == 'COMPLETE'


@pytest.mark.parametrize('point', CRASH_POINTS, ids=[f'{p}#{n}' for p, n in CRASH_POINTS])
def test_interrupted_official_campaign_can_never_resume(crash_matrix, point):
    run = crash_matrix['crashes'][point]
    assert run['first'].returncode == 91, f"fault point never reached: {run['first'].stderr[-2000:]}"
    crashed = run['after_crash']['state']
    assert crashed['state'] not in L.TERMINAL_STATES  # the dead owner left a non-terminal state
    # A later invocation: INCOMPLETE, the required notice, nonzero exit, and no new work.
    second = run['second']
    assert second.returncode == C.EXIT_INCOMPLETE, second.stdout + second.stderr[-2000:]
    assert L.INTERRUPTED_MESSAGE in second.stdout
    state = run['after_second']['state']
    assert state['state'] == 'INCOMPLETE' and state['outcome']['reason'].startswith('interrupted')
    assert crashed['state'] in state['outcome']['reason']
    assert [h['state'] for h in state['history']] == [h['state'] for h in crashed['history']] + ['INCOMPLETE']
    assert run['after_second']['tree'] == run['after_crash']['tree']  # nothing repaired, rerun or added
    # Terminal: every later invocation is refused and changes nothing.
    third = run['third']
    assert third.returncode == C.EXIT_REFUSED and 'already INCOMPLETE' in third.stdout
    assert state_of(run['directory']) == state
    assert_not_official(run['directory'])


def test_valid_resume_boundary_does_not_permit_official_continuation(crash_matrix):
    run = crash_matrix['crashes'][('after_generation_saved', 2)]
    resume = run['directory'] / 'runs/seed-42/generations/generation-0001/generation-0001.resume.pt'
    runner = load_resume_boundary(resume, strict_runtime=False, restore_global_rng=False)
    assert runner.completed_generations == 1  # a valid, loadable generation boundary exists
    assert run['second'].returncode == C.EXIT_INCOMPLETE and state_of(run['directory'])['state'] == 'INCOMPLETE'
    assert not (run['directory'] / 'runs/seed-42/generations/generation-0002').exists()


def test_lost_counters_cannot_authorize_continuation(crash_matrix):
    """R5: a game completed after the last counter snapshot is lost from the counters, and that is harmless."""
    run = crash_matrix['crashes'][('progress:game', 1)]
    counters = run['after_crash']['state']['counters']
    assert counters['training']['42']['selfplay_games_completed'] == 0  # a lower bound: one game physically ran
    assert run['after_second']['state']['state'] == 'INCOMPLETE'


def test_partial_sealed_results_are_never_official(crash_matrix):
    run = crash_matrix['crashes'][('before_campaign_complete', 1)]
    directory = run['directory']
    assert sorted(p.name for p in (directory / 'final').glob('seed-*.json')) == ['seed-42.json', 'seed-7.json']
    assert run['after_second']['state']['outcome']['reason'].startswith(
        'interrupted: owner pid') and 'RUNNING_FINAL_EVALUATION' in run['after_second']['state']['outcome']['reason']
    assert_not_official(directory)


def test_signal_stops_the_cli_and_the_campaign_is_incomplete(tmp_path, tiny_packages, launch_runtime):
    path, token = write_declaration(tiny_packages, declaration_document(tiny_packages, launch_runtime, games=40),
                                    'signal.json')
    directory = tmp_path / 'signal'
    process = subprocess.Popen([sys.executable, '-m', 'games.connect4.alphazero_v2.campaign', 'launch',
                                '--declaration', str(path), '--campaign-dir', str(directory),
                                '--authorize', token], cwd=ROOT, env=pinned_env(),
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    deadline = time.time() + 120
    while time.time() < deadline and not (directory / 'runs/seed-42/generations/generation-0000').exists():
        time.sleep(0.2)
    process.send_signal(signal.SIGINT)
    stdout, stderr = process.communicate(timeout=120)
    assert process.returncode == C.EXIT_INCOMPLETE, stdout + stderr
    state = state_of(directory)
    assert state['state'] == 'INCOMPLETE' and state['outcome']['reason'] == f'signal {int(signal.SIGINT)}'
    assert not (directory / 'runs/seed-42/generations/generation-0001').exists()
