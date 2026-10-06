"""Phase 4D.3B/4D.3B.2 torch-side preflight: evaluation agents, diagnostics, profiling helpers and the
fail-closed official campaign launcher (declaration binding, single owner, no resume, deadlines and budgets).
Every run uses tiny synthetic settings in temporary directories; no research training, learned
checkpoint or strength claim is produced."""
import hashlib
import inspect
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
from games.connect4.alphazero_v2 import official_evidence as OE
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
              'ff1ded6': 'ed641aea2fef86a29000df55d1e46d9a32bf48dc3ec8f5dbcc0f8449668d0bbb',   # Phase 4D.3B.2
              'bf48303': '34b4d899639d59d1bd4408ecc505b7eb20cb4d9c2fdc12095402c246887ac390',   # Phase 4D.3B.3
              '85dd988': '9f7259827e8abee6db81113147e48c2a3420248296d09df559f2783c5d7d85f4'}   # Phase 4D.3B.4
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
                         ladder=('random', 'negamax1'), ladder_openings=1, calibration_games=1, **budget_overrides):
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
        final=dict(ladder=list(ladder), ladder_openings=ladder_openings, calibration_games=calibration_games,
                   calibration_constant=None, one_time=True),
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
    return pin_fake_runtime(monkeypatch)


def pin_fake_runtime(patch):
    for name in THREAD_ENVIRONMENT:
        patch.setenv(name, '1')
    identity = dict(C.runtime_identity(), intra_op_threads=1, inter_op_threads=1, deterministic_algorithms=True,
                    deterministic_warn_only=False, thread_environment=required_thread_environment(1))
    current = dict(identity=identity)
    patch.setattr(C, 'runtime_identity', lambda: json.loads(json.dumps(current['identity'])))
    patch.setattr(C.Campaign, 'configure_runtime', lambda self: C.runtime_identity())
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
    assert sorted(declaration['launch_control']['rejected_declaration_tokens']) == sorted(OLD_TOKENS.values())
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
    assert C.required_seed_keys(declaration) == {'42', '314159'}  # acceptance's exact per-seed key set


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
                                                                  'packages', 'runs', 'source', 'state.json']
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


# Phase 4D.3B.4 strict COMPLETE schema: required fields, exact seed coverage, consistent copies ---------------
# One genuine two-seed COMPLETE campaign (the official seeds and per-seed limit; tiny synthetic work) is
# copied per test, and its state.json is forged. Every forged record must be refused.

SEEDS = {'42', '314159'}
LIMIT = 28800


@pytest.fixture(scope='module')
def official_complete(tmp_path_factory, tiny_packages):
    root = tmp_path_factory.mktemp('complete-schema')
    with pytest.MonkeyPatch.context() as patch:
        fake = pin_fake_runtime(patch)
        path, _ = in_process(tiny_packages, fake, 'complete-schema.json', seeds=C.OFFICIAL_SEEDS,
                             per_run_training_seconds=LIMIT)
        assert run_campaign(root / 'campaign', path, hashlib.sha256(path.read_bytes()).hexdigest()) == 'completed'
    return root / 'campaign'


@pytest.fixture
def complete_copy(official_complete, tmp_path):
    shutil.copytree(official_complete, tmp_path / 'campaign')
    return tmp_path / 'campaign'


def read_state(directory):
    return json.loads((directory / 'state.json').read_bytes())


def write_state(directory, document):
    (directory / 'state.json').write_bytes(L.view_bytes(document))


def refusal(directory):
    """The refusal message, or None if official_results accepted the record."""
    try:
        C.official_results(directory)
    except (C.NotOfficialEvidence, L.StateCorrupt) as error:
        return str(error)
    return None


def training_copies(document):
    return document['counters']['time']['training_seconds'], document['outcome']['completion']['training_seconds']


def test_exact_reproduced_exploit_is_refused(complete_copy):
    """The 4D.3B.3 blocker: 28,801 s for seed 42 is refused, and so is deleting seed 42 from both copies."""
    original = read_state(complete_copy)
    assert sorted(C.official_results(complete_copy)['results']) == ['314159', '42']
    over = json.loads(json.dumps(original))
    for copy in training_copies(over):
        copy['42'] = LIMIT + 1
    write_state(complete_copy, over)
    assert 'seed 42 training seconds exceed the per-seed budget' in refusal(complete_copy)
    deleted = json.loads(json.dumps(original))
    for copy in training_copies(deleted):
        del copy['42']
    write_state(complete_copy, deleted)
    message = refusal(complete_copy)
    assert message is not None, 'a COMPLETE record without seed 42 training time was accepted'
    assert "counters.time.training_seconds must cover exactly seeds ['314159', '42']; has ['314159']" in message
    assert "outcome.completion.training_seconds must cover exactly seeds ['314159', '42']; has ['314159']" in message


def rename(mapping, old, new):
    mapping[new] = mapping.pop(old)


TRAINING_FORGERIES = {
    'seed 314159 deleted from both': lambda t, c: (t.pop('314159'), c.pop('314159')),
    'both seeds deleted from both': lambda t, c: (t.clear(), c.clear()),
    'seed 42 deleted from counters only': lambda t, c: t.pop('42'),
    'seed 42 deleted from completion only': lambda t, c: c.pop('42'),
    'seed 314159 deleted from completion only': lambda t, c: c.pop('314159'),
    'unexpected extra seed in both': lambda t, c: (t.update({'7': 1.0}), c.update({'7': 1.0})),
    'non-canonical key 042 in both': lambda t, c: (rename(t, '42', '042'), rename(c, '42', '042')),
    'padded key " 42" in both': lambda t, c: (rename(t, '42', ' 42'), rename(c, '42', ' 42')),
    'counters and completion disagree': lambda t, c: (t.update({'42': 100.0}), c.update({'42': 200.0})),
    'negative seconds in both': lambda t, c: (t.update({'42': -1.0}), c.update({'42': -1.0})),
    'string seconds in both': lambda t, c: (t.update({'42': '100'}), c.update({'42': '100'})),
    'boolean seconds in both': lambda t, c: (t.update({'42': True}), c.update({'42': True})),
    'null seconds in both': lambda t, c: (t.update({'42': None}), c.update({'42': None})),
    'just over the limit in both': lambda t, c: (t.update({'42': LIMIT + 0.5}), c.update({'42': LIMIT + 0.5})),
    'mapping replaced by a list in both': None,
}


@pytest.mark.parametrize('name', sorted(TRAINING_FORGERIES))
def test_per_seed_training_time_requires_both_seeds_in_both_copies(complete_copy, name):
    document = read_state(complete_copy)
    time_copy, completion_copy = training_copies(document)
    if TRAINING_FORGERIES[name] is None:
        document['counters']['time']['training_seconds'] = list(time_copy.values())
        document['outcome']['completion']['training_seconds'] = list(completion_copy.values())
    else:
        TRAINING_FORGERIES[name](time_copy, completion_copy)
    write_state(complete_copy, document)
    assert refusal(complete_copy) is not None, f'forged COMPLETE accepted: {name}'


@pytest.mark.parametrize('token', ['NaN', 'Infinity', '-Infinity'])
def test_non_finite_training_time_is_refused(complete_copy, token):
    """view_bytes refuses NaN, but a forged state.json can still contain the token; json.loads would parse it."""
    document = read_state(complete_copy)
    for copy in training_copies(document):
        copy['42'] = 123456.789
    text = L.view_bytes(document).decode()
    assert text.count('123456.789') == 2
    (complete_copy / 'state.json').write_text(text.replace('123456.789', token))
    assert C.CampaignStateFile.load(complete_copy).state == 'COMPLETE'  # the ordinary loader accepts it
    assert 'non-finite JSON number' in refusal(complete_copy)


def test_duplicate_seed_keys_are_refused(complete_copy):
    """A duplicate key would otherwise resolve to its last value: an over-budget first copy could hide."""
    document = read_state(complete_copy)
    for copy in training_copies(document):
        copy['42'] = LIMIT + 1
    text = L.view_bytes(document).decode().replace(f'"42": {LIMIT + 1}', f'"42": {LIMIT + 1},\n   "42": 1.5')
    (complete_copy / 'state.json').write_text(text)
    assert training_copies(C.CampaignStateFile.load(complete_copy).document)[1]['42'] == 1.5
    assert 'duplicate JSON object keys' in refusal(complete_copy)


def test_exactly_the_per_seed_limit_is_accepted(complete_copy):
    document = read_state(complete_copy)
    for copy in training_copies(document):
        copy.update({'42': LIMIT, '314159': float(LIMIT)})
    write_state(complete_copy, document)
    assert C.official_results(complete_copy)['completion']['training_seconds'] == {'42': LIMIT, '314159': LIMIT}


def test_valid_two_seed_record_is_accepted(complete_copy):
    accepted = C.official_results(complete_copy)
    assert set(accepted['results']) == SEEDS and set(accepted['completion']['training_seconds']) == SEEDS
    assert all(accepted['results'][seed]['seed'] == int(seed) for seed in SEEDS)


def test_started_record_seed_coverage_is_checked_beyond_its_hash(complete_copy):
    """A consistently re-hashed started.json without seed 42's description is refused by the schema itself."""
    started_path = complete_copy / 'final/started.json'
    started = json.loads(started_path.read_bytes())
    del started['descriptions']['42']
    data = L.view_bytes(started)
    started_path.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    document = read_state(complete_copy)
    document['outcome']['started_sha256'] = digest
    next(h for h in document['history'] if h['state'] == L.RUNNING_FINAL)['started_sha256'] = digest
    write_state(complete_copy, document)
    assert "final/started.json descriptions must cover exactly seeds ['314159', '42']" in refusal(complete_copy)


def required_paths(document):
    """Every field of the COMPLETE record that acceptance requires. The only exclusions are diagnostic:
    individual phase names in counters.time.phase_seconds and provenance-only history/owner details."""
    def walk(value, prefix):
        if isinstance(value, dict):
            for key in value:
                path = prefix + (key,)
                if path[:3] == ('counters', 'time', 'phase_seconds') and len(path) > 3:
                    continue
                yield path
                yield from walk(value[key], path)
    yield from ((key,) for key in document)
    for part in ('outcome', 'counters'):
        yield from walk(document[part], (part,))
    for index, entry in enumerate(document['history']):
        if entry['state'] in (L.SELECTION_COMPLETE, L.RUNNING_FINAL):
            yield from walk(entry, ('history', index))
    # (history 'state'/'utc' and owner details are validated by CampaignStateFile or are provenance only)


def at_path(document, path):
    for key in path[:-1]:
        document = document[key]
    return document, path[-1]


def test_mutation_sweep_every_required_field_and_seed_key(complete_copy):
    """Delete each required field, and break each per-seed mapping (delete, extra or non-canonical seed), one at a
    time. Every mutated record must be refused; the untouched record and diagnostic-only deletions are accepted."""
    original = read_state(complete_copy)
    mutations = []
    for path in required_paths(original):
        if path[-1] in ('state', 'utc') and path[0] == 'history':
            continue
        mutations.append((f'delete {"/".join(map(str, path))}', path, 'delete'))
        parent, key = at_path(original, path)
        if isinstance(parent[key], dict) and set(parent[key]) == SEEDS:
            for change in ('delete-42', 'delete-314159', 'extra-7', 'rename-042'):
                mutations.append((f'{change} in {"/".join(map(str, path))}', path, change))
    accepted = []
    for name, path, change in mutations:
        document = json.loads(json.dumps(original))
        parent, key = at_path(document, path)
        if change == 'delete':
            del parent[key]
        elif change.startswith('delete-'):
            del parent[key][change[len('delete-'):]]
        elif change == 'extra-7':
            parent[key]['7'] = parent[key]['42']
        else:
            rename(parent[key], '42', '042')
        write_state(complete_copy, document)
        if refusal(complete_copy) is None:
            accepted.append(name)
    seed_mappings = sum(1 for _, _, change in mutations if change == 'extra-7')
    assert not accepted, f'{len(accepted)} of {len(mutations)} malformed COMPLETE records accepted: {accepted}'
    assert len(mutations) > 100 and seed_mappings == 6, (len(mutations), seed_mappings)
    # Diagnostic phase timings are typed but not a required set of names.
    document = json.loads(json.dumps(original))
    document['counters']['time']['phase_seconds'].pop('final')
    write_state(complete_copy, document)
    assert refusal(complete_copy) is None
    write_state(complete_copy, original)
    assert refusal(complete_copy) is None


@pytest.mark.parametrize('seeds,kind,valid', [
    ([42, 314159], 'research', True), ([42], 'research', False), ([314159, 42], 'research', False),
    ([42, 314159, 7], 'research', False), ([42, 7], 'test', True), ([42, 42], 'test', False),
    ([True, 2], 'test', False), (['42'], 'test', False), ([], 'test', False), ('42', 'test', False)])
def test_required_seed_keys_are_the_declared_seeds(seeds, kind, valid):
    declaration = dict(seeds=seeds, kind=kind)
    if valid:
        assert C.required_seed_keys(declaration) == {str(s) for s in seeds}
    else:
        with pytest.raises(C.MalformedRecord):
            C.required_seed_keys(declaration)


# Phase 4D.3B.5 official evidence contract: one schema, one result producer, schema-driven mutations -----------
# One genuine two-seed COMPLETE campaign with a richer tiny protocol than above: four ladder opponents (one
# reported-only, one v2 search agent), prefix openings and two calibration games. Mutations are generated from
# the produced artifacts themselves. Every path is deleted and retyped, and a member is added to every object.
# Only paths in OE.OPTIONAL_PATHS may be removed.

CONTRACT_LADDER = ('random', 'negamax1', 'negamax4', 'initial_v2_512')
STAGES = {'_final': 'A', '_complete': 'B', 'official_results': 'C'}


def validation_stage():
    for frame in inspect.stack():
        if frame.function in STAGES:
            return STAGES[frame.function]
    return None


@pytest.fixture(scope='module')
def contract_campaign(tmp_path_factory, tiny_packages):
    """The genuine campaign, with every contract validation recorded at its call site (A, B or C)."""
    root = tmp_path_factory.mktemp('evidence-contract')
    calls = []
    with pytest.MonkeyPatch.context() as patch:
        fake = pin_fake_runtime(patch)
        for name in ('seed_result_problems', 'started_problems'):
            real = getattr(OE, name)

            def spy(context, started, *args, _real=real, _name=name):
                problems = _real(context, started, *args)
                calls.append(dict(stage=validation_stage(), function=_name, args=json.loads(json.dumps(args)),
                                  started=json.loads(json.dumps(started)), problems=problems))
                return problems
            patch.setattr(OE, name, spy)
        path, token = in_process(tiny_packages, fake, 'evidence-contract.json', seeds=C.OFFICIAL_SEEDS,
                                 per_run_training_seconds=LIMIT, ladder=CONTRACT_LADDER, ladder_openings=3,
                                 calibration_games=2)
        assert run_campaign(root / 'campaign', path, token) == 'completed'
        official = C.official_results(root / 'campaign')
    return dict(directory=root / 'campaign', calls=calls, official=official, token=token)


@pytest.fixture
def contract_copy(contract_campaign, tmp_path):
    shutil.copytree(contract_campaign['directory'], tmp_path / 'campaign')
    return tmp_path / 'campaign'


@pytest.fixture
def cached_derivation(monkeypatch):
    """Memoize the result producer. It is a pure function of its inputs, so validation is unchanged; the
    thousands of mutated copies of a result share a few raw-evidence inputs, and bootstraps are slow."""
    real, cache = OE.derive_seed_result, {}

    def cached(context, seed, started, raw):
        key = json.dumps([context.token, seed, started, raw], sort_keys=True)
        if key not in cache:
            cache[key] = real(context, seed, started, raw)
        return cache[key]
    monkeypatch.setattr(OE, 'derive_seed_result', cached)


def read_json(path):
    return json.loads(Path(path).read_bytes())


def write_json(path, value):
    data = L.view_bytes(value)
    Path(path).write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def contract_inputs(directory):
    context, problems = OE.load_context(directory, read_state(directory)['declaration_sha256'])
    assert problems == []
    return context, read_json(directory / 'final/started.json')


def rehash_result(directory, seed, result):
    document = read_state(directory)
    document['outcome']['final_results'][seed] = write_json(directory / f'final/seed-{seed}.json', result)
    write_state(directory, document)


def rehash_started(directory, started):
    digest, document = write_json(directory / 'final/started.json', started), read_state(directory)
    document['outcome']['started_sha256'] = digest
    next(h for h in document['history'] if h['state'] == L.RUNNING_FINAL)['started_sha256'] = digest
    write_state(directory, document)


def strict_refusal(directory):
    """The refusal message (None if accepted). An internal validator error is a test failure: every validator
    must refuse malformed input by inspecting it, not by crashing into the fail-closed backstop."""
    message = refusal(directory)
    assert message is None or 'internal validation error' not in message, message
    return message


def json_paths(value, prefix=()):
    if isinstance(value, dict):
        items = value.items()
    elif isinstance(value, list):
        items = enumerate(value)
    else:
        return
    for key, child in items:
        yield prefix + (key,)
        yield from json_paths(child, prefix + (key,))


def schema_paths(document):
    """Every schema path of an artifact: list indices collapse to one schema position, mutated at its first and
    last occurrence (dict keys never collapse: seeds, row IDs and opponents are declaration-derived identities)."""
    groups = {}
    for path in json_paths(document):
        groups.setdefault(tuple('*' if isinstance(k, int) else k for k in path), []).append(path)
    for paths in groups.values():
        yield from dict.fromkeys((paths[0], paths[-1]))


def wrong_types(value):
    """Replacements of another JSON type. No field of the contract admits both of any such pair."""
    if isinstance(value, bool):
        return [int(value)]
    if isinstance(value, int):
        return [str(value), bool(value)]
    if isinstance(value, float):
        return [str(value)]
    if isinstance(value, str):
        return [len(value)]
    if value is None:
        return ['null']
    if isinstance(value, list):
        return [{str(i): v for i, v in enumerate(value)}]
    return [list(value.values())]


DELETE = object()


def mutated(document, path, value):
    copy = json.loads(json.dumps(document))
    parent = copy
    for key in path[:-1]:
        parent = parent[key]
    if value is DELETE:
        del parent[path[-1]]
    else:
        parent[path[-1]] = value
    return copy


def schema_mutations(document, artifact, paths=None):
    """(name, mutated copy, must refuse) for every schema path of a genuine artifact: delete it, retype it, and add
    an unexpected member to it if it is an object. Only declared optional diagnostics may be removed or extended."""
    for path in (schema_paths(document) if paths is None else paths):
        label = '/'.join(map(str, path))
        parent = document
        for key in path[:-1]:
            parent = parent[key]
        value = parent[path[-1]]
        yield f'delete {label}', mutated(document, path, DELETE), not OE.is_optional(artifact, path)
        for wrong in wrong_types(value):
            yield f'retype {label} as {type(wrong).__name__}', mutated(document, path, wrong), True
        if isinstance(value, dict):
            extended = dict(value, **{'unexpected-member': 0})
            yield (f'add a member to {label}', mutated(document, path, extended),
                   not OE.is_optional(artifact, path + ('unexpected-member',)))


def run_sweep(mutations, refused):
    """Apply every mutation; return (count, wrongly accepted, wrongly refused)."""
    count, accepted, rejected = 0, [], []
    for name, document, must_refuse in mutations:
        count += 1
        if refused(document) != must_refuse:
            (accepted if must_refuse else rejected).append(name)
    return count, accepted, rejected


def test_reproduced_exploit_certified_result_without_scientific_fields_is_refused(contract_copy):
    """The 4D.3B.4 blocker: final/seed-42.json without ladder, tactical, solved, evidence and calibration, its new
    SHA-256 written into the COMPLETE record, was accepted. Every required top-level field is now required."""
    original = read_json(contract_copy / 'final/seed-42.json')
    stripped = {k: v for k, v in original.items() if k not in ('ladder', 'tactical', 'solved', 'evidence',
                                                                 'calibration')}
    rehash_result(contract_copy, '42', stripped)
    message = strict_refusal(contract_copy)
    assert message is not None, 'a certified result without its scientific fields was accepted'
    assert "final/seed-42.json must have exactly keys" in message and "'calibration', 'evidence'" in message
    for key in sorted(OE.SEED_RESULT_KEYS):
        rehash_result(contract_copy, '42', {k: v for k, v in original.items() if k != key})
        assert strict_refusal(contract_copy) is not None, f'accepted without {key}'
    rehash_result(contract_copy, '42', original)
    assert strict_refusal(contract_copy) is None


def test_round_trip_produce_validate_serialize_parse_validate_complete_accept(contract_campaign):
    """produce -> validate (A) -> serialize -> strict parse -> validate (B) -> COMPLETE -> official_results (C)."""
    directory, calls = contract_campaign['directory'], contract_campaign['calls']
    results = [c for c in calls if c['function'] == 'seed_result_problems']
    assert [c['stage'] for c in results] == ['A', 'A', 'B', 'B', 'C', 'C']
    assert all({c['args'][0] for c in results if c['stage'] == stage} == SEEDS for stage in 'ABC')
    assert [c['args'][0] for c in results if c['stage'] == 'A'] == ['42', '314159']  # declared order, as run
    assert [c['stage'] for c in calls if c['function'] == 'started_problems'] == ['A', 'B', 'C']
    assert all(c['problems'] == [] for c in calls)
    official = contract_campaign['official']
    assert read_state(directory)['state'] == 'COMPLETE' and set(official['results']) == SEEDS
    for seed in sorted(SEEDS):
        produced, read_back, accepted = (c['args'][1] for c in results if c['args'][0] == seed)
        data = (directory / f'final/seed-{seed}.json').read_bytes()
        assert L.view_bytes(produced) == data  # the validated object is exactly the published bytes
        assert C.strict_json(data) == read_back == accepted == official['results'][seed]
        assert hashlib.sha256(data).hexdigest() == read_state(directory)['outcome']['final_results'][seed]
    started = [c for c in calls if c['function'] == 'started_problems']
    assert started[0]['started'] == started[1]['started'] == read_json(directory / 'final/started.json')


def test_produced_artifacts_have_exactly_the_contract_shape(contract_campaign):
    """The genuine artifacts carry exactly the declared key sets: a producer field without acceptance semantics
    would already have been refused before publication (see the next test)."""
    directory = contract_campaign['directory']
    context, started = contract_inputs(directory)
    for seed in sorted(SEEDS):
        result = read_json(directory / f'final/seed-{seed}.json')
        assert set(result) == OE.SEED_RESULT_KEYS and set(result['evidence']) == OE.EVIDENCE_KEYS
        assert set(result['ladder']) == set(CONTRACT_LADDER) and result['ladder']['negamax4']['gate'] is None
        assert all(set(e) == OE.TACTICAL_EVIDENCE_KEYS for e in result['evidence']['tactical'].values())
        assert all(set(e) == OE.SOLVED_EVIDENCE_KEYS for e in result['evidence']['solved'].values())
        assert {r['game'] for r in result['evidence']['calibration']} == {0, 1}
        records = [r for entry in result['ladder'].values() for r in entry['records']]
        assert all(set(r) == OE.ARENA_RECORD_KEYS for r in records) and len(records) == len(CONTRACT_LADDER) * 6
        assert {r['stratum'] for r in records} == {'empty', 'prefix'}
        assert OE.seed_result_problems(context, started, seed, result) == []
    assert read_state(directory)['counters']['evaluation']['by_kind'] == {
        kind: dict(started=count, completed=count) for kind, count in OE.expected_units(context).items()}


@pytest.mark.parametrize('where', ['search row', 'arena record', 'calibration record'])
def test_new_producer_field_without_acceptance_semantics_stops_publication(tmp_path, tiny_packages, fake_runtime,
                                                                           monkeypatch, where):
    """Adding a raw-evidence field to the producer without declaring it in the contract fails validation at A:
    the result is never published and the campaign ends INCOMPLETE."""
    import games.connect4.alphazero_v2.evaluation as E2
    if where == 'search row':
        real = E2.search_row
        monkeypatch.setattr(E2, 'search_row', lambda *a, **k: dict(real(*a, **k), undeclared=1))
    elif where == 'arena record':
        real = C.paired_game
        monkeypatch.setattr(C, 'paired_game', lambda *a, **k: dict(real(*a, **k), undeclared=1))
    else:
        real = E2.calibration_games
        monkeypatch.setattr(E2, 'calibration_games', lambda *a, **k: [dict(r, undeclared=1) for r in real(*a, **k)])
    path, token = in_process(tiny_packages, fake_runtime, f'undeclared-{where[:5]}.json')
    with pytest.raises(RuntimeError, match=r'final/seed-42.json violates the official evidence contract'):
        run_campaign(tmp_path / 'c', path, token)
    state = state_of(tmp_path / 'c')
    assert state['state'] == 'INCOMPLETE' and 'undeclared' in state['outcome']['reason']
    assert not (tmp_path / 'c/final/seed-42.json').exists()
    assert_not_official(tmp_path / 'c')


@pytest.mark.parametrize('seed', sorted(SEEDS))
def test_seed_result_schema_mutation_sweep(contract_campaign, cached_derivation, seed):
    """Every path of a genuine sealed result: deleted, retyped (and an int made a bool), and every object given
    an unexpected member. The shared validator must refuse each by inspection; nothing in a result is optional."""
    directory = contract_campaign['directory']
    context, started = contract_inputs(directory)
    original = read_json(directory / f'final/seed-{seed}.json')
    summarized = []

    def refused(document):
        problems = OE.seed_result_problems(context, started, seed, document)
        summarized.extend(p for p in problems if 'cannot be summarized' in p)
        return bool(problems)
    assert not refused(original)
    count, accepted, rejected = run_sweep(schema_mutations(original, 'final/seed-S.json'), refused)
    assert not accepted, f'{len(accepted)} of {count} malformed results accepted: {accepted[:20]}'
    assert not rejected and not summarized, (rejected, summarized[:3])
    assert count > 1800, count


def test_state_record_schema_mutation_sweep(contract_copy, cached_derivation):
    """Every path of the genuine COMPLETE record, through official_results. Only the optional diagnostic phase
    timings may be removed, and only that mapping may gain a member."""
    original = read_state(contract_copy)

    def refused(document):
        write_state(contract_copy, document)
        return strict_refusal(contract_copy) is not None
    count, accepted, rejected = run_sweep(schema_mutations(original, 'state.json'), refused)
    assert not accepted, f'{len(accepted)} of {count} malformed COMPLETE records accepted: {accepted[:20]}'
    assert not rejected, f'optional diagnostics refused: {rejected}'
    assert count > 350, count
    write_state(contract_copy, original)
    assert strict_refusal(contract_copy) is None


def test_started_record_schema_mutation_sweep(contract_copy, cached_derivation):
    """Every path of final/started.json, consistently re-hashed into the COMPLETE record."""
    original = read_json(contract_copy / 'final/started.json')

    def refused(document):
        rehash_started(contract_copy, document)
        return strict_refusal(contract_copy) is not None
    count, accepted, rejected = run_sweep(schema_mutations(original, 'final/started.json'), refused)
    assert not accepted and not rejected, f'{len(accepted)} of {count} malformed started records accepted: {accepted}'
    assert count > 250, count


def retoken(directory, declaration):
    """Coherently re-certify every artifact under a new declaration (the token and every copy of it)."""
    data = L.view_bytes(declaration)
    (directory / 'declaration.json').write_bytes(data)
    token = hashlib.sha256(data).hexdigest()
    started = read_json(directory / 'final/started.json')
    started['declaration_sha256'] = token
    for selection in started['selections'].values():
        selection['declaration_sha256'] = token
    document = read_state(directory)
    document['declaration_sha256'] = token
    for entry in document['history']:
        for selection in entry.get('selections', {}).values():
            selection['declaration_sha256'] = token
    for seed in sorted(SEEDS):
        result = read_json(directory / f'final/seed-{seed}.json')
        result['selection']['declaration_sha256'] = token
        document['outcome']['final_results'][seed] = write_json(directory / f'final/seed-{seed}.json', result)
    write_state(directory, document)
    rehash_started(directory, started)


def test_declaration_schema_mutation_sweep(contract_copy, cached_derivation):
    """Every path of the declaration, coherently re-tokened (every copy of the token and every certified hash
    rewritten): the declaration contract must refuse. A COMPLETE campaign's declaration is always one that could
    have launched. Only free-text notes and non-research frozen-record entries may be removed (another valid
    declaration). Without re-tokening, any change to the declaration's bytes is refused by its hash."""
    original = read_json(contract_copy / 'declaration.json')
    pristine = {name: (contract_copy / name).read_bytes() for name in ('state.json', 'final/started.json',
                                                                        'final/seed-42.json', 'final/seed-314159.json')}
    retoken(contract_copy, original)  # re-serialized: a new token, still the same declaration
    assert strict_refusal(contract_copy) is None

    def refused(document):
        retoken(contract_copy, document)
        return strict_refusal(contract_copy) is not None
    mutations = ((name, document, must_refuse and not (name.startswith('delete ') and OE.removable_declaration_item(
        original, tuple(int(k) if k.isdigit() else k for k in name[len('delete '):].split('/')))))
                 for name, document, must_refuse in schema_mutations(original, 'declaration.json'))
    count, accepted, rejected = run_sweep(mutations, refused)
    assert not accepted, f'{len(accepted)} of {count} re-tokened declarations accepted: {accepted}'
    assert not rejected, f'free declaration items refused: {rejected}'
    assert count > 500, count
    for name, data in pristine.items():
        (contract_copy / name).write_bytes(data)
    (contract_copy / 'declaration.json').write_bytes(L.view_bytes(dict(original, notes=['edited'])))
    assert 'declaration.json differs' in strict_refusal(contract_copy)


def test_optional_diagnostics_may_be_removed_but_stay_typed(contract_copy):
    """The contract's only optional field: the individual phase timings. Each may be removed (and new ones added);
    a retyped one is refused. Any other optional path would have to be declared in OE.OPTIONAL_PATHS."""
    assert OE.OPTIONAL_PATHS == {'state.json': (('counters', 'time', 'phase_seconds', '*'),)}
    original = read_state(contract_copy)
    phases = original['counters']['time']['phase_seconds']
    assert len(phases) >= 5
    for name in sorted(phases):
        write_state(contract_copy, mutated(original, ('counters', 'time', 'phase_seconds', name), DELETE))
        assert strict_refusal(contract_copy) is None, name
    write_state(contract_copy, mutated(original, ('counters', 'time', 'phase_seconds'), {}))
    assert strict_refusal(contract_copy) is None
    write_state(contract_copy, mutated(original, ('counters', 'time', 'phase_seconds', 'final'), 'slow'))
    assert 'phase_seconds is not a mapping of phase names to seconds' in strict_refusal(contract_copy)


def test_declaration_derived_seed_sets(contract_copy):
    """The certified result set covers exactly the declared seeds; each file is the result of its own seed."""
    document = read_state(contract_copy)
    results = document['outcome']['final_results']
    forgeries = {
        'seed removed': dict(results.items() - {('42', results['42'])}),
        'seed added': dict(results, **{'7': results['42']}),
        'seed renamed 042': {'042': results['42'], '314159': results['314159']},
        'seed results swapped': {'42': results['314159'], '314159': results['42']},
    }
    for name, final_results in forgeries.items():
        write_state(contract_copy, dict(document, outcome=dict(document['outcome'], final_results=final_results)))
        assert strict_refusal(contract_copy) is not None, name
    write_state(contract_copy, document)
    other = read_json(contract_copy / 'final/seed-314159.json')
    rehash_result(contract_copy, '42', other)  # seed 314159's genuine result certified as seed 42's
    assert 'final/seed-42.json' in strict_refusal(contract_copy)


def test_declaration_derived_evidence_sets(contract_copy, cached_derivation):
    """Ladder opponents, sealed rows, paired games and calibration games are exactly the declared sets, even when
    every summary is re-derived consistently from the forged evidence (only the evidence coverage can refuse)."""
    context, started = contract_inputs(contract_copy)
    original = read_json(contract_copy / 'final/seed-42.json')

    def consistent(change):
        """Change the raw evidence, then re-derive every summary from it with the real producer."""
        forged = json.loads(json.dumps(original))
        change(forged)
        tactical, solved = forged['evidence']['tactical'], forged['evidence']['solved']
        rows = {name: [r for r in context.rows(name) if r['id'] in evidence]
                for name, evidence in (('tactical-sealed', tactical), ('solved-sealed', solved))}
        forged_context = OE.OfficialContext(context.declaration, context.token, dict(context.packages, **rows))
        forged_context.declaration = dict(context.declaration, final=dict(
            context.declaration['final'], ladder=sorted(forged['ladder']),
            calibration_games=len({r['game'] for r in forged['evidence']['calibration']})))
        derived = OE.derive_seed_result(forged_context, 42, dict(started, descriptions={'42': dict(
            started['descriptions']['42'], opponents={k: started['descriptions']['42']['opponents'].get(
                k, started['descriptions']['42']['opponents']['random']) for k in forged['ladder']})}), dict(
            overlap_families=forged['overlap_families'], tactical=tactical, solved=solved,
            calibration=forged['evidence']['calibration'],
            ladder={k: v['records'] for k, v in forged['ladder'].items()},
            nn_only_vs_random=forged['nn_only_vs_random']['records']))
        return json.loads(json.dumps(derived))
    first_tactical, first_solved = context.rows('tactical-sealed')[0]['id'], context.rows('solved-sealed')[0]['id']
    forgeries = {
        'ladder opponent removed': lambda r: r['ladder'].pop('negamax1'),
        'ladder opponent added': lambda r: r['ladder'].update(negamax2=r['ladder']['random']),
        'ladder opponent renamed': lambda r: r['ladder'].update(negamax2=r['ladder'].pop('negamax4')),
        'tactical row removed': lambda r: r['evidence']['tactical'].pop(first_tactical),
        'solved row removed': lambda r: r['evidence']['solved'].pop(first_solved),
        'calibration game removed': lambda r: r['evidence'].update(calibration=[
            c for c in r['evidence']['calibration'] if c['game'] == 0]),
        'calibration truncated by one ply': lambda r: r['evidence']['calibration'].pop(),
        'arena game removed': lambda r: r['ladder']['random']['records'].pop(),
        'nn-only game removed': lambda r: r['nn_only_vs_random']['records'].pop(0),
    }
    for name, change in forgeries.items():
        forged = consistent(change)
        problems = OE.seed_result_problems(context, started, '42', forged)
        assert problems, f'accepted: {name}'
        assert not any('is not what its evidence derives' in p for p in problems), (name, problems)
        rehash_result(contract_copy, '42', forged)
        assert strict_refusal(contract_copy) is not None, name
    # Truncating a decisive calibration game by two plies keeps its parity and last-mover outcome; with its
    # summary left as published, it is refused. (Records carry no moves: re-deriving every summary from the
    # truncated records as well is a coherent forgery, outside the contract's scope; see the Phase 4D.3B.5 notes.)
    truncated = json.loads(json.dumps(original))
    del truncated['evidence']['calibration'][-2:]
    assert any('is not what its evidence derives: result.calibration' in p
               for p in OE.seed_result_problems(context, started, '42', truncated))
    rehash_result(contract_copy, '42', truncated)
    assert strict_refusal(contract_copy) is not None
    rehash_result(contract_copy, '42', original)
    assert strict_refusal(contract_copy) is None


def test_summaries_are_cross_checked_against_their_evidence(contract_copy, cached_derivation):
    """A summary that disagrees with its evidence is refused, and so is evidence that disagrees with itself
    (replayed games, root network values) even after its summary is re-derived to match."""
    context, started = contract_inputs(contract_copy)
    original = read_json(contract_copy / 'final/seed-42.json')

    def derive_from(result):
        raw, problems = OE.raw_evidence(context, '42', result)
        return problems, raw and json.loads(json.dumps(OE.derive_seed_result(context, 42, started, raw)))

    def flip_first_result(r):
        record = r['ladder']['random']['records'][0]
        record['result'] = {'win': 'loss', 'loss': 'win', 'draw': 'win'}[record['result']]
    summary_forgeries = {
        'ladder score': lambda r: r['ladder']['random']['summary'].update(score=1.0),
        'ladder gate': lambda r: r['ladder']['random']['gate'].update(passed=not r['ladder']['random']['gate']['passed']),
        'reported-only gate added': lambda r: r['ladder']['negamax4'].update(gate=r['ladder']['random']['gate']),
        'nn-only interval': lambda r: r['nn_only_vs_random']['summary'].update(interval95=[0.9, 1.0]),
        'tactical metric': lambda r: r['tactical']['complete_set'].update(immediate_win=1.0),
        'nn-only tactical metric': lambda r: r['tactical_nn_only']['complete_set'].update(safe_response=1.0),
        'solved metric': lambda r: r['solved']['complete_set'].update(optimal_preserving=1.0),
        'value metric': lambda r: r['value']['complete_set'].update(class_balanced_mse=0.0),
        'calibration summary': lambda r: r['calibration'].update(improvement_over_zero=1.0),
        'overlap sensitivity': lambda r: r['tactical'].update(overlap_excluded=None),
        'selection': lambda r: r['selection'].update(generation=0),
        'agent': lambda r: r['agent'].update(weights_sha256='0' * 64),
        'opponent description': lambda r: r['ladder']['negamax1']['opponent'].update(depth=2),
        'arena result alone': flip_first_result,
    }
    for name, change in summary_forgeries.items():
        forged = json.loads(json.dumps(original))
        change(forged)
        problems = OE.seed_result_problems(context, started, '42', forged)
        assert problems, f'accepted: {name}'
    self_inconsistent = {
        'arena result flipped, summary re-derived': flip_first_result,
        'arena winner changed, summary re-derived': lambda r: r['ladder']['random']['records'][0].update(winner=-1),
        'arena moves truncated': lambda r: r['ladder']['random']['records'][0]['moves'].pop(),
        'solved raw value changed, value summary re-derived': lambda r: r['evidence']['solved'][
            context.rows('solved-sealed')[0]['id']].update(raw_value=0.0),
        'search choice not the maximum-visit action': lambda r: r['evidence']['tactical'][
            context.rows('tactical-sealed')[0]['id']].update(choices=[next(
                a for a in range(7) if a in engine_position(context.rows('tactical-sealed')[0]['moves'])
                .get_valid_moves() and r['evidence']['tactical'][context.rows('tactical-sealed')[0]['id']]['visits'][0][a]
                < max(r['evidence']['tactical'][context.rows('tactical-sealed')[0]['id']]['visits'][0]))]),
        'calibration outcome changed': lambda r: r['evidence']['calibration'][0].update(
            outcome=-r['evidence']['calibration'][0]['outcome']),
    }
    for name, change in self_inconsistent.items():
        forged = json.loads(json.dumps(original))
        change(forged)
        problems, derived = derive_from(forged)
        assert problems, f'self-inconsistent evidence passed the structural checks: {name}'
        assert OE.seed_result_problems(context, started, '42', forged), name


def test_coherently_reduced_unit_counts_are_refused(contract_copy):
    """Every unit kind's count is fixed by the declaration and packages; an account consistently reduced by one
    unit (aggregates, totals and every copy of games started rewritten) is refused."""
    original = read_state(contract_copy)
    for kind in OE.GAME_KINDS + OE.ROW_KINDS:
        document = json.loads(json.dumps(original))
        evaluation = document['counters']['evaluation']
        evaluation['by_kind'][kind] = {k: v - 1 for k, v in evaluation['by_kind'][kind].items()}
        if kind in OE.GAME_KINDS:
            group = next(name for name, kinds in (('development_games', OE.DEVELOPMENT_GAME_KINDS),
                                                  ('sealed_games', OE.SEALED_GAME_KINDS),
                                                  ('calibration_games', OE.CALIBRATION_GAME_KINDS)) if kind in kinds)
            evaluation[group] -= 1
            for key in ('total_games_started', 'total_games_completed'):
                evaluation[key] -= 1
            for part in (document['counters']['time'], document['outcome']['completion']):
                part['evaluation_games_started'] -= 1
        write_state(contract_copy, document)
        assert f'{kind}: ' in (strict_refusal(contract_copy) or ''), kind
    write_state(contract_copy, original)
    assert strict_refusal(contract_copy) is None


def test_research_ladder_descriptions_satisfy_the_contract():
    """The full research ladder (all seven opponents, the retained 4D.2f checkpoint) as the producer describes
    it, against the declaration-derived opponent schema the sealed phase will be validated with."""
    declaration = C.build_declaration({}, runtime_identity={})
    checkpoint = ROOT / declaration['phase4d2f_checkpoint']['path']
    if not checkpoint.exists():
        pytest.skip('retained Phase 4D.2f checkpoint not present locally')
    context = OE.OfficialContext(declaration, '0' * 64, {})
    opponents = C.ladder_opponents(declaration, 42)
    assert sorted(opponents) == sorted(declaration['final']['ladder']) == sorted(OE.LADDER_NAMES)
    for name, opponent in opponents.items():
        description = json.loads(json.dumps(opponent.describe()))
        assert OE.opponent_problems(context, name, description) == [], name
        for key in description:
            assert OE.opponent_problems(context, name, {k: v for k, v in description.items() if k != key}), key
    wrong = json.loads(json.dumps(opponents['phase4d2f_512'].describe()))
    assert OE.opponent_problems(context, 'phase4d2f_512', dict(wrong, sha256='0' * 64))


def test_campaign_directory_holds_the_declared_packages(contract_copy):
    """Acceptance derives sealed rows from the campaign's own hash-verified package copies."""
    declaration = read_json(contract_copy / 'declaration.json')
    assert sorted(p.stem for p in (contract_copy / OE.PACKAGE_DIRECTORY).iterdir()) == sorted(declaration['packages'])
    package = contract_copy / OE.PACKAGE_DIRECTORY / 'tactical-sealed.json'
    package.write_bytes(package.read_bytes() + b' ')
    assert 'packages/tactical-sealed.json differs from the package the declaration binds' in strict_refusal(
        contract_copy)
    package.unlink()
    assert strict_refusal(contract_copy) is not None


def test_validators_are_total_on_arbitrary_json():
    """Every contract validator refuses any JSON value by inspection, never by raising."""
    declaration = C.build_declaration({}, runtime_identity={})
    context = OE.OfficialContext(declaration, '0' * 64, {})
    values = [None, True, 0, 1.5, 'x', [], {}, [None], {'seed': None}, {k: None for k in OE.SEED_RESULT_KEYS}]
    for value in values:
        assert OE.declaration_problems(value)
        assert OE.selection_problems(context, 'selection', value, '42')
        assert OE.started_problems(context, value, {})
        assert OE.seed_result_problems(context, {}, '42', value)
        assert OE.keys_problems('x', value, {'a'}) or value == {'a': None}


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
