"""Phase 4D.3B/4D.3B.1 torch-side preflight: evaluation agents, diagnostics, profiling helpers and the
bounded campaign launcher (declaration binding, crash recovery, deadlines and durable budgets).
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
from games.connect4.alphazero_v2.launch_control import LEASE_SECONDS
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


# Phase 4D.3B.1 launch control: shared fixtures ------------------------------------------------------

OLD_TOKEN = '2741314399741c1b20510b8a2beb42e0938b0aeafc9dd0c487201e57208867d8'
OLD_DECLARATION_COMMIT = 'a663454'  # the reviewed HEAD holding the rejected format-1 declaration
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
    if ceiling is not None:
        declaration['budgets']['evaluation_games_ceiling'] = (
            declaration['budgets']['planned_evaluation_games'] if ceiling == 'planned' else ceiling)
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


def run_to_end(directory, declaration, token, seeds=(42,), **options):
    statuses = []
    for seed in seeds:
        campaign = launch(directory, declaration, token, **options)
        try:
            statuses.append(campaign.run_seed(seed))
        finally:
            campaign.close()
    campaign = launch(directory, declaration, token, **options)
    try:
        statuses.append(campaign.final_evaluate())
    finally:
        campaign.close()
    return statuses


def journal_records(directory):
    return [json.loads(line)['record'] for line in (Path(directory) / 'journal.jsonl').read_text().splitlines()]


# Declaration and token ----------------------------------------------------------------------------------

def test_full_declaration_plans_exactly_the_reviewed_evaluation_games():
    declaration = C.build_declaration({})
    assert declaration['budgets'] == dict(per_run_training_seconds=28800, campaign_seconds=86400,
                                          evaluation_games_ceiling=6400, planned_evaluation_games=6272)
    assert declaration['seeds'] == [42, 314159] and declaration['primary_seed'] == 42
    assert declaration['generations'] == 20 and declaration['champion']['schedule'] == [5, 10, 15, 20]
    assert V2Config.from_dict(dict(declaration['config'], seed=42)) == V2Config(seed=42)
    assert len(declaration['champion']['baseline_rows']) == 20
    assert declaration['format_version'] == 2 and declaration['launch_control'] == C.launch_control_declaration()
    assert OLD_TOKEN in declaration['launch_control']['rejected_declaration_tokens']


def test_declaration_validation_rejects_inconsistent_bindings(tiny_packages, launch_runtime):
    declaration = declaration_document(tiny_packages, launch_runtime)
    assert C.validate_declaration(declaration, tiny_packages)
    source = declaration['execution_source']
    tampered_files = dict(source['files'], **{'games/connect4/alphazero_v2/search.py': '0' * 64})
    changes = [
        dict(budgets=dict(declaration['budgets'], planned_evaluation_games=1)), dict(generations=3),
        dict(seeds=[42, 42]), dict(format_version=1), dict(champion=dict(declaration['champion'], schedule=[2, 1])),
        dict(packages=dict(declaration['packages'], **{'tactical-sealed': dict(
            declaration['packages']['tactical-sealed'], sha256='0' * 64)})),
        dict(champion=dict(declaration['champion'], gate=dict(declaration['champion']['gate'], score=0.5))),
        dict(thresholds=dict(declaration['thresholds'], solved=dict(optimal_preserving=0.5, avoidable_loss_max=0.5))),
        dict(launch_control=dict(declaration['launch_control'], lease_seconds=10_000)),
        dict(launch_control=dict(declaration['launch_control'], rejected_declaration_tokens=[])),
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


def test_rejected_token_and_original_declaration_never_authorize(tmp_path, tiny_packages, launch_runtime):
    path, token = write_declaration(tiny_packages, declaration_document(tiny_packages, launch_runtime), 'ok.json')
    with pytest.raises(C.LaunchRefused, match='REJECTED'):
        C.Campaign(tmp_path / 'c', path, OLD_TOKEN)
    old = subprocess.run(['git', '-C', str(ROOT), 'show', f'{OLD_DECLARATION_COMMIT}:{FROZEN_DECLARATION.relative_to(ROOT)}'],
                         capture_output=True)
    if old.returncode != 0:
        pytest.skip('original declaration bytes unavailable from git')
    assert hashlib.sha256(old.stdout).hexdigest() == OLD_TOKEN
    copy = tmp_path / 'campaign-declaration.json'
    copy.write_bytes(old.stdout)
    for loader in (lambda: C.load_declaration(copy), lambda: C.Campaign(tmp_path / 'c', copy, OLD_TOKEN)):
        with pytest.raises(C.LaunchRefused, match='REJECTED / NOT AUTHORIZED'):
            loader()
    report = C.preflight(copy)
    assert report['ok'] is False and 'REJECTED' in report['gates'][0]['detail']
    assert report['declaration_sha256'] == OLD_TOKEN
    # The CLI refuses it in a fresh interpreter (exit 2) even with a pinned runtime.
    result = subprocess.run([sys.executable, '-m', 'games.connect4.alphazero_v2.campaign', 'run', '--declaration',
                             str(copy), '--campaign-dir', str(tmp_path / 'cli'), '--authorize', OLD_TOKEN, '--seed',
                             '42'], capture_output=True, text=True, timeout=120, cwd=ROOT, env=pinned_env())
    assert result.returncode == C.EXIT_REFUSED and 'REJECTED' in result.stdout
    assert not (tmp_path / 'cli').exists()


def test_frozen_declaration_is_the_new_format_2_declaration():
    """The repository's frozen declaration is the re-frozen one: valid, bound, and not the rejected token."""
    token = hashlib.sha256(FROZEN_DECLARATION.read_bytes()).hexdigest()
    assert token != OLD_TOKEN
    declaration, digest = C.load_declaration(FROZEN_DECLARATION)
    assert digest == token and declaration['format_version'] == 2 and declaration['kind'] == 'research'
    assert declaration['execution_source'] == C.execution_source_declaration()  # matches this tree's sources
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
        finally:
            target.write_bytes(original)


# Source binding -----------------------------------------------------------------------------------------

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
    tree = tmp_path / 'tree'
    for relative in provenance.execution_source_files():
        (tree / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, tree / relative)
    shutil.copytree(tiny_packages, tree / 'packages')
    path, token = write_declaration(tree / 'packages', declaration_document(tree / 'packages', launch_runtime),
                                    'declaration.json')

    def attempt(name):
        result = subprocess.run([sys.executable, '-c', COPIED_LAUNCH, str(path), token, str(tmp_path / name)],
                                capture_output=True, text=True, timeout=180, cwd=tree, env=pinned_env())
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout.strip().splitlines()[-1])
    assert attempt('clean') == dict(accepted=True, preflight=True)
    (tree / 'docs').mkdir()
    (tree / 'docs/notes.md').write_text('documentation only\n')
    (tree / 'tests').mkdir()
    (tree / 'tests/test_extra.py').write_text('def test_nothing():\n    pass\n')
    (tree / 'games/connect4/agents/negamax_agent.py').write_text('# legacy, outside the execution closure\n')
    assert attempt('docs-only') == dict(accepted=True, preflight=True)
    for relative, group in (('games/connect4/alphazero_v2/search.py', 'training'),
                            ('games/connect4/alphazero_v2/evaluation.py', 'evaluation_and_launch')):
        original = (tree / relative).read_bytes()
        (tree / relative).write_bytes(original + b'#')
        try:
            outcome = attempt(f'edited-{group}')
            assert outcome['accepted'] is False and f'{group} source differs' in outcome['error'] \
                and relative in outcome['error']
            assert not (tmp_path / f'edited-{group}').exists()
        finally:
            (tree / relative).write_bytes(original)


def test_source_drift_mid_campaign_stops_before_publication(tmp_path, tiny_packages, fake_runtime, monkeypatch):
    path, token = in_process(tiny_packages, fake_runtime, 'drift.json')
    real = C.execution_source_identity
    state = dict(drift=False)

    def drift(phase, context):
        if context.get('generation') == 2:
            state['drift'] = True
    monkeypatch.setattr(C, 'execution_source_identity',
                        lambda: tampered_identity('games/connect4/alphazero_v2/data.py') if state['drift'] else real())
    campaign = launch(tmp_path / 'c', path, token)
    with pytest.raises(C.LaunchRefused, match='during the campaign.*training source differs.*data.py'):
        campaign.run_seed(42, extra_check=drift)
    campaign.close()
    records = journal_records(tmp_path / 'c')
    assert not any(r['event'] == 'generation_committed' and r['generation'] == 2 for r in records)
    assert records[-1]['event'] == 'attempt_end' and records[-1]['status'].startswith('failed: LaunchRefused')
    state['drift'] = False
    assert run_to_end(tmp_path / 'c', path, token) == ['completed', 'completed']


# Runtime binding -----------------------------------------------------------------------------------------

@pytest.mark.parametrize('key', C.RUNTIME_IDENTITY_KEYS)
def test_every_runtime_identity_field_refuses_launch_and_resume(tmp_path, tiny_packages, fake_runtime, key):
    path, token = in_process(tiny_packages, fake_runtime, 'runtime.json')
    campaign = launch(tmp_path / 'c', path, token)
    assert campaign.run_seed(42, extra_check=stop_at('training', 1)).startswith('stopped')
    campaign.close()
    before = (tmp_path / 'c' / 'journal.jsonl').read_bytes()
    original = fake_runtime['identity']
    fake_runtime['identity'] = dict(original, **{key: 'changed'})
    try:
        with pytest.raises(C.LaunchRefused, match=f'runtime differs from the declaration.*{key}'):
            launch(tmp_path / 'c', path, token)
        assert (tmp_path / 'c' / 'journal.jsonl').read_bytes() == before  # refused before any new work
    finally:
        fake_runtime['identity'] = original


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


def test_runtime_change_between_training_and_final_evaluation_is_refused(tmp_path, tiny_packages, fake_runtime,
                                                                          monkeypatch):
    path, token = in_process(tiny_packages, fake_runtime, 'final-identity.json')
    campaign = launch(tmp_path / 'c', path, token)
    assert campaign.run_seed(42) == 'completed'
    campaign.close()
    original = fake_runtime['identity']
    fake_runtime['identity'] = dict(original, torch='2.10.1')
    with pytest.raises(C.LaunchRefused, match='torch'):
        launch(tmp_path / 'c', path, token)
    fake_runtime['identity'] = original
    monkeypatch.setattr(C, 'execution_source_identity',
                        lambda: tampered_identity('games/connect4/alphazero_v2/arena.py'))
    with pytest.raises(C.LaunchRefused, match='evaluation_and_launch source differs.*arena.py'):
        launch(tmp_path / 'c', path, token)
    assert not (tmp_path / 'c' / 'final').exists()


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


def test_concurrent_invocation_is_refused(tmp_path, tiny_packages, fake_runtime):
    path, token = in_process(tiny_packages, fake_runtime, 'lock.json')
    first = launch(tmp_path / 'c', path, token)
    try:
        with pytest.raises(C.CampaignLocked):
            launch(tmp_path / 'c', path, token)
        status = C.campaign_status(tmp_path / 'c')  # read-only status works while the writer holds the lock
        assert status['journal']['torn_tail_bytes'] == 0
    finally:
        first.close()
    launch(tmp_path / 'c', path, token).close()


def test_campaign_directory_stays_bound_to_its_declaration(tmp_path, tiny_packages, fake_runtime):
    path, token = in_process(tiny_packages, fake_runtime, 'bound-a.json')
    other, other_token = in_process(tiny_packages, fake_runtime, 'bound-b.json', games=3)
    launch(tmp_path / 'c', path, token).close()
    with pytest.raises(FileExistsError, match='different declaration'):
        launch(tmp_path / 'c', other, other_token)
    (tmp_path / 'c/declaration.json').unlink()
    shutil.copy2(other, tmp_path / 'c/declaration.json')  # a swapped copy cannot adopt the journal
    with pytest.raises(C.InconsistentCampaign, match='different declaration'):
        launch(tmp_path / 'c', other, other_token)
    (tmp_path / 'unrelated').mkdir()
    (tmp_path / 'unrelated/data.txt').write_text('not a campaign')
    with pytest.raises(FileExistsError, match='not a campaign directory'):
        launch(tmp_path / 'unrelated', path, token)


def test_final_evaluation_refuses_an_attempt_recorded_under_another_identity(tmp_path, tiny_packages,
                                                                               fake_runtime):
    path, token = in_process(tiny_packages, fake_runtime, 'history.json')
    campaign = launch(tmp_path / 'c', path, token)
    assert campaign.run_seed(42) == 'completed'
    campaign.close()
    # Append, with valid checksums, an attempt that an unchecked launcher could have run under another runtime.
    from games.connect4.alphazero_v2 import launch_control as L
    declaration = json.loads(path.read_text())
    state = L.CampaignState(declaration['generations'], declaration['champion']['schedule'])
    journal = L.Journal(tmp_path / 'c/journal.jsonl', state)
    number = len(state.attempts) + 1
    journal.append(dict(event='attempt_start', attempt=number, scope='42', declaration_sha256=token,
                        runtime=dict(fake_runtime['identity'], intra_op_threads=8),
                        execution_sha256=declaration['execution_source']['sha256']))
    journal.append(dict(event='attempt_end', attempt=number, scope='42', status='stopped', total_seconds=1,
                        training_seconds=0))
    campaign = launch(tmp_path / 'c', path, token)
    try:
        with pytest.raises(C.LaunchRefused, match=f'Attempt {number} ran under a different identity'):
            campaign.final_evaluate()
    finally:
        campaign.close()
    assert not (tmp_path / 'c/final').exists()


def test_torn_journal_tail_is_recovered_at_launch(tmp_path, tiny_packages, fake_runtime):
    path, token = in_process(tiny_packages, fake_runtime, 'torn.json')
    plain = run_to_end(tmp_path / 'plain', path, token)
    campaign = launch(tmp_path / 'c', path, token)
    assert campaign.run_seed(42, extra_check=stop_at('training', 2)).startswith('stopped')
    campaign.close()
    with open(tmp_path / 'c/journal.jsonl', 'ab') as stream:
        stream.write(b'{"seq": 999, "record": {"event": "unit_begin"')  # crash mid-append
    assert C.campaign_status(tmp_path / 'c')['journal']['torn_tail_bytes'] > 0
    assert run_to_end(tmp_path / 'c', path, token) == plain
    records = journal_records(tmp_path / 'c')
    torn = [r for r in records if r['event'] == 'torn_tail_discarded']
    assert len(torn) == 1 and (tmp_path / 'c' / torn[0]['saved_as']).exists()
    assert fingerprint(tmp_path / 'c') == fingerprint(tmp_path / 'plain')


# Durable transitions in-process --------------------------------------------------------------------------

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


def test_cooperative_stops_resume_without_rerunning_completed_units(tmp_path, tiny_packages, fake_runtime):
    path, token = in_process(tiny_packages, fake_runtime, 'stops.json')
    plain = run_to_end(tmp_path / 'plain', path, token)
    campaign = launch(tmp_path / 'c', path, token)
    status = campaign.run_seed(42, extra_check=stop_at('evaluation', kind='development_arena_game', after=3))
    campaign.close()
    assert status == 'stopped: test stop at evaluation'
    records = journal_records(tmp_path / 'c')
    completed_before = {r['unit'] for r in records if r['event'] == 'unit_complete'}
    abandoned = [r for r in records if r['event'] == 'unit_abandoned']
    assert len(abandoned) == 1 and abandoned[0]['partial']['abandoned']
    assert run_to_end(tmp_path / 'c', path, token) == plain
    records = journal_records(tmp_path / 'c')
    begins = [r['unit'] for r in records if r['event'] == 'unit_begin']
    assert not any(begins.count(u) > 1 for u in completed_before)  # completed units never rerun
    assert begins.count(abandoned[0]['unit']) == 2               # the abandoned unit is rerun once
    assert fingerprint(tmp_path / 'c') == fingerprint(tmp_path / 'plain')


def test_inconsistent_published_outputs_are_refused(tmp_path, tiny_packages, fake_runtime):
    path, token = in_process(tiny_packages, fake_runtime, 'inconsistent.json')
    campaign = launch(tmp_path / 'c', path, token)
    assert campaign.run_seed(42, extra_check=stop_at('training', 2)).startswith('stopped')
    campaign.close()
    artifact = tmp_path / 'c/runs/seed-42/generations/generation-0001/generation-0001.inference.pt'
    original = artifact.read_bytes()
    artifact.write_bytes(original + b'x')

    def resume():
        campaign = launch(tmp_path / 'c', path, token)
        try:
            return campaign.run_seed(42)
        finally:
            campaign.close()
    with pytest.raises(C.InconsistentCampaign, match='generation 1'):
        resume()
    artifact.write_bytes(original)
    view = tmp_path / 'c/runs/seed-42/generations/generation-0001/summary.json'
    view.write_text(view.read_text().replace('"generation": 1', '"generation": 9'))
    with pytest.raises(C.InconsistentCampaign, match='disagrees with the journal'):
        resume()
    assert not any(r['event'] == 'generation_committed' and r['generation'] == 2
                   for r in journal_records(tmp_path / 'c'))  # no new work over inconsistent outputs


# Deadlines (fake clock) -------------------------------------------------------------------------------------

def outcome(directory):
    path = Path(directory) / 'outcome.json'
    return json.loads(path.read_text()) if path.exists() else None


def test_final_calibration_receives_checks_and_an_overrun_is_incomplete(tmp_path, tiny_packages, fake_runtime,
                                                                        monkeypatch):
    """B3 regression: the last calibration game overruns the campaign cap after its final check."""
    clock = Clock()
    path, token = in_process(tiny_packages, fake_runtime, 'calibration.json', campaign_seconds=1000)
    import games.connect4.alphazero_v2.evaluation as E2
    seen = {}
    real = E2.calibration_games

    def overrunning(*args, check=None, **kwargs):
        seen['check'] = check is not None
        records = real(*args, check=check, **kwargs)
        clock.now = 1000.5
        return records
    directory = tmp_path / 'overrun'
    campaign = launch(directory, path, token, clock=clock)
    clock.now = 0.0
    assert campaign.run_seed(42) == 'completed'
    campaign.close()
    monkeypatch.setattr(E2, 'calibration_games', overrunning)
    campaign = launch(directory, path, token, clock=clock)
    status = campaign.final_evaluate()
    campaign.close()
    assert seen['check'] is True
    assert status == 'incomplete: campaign wall-clock budget exceeded before the phase completed'
    assert not (directory / 'final/seed-42.json').exists()
    assert outcome(directory)['status'] == 'INCOMPLETE'
    assert json.loads((directory / 'final/result.json').read_text())['status'] == 'INCOMPLETE'
    records = journal_records(directory)
    calibration = [r for r in records if r['event'] == 'unit_complete' and r['kind'] == 'calibration_game']
    assert calibration and calibration[-1]['within_budget'] is False
    assert not any(r['event'] in ('final_seed_complete', 'campaign_complete') for r in records)
    # INCOMPLETE is final: a later invocation does no work and cannot extend the deadline.
    clock.now = 0.0
    campaign = launch(directory, path, token, clock=Clock())
    assert campaign.final_evaluate().startswith('incomplete')
    campaign.close()
    assert len(journal_records(directory)) == len(records)


def test_stop_during_the_last_unit_defers_publication(tmp_path, tiny_packages, fake_runtime, monkeypatch):
    path, token = in_process(tiny_packages, fake_runtime, 'last-stop.json')
    campaign = launch(tmp_path / 'c', path, token)
    assert campaign.run_seed(42) == 'completed'
    campaign.close()
    import games.connect4.alphazero_v2.evaluation as E2
    real = E2.calibration_games
    campaign = launch(tmp_path / 'c', path, token)

    def stopping(*args, **kwargs):
        records = real(*args, **kwargs)
        campaign._budget.request_stop('signal 15')
        return records
    monkeypatch.setattr(E2, 'calibration_games', stopping)
    assert campaign.final_evaluate() == 'stopped: signal 15'
    campaign.close()
    assert not (tmp_path / 'c/final/seed-42.json').exists() and outcome(tmp_path / 'c') is None
    monkeypatch.setattr(E2, 'calibration_games', real)
    begun = sum(r['event'] == 'unit_begin' for r in journal_records(tmp_path / 'c'))
    campaign = launch(tmp_path / 'c', path, token)
    assert campaign.final_evaluate() == 'completed'
    campaign.close()
    assert sum(r['event'] == 'unit_begin' for r in journal_records(tmp_path / 'c')) == begun  # nothing rerun
    assert outcome(tmp_path / 'c')['status'] == 'COMPLETE'


def test_last_update_overrunning_the_training_cap_discards_the_generation(tmp_path, tiny_packages, fake_runtime,
                                                                          monkeypatch):
    clock = Clock()
    path, token = in_process(tiny_packages, fake_runtime, 'training-cap.json', per_run_training_seconds=100)
    from games.connect4.alphazero_v2 import generation as G
    real = G.GenerationRunner.run_generation

    def slow_last_update(self, *args, **kwargs):
        summary = real(self, *args, **kwargs)
        clock.now += 101.0  # the final optimizer update ran past the per-run cap
        return summary
    monkeypatch.setattr(G.GenerationRunner, 'run_generation', slow_last_update)
    campaign = launch(tmp_path / 'c', path, token, clock=clock)
    status = campaign.run_seed(42)
    campaign.close()
    assert status == ('incomplete: per-run collection+optimization budget exceeded before the generation '
                      'completed')
    records = journal_records(tmp_path / 'c')
    assert [r['generation'] for r in records if r['event'] == 'generation_committed'] == [0]
    assert [r['generation'] for r in records if r['event'] == 'generation_discarded'] == [1]
    assert outcome(tmp_path / 'c')['scope'] == '42'


def test_deadline_during_development_evaluation_preserves_evidence(tmp_path, tiny_packages, fake_runtime):
    clock = Clock()
    path, token = in_process(tiny_packages, fake_runtime, 'dev-deadline.json', campaign_seconds=1000)

    def expire(point, campaign):
        if point == 'before_unit_complete:development_arena_game':
            clock.now = 1000.0
    campaign = launch(tmp_path / 'c', path, token, clock=clock, fault=expire)
    status = campaign.run_seed(42)
    campaign.close()
    assert status == 'incomplete: campaign wall-clock budget exhausted'
    records = journal_records(tmp_path / 'c')
    completed = [r for r in records if r['event'] == 'unit_complete']
    assert len(completed) == 1 and completed[0]['within_budget'] is True  # finished exactly at the limit
    assert not any(r['event'] == 'champion_decision' for r in records)
    status = C.campaign_status(tmp_path / 'c')
    assert status['outcome']['event'] == 'campaign_incomplete'
    assert status['progress']['seeds']['42']['pending_checks'] == [1, 2]


def test_exact_game_ceiling_completes_and_one_lost_slot_is_incomplete(tmp_path, tiny_packages, fake_runtime):
    path, token = in_process(tiny_packages, fake_runtime, 'ceiling.json', ceiling='planned')
    assert run_to_end(tmp_path / 'exact', path, token) == ['completed', 'completed']
    planned = json.loads(path.read_text())['budgets']['planned_evaluation_games']
    consumption = C.campaign_status(tmp_path / 'exact')['consumption']
    assert consumption['evaluation_games_charged'] == consumption['evaluation_games_completed'] == planned
    # One abandoned game keeps its charge, so the final game no longer fits under the ceiling.
    campaign = launch(tmp_path / 'short', path, token)
    assert campaign.run_seed(42, extra_check=stop_at('evaluation', kind='development_arena_game', after=1)) \
        .startswith('stopped')
    campaign.close()
    statuses = run_to_end(tmp_path / 'short', path, token)
    assert statuses == ['completed', 'incomplete: evaluation-game ceiling reached']
    consumption = C.campaign_status(tmp_path / 'short')['consumption']
    assert consumption['evaluation_games_charged'] == planned
    assert consumption['evaluation_games_completed'] == planned - 1
    assert outcome(tmp_path / 'short')['status'] == 'INCOMPLETE'


# Crash recovery (hard interruption at every transition, real fresh interpreters) ----------------------------

CRASH_SCRIPT = '''
import json, os, sys
from pathlib import Path
from games.connect4.alphazero_v2 import campaign as C
declaration, token, directory, fault = sys.argv[1], sys.argv[2], Path(sys.argv[3]), json.loads(sys.argv[4])
counts = {}

def crash(point, campaign):
    scope = campaign.state.attempts[campaign.attempt]["scope"] if campaign.attempt else None
    if fault and point == fault[0] and scope == fault[1]:
        counts[point] = counts.get(point, 0) + 1
        if counts[point] == fault[2]:
            os._exit(91)  # hard interruption: no attempt_end, no cleanup
statuses = []
seeds = json.loads(Path(declaration).read_text())["seeds"]
for seed in seeds:
    while True:
        campaign = C.Campaign(directory, declaration, token, fault=crash)
        try:
            if campaign.state.seed(seed)["selected"] is not None:
                break
            statuses.append(campaign.run_seed(seed))
        finally:
            campaign.close()
        if statuses[-1] != "completed":
            break
campaign = C.Campaign(directory, declaration, token, fault=crash)
statuses.append(campaign.final_evaluate())
campaign.close()
print(json.dumps(statuses))
'''

CRASH_POINTS = [
    ('before_generation_begin', '42', 2), ('progress:game', '42', 1), ('progress:collected', '42', 1),
    ('progress:update', '42', 2), ('after_training', '42', 1), ('after_boundary_files', '42', 2),
    ('before_commit_record', '42', 2), ('after_commit_record', '42', 2), ('after_commit_install', '42', 2),
    ('before_diagnostics_record', '42', 1), ('after_diagnostics_record', '42', 2), ('after_prune_record', '42', 1),
    ('after_unit_begin:development_arena_game', '42', 2),
    ('before_unit_complete:development_baseline_game', '42', 1),
    ('before_unit_complete:development_tactics_row', '42', 3),
    ('before_decision_record', '42', 1), ('after_decision_record', '42', 2),
    ('before_selection_record', '42', 1), ('after_selection_record', '42', 1),
    ('before_commit_record', '7', 1), ('progress:update', '7', 1),  # between seeds / second seed
    ('before_final_started', 'final', 1), ('after_final_started', 'final', 1),
    ('after_unit_begin:final_tactical_row', 'final', 1), ('before_unit_complete:final_solved_row', 'final', 2),
    ('before_unit_complete:final_ladder_game', 'final', 3), ('after_unit_begin:calibration_game', 'final', 1),
    ('before_final_seed_record', 'final', 1), ('after_final_seed_record', 'final', 1),
    ('before_campaign_complete', 'final', 1),
]


# A hard interruption inside an open generation discards it (the generation-0 boundary has no open generation).
DISCARDS_GENERATION = {p for p in CRASH_POINTS if p[0] in (
    'progress:game', 'progress:collected', 'progress:update', 'after_training', 'after_boundary_files',
    'before_commit_record') and p != ('before_commit_record', '7', 1)}


def strip_timing(value):
    if isinstance(value, dict):
        return {k: strip_timing(v) for k, v in value.items() if k not in ('seconds', 'utc')}
    if isinstance(value, list):
        return [strip_timing(v) for v in value]
    return value


def fingerprint(directory):
    """Everything that must equal the uninterrupted campaign (timings and attempt numbers excluded)."""
    directory = Path(directory)
    records = journal_records(directory)
    pick = lambda event, *keys: sorted(tuple(json.dumps(r[k], sort_keys=True) for k in keys)  # noqa: E731
                                       for r in records if r['event'] == event)
    finals = {p.name: strip_timing(json.loads(p.read_text())) for p in sorted((directory / 'final').glob('seed-*.json'))}
    for result in finals.values():
        for name in ('agent', 'selection'):
            result.pop(name, None)
    return dict(
        commits=pick('generation_committed', 'seed', 'generation', 'learner_weights_sha256', 'state_sha256', 'summary'),
        games=sorted((r['seed'], r['generation'], r['files'].get('games', {}).get('sha256'))
                     for r in records if r['event'] == 'generation_committed'),
        diagnostics=pick('generation_diagnostics', 'seed', 'generation', 'development_raw_value'),
        decisions=pick('champion_decision', 'seed', 'generation', 'previous', 'promote', 'decision', 'arena',
                       'baselines', 'candidate_tactics', 'champion_tactics'),
        selections=pick('seed_selected', 'seed', 'generation', 'inference'),
        evidence={r['unit']: strip_timing(r['evidence']) for r in records if r['event'] == 'unit_complete'},
        finals=finals, outcome=[r['status'] for r in records if r['event'] in ('campaign_complete',
                                                                              'campaign_incomplete')])


@pytest.fixture(scope='module')
def crash_matrix(tmp_path_factory, tiny_packages, launch_runtime):
    """Run the uninterrupted tiny two-seed campaign and one hard crash + resume per transition, in parallel."""
    from concurrent.futures import ThreadPoolExecutor
    root = tmp_path_factory.mktemp('crash-matrix')
    path, token = write_declaration(tiny_packages, declaration_document(
        tiny_packages, launch_runtime, seeds=(42, 7), schedule=(1, 2)), 'crash.json')

    def run(directory, fault):
        result = subprocess.run([sys.executable, '-c', CRASH_SCRIPT, str(path), token, str(directory),
                                 json.dumps(fault)], capture_output=True, text=True, timeout=900, cwd=ROOT,
                                env=pinned_env())
        return result.returncode, result.stdout, result.stderr

    def scenario(index_point):
        index, point = index_point
        directory = root / f'crash-{index:02d}'
        first = run(directory, list(point))
        second = run(directory, None) if first[0] == 91 else None
        return point, first, second
    with ThreadPoolExecutor(max_workers=max(2, (os.cpu_count() or 2) // 2)) as pool:
        plain = pool.submit(run, root / 'plain', None)
        crashes = list(pool.map(scenario, enumerate(CRASH_POINTS)))
    return dict(root=root, plain=plain.result(), crashes={tuple(p): (i, f, s) for i, (p, f, s) in enumerate(crashes)})


def test_uninterrupted_tiny_campaign_completes(crash_matrix):
    code, stdout, stderr = crash_matrix['plain']
    assert code == 0, stderr[-3000:]
    assert json.loads(stdout.strip().splitlines()[-1]) == ['completed', 'completed', 'completed']
    assert outcome(crash_matrix['root'] / 'plain')['status'] == 'COMPLETE'
    consumption = C.campaign_status(crash_matrix['root'] / 'plain')['consumption']
    for seed in consumption['seeds'].values():  # uninterrupted: every attempted unit of work was accepted
        assert seed['selfplay_games_completed'] == seed['accepted_games'] == 2 * 2
        assert seed['training_plies_completed'] == seed['accepted_plies'] > 0
        assert seed['optimizer_steps_attempted'] == seed['accepted_optimizer_steps'] > 0
        assert seed['generations_begun'] == seed['generations_committed'] == 2
    assert consumption['evaluation_games_charged'] == consumption['evaluation_games_completed'] > 0
    assert all(u['attempted'] == u['completed'] for u in consumption['units'].values())


@pytest.mark.parametrize('point', CRASH_POINTS, ids=[f'{p}@{s}#{n}' for p, s, n in CRASH_POINTS])
def test_hard_crash_recovers_to_the_uninterrupted_result(crash_matrix, point):
    index, first, second = crash_matrix['crashes'][tuple(point)]
    assert first[0] == 91, f'fault point never reached: {first[2][-2000:]}'
    assert second is not None and second[0] == 0, second[2][-3000:]
    assert json.loads(second[1].strip().splitlines()[-1])[-1] == 'completed'
    directory = crash_matrix['root'] / f'crash-{index:02d}'
    plain = crash_matrix['root'] / 'plain'
    assert fingerprint(directory) == fingerprint(plain)
    status, reference = C.campaign_status(directory), C.campaign_status(plain)
    consumption = status['consumption']
    assert consumption['attempts']['recovered_after_hard_interruption'] == 1 and consumption['attempts']['open'] == 0
    # The crashed attempt is charged through its lease: time can only grow, never reset.
    assert consumption['charged_seconds'] >= reference['consumption']['charged_seconds'] + 0.5 * LEASE_SECONDS
    games, plain_games = consumption['evaluation_games_charged'], reference['consumption']['evaluation_games_charged']
    if point[0].endswith(('_game', 'calibration_game')) and 'unit' in point[0]:
        assert games == plain_games + 1  # the interrupted game stays charged and is replayed once
    else:
        assert games == plain_games
    assert consumption['evaluation_games_completed'] == reference['consumption']['evaluation_games_completed']
    for name, seed in consumption['seeds'].items():
        accepted = reference['consumption']['seeds'][name]
        for key in ('accepted_games', 'accepted_plies', 'accepted_optimizer_steps', 'generations_committed'):
            assert seed[key] == accepted[key], key
        assert seed['selfplay_games_completed'] >= accepted['selfplay_games_completed']
        assert seed['optimizer_steps_attempted'] >= accepted['optimizer_steps_attempted']
    if point == ('progress:game', '42', 1):  # one self-play game completed inside the discarded generation
        attempted, accepted = consumption['seeds']['42'], reference['consumption']['seeds']['42']
        assert attempted['selfplay_games_completed'] == accepted['selfplay_games_completed'] + 1
    if point == ('progress:update', '42', 2):
        attempted, accepted = consumption['seeds']['42'], reference['consumption']['seeds']['42']
        assert attempted['optimizer_steps_attempted'] == accepted['optimizer_steps_attempted'] + 2
    records = journal_records(directory)
    discarded = [r for r in records if r['event'] == 'generation_discarded']
    if point in DISCARDS_GENERATION:
        assert len(discarded) == 1 and 'hard interruption' in discarded[0]['reason']
    else:
        assert discarded == []
    if point[0] in ('after_boundary_files', 'before_commit_record'):
        assert list((directory / f'runs/seed-{point[1]}/staging').iterdir())  # orphan staging, never authoritative


def test_signal_stops_the_cli_cooperatively(tmp_path, tiny_packages, launch_runtime):
    path, token = write_declaration(tiny_packages, declaration_document(tiny_packages, launch_runtime, games=40),
                                    'signal.json')
    directory = tmp_path / 'signal'
    process = subprocess.Popen([sys.executable, '-m', 'games.connect4.alphazero_v2.campaign', 'run',
                                '--declaration', str(path), '--campaign-dir', str(directory),
                                '--authorize', token, '--seed', '42'], cwd=ROOT, env=pinned_env(),
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    deadline = time.time() + 120
    while time.time() < deadline and not any(
            '"selfplay_game"' in line for line in ((directory / 'journal.jsonl').read_text().splitlines()
                                                   if (directory / 'journal.jsonl').exists() else [])):
        time.sleep(0.2)
    process.send_signal(signal.SIGINT)
    stdout, stderr = process.communicate(timeout=120)
    assert process.returncode == C.EXIT_STOPPED, stdout + stderr
    end = [r for r in journal_records(directory) if r['event'] == 'attempt_end'][-1]
    assert end['status'] == f'stopped: signal {int(signal.SIGINT)}'
    assert any(r['event'] == 'generation_discarded' for r in journal_records(directory))


def test_quantiles_are_nearest_rank_including_minimum():
    values = list(range(1, 101))
    assert D.quantiles(values, (0.0, 0.5, 0.95, 1.0)) == dict(p0=1, p50=50, p95=95, p100=100)
