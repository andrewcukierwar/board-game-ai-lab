"""Phase 4D.3B torch-side preflight: evaluation agents, diagnostics, profiling helpers and the
bounded campaign launcher. Every run uses tiny synthetic settings in temporary directories;
no research training, learned checkpoint or strength claim is produced."""
import hashlib
import json
import os
from pathlib import Path
import random
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
from games.connect4.alphazero_v2.arena import RandomOpponent, run_paired_arena
from games.connect4.alphazero_v2.config import V2Config
from games.connect4.alphazero_v2.data import V2Example, encode_board, finalize_game, pre_move_ply
from games.connect4.alphazero_v2.generation import GenerationRunner, initial_model, load_resume_boundary
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


# Budget accounting --------------------------------------------------------------------------------

class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def budget_declaration(**budgets):
    return dict(budgets=dict(dict(per_run_training_seconds=100, campaign_seconds=300, evaluation_games_ceiling=3,
                                  planned_evaluation_games=3), **budgets))


def test_budget_enforces_training_campaign_and_evaluation_limits(tmp_path):
    ledger, clock = C.JsonLog(tmp_path / 'ledger.jsonl'), Clock()
    budget = C.Budget(ledger, budget_declaration(), 42, 1, clock=clock)
    budget.begin_training()
    clock.now = 99
    budget.check('training')
    clock.now = 100
    with pytest.raises(C.CampaignStop, match='per-run'):
        budget.check('training')
    budget.end_training()
    for _ in range(3):
        budget.check('evaluation')
        budget.count_evaluation_game({})
    with pytest.raises(C.CampaignStop, match='ceiling'):
        budget.check('evaluation')
    ledger.append(dict(event='attempt_end', status='stopped', **budget.counters()))
    # A second attempt inherits consumption: training and evaluation are already exhausted.
    clock2 = Clock()
    second = C.Budget(ledger, budget_declaration(), 42, 2, clock=clock2)
    with pytest.raises(C.CampaignStop, match='per-run'):
        second.check('training')
    other_seed = C.Budget(ledger, budget_declaration(), 7, 1, clock=clock2)
    other_seed.check('training')  # per-run budgets are per seed
    clock2.now = 201
    with pytest.raises(C.CampaignStop, match='campaign wall-clock'):
        other_seed.check('training')
    other_seed.request_stop('signal 2')
    with pytest.raises(C.CampaignStop, match='signal'):
        other_seed.check('evaluation')
    assert C.consumed(ledger.records())['evaluation_games'] == 3


# Campaign launcher -----------------------------------------------------------------------------

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
    return directory


def tiny_declaration(package_dir, name='declaration.json', games=2, **budget_overrides):
    manifest = json.loads((Path(package_dir) / 'manifest.json').read_text())
    config = V2Config(self_play_simulations=2, games_per_generation=games, replay_generations=2,
                      replay_max_games=2 * games, batch_size=8, max_generations=2).to_dict()
    declaration = C.build_declaration(
        {k: v for k, v in manifest.items() if k != 'exclusions'}, name='tiny-test', kind='test', config=config,
        seeds=(42,), primary_seed=42,
        budgets=dict(dict(per_run_training_seconds=3600, campaign_seconds=7200, evaluation_games_ceiling=200,
                          planned_evaluation_games=0), **budget_overrides),
        evaluation=dict(simulations=2, tactical_seeds=[0], root_noise=False, tactical_guard=False, temperature=0,
                        ties='seeded uniform'),
        champion=dict(schedule=[1, 2], arena_openings=2, baseline_opponents=['random'],
                      baseline_rows=C.baseline_rows(empty_pairs=1, prefix_families=0),
                      gate={k: v for k, v in C.statistics.CHAMPION_GATE.items() if k != 'schedule'},
                      tactical_package='tactical-development'),
        final=dict(ladder=['random', 'negamax1', 'initial_v2_512'], ladder_openings=2, calibration_games=2,
                   calibration_constant=None, one_time=True))
    path = Path(package_dir) / name
    path.write_text(json.dumps(declaration, indent=1, sort_keys=True))
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def run_script(body, timeout=600):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', **required_thread_environment(1))
    result = subprocess.run([sys.executable, '-c', textwrap.dedent(body)], capture_output=True, text=True,
                            timeout=timeout, cwd=ROOT, env=env)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


SCENARIO = '''
import json
from pathlib import Path
from games.connect4.alphazero_v2 import campaign as C
from games.connect4.alphazero_v2.network import load_inference_checkpoint, weights_sha256
declaration, token, directory = {declaration!r}, {token!r}, Path({directory!r})
statuses = []
interruptions = {interruptions!r}

def make_check():
    def check(phase, runner):
        if not interruptions:
            return
        when = interruptions[0]
        completed = None if runner is None else runner.completed_generations
        if phase == when[0] and completed == when[1]:
            interruptions.pop(0)
            raise C.CampaignStop("test interruption " + str(when))
    return check
while True:
    campaign = C.Campaign(directory, declaration, token)
    status = campaign.run_seed(42, extra_check=make_check())
    statuses.append(status)
    if status == "completed":
        break
final = C.Campaign(directory, declaration, token).final_evaluate()
run = directory / "runs" / "seed-42"
selection = json.loads((run / "selection.json").read_text())
champion = json.loads((run / "champion.json").read_text())
latest = [json.loads(l) for l in (run / "artifacts.jsonl").read_text().splitlines()]
inference = [r for r in latest if r["kind"] == "inference" and r["generation"] == 2][0]
weights = weights_sha256(load_inference_checkpoint(run / inference["path"]).model)
ledger = [json.loads(l) for l in (directory / "ledger.jsonl").read_text().splitlines()]
resumes = sorted(r["path"] for r in latest if r["kind"] == "resume")
pruned = sorted(r["path"] for r in latest if r["kind"] == "pruned")
print(json.dumps(dict(statuses=statuses, final=final, weights=weights, selected=selection["selected_generation"],
                      decisions=[(d["generation"], d["decision"]["promote"]) for d in champion["decisions"]],
                      incomplete=[c["generation"] for c in champion["incomplete_checks"]],
                      attempts=sorted({{r["attempt"] for r in ledger if r.get("seed") == 42}}),
                      consumed=C.consumed(ledger), resumes=resumes, pruned=pruned,
                      on_disk=sorted(str(p.relative_to(run)) for p in run.glob("attempt-*/*.resume.pt")),
                      final_files=sorted(p.name for p in (directory / "final").iterdir()))))
'''


def scenario(tmp_path, tiny_packages, name, interruptions):
    declaration, token = tiny_declaration(tiny_packages)
    return run_script(SCENARIO.format(declaration=str(declaration), token=token,
                                      directory=str(tmp_path / name), interruptions=interruptions))


def test_campaign_interruptions_resume_to_the_uninterrupted_result(tmp_path, tiny_packages):
    plain = scenario(tmp_path, tiny_packages, 'plain', [])
    assert plain['statuses'] == ['completed'] and plain['final'] == 'completed'
    assert [g for g, _ in plain['decisions']] == [1, 2] and plain['attempts'] == [1]
    assert plain['final_files'] == ['seed-42.json', 'started.json']
    assert len(plain['on_disk']) == 2 and plain['pruned'] == ['attempt-001/generation-0000.resume.pt']
    # Stop during the generation-1 champion check, then during generation-2 training.
    resumed = scenario(tmp_path, tiny_packages, 'resumed', [['evaluation', 1], ['training', 1]])
    assert resumed['statuses'] == [
        "stopped: development evaluation incomplete: test interruption ['evaluation', 1]",
        "stopped: test interruption ['training', 1]", 'completed']
    assert resumed['attempts'] == [1, 2, 3] and resumed['incomplete'] == [1]
    assert resumed['decisions'] == plain['decisions'] and resumed['selected'] == plain['selected']
    assert resumed['weights'] == plain['weights']
    assert resumed['consumed']['training_games'] >= plain['consumed']['training_games']
    assert resumed['consumed']['evaluation_games'] >= plain['consumed']['evaluation_games']


def test_campaign_refuses_without_authorization_runtime_or_selection(tmp_path, tiny_packages, monkeypatch):
    declaration, token = tiny_declaration(tiny_packages, name='refusals.json')
    with pytest.raises(PermissionError):
        C.Campaign(tmp_path / 'c', declaration, '0' * 64)
    campaign = C.Campaign(tmp_path / 'c', declaration, token)
    for name in THREAD_ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RuntimeError, match='thread environment'):
        campaign.require_runtime()
    with pytest.raises(RuntimeError, match='selection'):
        campaign.final_evaluate()
    other, other_token = tiny_declaration(tiny_packages, name='other.json', games=3)
    with pytest.raises(FileExistsError):
        C.Campaign(tmp_path / 'c', other, other_token)


def test_declaration_validation_rejects_inconsistent_plans(tmp_path, tiny_packages):
    path, _ = tiny_declaration(tiny_packages, name='valid.json')
    declaration = json.loads(path.read_text())
    for change in (dict(budgets=dict(declaration['budgets'], planned_evaluation_games=1)),
                   dict(generations=3), dict(seeds=[42, 42]),
                   dict(packages=dict(declaration['packages'], **{'tactical-sealed': dict(
                       declaration['packages']['tactical-sealed'], sha256='0' * 64)})),
                   dict(champion=dict(declaration['champion'], schedule=[2, 1]))):
        with pytest.raises(ValueError):
            C.validate_declaration(dict(declaration, **change), tiny_packages)


def test_full_declaration_plans_exactly_the_reviewed_evaluation_games():
    declaration = C.build_declaration({})
    assert declaration['budgets'] == dict(per_run_training_seconds=28800, campaign_seconds=86400,
                                          evaluation_games_ceiling=6400, planned_evaluation_games=6272)
    assert declaration['seeds'] == [42, 314159] and declaration['primary_seed'] == 42
    assert declaration['generations'] == 20 and declaration['champion']['schedule'] == [5, 10, 15, 20]
    assert V2Config.from_dict(dict(declaration['config'], seed=42)) == V2Config(seed=42)
    assert len(declaration['champion']['baseline_rows']) == 20


def test_signal_stops_the_cli_cooperatively(tmp_path, tiny_packages):
    declaration, token = tiny_declaration(tiny_packages, name='signal.json', games=40)
    directory = tmp_path / 'signal'
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', **required_thread_environment(1))
    process = subprocess.Popen([sys.executable, '-m', 'games.connect4.alphazero_v2.campaign', 'run',
                                '--declaration', str(declaration), '--campaign-dir', str(directory),
                                '--authorize', token, '--seed', '42'], cwd=ROOT, env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    ledger = directory / 'ledger.jsonl'
    deadline = time.time() + 120
    while time.time() < deadline and not (directory / 'runs' / 'seed-42' / 'artifacts.jsonl').exists():
        time.sleep(0.2)
    time.sleep(1.0)
    process.send_signal(signal.SIGINT)
    stdout, stderr = process.communicate(timeout=120)
    assert process.returncode == 3, stdout + stderr
    end = [json.loads(l) for l in ledger.read_text().splitlines() if '"attempt_end"' in l][-1]
    assert end['status'] == f'stopped: signal {int(signal.SIGINT)}'


def test_quantiles_are_nearest_rank_including_minimum():
    values = list(range(1, 101))
    assert D.quantiles(values, (0.0, 0.5, 0.95, 1.0)) == dict(p0=1, p50=50, p95=95, p100=100)
