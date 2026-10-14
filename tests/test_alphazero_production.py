"""Phase 4D.4K-A production AlphaZero artifact, decision adapter and research equivalence (torch)."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import platform
import shutil

import pytest

from games.connect4.connect4 import Connect4

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'models/alphazero/az2-hc-g6-cnn-s42-v1.json'
ARTIFACT = ROOT / 'models/alphazero/az2-hc-g6-cnn-s42-v1.pt'
REFERENCE = ROOT / 'benchmarks/alphazero/phase4d4k-a/research-reference.json'
ARTIFACT_SHA256 = '4c3cf26aa58495f43094436f231486969658d70b94b246a912a5bb54756dc528'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def engine(moves):
    game = Connect4()
    for move in moves:
        assert game.make_move(move)
    return game


# Torch-required production path -----------------------------------------------------------------

torch = pytest.importorskip('torch')
from games.connect4.alphazero_v2 import production  # noqa: E402
from games.connect4.alphazero_v2.bitboard import BitboardState  # noqa: E402
from games.connect4.alphazero_v2.data import seeded_rng  # noqa: E402
from games.connect4.alphazero_v2.production import ArtifactError, load_once, load_production_model  # noqa: E402
from games.connect4.agents.alphazero_agent import AlphaZeroAgent  # noqa: E402


@pytest.fixture(scope='module')
def model():
    return load_once(MANIFEST)


@pytest.fixture(scope='module')
def reference():
    return json.loads(REFERENCE.read_text())


def same_runtime(model):
    export = model.manifest['export']['runtime']
    return (platform.python_version(), str(torch.__version__), platform.platform()) == (
        export['python'], export['torch'], export['platform'])


def copy_artifact(tmp_path, edit=None, artifact_bytes=None):
    manifest = json.loads(MANIFEST.read_text())
    if edit:
        edit(manifest)
    (tmp_path / MANIFEST.name).write_text(json.dumps(manifest))
    (tmp_path / ARTIFACT.name).write_bytes(ARTIFACT.read_bytes() if artifact_bytes is None else artifact_bytes)
    return tmp_path / MANIFEST.name


def test_approved_hash_is_pinned_in_code():
    assert production.APPROVED_ARTIFACTS == {'az2-hc-g6-cnn-s42-v1': ARTIFACT_SHA256}


def test_model_loads_once_in_eval_inference_mode(model):
    assert load_once(MANIFEST) is model
    assert not any(m.training for m in model.inference.model.modules())
    assert not any(p.requires_grad for p in model.inference.model.parameters())
    assert torch.get_num_threads() == production.TORCH_THREADS
    assert model.model_id == 'az2-hc-g6-cnn-s42-v1' and model.artifact_sha256 == ARTIFACT_SHA256


def test_reference_predictions_pass_self_check(model):
    worst = production.check_reference_predictions(model.inference, model.manifest['reference_predictions'])
    assert worst <= production.REFERENCE_TOLERANCE
    if same_runtime(model):
        assert worst == 0.0


@pytest.mark.parametrize('case,code', [
    ('missing', 'artifact_missing'), ('flipped_byte', 'artifact_integrity'), ('truncated', 'artifact_integrity'),
    ('unapproved_hash', 'artifact_integrity'), ('architecture', 'unsupported_architecture'),
    ('inference_contract', 'unsupported_architecture'), ('parameters', 'unsupported_architecture'),
    ('traversal', 'artifact_contract'), ('search_settings', 'artifact_contract'), ('extra_field', 'artifact_contract'),
    ('format_version', 'artifact_contract'), ('weights_hash', 'artifact_integrity'),
    ('reference_drift', 'self_check_failed'), ('reference_illegal', 'artifact_contract'),
    ('not_json', 'artifact_contract'), ('no_manifest', 'artifact_missing'),
])
def test_invalid_artifacts_fail_with_safe_codes(tmp_path, case, code):
    data = ARTIFACT.read_bytes()
    edits = {
        'unapproved_hash': lambda m: m['artifact'].update(sha256='1' * 64),
        'architecture': lambda m: m['architecture'].update(identity='connect4-resnet3x64-bn-policy32-value32x64-scalar-tanh-v1'),
        'inference_contract': lambda m: m['inference_contract'].update(format_version=1),
        'parameters': lambda m: m['architecture'].update(parameters=322440),
        'traversal': lambda m: m['artifact'].update(file='../az2-hc-g6-cnn-s42-v1.pt'),
        'search_settings': lambda m: m['search'].update(simulations=512),
        'extra_field': lambda m: m.update(optimizer={}),
        'format_version': lambda m: m.update(format_version=2),
        'weights_hash': lambda m: m['architecture'].update(weights_sha256='2' * 64),
        'reference_drift': lambda m: m['reference_predictions'][3].update(value=m['reference_predictions'][3]['value'] + 0.01),
        'reference_illegal': lambda m: m['reference_predictions'][0].update(moves=[0] * 7),
    }
    if case == 'missing':
        path = copy_artifact(tmp_path)
        (tmp_path / ARTIFACT.name).unlink()
    elif case == 'flipped_byte':
        path = copy_artifact(tmp_path, artifact_bytes=data[:-100] + bytes([data[-100] ^ 1]) + data[-99:])
    elif case == 'truncated':
        path = copy_artifact(tmp_path, artifact_bytes=data[:-1])
    elif case == 'not_json':
        path = tmp_path / MANIFEST.name
        path.write_text('{"format": NaN}')
    elif case == 'no_manifest':
        path = tmp_path / 'absent.json'
    else:
        path = copy_artifact(tmp_path, edits[case])
    with pytest.raises(ArtifactError) as error:
        load_production_model(path)
    assert error.value.code == code
    assert str(tmp_path) not in str(error.value)


def test_strict_loader_rejects_arbitrary_pickles_and_wrong_tensors(tmp_path):
    class Payload:
        def __reduce__(self):
            return (print, ('unsafe code executed',))
    for name, payload in [('pickle', {'contract': {}, 'model_state_dict': Payload()}),
                          ('empty', {'contract': production.INFERENCE_CONTRACT, 'model_state_dict': {}})]:
        target = tmp_path / f'{name}.pt'
        torch.save(payload, target)
        digest = sha256(target)
        (tmp_path / name).mkdir()
        path = copy_artifact(tmp_path / name,
                             lambda m: m['artifact'].update(sha256=digest, size_bytes=target.stat().st_size),
                             artifact_bytes=target.read_bytes())
        with pytest.raises(ArtifactError) as error:
            load_production_model(path, approved={'az2-hc-g6-cnn-s42-v1': digest})
        assert error.value.code == 'unsupported_architecture'


def test_failed_loads_are_not_cached(tmp_path):
    bad = copy_artifact(tmp_path, artifact_bytes=b'x' * ARTIFACT.stat().st_size)
    for _ in range(2):
        with pytest.raises(ArtifactError):
            load_once(bad)
    assert all(key[0] != str(bad.resolve()) for key in production._cache)


# Decisions ----------------------------------------------------------------------------------------

POSITIONS = {
    'empty': [], 'second_player': [3], 'early': [3, 3, 2, 4], 'midgame': [2, 0, 1, 5, 1, 0, 1, 1, 0, 2, 3, 2],
    'full_column': [3, 3, 3, 3, 3, 3, 2, 4],
    'near_terminal': [1, 5, 2, 4, 4, 4, 2, 1, 3, 4, 5, 5, 6, 5, 6, 1, 2, 2, 1, 1, 4, 5, 2, 6, 2, 6, 5, 0, 6, 6, 4,
                      1, 0, 3, 3, 0, 0, 0],
}


@pytest.mark.parametrize('name', sorted(POSITIONS))
@pytest.mark.parametrize('guarded', [False, True])
def test_decisions_are_legal_deterministic_and_preserve_original_search(model, name, guarded):
    game = engine(POSITIONS[name])
    before = [row[:] for row in game.board], game.current_player
    first = model.decide(game, tactical_guard=guarded)
    again = model.decide(Connect4(game.board, game.current_player), tactical_guard=guarded)
    assert ([row[:] for row in game.board], game.current_player) == before  # caller state untouched
    assert first.final_action in game.get_valid_moves() and first.final_action == again.final_action
    assert first.visits == again.visits and sum(first.visits) == 256
    assert all(first.visits[c] == 0 for c in range(7) if not game.is_valid_move(c))
    unguarded = model.decide(game, tactical_guard=False)
    assert first.original_action == unguarded.original_action == unguarded.final_action
    assert first.visits == unguarded.visits and first.root_logits == unguarded.root_logits
    assert (first.guard is not None) is guarded


def test_mirrored_positions_and_both_colors(model):
    moves = POSITIONS['midgame']  # X to move; its prefix leaves O to move
    for sequence in (moves, [6 - m for m in moves]):
        for prefix in (sequence, sequence[:-1]):
            game = engine(prefix)
            decision = AlphaZeroAgent(model, tactical_guard=True).choose_move(game)
            assert decision in game.get_valid_moves()


@pytest.mark.parametrize('moves', [[3, 4, 3, 4, 3, 4, 3], [6, 0, 6, 1, 5, 2, 0, 3]])
def test_terminal_and_malformed_positions_are_rejected(model, moves):
    for guarded in (False, True):
        with pytest.raises(ValueError):
            model.decide(engine(moves), tactical_guard=guarded)
    inconsistent = Connect4()
    inconsistent.board[5][0] = 'X'  # X stone but X to move: impossible turn count
    floating = Connect4()
    floating.board[0][0] = 'X'
    for bad in (inconsistent, floating):
        with pytest.raises(ValueError):
            model.decide(bad, tactical_guard=True)
    with pytest.raises(TypeError):
        model.decide(Connect4(), tactical_guard='true')


def rows_by_reason(reference, reason):
    return [r for r in reference['rows'] if not r['terminal'] and r['guarded']['guard']['reason'] == reason]


# O's open bottom-row three (columns 1-3) wins at 0 or 4: every X move allows a winning reply.
ALL_UNSAFE = [5, 1, 6, 2, 5, 3]


@pytest.mark.parametrize('reason', ['IMMEDIATE_WIN', 'AVOID_IMMEDIATE_LOSS', 'NO_SAFE_MOVE', 'UNCHANGED'])
def test_guard_rules_on_research_positions(model, reference, reason):
    rows = [dict(moves=ALL_UNSAFE)] if reason == 'NO_SAFE_MOVE' else rows_by_reason(reference, reason)
    assert rows, reason
    for row in rows[:6]:
        game = engine(row['moves'])
        decision = model.decide(game, tactical_guard=True)
        guard = decision.guard
        assert guard['reason'] == reason and decision.visits == tuple(guard['original_visits'])
        if reason == 'IMMEDIATE_WIN':
            after = Connect4(game.board, game.current_player)
            after.make_move(decision.final_action)
            assert after.check_winner() == game.current_player
        elif reason == 'AVOID_IMMEDIATE_LOSS':
            assert decision.final_action in guard['safe_actions'] and decision.original_action in guard['unsafe_actions']
            assert decision.final_action == min(guard['safe_actions'], key=lambda a: (-decision.visits[a], a))
        else:
            assert decision.final_action == decision.original_action
            if reason == 'NO_SAFE_MOVE':
                assert guard['safe_actions'] == [] and guard['no_safe_action'] and not guard['changed']


def test_production_matches_research_reference_subset(model, reference):
    """Same runtime: exact. Other runtime: floats within the declared 1e-4; visits/actions always exact."""
    exact = same_runtime(model)
    chosen = [r for i, r in enumerate(reference['rows'])
              if r['terminal'] or i % 6 == 0 or r['guarded']['guard']['changed']]
    assert len(chosen) >= 60
    for row in chosen:
        game = engine(row['moves'])
        if row['terminal']:
            with pytest.raises(ValueError):
                model.decide(game, tactical_guard=False)
            continue
        for variant, guarded in (('control', False), ('guarded', True)):
            rng = seeded_rng(f"{reference['rng']['domain']}:{row['id']}", reference['rng']['seed'])
            decision = model.decide(game, tactical_guard=guarded, rng=rng)
            expected = row[variant]
            assert (decision.original_action, decision.final_action, list(decision.visits)) == (
                expected['original_action'], expected['final_action'], expected['visits']), row['id']
            for actual, wanted in ((decision.root_logits, expected['root_logits']),
                                   (decision.root_prior, expected['root_prior']),
                                   ((decision.root_value,), (expected['root_value'],))):
                if exact:
                    assert list(actual) == list(wanted)
                else:
                    assert max(abs(a - b) for a, b in zip(actual, wanted)) <= production.REFERENCE_TOLERANCE
            if guarded:
                assert {k: v for k, v in decision.guard.items() if not k.endswith('_seconds')} == expected['guard']


def test_concurrent_decisions_match_sequential(model):
    games = [engine(moves) for moves in POSITIONS.values()] * 3
    sequential = [model.decide(g, tactical_guard=bool(i % 2)) for i, g in enumerate(games)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        concurrent = list(pool.map(lambda pair: model.decide(pair[1], tactical_guard=bool(pair[0] % 2)),
                                   enumerate(games)))
    assert [(d.final_action, d.visits, d.root_logits) for d in concurrent] == [
        (d.final_action, d.visits, d.root_logits) for d in sequential]


def test_tie_break_rng_is_per_position_and_never_shared(model):
    state = BitboardState.from_game(engine([3]))
    assert production.position_rng(state).random() == production.position_rng(state).random()
    other = BitboardState.from_game(engine([2]))
    assert production.position_rng(state).random() != production.position_rng(other).random()


def test_agent_requires_loaded_model_and_boolean_guard(model):
    with pytest.raises(TypeError):
        AlphaZeroAgent(object())
    with pytest.raises(TypeError):
        AlphaZeroAgent(model, tactical_guard=1)
    agent = AlphaZeroAgent(model, tactical_guard=True)
    column = agent.choose_move(Connect4())
    assert column == agent.last_decision.final_action and agent.last_decision.guard is not None


def test_artifact_copy_is_independent_of_original_location(tmp_path, model):
    for name in (MANIFEST.name, ARTIFACT.name):
        shutil.copy(ROOT / 'models/alphazero' / name, tmp_path / name)
    copied = load_production_model(tmp_path / MANIFEST.name)
    game = engine(POSITIONS['early'])
    assert copied.decide(game, tactical_guard=False).visits == model.decide(game, tactical_guard=False).visits
