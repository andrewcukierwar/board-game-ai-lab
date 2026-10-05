"""Engine/replay audit correctness; scripted histories only, no learning runs."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from games.connect4.connect4 import Connect4
from games.connect4 import neural_value_target_audit as audit

DRAW = [2, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
        3, 3, 3, 3, 3, 3, 6, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5,
        6, 6, 6, 6, 6]
X_WIN = [0, 1, 0, 1, 0, 2, 0]
O_WIN = [0, 1, 0, 1, 2, 1, 2, 1]


def position(moves):
    game = Connect4()
    for move in moves:
        assert not game.is_game_over() and move in game.get_valid_moves()
        assert game.make_move(move)
    return game


@pytest.mark.parametrize("moves,value,actor", [
    (X_WIN[:-1], 1, 0), (O_WIN[:-1], 1, 1),
    ([6, 1, 6, 2, 5, 3], -1, 0), ([1, 6, 2, 6, 3], -1, 1),
    ([0, 1, 0, 1, 2, 1], None, 0), ([0, 1, 0, 1, 0], None, 1),
    ([], None, 0), ([3], None, 1), (DRAW[:-1], None, 1),
])
@pytest.mark.parametrize("mirror", [False, True])
def test_conservative_proofs_both_players_and_draw_escape(moves, value, actor, mirror):
    game = position([6 - m if mirror else m for m in moves])
    before = deepcopy(game.__dict__)
    proof = audit.tactical_proof(game)
    assert proof["value"] == value and game.current_player == actor
    assert game.__dict__ == before
    if value == 1:
        for move in proof["winning_actions"]:
            assert position([6 - m if mirror else m for m in moves] + [move]).check_winner() == actor
    if value == -1:
        assert set(map(int, proof["reply_witnesses"])) == set(game.get_valid_moves())
        for move, replies in proof["reply_witnesses"].items():
            after = deepcopy(game)
            assert after.make_move(int(move)) and not after.is_game_over()
            for reply in replies:
                terminal = deepcopy(after)
                assert terminal.make_move(reply) and terminal.check_winner() == 1 - actor


def test_immediate_win_takes_precedence_over_opponent_threats():
    game = position([6, 1, 6, 2, 6, 3])
    assert audit.tactical_proof(game)["value"] == 1


def test_terminal_proof_rejected():
    with pytest.raises(ValueError, match="nonterminal"):
        audit.tactical_proof(position(X_WIN))


def synthetic_episode(moves):
    game, events, examples = Connect4(), [], []
    for ply, move in enumerate(moves, 1):
        policy = [float(a == move) for a in range(7)]
        example = dict(observation=audit.canonical_board(game), acting_player=game.current_player,
                       policy=policy, temperature=1.0, tactical_guard=False, outcome=None,
                       encoding=audit.ENCODING, policy_target=audit.POLICY_TARGET)
        events.append(dict(event="ply", game=1, game_ply=ply, actual_ply=ply, updates=0,
                           actor=game.current_player, action=move, observation=example["observation"],
                           policy=policy, visits=[32 if a == move else 0 for a in range(7)],
                           temperature=1.0, tactical_guard=False, guard_applied="disabled",
                           example_sha256=hashlib.sha256(json.dumps(example, sort_keys=True,
                                                                    allow_nan=False).encode()).hexdigest(),
                           predicted_wdl=[0.3, 0.4, 0.3]))
        examples.append(example)
        assert game.make_move(move)
    winner = game.check_winner()
    rows = []
    for e in examples:
        e["outcome"] = 0 if winner == -1 else 1 if winner == e["acting_player"] else -1
        rows.append(dict(outcome=e["outcome"], actor=e["acting_player"]))
    completed = dict(event="completed_game", index=1, moves=moves, plies=len(moves),
                     winner=winner, update_start=0, examples=examples, labels=audit.counts(rows),
                     labels_by_actor={str(a): audit.counts([r for r in rows if r["actor"] == a])
                                      for a in (0, 1)})
    events.append(completed)
    report = dict(completed_games=1, updates=0, actual_plies=len(moves), collected_plies=len(moves),
                  labels=audit.counts(rows), games=[dict(completed, update_end=0)])
    config = dict(training_sampling_seed=42, simulations=32, temperature=1.0, batch_size=32)
    return events, config, report


@pytest.mark.parametrize("moves", [X_WIN, O_WIN, DRAW])
def test_exact_replay_labels_immutable_and_serializable(moves):
    events, config, report = synthetic_episode(moves)
    before = deepcopy((events, config, report))
    rows, updates = audit.replay_events(events, config, report, "synthetic")
    assert not updates and len(rows) == len(moves)
    assert (events, config, report) == before
    for row in rows:
        assert row["outcome"] == (0 if row["winner"] == -1 else
                                  1 if row["winner"] == row["actor"] else -1)
    result = dict(rows=rows, summary=audit.summary(rows))
    assert json.loads(json.dumps(result, allow_nan=False)) == result


@pytest.mark.parametrize("corruption", ["board", "actor", "label", "fingerprint", "visits", "action"])
def test_corruption_rejected_without_repair(corruption):
    events, config, report = synthetic_episode(X_WIN)
    if corruption == "board":
        events[0]["observation"][0][0] = 1
    elif corruption == "actor":
        events[0]["actor"] = 1
    elif corruption == "label":
        events[-1]["examples"][0]["outcome"] = -1
    elif corruption == "fingerprint":
        events[0]["example_sha256"] = "bad"
    elif corruption == "visits":
        events[0]["visits"][0] = 31
    else:
        events[0]["action"] = 7
    before = deepcopy(events)
    with pytest.raises(ValueError):
        audit.replay_events(events, config, report, "synthetic")
    assert events == before


def test_manifest_failure_and_no_old_evidence_writes(tmp_path):
    evidence = tmp_path / "old"
    evidence.mkdir()
    (evidence / "events.jsonl").write_text("original\n")
    manifest = evidence / "manifest.json"
    manifest.write_text(json.dumps(dict(format_version=1,
        files_sha256={"events.jsonl": audit.sha256(evidence / "events.jsonl")})))
    before = {p.name: p.read_bytes() for p in evidence.iterdir()}
    assert audit.verify_manifest(manifest)["verified_entries"] == 1
    assert before == {p.name: p.read_bytes() for p in evidence.iterdir()}
    (evidence / "events.jsonl").write_text("corrupt\n")
    with pytest.raises(ValueError, match="Manifest mismatch"):
        audit.verify_manifest(manifest)
    assert (evidence / "events.jsonl").read_text() == "corrupt\n"


def test_output_overwrite_and_input_overlap_rejected(tmp_path):
    old = tmp_path / audit.RUNS["4D.2b"]
    old.mkdir()
    existing = tmp_path / "existing.json"
    existing.write_text("preserved")
    with pytest.raises(ValueError, match="must be new"):
        audit.main(["--evidence-root", str(tmp_path), "--output", str(existing)])
    with pytest.raises(ValueError, match="overlaps"):
        audit.main(["--evidence-root", str(tmp_path), "--output", str(old / "analysis.json")])
    assert existing.read_text() == "preserved"


def test_scores_separate_outcomes_from_proofs_and_handle_zero_probability():
    row = dict(outcome=-1, predicted_wdl=[1., 0., 0.], proof=dict(value=1))
    actual = audit.scores([row], lambda r: r["outcome"])
    exact = audit.scores([row], lambda r: r["proof"]["value"])
    assert actual["brier"] == 2 and actual["nll_infinite"] and actual["nll"] is None
    assert exact["brier"] == exact["nll"] == exact["scalar_mse"] == 0
    json.dumps(actual, allow_nan=False)


def test_sampling_counts_indices_not_collection_proportions():
    rows = [dict(outcome=0 if i == 0 else 1, collection_index=i, cohort="21-50",
                 proof=dict(value=None)) for i in range(4)]
    sampled = audit.sampling_summary(rows, [dict(sampled_indices=[1, 2, 3]),
                                          dict(sampled_indices=[0, 1, 2])])
    assert sampled["outcomes"] == {"1": 5, "0": 1, "-1": 0}
    assert sampled["draw_fraction"] == 1 / 6
    assert sampled["updates_with_draw"] == 1


def test_discarded_logits_ce_gradient_matches_actual_target():
    torch = pytest.importorskip("torch")
    from games.connect4.train_mcts_nn import training_loss
    logits = torch.tensor([[2., -3., -2.]], requires_grad=True)
    p = logits.softmax(1).detach()
    policy_logits = torch.zeros(1, 7)
    policies = torch.full((1, 7), 1 / 7)
    _, _, loss = training_loss(policy_logits, logits, policies, torch.tensor([2]),
                               return_components=True)
    actual_gradient, = torch.autograd.grad(loss, logits)
    assert torch.allclose(actual_gradient, p - torch.tensor([[0., 0., 1.]]))
    _, _, exact_loss = training_loss(policy_logits, logits, policies, torch.tensor([0]),
                                     return_components=True)
    exact_gradient, = torch.autograd.grad(exact_loss, logits)
    assert torch.allclose(actual_gradient - exact_gradient, torch.tensor([[1., 0., -1.]]))
    # No model and no optimizer exist in this discarded calculation.
