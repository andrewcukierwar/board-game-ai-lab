"""Mutation sweep: apply one source mutation, run the v2 tests, restore. Reports kill/survive."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable
V2 = 'games/connect4/alphazero_v2/'
AG = 'games/connect4/agents/mcts_nn_agent.py'

MUTATIONS = [
    ('puct_sign', AG, 'return (-self.q_value + exploration', 'return (self.q_value + exploration'),
    ('backup_no_alternation', AG, '        value = -value\n', '        value = value\n'),
    ('terminal_sign', AG, '(1.0 if winner == game.current_player else -1.0)', '(-1.0 if winner == game.current_player else 1.0)'),
    ('outcome_sign', V2 + 'data.py', '(1 if winner == actor else -1)', '(-1 if winner == actor else 1)'),
    ('schedule_le', V2 + 'config.py', 'return 1.0 if ply < exploratory_plies else 0.0', 'return 1.0 if ply <= exploratory_plies else 0.0'),
    ('target_from_action_temp', V2 + 'data.py', '    def policy_target(self):\n        return visit_target(self.visits)',
     '    def policy_target(self):\n        from ..agents.mcts_nn_agent import visit_policy\n        return tuple(float(p) for p in visit_policy(self.visits, self.action_temperature))'),
    ('noise_global_numpy', V2 + 'search.py', 'noise[moves] = self._rng.dirichlet(', 'noise[moves] = np.random.dirichlet('),
    ('noise_two_draws', V2 + 'search.py', '        self.draws += 1\n        return mixed / mixed.sum(), noise',
     '        self._rng.random()\n        self.draws += 1\n        return mixed / mixed.sum(), noise'),
    ('snapshot_is_learner', V2 + 'generation.py', 'snapshot = frozen_copy(self.model)', 'snapshot = self.model'),
    ('replay_window_plus_one', V2 + 'data.py', '[-self.window_generations:]', '[-(self.window_generations + 1):]'),
    ('sample_with_replacement', V2 + 'data.py', 'positions = rng.sample(range(len(self)), batch_size)',
     'positions = [rng.randrange(len(self)) for _ in range(batch_size)]'),
    ('sample_newest_only', V2 + 'data.py', 'positions = rng.sample(range(len(self)), batch_size)',
     'newest = [i for i, (gi, _, _) in enumerate(self._index) if gi == len(self._generations) - 1]\n        positions = rng.sample(newest, batch_size)'),
    ('sampling_uses_search_rng', V2 + 'generation.py', 'self.replay.sample(config.batch_size, self.sampling_rng)',
     'self.replay.sample(config.batch_size, self.search_rng)'),
    ('optimizer_moments_reset', V2 + 'training.py', '            self.optimizer.step()\n',
     '            self.optimizer.state.clear()\n            self.optimizer.step()\n'),
    ('bias_decay', V2 + 'training.py', 'dict(params=[parameters[n] for n in no_decay], weight_decay=0.0)',
     'dict(params=[parameters[n] for n in no_decay], weight_decay=config.weight_decay)'),
    ('no_clipping', V2 + 'training.py', 'clip_grad_norm_(parameters, self.config.max_grad_norm,', 'clip_grad_norm_(parameters, 1e30,'),
    ('updates_floor', V2 + 'config.py', 'return -(-self.samples_per_new_position * new_positions // self.batch_size)',
     'return max(1, self.samples_per_new_position * new_positions // self.batch_size)'),
    ('softmax_then_mask', V2 + 'network.py', '    shifted = raw[legal] - raw[legal].max()\n    weights = np.exp(shifted)  # max term is exactly 1, so the sum is in [1, 7]\n    policy = np.zeros(7, dtype=np.float64)\n    policy[legal] = weights / weights.sum()',
     '    full = np.exp(raw - raw.max()); full /= full.sum()\n    policy = np.zeros(7, dtype=np.float64)\n    policy[legal] = full[legal]\n    policy = policy / policy.sum() if policy.sum() > 0 else policy'),
    ('reflection_keeps_visits', V2 + 'data.py', 'visits=example.visits[::-1], action=6 - example.action', 'visits=example.visits, action=example.action'),
    ('resume_skip_optimizer', V2 + 'generation.py', '    runner.trainer.optimizer.load_state_dict(payload["optimizer_state_dict"])\n', '    pass\n'),
    ('resume_skip_noise_state', V2 + 'generation.py', '    runner.root_noise.set_state(_pcg64_from_storage(rng["root_noise"]))\n', '    pass\n'),
    ('resume_skip_aug_rng', V2 + 'generation.py', '    runner.augmenter.rng.setstate(_python_rng_tuple(rng["augmentation"]))\n', '    pass\n'),
    ('strict_drop_intra', V2 + 'generation.py', 'STRICT_RUNTIME_KEYS = ("python", "torch", "numpy", "machine", "intra_op_threads")',
     'STRICT_RUNTIME_KEYS = ("python", "torch", "numpy", "machine")'),
    ('inference_loader_no_contract', V2 + 'network.py', '            or checkpoint["contract"] != INFERENCE_CONTRACT):', '            or False):'),
    ('example_temp_constant', V2 + 'data.py', 'decision.action.move, decision.action.temperature)', 'decision.action.move, 1.0)'),
    ('eval_temperature_one', V2 + 'search.py', 'return result, select_action(result.visits, 0.0, rng)', 'return result, select_action(result.visits, 1.0, rng)'),
    ('label_by_final_mover', V2 + 'data.py', 'outcome_for_actor(winner, e.actor)', 'outcome_for_actor(winner, game.current_player)'),
    ('evict_before_check_partial', V2 + 'generation.py', '            evicted = self.replay.add_generation(generation, games)\n',
     '            evicted = self.replay.add_generation(generation, games[:-1] if len(games) > 1 else games)\n'),
    ('resume_skip_counters_steps', V2 + 'generation.py', '    runner.trainer.steps = counters["trainer_steps"]\n', '    pass\n'),
    ('noise_eps_ignored', V2 + 'search.py', 'mixed = (1 - self.epsilon) * priors + self.epsilon * noise', 'mixed = 0.5 * priors + 0.5 * noise'),
    ('runtime_ignore_interop', V2 + 'provenance.py', 'for key in set(recorded) | set(current) if', 'for key in set(recorded) | set(current) if key != "inter_op_threads" and'),
    ('runtime_drop_deterministic', V2 + 'provenance.py', '        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),\n', ''),
    ('source_gate_disabled', V2 + 'generation.py', '    if strict_source and source_changes:', '    if False:'),
    ('runtime_gate_disabled', V2 + 'generation.py', '    if strict_runtime and runtime_changes:', '    if False:'),
    ('drift_check_removed', V2 + 'generation.py', '        changed = runtime_differences(self.runtime, runtime_identity())', '        changed = []'),
    ('disk_check_removed', V2 + 'generation.py', '    changed = source_differences(PROCESS_EXECUTION_SOURCE, execution_source_identity())', '    changed = []'),
    ('schedule_check_removed', V2 + 'generation.py', '        if float(temperature) != action_temperature(pre_move_ply(game), exploratory_plies):', '        if False:'),
    ('lineage_not_saved', V2 + 'generation.py', '                    lineage=deepcopy(self.lineage))', '                    lineage=[])'),
    ('closure_missing_board', V2 + 'provenance.py', '    "games/connect4/board.py",\n', ''),
]


def run_tests(test_files):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    proc = subprocess.run([PY, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', *test_files], cwd=ROOT,
                          capture_output=True, text=True, env=env, timeout=900)
    tail = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else proc.stderr[-300:]
    failed = re.search(r'(\d+) failed', tail)
    errors = re.search(r'(\d+) error', tail)
    return proc.returncode, (int(failed.group(1)) if failed else 0) + (int(errors.group(1)) if errors else 0), tail


def main():
    names = set(sys.argv[2:])
    test_files = sys.argv[1].split(',')
    results = []
    for name, rel, old, new in MUTATIONS:
        if names and name not in names:
            continue
        path = ROOT / rel
        original = path.read_text()
        if original.count(old) != 1:
            results.append((name, 'MUTATION-NOT-APPLICABLE', original.count(old)))
            print(name, 'not applicable', original.count(old), flush=True)
            continue
        backup = Path(__file__).with_name('mutation-backup')
        backup.mkdir(exist_ok=True)
        (backup / (rel.replace('/', '__') + '.orig')).write_text(original)  # survives a hard kill
        try:
            path.write_text(original.replace(old, new))
            code, failures, tail = run_tests(test_files)
        finally:
            path.write_text(original)
        status = 'KILLED' if code != 0 else 'SURVIVED'
        results.append((name, status, failures))
        print(f'{name:32s} {status:9s} failures={failures}  {tail}', flush=True)
    print(json.dumps(results))


if __name__ == '__main__':
    main()
