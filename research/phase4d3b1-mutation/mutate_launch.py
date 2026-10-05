"""Phase 4D.3B.1 mutation sweep over launch control: apply one plausible defect, run the focused tests
(stopping at the first failure), restore. Reports KILLED / SURVIVED per mutation.

    PYTHONPATH=. python research/phase4d3b1-mutation/mutate_launch.py [NAMES...]

The source file is backed up before every mutation (survives a hard kill). Never run this while another
process imports the package.
"""
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
V2 = 'games/connect4/alphazero_v2/'
CAMPAIGN, LAUNCH, PROVENANCE = V2 + 'campaign.py', V2 + 'launch_control.py', V2 + 'provenance.py'
TESTS = ['tests/test_alphazero_v2_launch_control.py', 'tests/test_alphazero_v2_campaign.py']
SKIP = 'not phase4d2f_adapter_is_hash_pinned and not frozen_declaration_is_the_new'

MUTATIONS = [
    # Source binding
    ('source_not_compared', CAMPAIGN,
     '    declared, current = declaration["execution_source"], execution_source_identity()',
     '    declared = current = execution_source_identity()'),
    ('source_group_misreported', PROVENANCE,
     '    return "training" if relative in TRAINING_SOURCE_FILES else "evaluation_and_launch"',
     '    return "training"'),
    ('training_group_missing_oracle', PROVENANCE, '"network", "oracle",', '"network",'),
    ('closure_not_imported_at_launch', CAMPAIGN,
     '        importlib.import_module(module[:-len(".__init__")] if module.endswith(".__init__") else module)',
     '        pass'),
    ('mid_campaign_identity_unchecked', CAMPAIGN,
     '        problems = identity_problems(self.declaration, runtime_identity())\n        if problems:',
     '        problems = []\n        if problems:'),
    ('declared_source_consistency_unchecked', CAMPAIGN,
     '            or source["groups"] != source_groups(source["files"])):',
     '            and False):'),
    # Runtime binding
    ('runtime_not_compared', CAMPAIGN,
     '    differing = runtime_differences(declaration["runtime_identity"], runtime)',
     '    differing = []'),
    ('unavailable_fields_allowed', CAMPAIGN,
     '    missing = unavailable_runtime_fields(runtime)\n    if missing:\n        problems',
     '    missing = []\n    if missing:\n        problems'),
    ('preflight_does_not_configure', CAMPAIGN,
     '        identity = configure_deterministic_runtime(declaration["runtime"]["threads"])\n        report["runtime"]',
     '        identity = runtime_identity()\n        report["runtime"]'),
    ('cpu_probe_subprocess_only', PROVENANCE,
     '            probed = _sysctl_string("machdep.cpu.brand_string")', '            probed = None'),
    ('final_skips_attempt_identity', CAMPAIGN,
     '            if (runtime_differences(self.declaration["runtime_identity"], start["runtime"])',
     '            if False and (runtime_differences(self.declaration["runtime_identity"], start["runtime"])'),
    # Old token and resumed-token verification
    ('rejected_list_empty', LAUNCH,
     'REJECTED_DECLARATION_TOKENS = ("2741314399741c1b20510b8a2beb42e0938b0aeafc9dd0c487201e57208867d8",)',
     'REJECTED_DECLARATION_TOKENS = ()'),
    ('load_skips_rejection', CAMPAIGN,
     '    if digest in REJECTED_DECLARATION_TOKENS:\n        raise LaunchRefused(f"Declaration token',
     '    if False:\n        raise LaunchRefused(f"Declaration token'),
    ('token_not_compared', CAMPAIGN, '        if authorization != digest:', '        if False:'),
    ('directory_declaration_unchecked', CAMPAIGN,
     '                if sha256_file(copy) != self.declaration_sha256:', '                if False:'),
    ('journal_declaration_unchecked', CAMPAIGN,
     '            elif self.state.declaration_sha256 != self.declaration_sha256:', '            elif False:'),
    # Durable transitions and recovery
    ('install_skips_hash_check', CAMPAIGN,
     '            if not path.is_file() or sha256_file(path) != info["sha256"]:', '            if not path.is_file():'),
    ('no_staging_recovery', CAMPAIGN,
     '        source = final_dir if final_dir.exists() else staging', '        source = final_dir'),
    ('completed_units_rerun', CAMPAIGN,
     '        evidence = self.state.evidence(unit)\n        if evidence is not None:',
     '        evidence = self.state.evidence(unit)\n        if False:'),
    ('views_not_verified', LAUNCH,
     '        if path.read_bytes() != data:', '        if False:'),
    ('duplicate_decision_allowed', LAUNCH,
     '        self._require(generation not in seed["decisions"], f"duplicate champion decision',
     '        self._require(True, f"duplicate champion decision'),
    ('out_of_order_commit_allowed', LAUNCH,
     '        self._require(generation == (0 if latest is None else latest + 1),',
     '        self._require(True,'),
    ('open_attempts_not_recovered', LAUNCH, '    for number in state.open_attempts():', '    for number in []:'),
    ('torn_tail_not_repaired', CAMPAIGN, '            self.journal.repair()\n', ''),
    ('unterminated_record_accepted', LAUNCH,
     '            parsed = self._parse(data[offset:end]) if newline >= 0 else None',
     '            parsed = self._parse(data[offset:end])'),
    ('damage_before_valid_accepted', LAUNCH,
     '                if damaged_at is not None:\n                    raise JournalCorrupt',
     '                if False:\n                    raise JournalCorrupt'),
    ('concurrent_writer_allowed', LAUNCH,
     '            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)', '            pass'),
    # Deadline persistence
    ('lease_not_written', LAUNCH,
     '            if renew_training:\n                self.reserved_training = training + LEASE_SECONDS\n            self._write_lease()',
     '            if renew_training:\n                self.reserved_training = training + LEASE_SECONDS'),
    ('lease_renewed_only_after_expiry', LAUNCH,
     '        renew_total = force or self.reserved_total - elapsed < RENEW_BELOW_SECONDS',
     '        renew_total = force or self.reserved_total - elapsed < 0'),
    ('crash_charged_zero', LAUNCH,
     '        settled = info["end"] or info["recovered"] or info["lease"]', '        settled = info["end"] or info["recovered"]'),
    ('clean_end_charged_lease', LAUNCH,
     '        settled = info["end"] or info["recovered"] or info["lease"]',
     '        settled = info["lease"] or info["end"] or info["recovered"]'),
    ('start_check_strict', LAUNCH,
     '        if self.total_seconds() >= self.limits["campaign_seconds"]:',
     '        if self.total_seconds() > self.limits["campaign_seconds"]:'),
    ('completion_not_rechecked', CAMPAIGN,
     '        problem = self._completion_problem(training=training)\n        if problem is not None:\n            raise problem',
     '        problem = None\n        if problem is not None:\n            raise problem'),
    ('completion_ignores_stop', CAMPAIGN,
     '        if self._budget.stop_reason is not None:\n            return CampaignStop(self._budget.stop_reason)',
     '        if False:\n            return CampaignStop(self._budget.stop_reason)'),
    ('training_completion_unchecked', CAMPAIGN,
     '            problem = self._completion_problem(training=True)', '            problem = None'),
    ('calibration_check_not_forwarded', CAMPAIGN,
     '                                                               seed=seed * 1_000_003 + index, check=check)]))',
     '                                                               seed=seed * 1_000_003 + index)]))'),
    ('incomplete_not_final', CAMPAIGN,
     '    def final_evaluate(self, *, extra_check=None):\n        if self.state.outcome is not None:',
     '    def final_evaluate(self, *, extra_check=None):\n        if False:'),
    # Budget persistence
    ('game_charged_at_completion', LAUNCH,
     '        if record["kind"] in GAME_KINDS:\n            self.games_charged += 1\n\n    def _close_unit',
     '\n    def _close_unit'),
    ('ceiling_not_enforced', LAUNCH,
     '        if self.state.games_charged >= self.limits["evaluation_games_ceiling"]:', '        if False:'),
    ('prior_training_ignored', LAUNCH,
     '        self.prior_training = self.state.charged_training_seconds(self.scope)', '        self.prior_training = 0.0'),
    ('selfplay_progress_not_journaled', CAMPAIGN,
     '                if event == "game":\n                    self.journal.append(',
     '                if False:\n                    self.journal.append('),
    ('abandoned_game_not_recorded', CAMPAIGN,
     '        if isinstance(evidence, dict) and evidence.get("abandoned"):\n            self.journal.append(',
     '        if isinstance(evidence, dict) and evidence.get("abandoned"):\n            raise self._last_stop\n            self.journal.append('),
]


def run_tests():
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    proc = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-x', '-p', 'no:cacheprovider', '-k', SKIP, *TESTS],
                          cwd=ROOT, capture_output=True, text=True, env=env, timeout=3600)
    lines = proc.stdout.strip().splitlines()
    tail = lines[-1] if lines else proc.stderr[-300:]
    first = next((line for line in lines if line.startswith('FAILED')), '')
    return proc.returncode, tail, first


def main():
    names = set(sys.argv[1:])
    results = []
    for name, rel, old, new in MUTATIONS:
        if names and name not in names:
            continue
        path = ROOT / rel
        original = path.read_text()
        if original.count(old) != 1:
            results.append(dict(name=name, status='NOT-APPLICABLE', matches=original.count(old)))
            print(f'{name:40s} NOT-APPLICABLE matches={original.count(old)}', flush=True)
            continue
        backup = Path(__file__).with_name('mutation-backup')
        backup.mkdir(exist_ok=True)
        (backup / (rel.replace('/', '__') + '.orig')).write_text(original)
        try:
            path.write_text(original.replace(old, new))
            code, tail, first = run_tests()
        finally:
            path.write_text(original)
        status = 'KILLED' if code != 0 else 'SURVIVED'
        results.append(dict(name=name, file=rel, status=status, first_failure=first, tail=tail))
        print(f'{name:40s} {status:8s} {first or tail}', flush=True)
    print(json.dumps(results, indent=1))


if __name__ == '__main__':
    main()
