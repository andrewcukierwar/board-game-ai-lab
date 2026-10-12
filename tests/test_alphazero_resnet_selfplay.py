"""Phase 4D.4H-C: decision rules, uncertainty and preservation logic (no model, no training)."""
import pytest

from research.alphazero_resnet_selfplay import analyze, common, prepare

RULES = prepare.ANALYSIS


def rows(a, b, clusters):
    return [(f'c{i % clusters}', x, y) for i, (x, y) in enumerate(zip(a, b))]


def test_contrast_is_paired_and_deterministic():
    data = {42: rows([1, 1, 0, 1] * 10, [0, 1, 0, 0] * 10, 8), 314159: rows([1, 0] * 20, [0, 0] * 20, 8)}
    first = analyze.contrast(data, 'x', 2000, 7)
    assert first == analyze.contrast(data, 'x', 2000, 7)
    assert first['per_seed']['42']['difference'] == pytest.approx(0.5)
    assert first['per_seed']['314159']['difference'] == pytest.approx(0.5)
    assert first['pooled']['difference'] == pytest.approx(0.5)
    low, high = first['pooled']['interval95']
    assert low <= 0.5 <= high and low > 0
    # Each label has its own stream, so unrelated contrasts never share resamples.
    uneven = {42: rows([0.1 * (i % 7) for i in range(60)], [0.05 * (i % 11) for i in range(60)], 12)}
    assert analyze.contrast(uneven, 'x', 2000, 7)['pooled']['interval95'] != analyze.contrast(uneven, 'y', 2000, 7)['pooled']['interval95']


def test_identical_arms_have_zero_width_interval():
    data = {42: rows([1, 0, 1], [1, 0, 1], 3), 314159: rows([0, 0, 1], [0, 0, 1], 3)}
    result = analyze.contrast(data, 'same', 500, 1)
    assert result['pooled'] == dict(difference=0.0, interval95=[0.0, 0.0])
    assert analyze.direction(result, True) == 'INCONCLUSIVE'


def test_paired_rows_refuses_membership_mismatch():
    a = [dict(id='p', cluster='c', v=1), dict(id='q', cluster='c', v=0)]
    with pytest.raises(ValueError):
        analyze.paired_rows(a, a[:1], 'v')
    assert analyze.paired_rows(a, list(reversed(a)), 'v') == [('c', 1.0, 1.0), ('c', 0.0, 0.0)]


def result(seed_a, seed_b, low, high):
    return dict(per_seed={'42': dict(difference=seed_a), '314159': dict(difference=seed_b)},
                pooled=dict(difference=(seed_a + seed_b) / 2, interval95=[low, high]))


def test_strength_rule():
    rule, steady = RULES['strength'], result(0.02, 0.02, -0.05, 0.09)
    assert analyze.strength_verdict(result(0.10, 0.08, 0.01, 0.17), steady, 0.5, rule)[0] == 'PASS'
    # A small positive difference with broad uncertainty is not proof of superiority.
    assert analyze.strength_verdict(result(0.04, 0.03, -0.08, 0.15), steady, 0.5, rule)[0] == 'INCONCLUSIVE'
    # Below the practical margin even when the interval excludes zero.
    assert analyze.strength_verdict(result(0.04, 0.04, 0.01, 0.07), steady, 0.5, rule)[0] == 'INCONCLUSIVE'
    # One seed disagrees.
    assert analyze.strength_verdict(result(0.30, -0.02, 0.02, 0.26), steady, 0.5, rule)[0] == 'INCONCLUSIVE'
    # Not a single-checkpoint result: the earlier checkpoint must not point the other way.
    assert analyze.strength_verdict(result(0.10, 0.08, 0.01, 0.17), result(-0.1, -0.1, -0.2, 0.0), 0.5, rule)[0] == 'INCONCLUSIVE'
    assert analyze.strength_verdict(result(-0.10, -0.08, -0.17, -0.01), steady, 0.5, rule)[0] == 'FAIL'
    assert analyze.strength_verdict(result(-0.10, -0.08, -0.17, 0.01), steady, 0.5, rule)[0] == 'INCONCLUSIVE'
    assert analyze.strength_verdict(result(0.10, 0.08, 0.01, 0.17), steady, 0.95, rule) == ('INCONCLUSIVE', 'endpoint saturated')


def test_tactical_gate():
    rule = RULES['tactical']
    good, bad, vague = result(0.0, 0.02, -0.04, 0.06), result(-0.08, -0.06, -0.13, -0.01), result(-0.04, -0.04, -0.12, 0.03)
    assert analyze.tactical_verdict([good, good], rule) == 'PASS'
    assert analyze.tactical_verdict([good, bad], rule) == 'FAIL'
    assert analyze.tactical_verdict([good, vague], rule) == 'INCONCLUSIVE'


def test_compute_claim_and_recommendation():
    rule = RULES['equal_compute']
    assert [analyze.compute_claim(r, rule) for r in (1.0, 1.10, 0.85, 1.25, 1.3, 0.7)] == [
        'APPROXIMATELY_EQUAL', 'APPROXIMATELY_EQUAL', 'ROUGHLY_COMPARABLE', 'ROUGHLY_COMPARABLE', 'NOT_EQUAL', 'NOT_EQUAL']
    assert analyze.recommendation('PASS', 'PASS', 'APPROXIMATELY_EQUAL') == 'REPLACE_WITH_RESNET3X64'
    assert analyze.recommendation('PASS', 'INCONCLUSIVE', 'ROUGHLY_COMPARABLE') == 'REPLACE_WITH_RESNET3X64'
    assert analyze.recommendation('PASS', 'FAIL', 'APPROXIMATELY_EQUAL') == 'KEEP_CNN'
    assert analyze.recommendation('PASS', 'PASS', 'NOT_EQUAL') == 'NO_DEMONSTRATED_ADVANTAGE_KEEP_CNN_DEFAULT'
    assert analyze.recommendation('INCONCLUSIVE', 'PASS', 'APPROXIMATELY_EQUAL') == 'NO_DEMONSTRATED_ADVANTAGE_KEEP_CNN_DEFAULT'
    assert analyze.recommendation('FAIL', 'PASS', 'APPROXIMATELY_EQUAL') == 'KEEP_CNN'


def test_direction_orients_losses_and_accuracies():
    lower = result(-0.1, -0.2, -0.25, -0.05)
    assert analyze.direction(lower, higher_is_better=False) == 'BETTER'
    assert analyze.direction(lower, higher_is_better=True) == 'WORSE'
    assert analyze.direction(result(-0.1, 0.2, 0.01, 0.2), True) == 'INCONCLUSIVE'


def test_protected_bytes_are_enforced_but_additions_and_commits_are_not():
    before = {label: {'a.pt': 'x', 'b.json': 'y'} for label, _ in common.PROTECTED}
    added = {label: dict(files, **{'new-by-another-session.md': 'z'}) for label, files in before.items()}
    assert common.compare_protected(before, added)['intact']
    changed = {label: dict(files, **{'a.pt': 'changed'}) for label, files in before.items()}
    report = common.compare_protected(before, changed)
    assert not report['intact'] and report['trees']['phase4d4g-output']['changed'] == ['a.pt']
    missing = {label: {'a.pt': 'x'} for label in before}
    assert common.compare_protected(before, missing)['trees']['phase4d4h-b-output']['missing'] == ['b.json']


def test_frozen_budgets_keep_the_declared_shape():
    cnn, resnet = prepare.BUDGETS['cnn'], prepare.BUDGETS['resnet3x64']
    assert cnn['games_per_generation'] * 2 == resnet['games_per_generation'] * 3
    assert cnn['optimizer_updates'] == cnn['games_per_generation'] and resnet['optimizer_updates'] == resnet['games_per_generation']
    assert len(prepare.ORDER) == prepare.GENERATIONS == 6 and prepare.ORDER[0] == list(reversed(prepare.ORDER[1]))
    assert all(sorted(map(tuple, order)) == sorted(common.ARMS) for order in prepare.ORDER)
    # Equal cumulative games: CNN generation 4 against ResNet generation 6, CNN 2 against ResNet 3.
    assert 4 * cnn['games_per_generation'] == 6 * resnet['games_per_generation']
    assert 2 * cnn['games_per_generation'] == 3 * resnet['games_per_generation']
