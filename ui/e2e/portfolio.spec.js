import { test, expect } from '@playwright/test';
import { status } from '../tests/fixtures/alphazero.js';
import { recommendedField } from '../src/connect4/competitionFields.js';
import { competitorLabel } from '../src/connect4/competitorConfig.js';

const victor = (process.env.VICTOR_LABS_UI ?? process.env.VICTOR_RELEASE_UI) === 'true';
const order = ['random', 'negamax', 'mcts', 'mcts_tactical', 'alphazero', ...(victor ? ['victor_research'] : [])];
test.beforeEach(async ({ page }) => {
  page.on('pageerror', error => { throw error; });
  // Discovery only: these checks never start a game.
  await page.route('**/v1/**', route => route.request().url().endsWith('/agents/alphazero') ? route.fulfill({ json: status }) : route.abort());
});
const fits = page => page.evaluate(() => document.documentElement.scrollWidth <= innerWidth);

test('AlphaZero precedes Victor on the homepage, Play and Match Lab', async ({ page }) => {
  await page.goto('/');
  await expect(page.locator('#agents h3')).toHaveText(['Random', 'Negamax', 'Classic MCTS', 'Tactical MCTS', 'AlphaZero', ...(victor ? ['Victor Research'] : [])]);
  const research = await page.locator('#research').textContent();
  expect(research.indexOf('Train a neural network')).toBeLessThan(research.indexOf('Strategic reasoning + exact proofs'));
  await expect(page.getByRole('link', { name: /Play against AlphaZero/ })).toHaveAttribute('href', '/connect4');
  await page.goto('/connect4'); await expect(page.locator('input[value="alphazero"]')).toBeVisible();
  expect(await page.locator('input[name="opponent"]').evaluateAll(inputs => inputs.map(i => i.value))).toEqual(order);
  await page.goto('/connect4/match-lab'); await expect(page.locator('input[name="player-2"][value="alphazero"]')).toBeVisible();
  for (const player of [1, 2]) expect(await page.locator(`input[name="player-${player}"]`).evaluateAll(inputs => inputs.map(i => i.value))).toEqual(['human', ...order]);
});

test('Tournament: modern default field, Human last in every slot, one-Human rule', async ({ page }) => {
  await page.goto('/connect4/tournament'); await expect(page.locator('#entrant-1 option', { hasText: 'AlphaZero' })).toHaveCount(2);
  const selected = async n => page.locator('.entrant-row select').evaluateAll((s, n) => s.slice(0, n).map(x => x.selectedOptions[0].textContent), n);
  expect(await selected(8)).toEqual(recommendedField(8, { alphaZero: true, victor }).map(competitorLabel));
  for (let i = 1; i <= 8; i++) {
    const labels = await page.locator(`#entrant-${i} option`).allTextContents();
    expect(labels.at(-1)).toBe('You · Human');
    expect(labels.indexOf('AlphaZero (Experimental) · safeguard On')).toBeLessThan(labels.length - 1);
    if (victor) expect(labels.indexOf('AlphaZero (Experimental) · safeguard On')).toBeLessThan(labels.indexOf('Victor Research (Experimental)'));
  }
  await page.locator('#entrant-8').selectOption({ label: 'You · Human' });
  await expect(page.locator('#entrant-1').getByRole('option', { name: 'You · Human', exact: true })).toBeDisabled();
  await page.locator('#tournament-size').selectOption('16');
  expect(await selected(16)).toEqual(recommendedField(16, { alphaZero: true, victor }).map(competitorLabel));
  await page.locator('#entrant-1').selectOption({ label: 'Negamax · depth 12' });
  await page.getByRole('button', { name: 'Reset default field' }).click();
  expect(await selected(16)).toEqual(recommendedField(16, { alphaZero: true, victor }).map(competitorLabel));
});

test('Season: AI-only menus in canonical order and modern defaults at every size', async ({ page }) => {
  await page.goto('/connect4/season'); await expect(page.locator('#season-entrant-1 option', { hasText: 'AlphaZero' })).toHaveCount(2);
  const labels = await page.locator('#season-entrant-1 option').allTextContents();
  expect(labels.some(l => l.includes('Human'))).toBe(false);
  expect(labels.at(-1)).toBe(victor ? 'Victor Research (Experimental)' : 'AlphaZero (Experimental) · safeguard On');
  await expect(page.locator('.tournament-setup').getByRole('link', { name: 'Tournament Lab' })).toHaveAttribute('href', '/connect4/tournament');
  for (const size of [4, 6, 8, 10, 12]) {
    await page.locator('#season-size').selectOption(String(size));
    expect(await page.locator('.entrant-row select').evaluateAll(s => s.map(x => x.selectedOptions[0].textContent))).toEqual(recommendedField(size, { alphaZero: true, victor }).map(competitorLabel));
  }
});

for (const width of [1440, 820, 375, 320]) test(`homepage and setup layouts fit at ${width}px`, async ({ page }) => {
  await page.setViewportSize({ width, height: 1000 });
  for (const path of ['/', '/connect4', '/connect4/tournament', '/connect4/season']) { await page.goto(path); expect(await fits(page)).toBe(true); }
});
