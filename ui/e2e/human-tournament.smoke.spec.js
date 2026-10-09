import { test, expect } from '@playwright/test';
import { mockTournament, seedTournament, KEY } from './fixtures/tournament.js';
import { humanFixture, resultFixture } from './fixtures/human-tournament.js';
test.beforeEach(async ({ page }) => { page.on('pageerror', error => { throw error; }); });
for (const color of [0, 1]) test(`Human ${color ? 'Yellow' : 'Red'} keyboard start, AI opener, live board and paused move`, async ({ page }) => {
  const s = await mockTournament(page); await seedTournament(page, humanFixture(color)); await page.goto('/connect4/tournament');
  await page.locator('#run-tournament').click();
  await expect(page.locator('.tournament-status')).toContainText('Waiting for Human'); expect(s.starts).toHaveLength(0);
  await page.locator('#play-your-match').focus(); await page.keyboard.press('Enter');
  await expect(page.locator('.tournament-viewer')).toBeFocused();
  await expect(page.locator('.board-revision')).toHaveText('Move 0');
  await expect(page.locator('.human-color-assignment')).toContainText(color ? 'You are Yellow' : 'You are Red');
  if (color) {
    await expect(page.locator('.cell:enabled')).toHaveCount(0);
    await page.locator('#match-speed').selectOption('fast'); await page.locator('#match-autoplay').click();
    await expect(page.locator('.human-match-status')).toContainText('Autoplay waiting for you');
    await page.locator('#match-autoplay').click(); expect(s.moves[0]).not.toHaveProperty('column');
  }
  await expect(page.locator('.human-match-status')).toContainText('Your turn'); await expect(page.locator('#match-next')).toBeDisabled();
  const cell = page.locator(`.cell[data-column="${color ? 1 : 0}"]`).first();
  await cell.focus(); await page.keyboard.press('Enter');
  await expect(page.locator('.board-revision')).toHaveText(`Move ${color ? 2 : 1}`);
  expect(s.moves.at(-1).column).toBe(color ? 1 : 0); expect(s.maxFlight).toBe(1);
  await expect(page.locator('#match-next')).toBeEnabled(); await expect(page.locator('.cell:enabled')).toHaveCount(0);
});
for (const width of [375, 320]) test(`mobile ${width}px Human setup, turn, replay, path navigation and champion`, async ({ page }) => {
  await page.setViewportSize({ width, height: 900 }); const s = await mockTournament(page);
  await page.goto('/connect4/tournament'); await page.locator('#entrant-1').selectOption({ label: 'You · Human' });
  await expect(page.locator('#entrant-2').getByRole('option', { name: 'You · Human', exact: true })).toBeDisabled();
  await page.locator('#tournament-seed').fill('1234'); await page.locator('#tournament-create').click();
  await expect(page.locator('.mobile-round-list .has-human')).toContainText('You');
  const ready = humanFixture(0); await page.evaluate(({ key, t }) => localStorage.setItem(key, JSON.stringify(t)), { key: KEY, t: ready }); await page.reload();
  await page.locator('#play-your-match').click(); await expect(page.locator('.human-match-status')).toContainText('Your turn');
  const target = page.locator('.cell[data-column="0"]').first(); const box = await target.boundingBox(); expect(box.width).toBeGreaterThanOrEqual(28); // seven columns fit a 320px device
  await target.click(); await page.locator('#match-next').click(); await expect(page.locator('.board-revision')).toHaveText('Move 2');
  await page.locator('#match-previous').click(); await expect(page.locator('.cell:enabled')).toHaveCount(0);
  await page.locator('#match-live').click(); await expect(page.locator('.cell:enabled')).toHaveCount(42);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await seedTournament(page, resultFixture(true, 64, true)); await page.reload();
  await expect(page.locator('.champion-banner')).toContainText('You are the Tournament Champion');
  await page.locator('#tournament-round').selectOption('5'); await expect(page.locator('.mobile-round-list .has-human')).toContainText('You advance');
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true); expect(s.starts).toHaveLength(1);
  await seedTournament(page, resultFixture(false, 64, true)); await page.reload();
  await expect(page.locator('.human-tournament-banner')).toContainText('You were eliminated');
  await expect(page.locator('.champion-banner')).not.toContainText('You are the Tournament Champion');
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
});
