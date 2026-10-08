import { test, expect } from '@playwright/test';
import { create, mockSeason, seedSeason, seasonFixture, DRAW } from './fixtures/season.js';
for (const width of [1440, 375, 320]) test(`Season setup, schedule, controls, standings, ratings, chart text, pairwise and replay at ${width}px`, async ({ page }, info) => {
  await page.setViewportSize({ width, height: 1000 }); const errors = []; page.on('pageerror', e => errors.push(e.message));
  const mock = await mockSeason(page); await create(page); await expect(page.getByRole('navigation', { name: 'Main navigation' }).getByRole('link', { name: 'Season Lab' })).toHaveAttribute('aria-current', 'page');
  expect(mock.starts).toHaveLength(0); await page.locator('#season-watch').focus(); await page.keyboard.press('Enter'); await expect(page.locator('#match-next')).toBeEnabled();
  await page.locator('#match-next').focus(); await page.keyboard.press('Enter'); await expect(page.locator('.board-revision')).toHaveText('Move 1'); await expect(page.locator('#match-next')).toBeEnabled();
  await page.locator('#season-round').selectOption('14'); await expect(page.locator('.season-fixture')).toHaveCount(4); await page.locator('.season-fixture').first().click(); await expect(page.locator('.season-viewer')).toContainText('Upcoming fixture');
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await seedSeason(page, seasonFixture(8, 4, 48)); await page.reload(); await expect(page.locator('.season-ratings')).toContainText('Observed score-rate'); await expect(page.locator('.season-ratings')).not.toContainText('Too few games');
  expect(await page.locator('.season-history svg text').first().evaluate(n => getComputedStyle(n).fill === getComputedStyle(n).color)).toBe(true);
  await page.locator('.season-history summary').click(); await expect(page.getByRole('region', { name: 'Game by game Elo history' })).toBeVisible();
  if (width <= 760) { await expect(page.locator('.pairwise-desktop')).toBeHidden(); await page.locator('#pairwise-entrant').selectOption('entrant-4'); await expect(page.locator('.pairwise-mobile li')).toHaveCount(7); }
  const tab = info.project.name === 'webkit' && process.platform === 'darwin' ? 'Alt+Tab' : 'Tab';
  await page.locator('#season-round').focus(); await page.keyboard.press(tab); await page.keyboard.press(`Shift+${tab}`); await expect(page.locator('#season-round')).toBeFocused(); expect(await page.locator('#season-round').evaluate(n => getComputedStyle(n).outlineStyle)).toBe('solid');
  await seedSeason(page, seasonFixture(12, 8, 528, DRAW)); await page.reload(); await expect(page.locator('.season-leaders')).toBeVisible(); await expect(page.locator('.tournament-summary')).toContainText('528 of 528');
  await page.locator('#season-round').selectOption('1'); await page.locator('.season-fixture').first().click(); await page.locator('#match-previous').click(); await expect(page.locator('.board-revision')).toHaveText('Move 41');
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true); expect(errors).toEqual([]);
});
