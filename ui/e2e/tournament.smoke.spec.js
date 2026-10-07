import { test, expect } from '@playwright/test';
import { create, mockTournament, seedTournament, tournamentFixture } from './fixtures/tournament.js';
for (const width of [1440, 1024, 820, 768, 375, 320]) test(`Tournament setup, active viewer, local replay and focus at ${width}px`, async ({ page }, info) => {
  await page.setViewportSize({ width, height: 1000 }); await page.emulateMedia({ reducedMotion: 'reduce' });
  const errors = []; page.on('pageerror', e => errors.push(e.message));
  const s = await mockTournament(page); await create(page);
  await expect(page.getByRole('heading', { level: 1 })).toHaveText('Tournament Lab');
  await expect(page.getByRole('navigation').getByRole('link', { name: 'Tournament Lab' })).toHaveAttribute('aria-current', 'page');
  expect(s.starts).toHaveLength(0);
  await page.locator('#tournament-watch').focus(); await page.keyboard.press('Enter');
  await expect(page.locator('.board-revision')).toHaveText('Move 0');
  await expect(page.locator('#match-next')).toBeEnabled();
  await page.locator('#match-next').focus(); await page.keyboard.press('Enter');
  await expect(page.locator('.board-revision')).toHaveText('Move 1'); await expect(page.locator('#match-next')).toBeEnabled();
  await page.locator('#match-previous').click(); await expect(page.locator('.board-revision')).toHaveText('Move 0');
  await expect(page.locator('#match-next')).toBeDisabled(); await page.locator('#match-live').click();
  expect(s.moves).toHaveLength(1);
  if (width <= 760) { await expect(page.locator('#tournament-round')).toBeVisible(); await page.locator('#tournament-round').selectOption('2');
    await expect(page.locator('.mobile-round-list')).toContainText('Final'); await page.locator('#tournament-round').selectOption('0'); }
  else await expect(page.locator('.desktop-bracket')).toBeVisible();
  const tab = info.project.name === 'webkit' && process.platform === 'darwin' ? 'Alt+Tab' : 'Tab';
  const control = page.locator('#match-previous'); await control.focus(); await page.keyboard.press(tab); await page.keyboard.press(`Shift+${tab}`);
  expect(await control.evaluate(n => getComputedStyle(n).outlineStyle)).toBe('solid');
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await seedTournament(page, tournamentFixture(16)); await page.reload();
  await expect(page.getByRole('region', { name: 'Tournament Champion', exact: true })).toBeVisible();
  const cards = page.locator(width <= 760 ? '.mobile-round-list .bracket-card' : '.desktop-bracket .bracket-card'); await cards.first().click();
  await page.locator('#match-previous').click(); await expect(page.locator('.board-revision')).toHaveText('Move 6');
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.screenshot({ path: info.outputPath(`tournament-${width}.png`), fullPage: true }); expect(errors).toEqual([]);
});
for (const width of [1440, 1024, 820, 768, 375, 320]) test(`64-player field and bracket round selection at ${width}px`, async ({ page }) => {
  await page.setViewportSize({ width, height: 1000 }); await create(page, 64);
  await expect(page.locator('.desktop-bracket .bracket-card')).toHaveCount(63);
  if (width <= 760) {
    await page.locator('#tournament-round').selectOption('5'); await expect(page.locator('.mobile-round-list .bracket-card')).toHaveCount(1);
  } else {
    await page.locator('.desktop-bracket').evaluate(n => { n.scrollLeft = n.scrollWidth; });
    await expect(page.locator('.bracket-destination')).toBeVisible();
  }
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await seedTournament(page, tournamentFixture(64)); await page.reload();
  await expect(page.getByRole('region', { name: 'Tournament Champion', exact: true })).toBeVisible();
  const cards = page.locator(width <= 760 ? '.mobile-round-list .bracket-card' : '.desktop-bracket .bracket-card');
  if (width <= 760) await page.locator('#tournament-round').selectOption('5');
  await cards.first().click(); await page.locator('#match-previous').click();
  await expect(page.locator('.board-revision')).toHaveText('Move 6');
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
});
