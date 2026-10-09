import { test, expect } from '@playwright/test';
import { mkdir } from 'node:fs/promises';
import { resolve } from 'node:path';
import { create, mockTournament, seedTournament, tournamentFixture, KEY } from './fixtures/tournament.js';
import { DRAW } from './fixtures/match.js';
test.beforeEach(async ({ page }) => { page.on('pageerror', error => { throw error; }); });
const scope = page => page.locator('.tournament-status');
async function start(page) { await page.locator('#tournament-watch').click(); await expect(page.locator('#match-next')).toBeEnabled(); }
async function accelerate(page) {
  await page.clock.install({ time: new Date('2026-10-07T12:00:00Z') }); await page.clock.pauseAt(new Date('2026-10-07T12:00:01Z'));
}
async function finish(page, limit = 700) {
  for (let i = 0; i < limit; i++) {
    if (await scope(page).textContent() === 'Paused' || await scope(page).textContent() === 'Tournament complete. Champion crowned.') break;
    await page.clock.runFor(851);
    await expect(page.locator('.tournament-controls')).toHaveAttribute('aria-busy', 'false');
    // Let each network transaction and authoritative read settle before another timer.
    await expect(page.locator('.tournament-status')).not.toHaveText('Waiting / recovering authoritative history');
  }
}
test('discoverable sibling route, strict seed form, independent field and deterministic bracket on reset', async ({ page }) => {
  await page.goto('/'); await page.getByRole('link', { name: 'Open Tournament Lab' }).click();
  await expect(page.getByRole('heading', { level: 1 })).toHaveText('Tournament Lab');
  await expect(page.locator('#entrant-1 option')).toHaveCount(10 + Number((process.env.VICTOR_LABS_UI ?? process.env.VICTOR_RELEASE_UI) === 'true'));
  await page.locator('#tournament-seed').fill('1.2'); await page.locator('#tournament-create').click(); await expect(page.getByRole('alert')).toContainText('whole number');
  await page.locator('#entrant-1').selectOption('8'); await page.locator('#entrant-2').selectOption('8');
  await page.locator('#tournament-seed').fill('1234'); await page.locator('#tournament-create').click();
  const before = await page.locator('.desktop-bracket .bracket-card').allTextContents();
  const data = await page.evaluate(key => JSON.parse(localStorage.getItem(key)), KEY);
  expect(data.entrants[0].config).toEqual({ type: 'mcts', simulations: 800 }); expect(data.entrants[1].config).toEqual(data.entrants[0].config);
  await page.getByRole('button', { name: 'New tournament', exact: true }).click();
  await page.locator('#tournament-seed').fill('1234'); await page.locator('#tournament-create').click();
  expect(await page.locator('.desktop-bracket .bracket-card').allTextContents()).toEqual(before);
});
for (const [mode, count] of [['matchup', 1], ['round', 4], ['tournament', 7]]) test(`Run ${mode}, completed advancement, compact persistence and local replay`, async ({ page }) => {
  const s = await mockTournament(page); await create(page); await accelerate(page);
  await page.locator(`#run-${mode}`).click(); await finish(page);
  expect(s.starts).toHaveLength(count); expect(s.moves).toHaveLength(count * 7); expect(s.maxFlight).toBe(1);
  if (mode === 'tournament') await expect(page.getByRole('region', { name: 'Tournament Champion', exact: true })).toBeVisible();
  const stored = await page.evaluate(key => JSON.parse(localStorage.getItem(key)), KEY);
  expect(stored.rounds.flat().filter(m => m.status === 'complete')).toHaveLength(count);
  expect(JSON.stringify(stored)).not.toContain('board_after');
  await page.locator('.desktop-bracket .bracket-card').first().click();
  await page.locator('#match-previous').click(); await expect(page.locator('.board-revision')).toHaveText('Move 6');
  const calls = s.moves.length; await page.locator('#match-live').click(); expect(s.moves).toHaveLength(calls);
  await page.reload(); await expect(page.locator('.tournament-summary')).toContainText(`${count} of 7`);
  await page.locator('.desktop-bracket .bracket-card').first().click(); await page.locator('#match-previous').click();
  await expect(page.locator('.board-revision')).toHaveText('Move 6'); expect(s.starts).toHaveLength(count);
});
test('in-flight Pause accepts the POST then stops without another ply', async ({ page }) => {
  const s = await mockTournament(page); await create(page); await start(page); let release;
  const gate = new Promise(r => { release = r; });
  await page.route('**/v1/connect4/make_move', async route => { await gate; await route.fallback(); });
  await page.locator('#match-speed').selectOption('fast'); await page.locator('#run-tournament').click();
  await expect(page.locator('#match-next')).toBeDisabled();
  await page.waitForTimeout(400); await page.locator('#tournament-pause').click(); release();
  await expect(page.locator('.board-revision')).toHaveText('Move 1'); await expect(page.locator('#match-next')).toBeEnabled();
  await page.waitForTimeout(700); expect(s.moves).toHaveLength(1);
});
test('reload session reconciliation and expiry preserve progress; explicit seeded restart', async ({ page }) => {
  const initial = tournamentFixture(8, 1); await seedTournament(page, initial);
  // Remove startup injector after initial navigation so later reloads use current persisted state.
  const s = await mockTournament(page); await page.goto('/connect4/tournament');
  await start(page); await page.locator('#match-next').click(); await expect(page.locator('.board-revision')).toHaveText('Move 1');
  await page.context().clearCookies(); // storage is deliberately independent of cookies
  // Reload through navigation without the initial-data override.
  const stored = await page.evaluate(key => localStorage.getItem(key), KEY);
  await page.addInitScript(({ key, stored }) => localStorage.setItem(key, stored), { key: KEY, stored });
  await page.reload(); await expect(page.locator('#match-next')).toBeEnabled(); await expect(page.locator('.board-revision')).toHaveText('Move 1');
  s.expire = true;
  await page.locator('#match-next').click(); await expect(page.locator('#tournament-restart')).toBeVisible();
  await expect(page.locator('.tournament-summary')).toContainText('1 of 7'); const old = s.starts[0];
  s.expire = false; await page.locator('#tournament-restart').click(); await expect(page.locator('.board-revision')).toHaveText('Move 0');
  expect(s.starts[1].rng_seed).toBe(old.rng_seed); expect(s.starts[1].player1).toEqual(old.player1); expect(s.starts[1].player2).toEqual(old.player2);
});
test('three draws show explicit seeded advancement and all three local replays', async ({ page }) => {
  const s = await mockTournament(page, DRAW); await create(page); await accelerate(page);
  await page.locator('#run-matchup').click(); await finish(page); expect(s.starts).toHaveLength(3);
  await page.locator('.desktop-bracket .bracket-card').first().click();
  await expect(page.locator('.tiebreak-note')).toContainText('Advanced by seeded tiebreak after three draws.');
  for (let i = 1; i <= 3; i++) { await page.getByRole('button', { name: `Game ${i} — Draw`, exact: true }).click(); await expect(page.locator('.board-revision')).toHaveText('Move 42'); }
  await page.locator('#match-previous').click(); await expect(page.locator('.board-revision')).toHaveText('Move 41');
});
test('invalid saved topology rejects whole record and allows new tournament', async ({ page }) => {
  const t = tournamentFixture(); t.rounds[0][0].winnerEntrantId = 'impossible'; await seedTournament(page, t); await page.goto('/connect4/tournament');
  await expect(page.getByRole('alert')).toContainText('could not be validated'); await expect(page.locator('#tournament-create')).toBeVisible();
  await page.locator('#tournament-seed').fill('0'); await page.locator('#tournament-create').click(); await expect(page.getByRole('alert')).toHaveCount(0);
});
test('representative polished Tournament Lab screenshots', async ({ page }) => {
  const folder = resolve('playwright-report/phase5b'); await mkdir(folder, { recursive: true });
  await page.setViewportSize({ width: 1440, height: 1040 }); await page.goto('/connect4/tournament'); await page.locator('#tournament-seed').fill('1234');
  await page.evaluate(() => { document.activeElement?.blur(); window.scrollTo(0, 0); }); await page.screenshot({ style: '.skip-link { visibility: hidden; }', path: `${folder}/01-desktop-setup.png`, fullPage: true });
  const s = await mockTournament(page); await page.locator('#tournament-create').click(); await start(page);
  for (let i = 1; i <= 6; i++) { await page.locator('#match-next').click(); await expect(page.locator('.board-revision')).toHaveText(`Move ${i}`); await expect(page.locator('#match-next')).toBeEnabled(); }
  await page.evaluate(() => { document.activeElement?.blur(); window.scrollTo(0, 0); }); await page.screenshot({ style: '.skip-link { visibility: hidden; }', path: `${folder}/02-desktop-active-bracket.png`, fullPage: true });
  await page.locator('.tournament-viewer').screenshot({ style: '.skip-link { visibility: hidden; }', path: `${folder}/03-desktop-live-viewer.png` });
  await page.setViewportSize({ width: 375, height: 1000 }); await page.evaluate(() => { document.activeElement?.blur(); window.scrollTo(0, 0); }); await page.screenshot({ style: '.skip-link { visibility: hidden; }', path: `${folder}/07-mobile-active.png`, fullPage: true });
  await page.setViewportSize({ width: 1440, height: 1040 }); await seedTournament(page, tournamentFixture(8)); await page.reload();
  await page.evaluate(() => { document.activeElement?.blur(); window.scrollTo(0, 0); }); await page.screenshot({ style: '.skip-link { visibility: hidden; }', path: `${folder}/04-desktop-champion.png`, fullPage: true });
  await page.locator('.desktop-bracket .bracket-card').first().click(); await page.locator('#match-previous').click();
  await page.locator('.tournament-viewer').screenshot({ style: '.skip-link { visibility: hidden; }', path: `${folder}/05-desktop-completed-replay.png` });
  await seedTournament(page, tournamentFixture(64, 32)); await page.reload(); await page.evaluate(() => { document.activeElement?.blur(); window.scrollTo(0, 0); }); await page.screenshot({ style: '.skip-link { visibility: hidden; }', path: `${folder}/06-desktop-64-bracket.png`, fullPage: true });
  await seedTournament(page, tournamentFixture(8)); await page.setViewportSize({ width: 375, height: 1000 }); await page.reload(); await page.locator('#tournament-round').selectOption('2');
  await page.evaluate(() => { document.activeElement?.blur(); window.scrollTo(0, 0); }); await page.screenshot({ style: '.skip-link { visibility: hidden; }', path: `${folder}/08-mobile-champion.png`, fullPage: true }); expect(s.maxFlight).toBeLessThanOrEqual(1);
});
test('navigation during in-flight mutation keeps one owner on return', async ({ page }) => {
  const s = await mockTournament(page); await create(page); await start(page);
  let release, entered; const gate = new Promise(r => { release = r; }), waiting = new Promise(r => { entered = r; });
  await page.route('**/v1/connect4/make_move', async route => { entered(); await gate; await route.fallback(); });
  await page.locator('#match-next').click(); await waiting;
  const nav = page.getByRole('navigation'); await nav.getByRole('link', { name: 'Match Lab', exact: true }).click();
  await nav.getByRole('link', { name: 'Tournament Lab', exact: true }).click();
  await expect(page.locator('#match-next')).toBeDisabled(); await expect(page.getByRole('button', { name: 'New tournament', exact: true })).toBeDisabled();
  release(); await expect(page.locator('.board-revision')).toHaveText('Move 1'); await expect(page.locator('#match-next')).toBeEnabled();
  expect(s.moves).toHaveLength(1); expect(s.starts).toHaveLength(1); expect(s.maxFlight).toBe(1);
});
test('active game Autoplay stops at game completion and keeps replay', async ({ page }) => {
  const s = await mockTournament(page); await create(page); await start(page); await accelerate(page);
  await page.locator('#match-autoplay').click(); await finish(page);
  expect(s.moves).toHaveLength(7); expect(s.starts).toHaveLength(1);
  await page.locator('.desktop-bracket .bracket-card').first().click(); await page.locator('#match-previous').click();
  await expect(page.locator('.board-revision')).toHaveText('Move 6');
});
