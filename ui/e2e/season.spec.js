import { test, expect } from '@playwright/test';
import { mkdir } from 'node:fs/promises';
import { resolve } from 'node:path';
import { create, mockSeason, seedSeason, seasonFixture, accelerate, finish, KEY, DRAW } from './fixtures/season.js';
test.beforeEach(async ({ page }) => page.on('pageerror', error => { throw error; }));
test('homepage/navigation, representative 8-player setup, strict seed, workload, all lengths/sizes and no Human', async ({ page }) => {
  const mock = await mockSeason(page); await page.goto('/'); await page.getByRole('link', { name: 'Open Season Lab' }).click();
  await expect(page.getByRole('heading', { level: 1 })).toHaveText('Season Lab'); await expect(page.locator('.entrant-row')).toHaveCount(8); await expect(page.locator('#season-entrant-1 option')).toHaveCount(9);
  await expect(page.locator('#season-entrant-1')).not.toContainText('Human');
  for (const [games, count] of [[2, 56], [4, 112], [8, 224]]) { await page.locator('#season-games').selectOption(String(games)); await expect(page.locator('.season-workload')).toContainText(`${count} scheduled games`); }
  await page.locator('#season-size').selectOption('12'); await expect(page.locator('.season-workload')).toContainText('528 scheduled games');
  for (const size of [4, 6, 8, 10, 12]) { await page.locator('#season-size').selectOption(String(size)); await expect(page.locator('.entrant-row')).toHaveCount(size); }
  await page.locator('#season-seed').fill('4294967296'); await page.locator('#season-create').click(); await expect(page.getByRole('alert')).toContainText('whole number');
  await page.locator('#season-size').selectOption('8'); await page.locator('#season-games').selectOption('4'); await page.locator('#season-seed').fill('1234'); await page.locator('#season-create').click();
  await expect(page.locator('.season-fixture')).toHaveCount(4); await expect(page.locator('.tournament-summary')).toContainText('112'); expect(mock.starts).toHaveLength(0);
  await page.locator('#season-round').selectOption('14'); await page.locator('.season-fixture').first().click(); await expect(page.locator('.season-viewer')).toContainText('Upcoming fixture');
  await page.getByRole('button', { name: 'Current round', exact: true }).click();
});
test('Next move, replay review disables move, Run round, safe Pause, reload, deterministic Run season and completion leaders', async ({ page }) => {
  const mock = await mockSeason(page); await create(page, 4); await page.locator('#season-watch').click(); await expect(page.locator('#match-next')).toBeEnabled();
  await page.locator('#match-next').click(); await expect(page.locator('.board-revision')).toHaveText('Move 1'); await expect(page.locator('#match-next')).toBeEnabled(); expect(mock.moves).toHaveLength(1);
  await page.locator('#match-previous').click(); await expect(page.locator('#match-next')).toBeDisabled(); await page.locator('#match-live').click();
  await page.reload(); await expect(page.locator('#match-next')).toBeEnabled(); await expect(page.locator('.season-status')).toHaveText('Paused'); await expect(page.locator('.board-revision')).toHaveText('Move 1');
  await accelerate(page); await page.locator('#season-run-season').click(); await expect(page.locator('.season-status')).toHaveText('Running season'); await page.locator('#season-pause').click();
  await page.clock.runFor(2000); expect(mock.moves).toHaveLength(1); await page.locator('#season-run-round').click(); await finish(page);
  expect(mock.starts).toHaveLength(2); await expect(page.locator('.tournament-summary')).toContainText('2 of 12'); await expect(page.locator('.season-ratings tbody')).not.toHaveText(/1500150015001500/);
  await page.locator('#season-round').selectOption('1'); await page.locator('.season-fixture').first().click(); await page.locator('#match-previous').click(); await expect(page.locator('.board-revision')).toHaveText('Move 6');
  await page.locator('#season-run-season').click(); await finish(page);
  expect(mock.starts).toHaveLength(12); expect(mock.moves).toHaveLength(84); expect(mock.maxFlight).toBe(1);
  await expect(page.getByRole('region', { name: 'Season Complete', exact: true })).toContainText('Standings Leader'); await expect(page.locator('.season-leaders')).toContainText('Elo Leader');
  await expect(page.locator('.season-standings tbody tr')).toHaveCount(4); await expect(page.getByRole('img', { name: 'Elo rating history by completed season game' })).toBeVisible();
  await expect(page.locator('.season-pairwise')).toContainText('Pairwise results');
  const saved = await page.evaluate(key => JSON.parse(localStorage.getItem(key)), KEY); expect(saved.completedGames).toHaveLength(12); expect(JSON.stringify(saved)).not.toContain('board');
  await page.reload(); await expect(page.locator('.season-status')).toHaveText('Season Complete'); await page.locator('#season-round').selectOption('1'); await page.locator('.season-fixture').first().click(); await page.locator('#match-previous').click(); await expect(page.locator('.board-revision')).toHaveText('Move 6');
});
test('Autoplay game completes one draw with no rematch; expiry and seeded restart preserve prior results', async ({ page }) => {
  const mock = await mockSeason(page, DRAW); await create(page, 4); await accelerate(page); await page.locator('#season-run-game').click(); await finish(page);
  expect(mock.starts).toHaveLength(1); expect(mock.moves).toHaveLength(42); await expect(page.locator('.tournament-summary')).toContainText('1 of 12');
  await page.locator('#season-watch').click(); await expect(page.locator('#match-next')).toBeEnabled(); mock.expire = true; await page.locator('#match-next').click(); await expect(page.locator('#season-restart')).toBeVisible();
  const old = mock.starts.at(-1); mock.expire = false; await page.locator('#season-restart').click(); await expect(page.locator('.board-revision')).toHaveText('Move 0');
  expect(mock.starts.at(-1).rng_seed).toBe(old.rng_seed); expect(mock.starts.at(-1).player1).toEqual(old.player1); await expect(page.locator('.tournament-summary')).toContainText('1 of 12');
});
test('lost committed response and agent_busy reconcile once and require explicit continuation', async ({ page }) => {
  const mock = await mockSeason(page); await create(page, 4); await page.locator('#season-watch').click(); await expect(page.locator('#match-next')).toBeEnabled();
  await page.route('**/v1/connect4/make_move', async route => { mock.active.columns.push(0); await route.abort(); });
  await page.locator('#match-next').click(); await expect(page.getByRole('alert')).toBeVisible(); await expect(page.locator('.board-revision')).toHaveText('Move 1'); await expect(page.locator('#match-next')).toBeDisabled();
  await page.unroute('**/v1/connect4/make_move'); await page.locator('#season-refresh').click(); await expect(page.locator('#match-next')).toBeEnabled();
  await page.route('**/v1/connect4/make_move', route => route.fulfill({ status: 503, json: { code: 'agent_busy', error: 'agent_busy' } }));
  await page.locator('#match-next').click(); await expect(page.getByRole('alert')).toContainText('agent_busy'); await expect(page.locator('.board-revision')).toHaveText('Move 1'); await expect(page.locator('.season-status')).toHaveText('Paused');
});
test('invalid storage rejects inconsistent schedule and derived aggregates', async ({ page }) => {
  const s = seasonFixture(4, 2, 2); s.schedule[0].gameSeed++; await seedSeason(page, s); await page.goto('/connect4/season'); await expect(page.getByRole('alert')).toContainText('could not be validated'); await expect(page.locator('#season-create')).toBeVisible();
});
test('route departure while a move is in flight keeps one mutation owner on return', async ({ page }) => {
  const mock = await mockSeason(page); await create(page, 4); await page.locator('#season-watch').click(); await expect(page.locator('#match-next')).toBeEnabled();
  let release, entered; const gate = new Promise(r => { release = r; }), started = new Promise(r => { entered = r; });
  await page.route('**/v1/connect4/make_move', async route => { entered(); await gate; await route.fallback(); });
  await page.locator('#match-next').click(); await started;
  const nav = page.getByRole('navigation', { name: 'Main navigation' }); await nav.getByRole('link', { name: 'Tournament Lab', exact: true }).click(); await nav.getByRole('link', { name: 'Season Lab', exact: true }).click();
  await expect(page.locator('#match-next')).toBeDisabled(); release(); await expect(page.locator('.board-revision')).toHaveText('Move 1'); await expect(page.locator('#match-next')).toBeEnabled(); expect(mock.moves).toHaveLength(1); expect(mock.maxFlight).toBe(1);
});
test('polished seeded season screenshots', async ({ page }) => {
  const folder = resolve('playwright-report/phase5d'); await mkdir(folder, { recursive: true });
  const capture = async (name, locator) => { await page.evaluate(() => { document.activeElement?.blur(); window.scrollTo(0, 0); }); await (locator ?? page).screenshot({ path: `${folder}/${name}.png`, fullPage: locator ? undefined : true, style: '.skip-link { visibility: hidden; }' }); };
  await page.setViewportSize({ width: 1440, height: 1040 }); await page.goto('/connect4/season'); await page.locator('#season-games').selectOption('4'); await page.locator('#season-seed').fill('1234'); await capture('01-setup-8x4');
  const mock = await mockSeason(page); await page.locator('#season-create').click(); await page.locator('#season-watch').click(); await expect(page.locator('#match-next')).toBeEnabled();
  for (let i = 1; i <= 6; i++) { await page.locator('#match-next').click(); await expect(page.locator('.board-revision')).toHaveText(`Move ${i}`); await expect(page.locator('#match-next')).toBeEnabled(); }
  await capture('02-active-schedule', page.locator('.season-schedule')); await capture('03-live-viewer', page.locator('.season-viewer'));
  await seedSeason(page, seasonFixture(8, 4, 48)); await page.reload(); await capture('04-standings', page.locator('.season-standings'));
  await capture('05-ratings-history', page.locator('.season-rating-group')); await capture('06-pairwise-sides', page.locator('.season-pairwise'));
  await seedSeason(page, seasonFixture(8, 4)); await page.reload(); await capture('07-complete-leaders');
  await seedSeason(page, seasonFixture(12, 8, 528, DRAW)); await page.reload(); await capture('08-large-12x8');
  await page.setViewportSize({ width: 375, height: 1000 }); await capture('09-mobile-standings', page.locator('.season-standings')); await capture('10-mobile-ratings-viewer'); await capture('11-mobile-current-fixture', page.locator('.season-viewer')); await capture('12-mobile-ratings', page.locator('.season-ratings'));  expect(mock.maxFlight).toBe(1);
});
test('completion presents separate standings and Elo leaders when rankings differ', async ({ page }) => {
  const s = seasonFixture(4, 2, 12, [0, 1, 0, 1, 0, 1, 0]); await seedSeason(page, s); await page.goto('/connect4/season');
  await expect(page.locator('.season-leaders article').nth(0)).toContainText('#1 Random');
  const raw = Object.fromEntries(s.entrants.map(e => [e.entrantId, 1500]));
  for (const g of s.completedGames) { const a = g.redEntrantId, b = g.yellowEntrantId, delta = 24 * (1 - 1 / (1 + 10 ** ((raw[b] - raw[a]) / 400))); raw[a] += delta; raw[b] -= delta; }
  const leader = Object.entries(raw).sort((a, b) => b[1] - a[1])[0]; expect(leader[0]).not.toBe('entrant-1');
  await expect(page.locator('.season-leaders article').nth(1)).toContainText(`${Math.round(leader[1])} Elo`);
  for (const [id, elo] of Object.entries(raw)) {
    const seedNumber = Number(id.split('-')[1]); const row = page.locator('.season-ratings tbody tr').filter({ has: page.getByRole('rowheader', { name: new RegExp(`^#${seedNumber} `) }) });
    await expect(row.locator('td').nth(1)).toHaveText(String(Math.round(elo)));
  }
});
test('lost final move response preserves its result and offers continuation for the next fixture', async ({ page }) => {
  const mock = await mockSeason(page); await create(page, 4); await page.locator('#season-watch').click(); await expect(page.locator('#match-next')).toBeEnabled();
  for (let i = 1; i <= 6; i++) { await page.locator('#match-next').click(); await expect(page.locator('.board-revision')).toHaveText(`Move ${i}`); await expect(page.locator('#match-next')).toBeEnabled(); }
  await page.route('**/v1/connect4/make_move', async route => { mock.active.columns.push(0); await route.abort(); }); await page.locator('#match-next').click();
  await expect(page.locator('.tournament-summary')).toContainText('1 of 12'); await expect(page.getByRole('button', { name: 'Continue after confirmed result' })).toBeVisible(); await expect(page.locator('#season-run-season')).toBeDisabled();
  await page.locator('#season-refresh').click(); await expect(page.locator('#season-run-season')).toBeEnabled(); expect(mock.starts).toHaveLength(1);
  await page.locator('#season-round').selectOption('1'); await page.locator('.season-fixture').first().click(); await page.locator('#match-previous').click(); await expect(page.locator('.board-revision')).toHaveText('Move 6');
});
