import { test, expect } from '@playwright/test';
import { mkdir } from 'node:fs/promises';
import { resolve } from 'node:path';
import { mockTournament, seedTournament, KEY } from './fixtures/tournament.js';
import { humanFixture, resultFixture } from './fixtures/human-tournament.js';
import { gamePlan, replayColumns } from '../src/tournament/model.js';
import { WIN, DRAW } from './fixtures/match.js';
const busy = page => expect(page.locator('.tournament-controls')).toHaveAttribute('aria-busy', 'false');
async function clock(page) { await page.clock.install(); await page.clock.pauseAt(new Date(Date.now() + 1000)); }
async function tick(page) { await page.clock.runFor(851); await busy(page); }
async function play(page, s, sequence = WIN) {
  await page.locator('#match-autoplay').click();
  for (let i = 0; i < 50 && await page.locator('#match-autoplay').count(); i++) {
    if (s.active.players[s.active.columns.length % 2].type === 'human') {
      await page.locator(`.cell[data-column="${sequence[s.active.columns.length]}"]`).first().click(); await busy(page);
    } else await tick(page);
  }
}
test.beforeEach(async ({ page }) => { page.on('pageerror', error => { throw error; }); });
test('AI matchups finish/persist in order until Human; no session starts without the CTA', async ({ page }) => {
  const s = await mockTournament(page); await seedTournament(page, humanFixture(0, 2)); await page.goto('/connect4/tournament'); await clock(page);
  await page.locator('#run-tournament').click();
  for (let i = 0; i < 20 && !await page.locator('#play-your-match').count(); i++) await tick(page);
  await expect(page.locator('.tournament-status')).toContainText('Waiting for Human'); expect(s.starts).toHaveLength(2); expect(s.moves).toHaveLength(14);
  const stored = await page.evaluate(k => JSON.parse(localStorage.getItem(k)), KEY); expect(stored.active).toBeNull();
  await page.locator('#play-your-match').click(); await expect(page.locator('.board-revision')).toHaveText('Move 0'); expect(s.starts).toHaveLength(3);
});
for (const color of [0, 1]) test(`Human ${color ? 'elimination' : 'advancement'} and compact local completed replay after reload`, async ({ page }) => {
  const s = await mockTournament(page); await seedTournament(page, humanFixture(color)); await page.goto('/connect4/tournament'); await clock(page);
  await page.locator('#play-your-match').click(); await play(page, s);
  await expect(page.locator('.tournament-status')).toHaveText('Paused'); expect(s.starts).toHaveLength(1);
  await expect(page.locator('.human-tournament-banner')).toContainText(color ? 'You were eliminated' : 'You advanced');
  await expect(page.locator('.human-match-status')).toContainText(color ? 'wins.' : 'You win!');
  const stored = await page.evaluate(k => localStorage.getItem(k), KEY);
  await page.addInitScript(({ key, stored }) => localStorage.setItem(key, stored), { key: KEY, stored });
  await page.reload(); await page.locator('.desktop-bracket .has-human').first().click();
  await page.locator('#match-previous').click(); await expect(page.locator('.board-revision')).toHaveText('Move 6');
  await expect(page.locator('.cell:enabled')).toHaveCount(0); expect(s.starts).toHaveLength(1);
  await expect(page.locator('.board-legend')).toContainText('You');
});
test('each drawn Human game waits for Play rematch, previews swapped colors and resolves seeded tiebreak', async ({ page }) => {
  const s = await mockTournament(page, DRAW); await seedTournament(page, humanFixture()); await page.goto('/connect4/tournament'); await clock(page);
  for (let number = 1; number <= 3; number++) {
    await page.locator('#play-your-match').click(); await play(page, s, DRAW);
    await expect(page.locator('.tournament-status')).toHaveText('Paused'); expect(s.starts).toHaveLength(number);
    if (number < 3) {
      await expect(page.locator('#play-your-match')).toHaveText('Play rematch');
      await expect(page.locator('.human-color-assignment')).toContainText(number === 1 ? 'You are Yellow' : 'You are');
      await page.locator('#run-tournament').click(); await tick(page); expect(s.starts).toHaveLength(number);
    }
  }
  await expect(page.locator('.tiebreak-note')).toContainText('Advanced by seeded tiebreak after three draws');
  expect(s.moves).toHaveLength(126); expect(s.maxFlight).toBe(1);
});
test('double-click Human POST, replay lock, lost committed response and live reload reconcile safely', async ({ page }) => {
  const s = await mockTournament(page); await seedTournament(page, humanFixture()); await page.goto('/connect4/tournament');
  await page.locator('#play-your-match').click();
  await page.route('**/v1/connect4/make_move', async route => {
    const body = route.request().postDataJSON(); s.moves.push(body); s.active.columns.push(body.column); await route.abort('failed');
  }, { times: 1 });
  await page.locator('.cell[data-column="0"]').first().evaluate(el => { el.click(); el.click(); });
  await expect(page.locator('.board-revision')).toHaveText('Move 1'); expect(s.moves).toHaveLength(1);
  await expect(page.getByRole('alert')).toBeVisible(); await page.locator('#match-refresh').click(); await busy(page);
  await page.locator('#match-next').click(); await expect(page.locator('.board-revision')).toHaveText('Move 2');
  await page.locator('#match-previous').click(); await expect(page.locator('.cell:enabled')).toHaveCount(0);
  await page.locator('#match-live').click(); await expect(page.locator('.cell:enabled')).toHaveCount(42);
  const stored = await page.evaluate(k => localStorage.getItem(k), KEY); await page.addInitScript(({ key, stored }) => localStorage.setItem(key, stored), { key: KEY, stored });
  await page.reload(); await expect(page.locator('.human-match-status')).toContainText('Your turn'); await expect(page.locator('.board-revision')).toHaveText('Move 2');
  expect(s.starts).toHaveLength(1); expect(s.moves).toHaveLength(2);
});
test('expired Human session explains restart from zero and retains planned seed/colors', async ({ page }) => {
  const s = await mockTournament(page); await seedTournament(page, humanFixture()); await page.goto('/connect4/tournament');
  await page.locator('#play-your-match').click(); await page.locator('.cell[data-column="0"]').first().click();
  await expect(page.locator('.board-revision')).toHaveText('Move 1'); s.expire = true; await page.locator('#match-next').click();
  await expect(page.getByRole('alert')).toContainText('restart from the beginning');
  await expect(page.locator('#tournament-restart')).toHaveText('Restart this game'); s.expire = false;
  await page.locator('#tournament-restart').click(); await expect(page.locator('.board-revision')).toHaveText('Move 0');
  expect(s.starts[1].rng_seed).toBe(s.starts[0].rng_seed); expect(s.starts[1].player1).toEqual(s.starts[0].player1); expect(s.starts[1].player2).toEqual(s.starts[0].player2);
});
test('polished Human tournament desktop and mobile screenshots', async ({ page }) => {
  const folder = resolve('playwright-report/phase5c'); await mkdir(folder, { recursive: true });
  async function shot(name, locator) { await page.evaluate(() => document.activeElement?.blur()); await (locator ?? page).screenshot({ style: '.skip-link { visibility: hidden; }', path: `${folder}/${name}.png`, ...(locator ? {} : { fullPage: true }) }); }
  await page.setViewportSize({ width: 1440, height: 1040 }); await page.goto('/connect4/tournament');
  await page.locator('#entrant-1').selectOption({ label: 'You · Human' }); await page.locator('#tournament-seed').fill('1234'); await shot('01-setup-human');
  const s = await mockTournament(page); await seedTournament(page, humanFixture()); await page.reload();
  await shot('02-desktop-human-path'); await shot('03-your-match-ready', page.locator('.human-tournament-banner'));
  await page.locator('#play-your-match').click(); await shot('04-human-red', page.locator('.tournament-viewer'));
  await page.setViewportSize({ width: 375, height: 1000 }); await shot('08-mobile-human-turn');
  await page.setViewportSize({ width: 1440, height: 1040 }); await seedTournament(page, humanFixture(1)); await page.reload(); await page.locator('#play-your-match').click();
  await page.locator('#match-next').click(); await expect(page.locator('.human-match-status')).toContainText('Your turn'); await shot('05-human-yellow-ai-opener', page.locator('.tournament-viewer'));
  await seedTournament(page, resultFixture()); await page.reload(); await shot('06-human-advanced');
  await seedTournament(page, resultFixture(true, 8, true)); await page.reload(); await shot('07-human-champion');
  await page.setViewportSize({ width: 320, height: 1000 }); await page.locator('#tournament-round').selectOption('2'); await shot('09-mobile-human-champion'); expect(s.maxFlight).toBeLessThanOrEqual(1);
});

test('real local API Human policy completes, compacts, replaces its session and replays locally', async ({ page, request }) => {
  const t = humanFixture(); t.entrants.forEach(e => { if (e.config.type !== 'human') e.config = { type: 'negamax', depth: 1 }; });
  const plan = gamePlan(t, t.rounds[0][0]), posts = [];
  page.on('request', r => { if (r.url().endsWith('/make_move')) posts.push(r.postDataJSON()); });
  await seedTournament(page, t); await page.goto('/connect4/tournament'); await clock(page);
  await page.locator('#play-your-match').click(); await expect(page.locator('.board-revision')).toHaveText('Move 0');
  const activeId = await page.evaluate(k => JSON.parse(localStorage.getItem(k)).active.gameId, KEY);
  await page.locator('#match-autoplay').click();
  for (let i = 0; i < 50; i++) {
    const saved = await page.evaluate(k => JSON.parse(localStorage.getItem(k)), KEY); if (!saved.active) break;
    const { game } = replayColumns(saved.active.columns, plan.playerConfigs);
    if (game.players[game.currentPlayer].type === 'human') {
      const col = [3, 2, 4, 1, 5, 0, 6].find(col => game.legalMoves.includes(col));
      await page.getByRole('button', { name: `Drop in column ${col + 1}`, exact: true }).click(); await busy(page);
    } else await tick(page);
  }
  await expect(page.locator('.tournament-status')).toHaveText('Paused');
  const saved = await page.evaluate(k => JSON.parse(localStorage.getItem(k)), KEY);
  expect(saved.active).toBeNull(); expect(saved.rounds[0][0].games).toHaveLength(1);
  expect(posts.length).toBeGreaterThan(6);
  posts.forEach((p, i) => { expect(p.revision).toBe(i); expect('column' in p).toBe(plan.playerConfigs[i % 2].type === 'human'); });
  expect(replayColumns(saved.rounds[0][0].games[0].columns, plan.playerConfigs).game.gameOver).toBe(true);
  await page.locator('.desktop-bracket [data-matchup="r1-m2"]').click(); await page.locator('#tournament-watch').click();
  await expect(page.locator('.board-revision')).toHaveText('Move 0');
  const api = process.env.PLAYWRIGHT_API_URL || 'http://localhost:8007';
  expect((await request.get(`${api}/v1/connect4/games/${activeId}/history`)).status()).toBe(404);
  const count = posts.length; await page.locator('.desktop-bracket [data-matchup="r1-m1"]').click();
  await page.locator('#match-previous').click(); await expect(page.locator('.board-legend')).toContainText('You'); expect(posts).toHaveLength(count);
});
