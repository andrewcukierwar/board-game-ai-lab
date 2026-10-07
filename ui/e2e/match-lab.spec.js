import { test, expect } from '@playwright/test';
import { mkdir } from 'node:fs/promises';
import { resolve } from 'node:path';
import { DRAW, WIN, matchFixture } from './fixtures/match.js';
const api = path => (process.env.PLAYWRIGHT_API_URL || '') + path;
async function configure(page, types = ['random', 'random']) {
  await page.goto('/connect4/match-lab');
  for (let i = 0; i < 2; i++) await page.locator(`input[name="player-${i + 1}"][value="${types[i]}"]`).check();
}
async function start(page, types) {
  await configure(page, types); await page.locator('#match-start').click();
  await expect(page.locator('.board-revision')).toHaveText('Move 0');
  await expect(page.locator('#match-start')).toBeEnabled();
}
const step = async (page, revision) => { await page.locator('#match-next').click(); await expect(page.locator('.board-revision')).toHaveText(`Move ${revision}`); await expect(page.locator('#match-start')).toBeEnabled(); };
const fits = page => page.evaluate(() => document.documentElement.scrollWidth <= innerWidth);

test('homepage and navigation discover a separate Match Lab without starting a session', async ({ page }) => {
  let requests = 0; page.on('request', r => { if (r.url().includes('/v1/')) requests++; });
  await page.goto('/'); await page.getByRole('link', { name: 'Open Match Lab' }).click();
  await expect(page).toHaveURL(/\/connect4\/match-lab$/);
  await expect(page.getByRole('heading', { level: 1 })).toHaveText('Match Lab');
  await expect(page.getByRole('navigation').getByRole('link', { name: 'Match Lab' })).toHaveAttribute('aria-current', 'page');
  await expect(page.getByRole('navigation').getByRole('link', { name: 'Play', exact: true })).not.toHaveAttribute('aria-current');
  expect(requests).toBe(0); await page.reload(); await expect(page.locator('#match-start')).toBeEnabled();
});

test('real Negamax vs MCTS starts paused, steps exactly once and reviews without network', async ({ page }) => {
  await configure(page, ['negamax', 'mcts']);
  await page.locator('#player-1-depth').selectOption('1'); await page.locator('#player-2-simulations').selectOption('100');
  const moves = []; page.on('request', r => { if (r.url().endsWith('/make_move')) moves.push(r.postDataJSON()); });
  const started = page.waitForResponse(r => r.url().endsWith('/start_game'));
  await page.locator('#match-start').click(); const game = await (await started).json();
  await expect(page.locator('#match-start')).toBeEnabled(); expect(game.revision).toBe(0); expect(moves).toHaveLength(0);
  await expect(page.locator('.cell:enabled')).toHaveCount(0);
  await step(page, 1); await step(page, 2);
  expect(moves).toEqual([{ game_id: game.game_id, revision: 0 }, { game_id: game.game_id, revision: 1 }]);
  await expect(page.locator('.move-list li')).toHaveCount(3);
  await expect(page.locator('.move-list')).toContainText('Red · Negamax · depth 1');
  await expect(page.locator('.move-list')).toContainText('Yellow · MCTS · 100 simulations');
  await page.locator('#match-previous').click(); await expect(page.locator('.board-revision')).toHaveText('Move 1');
  await expect(page.locator('#match-view-mode')).toHaveText('REVIEWING MOVE 1 OF 2Return to live');
  await expect(page.locator('#match-next')).toBeDisabled(); await expect(page.locator('#match-autoplay')).toBeDisabled();
  await page.locator('#match-previous').click(); await expect(page.locator('.circle.x, .circle.o')).toHaveCount(0);
  await page.locator('#match-forward').click(); await expect(page.locator('.circle.x')).toHaveCount(1);
  await page.locator('#match-live').click(); await expect(page.locator('.board-revision')).toHaveText('Move 2');
  expect((await (await page.request.get(api(`/v1/connect4/games/${game.game_id}`))).json()).revision).toBe(2);
  expect(moves).toHaveLength(2);
});

test('real AI autoplay is sequential; pause during a held committed response stops after reconciliation', async ({ page }) => {
  await start(page); let release, calls = 0, active = 0, maxActive = 0;
  const gate = new Promise(resolve => { release = resolve; });
  await page.route('**/v1/connect4/make_move', async route => {
    calls++; maxActive = Math.max(maxActive, ++active);
    const response = await route.fetch(); await gate; active--; await route.fulfill({ response });
  });
  await page.locator('#match-speed').selectOption('fast'); await page.locator('#match-autoplay').click();
  await expect(page.locator('#match-message')).toContainText('AI thinking');
  await page.locator('#match-autoplay').click(); release();
  await expect(page.locator('.board-revision')).toHaveText('Move 1'); await expect(page.locator('#match-start')).toBeEnabled();
  await page.waitForTimeout(650); expect(calls).toBe(1); expect(maxActive).toBe(1);
  await page.unroute('**/v1/connect4/make_move');
  await page.locator('#match-autoplay').click(); await expect(page.locator('.move-list li')).toHaveCount(4);
  await page.locator('#match-autoplay').click(); await expect(page.locator('#match-start')).toBeEnabled();
  const revision = await page.locator('.board-revision').textContent(); await page.waitForTimeout(650);
  await expect(page.locator('.board-revision')).toHaveText(revision);
});

for (const types of [['human', 'random'], ['random', 'human']]) test(`real autoplay waits for Human and resumes AI: ${types}`, async ({ page }) => {
  await start(page, types); await page.locator('#match-speed').selectOption('fast'); await page.locator('#match-autoplay').click();
  await expect(page.getByRole('status')).toHaveText(/Autoplay waiting for Human/);
  const before = types[0] === 'human' ? 0 : 1;
  await expect(page.locator('.board-revision')).toHaveText(`Move ${before}`);
  await page.waitForTimeout(500); await expect(page.locator('.board-revision')).toHaveText(`Move ${before}`);
  await expect(page.locator('#match-next')).toBeDisabled();
  await page.locator('.cell[data-column="3"]:enabled').last().click();
  await expect(page.locator('.board-revision')).toHaveText(`Move ${before + 2}`);
  await expect(page.getByRole('status')).toHaveText(/Autoplay waiting for Human/);
});

test('real Human/Human win preserves terminal replay and correct competitor/color result', async ({ page }) => {
  await start(page, ['human', 'human']); await expect(page.locator('#match-autoplay')).toBeDisabled();
  for (const [i, c] of WIN.entries()) {
    await page.locator(`.cell[data-column="${c}"]:enabled`).last().click(); await expect(page.locator('.board-revision')).toHaveText(`Move ${i + 1}`);
    await expect(page.locator('#match-start')).toBeEnabled();
  }
  await expect(page.getByRole('status')).toHaveText('Human wins as Red in 7 moves.');
  await expect(page.locator('.cell:enabled')).toHaveCount(0);
  await page.locator('#match-previous').click(); await expect(page.locator('.board-revision')).toHaveText('Move 6');
  await page.locator('#match-live').click(); await expect(page.getByRole('status')).toContainText('wins');
});

test('lost AI response after real server commit reconciles history without duplicate move', async ({ page }) => {
  await start(page); let calls = 0;
  await page.route('**/v1/connect4/make_move', async route => { calls++; await route.fetch(); await route.abort('failed'); });
  await page.locator('#match-autoplay').click();
  await expect(page.locator('#match-refresh')).toBeEnabled();
  await expect(page.locator('.board-revision')).toHaveText('Move 1'); await expect(page.locator('.move-list li')).toHaveCount(2);
  await expect(page.locator('#match-autoplay')).toHaveAttribute('aria-pressed', 'false');
  await page.waitForTimeout(1200); expect(calls).toBe(1);
  await page.locator('#match-refresh').click(); await expect(page.locator('#match-next')).toBeEnabled(); expect(calls).toBe(1);
  await page.locator('#match-previous').click(); await expect(page.locator('.circle.x')).toHaveCount(0);
});

test('agent_busy requires explicit refresh and continuation; no retry loop', async ({ page }) => {
  await start(page); let calls = 0;
  await page.route('**/v1/connect4/make_move', route => { calls++; return route.fulfill({ status: 503, json: { error: 'Another MCTS search is running.', code: 'agent_busy' } }); });
  await page.locator('#match-autoplay').click(); await expect(page.locator('#match-refresh')).toBeEnabled();
  await page.waitForTimeout(1200); expect(calls).toBe(1); await expect(page.locator('.board-revision')).toHaveText('Move 0');
  await expect(page.locator('#match-next')).toBeDisabled(); await page.locator('#match-refresh').click();
  await expect(page.locator('#match-next')).toBeEnabled(); expect(calls).toBe(1);
});

test('expired real session clears match and offers a fresh start', async ({ page }) => {
  await start(page); const state = await (await page.request.get(api('/v1/connect4/health'))).text(); expect(state).toBe('OK');
  let gid;
  await page.route('**/v1/connect4/make_move', route => { gid = route.request().postDataJSON().game_id; return route.fulfill({ status: 404, json: { error: 'Expired', code: 'session_not_found' } }); });
  await page.route('**/v1/connect4/games/*/history', async route => {
    await page.request.post(api('/v1/connect4/start_game'), { data: { replace_game_id: gid } }); return route.continue();
  });
  await page.locator('#match-next').click(); await expect(page.getByRole('status')).toContainText('Session expired');
  await expect(page.locator('#match-start')).toHaveText('Start match'); await expect(page.locator('.cell')).toHaveCount(0);
});

test('navigation cancels scheduled playback and ignores delayed move rendering', async ({ page }) => {
  await start(page); let release, calls = 0;
  const gate = new Promise(resolve => { release = resolve; });
  await page.route('**/v1/connect4/make_move', async route => { calls++; const response = await route.fetch(); await gate; await route.fulfill({ response }).catch(() => {}); });
  await page.locator('#match-autoplay').click(); await expect(page.locator('#match-message')).toContainText('AI thinking');
  await page.getByRole('link', { name: 'Back to Play' }).click(); release();
  await page.getByRole('navigation').getByRole('link', { name: 'Match Lab' }).click();
  await expect(page.locator('#match-start')).toHaveText('Start match'); await page.waitForTimeout(1200);
  expect(calls).toBe(1); await expect(page.locator('.cell')).toHaveCount(0);
});

async function mockMatch(page, sequence = DRAW) {
  let players, plies = [];
  await page.route('**/v1/connect4/start_game', route => {
    const body = route.request().postDataJSON(); players = [body.player1, body.player2]; plies = [];
    return route.fulfill({ status: 201, json: matchFixture(players).state });
  });
  await page.route('**/v1/connect4/make_move', route => {
    const body = route.request().postDataJSON(); plies.push(body.column ?? sequence[plies.length]);
    return route.fulfill({ json: matchFixture(players, plies).state });
  });
  await page.route('**/v1/connect4/games/*/history', route => route.fulfill({ json: matchFixture(players, plies) }));
}

test('complete controlled draw has 43 replay positions and a bounded timeline', async ({ page }) => {
  await mockMatch(page); await page.clock.install(); await start(page); await page.locator('#match-speed').selectOption('fast'); await page.locator('#match-autoplay').click();
  for (let i = 1; i <= 42; i++) { await page.clock.runFor(300); await expect(page.locator('.board-revision')).toHaveText(`Move ${i}`); await expect(page.locator('#match-start')).toBeEnabled(); }
  await expect(page.getByRole('status')).toHaveText('Draw after 42 moves.'); await expect(page.locator('.move-list li')).toHaveCount(43);
  expect(await page.locator('.move-list').evaluate(n => n.clientHeight)).toBeLessThanOrEqual(340);
  await page.locator('.move-list button').first().click(); await expect(page.locator('.circle.x, .circle.o')).toHaveCount(0);
  await page.locator('#match-live').click(); await expect(page.locator('.circle.x, .circle.o')).toHaveCount(42);
});

test('representative Match Lab screenshots: desktop setup, autoplay, replay and mobile', async ({ page }) => {
  const dir = resolve('playwright-report/phase5a'); await mkdir(dir, { recursive: true });
  const capture = async name => { await page.evaluate(() => { document.activeElement?.blur(); scrollTo(0, 0); }); await page.mouse.move(0, 0); await page.screenshot({ path: `${dir}/${name}`, fullPage: true }); };
  await mockMatch(page, [3, 2, 4, 3, 2, 4, 3, 4, 5, 1]);
  await page.setViewportSize({ width: 1440, height: 1120 }); await configure(page, ['negamax', 'mcts']);
  await page.locator('#player-1-depth').selectOption('6'); await page.locator('#player-2-simulations').selectOption('400');
  await capture('01-desktop-configured.png');
  await page.locator('#match-start').click(); await expect(page.locator('#match-start')).toBeEnabled();
  for (let i = 1; i <= 8; i++) await step(page, i);
  await page.locator('#match-speed').selectOption('slow'); await page.locator('#match-autoplay').click();
  await capture('02-desktop-autoplay.png');
  await page.locator('#match-previous').click(); await expect(page.locator('#match-autoplay')).toHaveAttribute('aria-pressed', 'false');
  await page.locator('.move-list button').nth(4).click(); await expect(page.locator('.board-revision')).toHaveText('Move 4');
  await capture('03-desktop-replay.png');
  await page.setViewportSize({ width: 375, height: 812 }); await page.locator('#match-live').click();
  await page.evaluate(() => scrollTo(0, 0)); expect(await fits(page)).toBe(true);
  await capture('04-mobile-active.png');
  await page.locator('#match-previous').click(); await page.locator('.move-list button').nth(4).click();
  await page.evaluate(() => scrollTo(0, 0)); await capture('05-mobile-replay.png');
});
