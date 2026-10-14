import { test, expect } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { verifyEvaluationExport } from '../src/evaluation/verify.js';
const api = process.env.PLAYWRIGHT_API_URL;
const isolated = api && new URL(api).port !== '8000' && ['127.0.0.1', 'localhost'].includes(new URL(api).hostname);
test.skip(!isolated, 'AlphaZero tests require an explicitly configured isolated localhost staging API, never port 8000.');
const option = page => page.locator('input[value="alphazero"]');
const turn = page => page.locator('#message');
const column = page => page.locator('.cell[data-column="3"]:enabled').first();
async function play(page, guard = 'off', first = 'human') {
  await page.goto('/connect4'); await expect(option(page)).toBeVisible(); await option(page).check();
  await page.locator('#opponent-guard').selectOption(guard); await page.locator(`#first-${first}`).check();
  const response = page.waitForResponse(r => r.url().endsWith('/start_game'));
  await page.locator('#start-button').click(); const game = await (await response).json();
  await expect(page.locator('#loading')).toBeHidden(); return game;
}
async function setupMatch(page, opponent = 'alphazero') {
  await page.goto('/connect4/match-lab'); await expect(page.locator('input[name="player-1"][value="alphazero"]')).toBeVisible();
  await page.locator('input[name="player-1"][value="alphazero"]').check();
  await page.locator(`input[name="player-2"][value="${opponent}"]`).check();
  if (opponent === 'negamax') await page.locator('#player-2-depth').selectOption('1');
  if (['mcts', 'mcts_tactical'].includes(opponent)) await page.locator('#player-2-simulations').selectOption(opponent === 'mcts' ? '500' : '200');
  if (opponent === 'alphazero') await page.locator('#player-2-guard').selectOption('on');
  await page.locator('#match-start').click(); await expect(page.locator('.board-revision')).toHaveText('Move 0');
}
for (const kind of ['404', 'disabled', 'unavailable', 'network', 'malformed']) test(`quiet discovery ${kind} keeps existing agents usable`, async ({ page }) => {
  await page.route('**/agents/alphazero', route => kind === 'network' ? route.abort('failed') : route.fulfill({ status: kind === '404' ? 404 : 200,
    json: kind === 'malformed' ? { available: true } : { type: 'alphazero', enabled: kind === 'unavailable', available: false } }));
  await page.goto('/connect4'); await expect(option(page)).toHaveCount(0); await page.locator('input[value="random"]').check();
  await page.locator('#start-button').click(); await column(page).click(); await expect(page.locator('#loading')).toBeHidden();
  await expect(page.locator('.circle.x')).toHaveCount(1); await expect(page.locator('.circle.o')).toHaveCount(1);
});
for (const guard of ['off', 'on']) for (const first of ['human', 'ai']) test(`real Play safeguard ${guard}, ${first} first; fixed config and restart`, async ({ page, request }) => {
  const game = await play(page, guard, first); const index = first === 'ai' ? 0 : 1;
  expect(game.players[index]).toEqual({ type: 'alphazero', tactical_guard: guard === 'on' });
  await expect(turn(page)).toContainText('Your turn'); await column(page).click(); await expect(page.locator('#loading')).toBeHidden();
  const state = await (await request.get(`${api}/v1/connect4/games/${game.game_id}`)).json();
  expect(state.revision).toBe(first === 'ai' ? 3 : 2);
  await page.locator('#opponent-guard').selectOption(guard === 'on' ? 'off' : 'on');
  expect((await (await request.get(`${api}/v1/connect4/games/${game.game_id}`)).json()).players).toEqual(game.players);
  await expect(page.locator('.settings-note')).toContainText('Settings apply when you start a new game');
  await page.locator('#restart-button').click(); await expect(page.locator('#loading')).toBeHidden(); await expect(turn(page)).toContainText('Your turn');
});
for (const opponent of ['human', 'negamax', 'mcts', 'mcts_tactical', 'victor_research', 'alphazero']) test(`real Match AlphaZero vs ${opponent}, step/timeline/replay`, async ({ page }) => {
  await setupMatch(page, opponent); await page.locator('#match-next').click(); await expect(page.locator('.board-revision')).toHaveText('Move 1');
  if (opponent === 'human') await column(page).click(); else await page.locator('#match-next').click();
  await expect(page.locator('.board-revision')).toHaveText('Move 2'); await page.locator('#match-previous').click();
  await expect(page.locator('.board-revision')).toHaveText('Move 1'); await page.locator('#match-forward').click();
  await expect(page.locator('#match-view-mode')).toContainText('LIVE');
});
test('Match autoplay busy bounded retries and pause cancels old retry', async ({ page, request }) => {
  await setupMatch(page); let posts = 0, release;
  const pendingBusy = new Promise(resolve => { release = resolve; });
  await page.route('**/make_move', async route => { posts++; await pendingBusy; return route.fulfill({ status: 503, headers: { 'Retry-After': '1' }, json: { code: 'agent_busy', error: 'Busy' } }); });
  await page.locator('#match-autoplay').click(); await expect.poll(() => posts).toBe(1); await page.locator('#match-autoplay').click(); release();
  await page.waitForTimeout(1300); expect(posts).toBe(1); await expect(page.locator('.board-revision')).toHaveText('Move 0');
  await page.unroute('**/make_move'); let busyOnce = true;
  await page.route('**/make_move', route => { if (busyOnce) { busyOnce = false; return route.fulfill({ status: 503, json: { code: 'agent_busy', error: 'Busy' } }); } return route.continue(); });
  await page.locator('#match-next').click(); await expect(page.locator('.board-revision')).toHaveText('Move 1');
  await page.unroute('**/make_move'); await page.locator('#match-speed').selectOption('fast'); await page.locator('#match-autoplay').click();
  await expect(page.locator('.board-revision')).toHaveText('Move 3'); await page.locator('#match-autoplay').click();
  await expect(page.locator('#match-next')).toBeEnabled(); const revision = await page.locator('.board-revision').textContent(); await page.waitForTimeout(700);
  expect(await page.locator('.board-revision').textContent()).toBe(revision);
});
test('uncertain network failure reconciles committed AI move without another POST', async ({ page }) => {
  let aiPosts = 0;
  await page.route('**/make_move', async route => {
    if (!('column' in route.request().postDataJSON())) { aiPosts++; await route.fetch(); return route.abort('failed'); }
    return route.continue();
  });
  await play(page, 'off', 'ai'); await expect(page.locator('.circle.x')).toHaveCount(1); expect(aiPosts).toBe(1);
  await expect(page.locator('#message')).toContainText(/Network Error|could not/);
});
test('unavailable active game stops autoplay and retains history; refresh discovery changes options', async ({ page }) => {
  await setupMatch(page); await page.route('**/make_move', route => route.fulfill({ status: 503, json: { code: 'agent_unavailable', error: 'Unavailable' } }));
  await page.locator('#match-autoplay').click(); await expect(page.locator('#match-message')).toContainText('unavailable');
  await expect(page.locator('.board-revision')).toHaveText('Move 0'); await expect(page.locator('#match-autoplay')).toHaveAttribute('aria-pressed', 'false');
  await expect(option(page)).toHaveCount(0);
  await page.getByRole('button', { name: 'Refresh agent availability' }).click(); await expect(option(page)).toHaveCount(2);
});
for (const lab of ['tournament', 'season']) test(`real ${lab} guarded/unguarded field, autoplay, compact persistence and export`, async ({ page }) => {
  await page.goto(`/connect4/${lab}`); if (lab === 'season') await page.locator('#season-size').selectOption('4');
  const entrantPrefix = lab === 'season' ? 'season-entrant' : 'entrant';
  const count = lab === 'season' ? 4 : 8;
  await expect(page.locator(`#${entrantPrefix}-1 option`).filter({ hasText: 'AlphaZero' })).toHaveCount(2);
  for (let i = 1; i <= count; i++) await page.locator(`#${entrantPrefix}-${i}`).selectOption({ label: `AlphaZero (Experimental) · safeguard ${i % 2 ? 'off' : 'on'}` });
  await page.locator(`#${lab}-create`).click(); await expect(page.locator('.tournament-summary')).toBeVisible();
  await page.locator('#'+(lab === 'season' ? 'season-watch' : 'tournament-watch')).click(); await expect(page.locator('#match-next')).toBeEnabled();
  await page.locator('#match-speed').selectOption('fast'); await page.locator('#match-autoplay').click();
  await expect.poll(() => page.evaluate(lab => { const s = JSON.parse(localStorage.getItem(`board-game-ai-lab:${lab}:v1`)); return lab === 'season' ? s.completedGames.length : s.rounds.flat().reduce((n, m) => n + m.games.length, 0); }, lab), { timeout: 30000 }).toBe(1);
  const saved = await page.evaluate(lab => JSON.parse(localStorage.getItem(`board-game-ai-lab:${lab}:v1`)), lab);
  expect(saved.version).toBe(2); expect(saved.alphazeroProvenance.model.id).toBe('az2-hc-g6-cnn-s42-v1');
  expect(saved.entrants.map(e => e.config.tactical_guard)).toContain(true); expect(saved.entrants.map(e => e.config.tactical_guard)).toContain(false);
  await page.reload(); await expect(page.locator('.tournament-summary')).toBeVisible();
  if (lab === 'season') {
    await expect(page.getByRole('button', { name: 'Export evaluation JSON' })).toBeEnabled();
    const downloading = page.waitForEvent('download'); await page.getByRole('button', { name: 'Export evaluation JSON' }).click();
    const download = await downloading, artifact = JSON.parse(await readFile(await download.path(), 'utf8'));
    expect(artifact.schema_version).toBe(3); expect((await verifyEvaluationExport(artifact, { digest: async text => createHash('sha256').update(text).digest('hex') })).ok).toBe(true);
  }
  if (lab === 'season') {
    await page.route('**/agents/alphazero', route => route.fulfill({ status: 404, json: {} }));
    await page.reload();
    await expect(page.locator('#season-run-season')).toBeDisabled();
    await expect(page.locator('main')).toContainText('History remains reviewable');
    await expect(page.getByRole('button', { name: 'Export evaluation JSON' })).toBeEnabled();
    expect(await page.evaluate(() => JSON.parse(localStorage.getItem('board-game-ai-lab:season:v1')).entrants.every(e => e.config.type === 'alphazero'))).toBe(true);
    await page.locator('#season-round').selectOption('1'); await page.locator('.season-fixture').filter({ hasText: 'Open local replay' }).first().click(); }
  else { await page.locator('.desktop-bracket [data-matchup="r1-m1"]').click(); await page.getByRole('button', { name: 'Game 1', exact: false }).first().click(); }
  await page.locator('#match-previous').click(); await expect(page.locator('.cell:enabled')).toHaveCount(0);
});
for (const width of [375, 320]) test(`AlphaZero mobile and keyboard controls at ${width}px`, async ({ page }, info) => {
  await page.setViewportSize({ width, height: 1000 }); await page.emulateMedia({ reducedMotion: 'reduce' }); await page.goto('/connect4');
  await expect(option(page)).toBeVisible(); await option(page).focus(); await page.keyboard.press('Space'); await expect(option(page)).toBeChecked();
  await page.locator('#opponent-guard').focus(); await page.keyboard.press('o'); await expect(page.locator('#opponent-guard')).toHaveValue('on');
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.screenshot({ path: info.outputPath(`alphazero-${width}.png`), fullPage: true });
});

test('Human first in Match Lab versus guarded AlphaZero', async ({ page }) => {
  await page.goto('/connect4/match-lab'); await expect(page.locator('input[name="player-2"][value="alphazero"]')).toBeVisible();
  await page.locator('input[name="player-1"][value="human"]').check();
  await page.locator('input[name="player-2"][value="alphazero"]').check(); await page.locator('#player-2-guard').selectOption('on');
  await page.locator('#match-start').click(); await expect(page.locator('.board-revision')).toHaveText('Move 0');
  await column(page).click(); await expect(page.locator('.board-revision')).toHaveText('Move 1');
  await page.locator('#match-next').click(); await expect(page.locator('.board-revision')).toHaveText('Move 2');
  await expect(page.locator('.move-list')).toContainText('safeguard on');
});

test('AlphaZero game uses existing position and last-AI explanation UI without provider calls', async ({ page }) => {
  const { explained } = await import('./fixtures/analysis.js'); let calls = 0;
  await page.route('**/explain', route => { calls++; return route.fulfill({ json: explained(route.request().postDataJSON()) }); });
  await play(page, 'on', 'ai');
  await page.getByRole('button', { name: 'Analyze Position', exact: true }).click();
  await expect(page.locator('#explanation-result')).toContainText('Verified tactical evidence');
  await page.getByRole('button', { name: 'Analyze Last AI Move', exact: true }).click();
  await expect(page.locator('#explanation-panel')).toHaveAttribute('aria-busy', 'false'); expect(calls).toBe(2);
});
