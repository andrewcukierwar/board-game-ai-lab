import { test, expect } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { verifyEvaluationExport } from '../src/evaluation/verify.js';
const api = process.env.PLAYWRIGHT_API_URL;
const victorEnabled = (process.env.VICTOR_LABS_UI ?? process.env.VICTOR_RELEASE_UI) === 'true';
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
async function setupMatch(page, opponent = 'alphazero', navigate = true) {
  if (navigate) await page.goto('/connect4/match-lab');
  await expect(page.locator('input[name="player-1"][value="alphazero"]')).toBeVisible();
  await page.locator('input[name="player-1"][value="alphazero"]').check();
  await page.locator(`input[name="player-2"][value="${opponent}"]`).check();
  if (opponent === 'negamax') await page.locator('#player-2-depth').selectOption('1');
  if (['mcts', 'mcts_tactical'].includes(opponent)) await page.locator('#player-2-simulations').selectOption(opponent === 'mcts' ? '500' : '200');
  if (opponent === 'alphazero') await page.locator('#player-2-guard').selectOption('on');
  const started = page.waitForResponse(r => r.url().endsWith('/start_game'));
  await page.locator('#match-start').click(); const game = await (await started).json();
  await expect(page.locator('.board-revision')).toHaveText('Move 0');
  await expect(page.locator('#match-next')).toBeEnabled();
  return game;
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
  test.skip(opponent === 'victor_research' && !victorEnabled, 'Victor matchup requires the enabled frontend feature flag.');
  // Check both selectors against the declared mode, so a misconfigured enabled
  // build fails instead of silently dropping the Victor integration.
  await page.goto('/connect4/match-lab');
  await expect(page.locator('input[value="victor_research"]')).toHaveCount(victorEnabled ? 2 : 0);
  if (!victorEnabled && opponent === 'human') {
    for (const type of ['human', 'random', 'negamax', 'mcts', 'mcts_tactical', 'alphazero']) {
      for (const player of [1, 2]) {
        const input = page.locator(`input[name="player-${player}"][value="${type}"]`);
        await expect(input).toBeVisible(); await input.check(); await expect(input).toBeChecked();
      }
    }
  }
  await setupMatch(page, opponent); await page.locator('#match-next').click(); await expect(page.locator('.board-revision')).toHaveText('Move 1');
  if (opponent === 'human') await column(page).click(); else await page.locator('#match-next').click();
  await expect(page.locator('.board-revision')).toHaveText('Move 2'); await page.locator('#match-previous').click();
  await expect(page.locator('.board-revision')).toHaveText('Move 1'); await page.locator('#match-forward').click();
  await expect(page.locator('#match-view-mode')).toContainText('LIVE');
});
const deferred = () => { let resolve; const promise = new Promise(r => { resolve = r; }); return { promise, resolve }; };
const busyResponse = { status: 503, headers: { 'Retry-After': '1' }, json: { code: 'agent_busy', error: 'Busy' } };

// Every POST stops before reaching the server. A test explicitly authorizes
// either a busy response or one real move, optionally holding its response.
async function controlledMoves(page) {
  const calls = []; let inFlight = 0, maxFlight = 0, closing = false;
  const handler = async route => {
    if (closing) return await route.abort();
    const call = { body: route.request().postDataJSON(), action: deferred(), committed: deferred(), response: deferred() };
    calls.push(call); maxFlight = Math.max(maxFlight, ++inFlight);
    try {
      const action = await call.action.promise;
      if (closing) return await route.abort();
      if (action === 'busy') return await route.fulfill(busyResponse);
      const response = await route.fetch({ maxRetries: 0, timeout: 10000 });
      call.committed.resolve({ state: await response.json() });
      await call.response.promise;
      if (closing) return await route.abort();
      await route.fulfill({ response });
    } catch (error) {
      call.committed.resolve({ error });
      if (!closing) throw error;
    } finally { inFlight--; }
  };
  await page.route('**/make_move', handler);
  return {
    calls,
    async at(count) { await expect.poll(() => calls.length).toBe(count); return calls[count - 1]; },
    async accepted(call) {
      const result = await call.committed.promise;
      if (result.error) throw result.error;
      return result.state;
    },
    async settled() { await expect.poll(() => inFlight).toBe(0); expect(maxFlight).toBe(1); },
    async close() {
      closing = true;
      for (const call of calls) { call.action.resolve(); call.response.resolve(); }
      await page.unrouteAll({ behavior: 'wait' });
    },
  };
}
async function frozenMatch(page) {
  const time = new Date('2026-10-15T12:00:00Z');
  await page.clock.install({ time });
  const game = await setupMatch(page);
  await page.clock.pauseAt(new Date(time.getTime() + 60000));
  // Observe retry timer registration before advancing virtual time or pausing.
  // Delegates unchanged to Playwright's clock; no production delay is added.
  await page.evaluate(() => {
    const schedule = window.setTimeout;
    window.busyWaits = 0;
    window.setTimeout = (fn, delay, ...args) => {
      if (delay === 1000) window.busyWaits++;
      return schedule(fn, delay, ...args);
    };
  });
  await page.locator('#match-speed').selectOption('fast');
  return game;
}
for (const timing of ['pause before busy response', 'pause during retry wait', 'split retry clock']) {
  test(`Match autoplay busy bounded retries and pause cancels old retry: ${timing}`, async ({ page, request }) => {
    const game = await frozenMatch(page), moves = await controlledMoves(page);
    const revision = page.locator('.board-revision'), autoplay = page.locator('#match-autoplay');
    const server = async () => {
      const response = await request.get(`${api}/v1/connect4/games/${game.game_id}/history`);
      expect(response.ok()).toBe(true); return response.json();
    };
    let waits = 0;
    const busy = async call => {
      call.action.resolve('busy');
      await expect.poll(() => page.evaluate(() => window.busyWaits)).toBe(++waits);
    };
    const advanceRetry = async () => {
      if (timing === 'split retry clock') { await page.clock.runFor(999); await page.clock.runFor(1); }
      else await page.clock.runFor(1000);
    };
    const paused = async (count, rev) => {
      await expect(autoplay).toHaveAttribute('aria-pressed', 'false');
      await expect(page.locator('#match-next')).toBeEnabled();
      await page.clock.runFor(10000); await moves.settled();
      expect(moves.calls).toHaveLength(count); await expect(revision).toHaveText(`Move ${rev}`);
      const history = await server(); expect(history.revision).toBe(rev); expect(history.moves).toHaveLength(rev);
      expect(history.moves.map(m => m.revision)).toEqual(Array.from({ length: rev }, (_, i) => i + 1));
    };
    try {
      // Exhaust the exact bound (initial POST + two retries), without a commit.
      await autoplay.click(); await page.clock.runFor(300);
      await busy(await moves.at(1)); await advanceRetry();
      await busy(await moves.at(2)); await advanceRetry();
      (await moves.at(3)).action.resolve('busy');
      await expect(page.locator('#match-message')).toContainText('busy');
      await expect(page.locator('#match-start')).toBeEnabled();
      await page.clock.runFor(10000); expect(moves.calls).toHaveLength(3);
      expect((await server()).revision).toBe(0);
      await page.locator('#match-refresh').click();
      await expect(page.locator('#match-next')).toBeEnabled();

      // Cancel a busy response/retry at a known boundary, then advance past
      // every retry and autoplay deadline. A cancelled retry must not POST.
      await autoplay.click(); await page.clock.runFor(300); const cancelled = await moves.at(4);
      if (timing === 'pause before busy response') { await autoplay.click(); await busy(cancelled); }
      else { await busy(cancelled); await autoplay.click(); }
      await advanceRetry(); await paused(4, 0);

      // Explicit continuation retries once and commits once: cancellation has
      // released the controller's reservation and has not poisoned future work.
      await page.locator('#match-next').click(); await busy(await moves.at(5)); await advanceRetry();
      const stepped = await moves.at(6); stepped.action.resolve('move'); stepped.response.resolve();
      await expect(revision).toHaveText('Move 1'); await paused(6, 1);

      // Allow autoplay to make progress, then hold the following accepted move.
      // Pause preserves that authorized in-flight commit, but starts no new ply.
      await autoplay.click(); await page.clock.runFor(300);
      const playing = await moves.at(7); playing.action.resolve('move'); playing.response.resolve();
      await expect(revision).toHaveText('Move 2');
      await expect(page.locator('#game-board')).toHaveAttribute('aria-busy', 'false');
      await page.clock.runFor(300); const held = await moves.at(8); held.action.resolve('move');
      expect((await moves.accepted(held)).revision).toBe(3);
      expect((await server()).revision).toBe(3);
      await autoplay.click(); await expect(revision).toHaveText('Move 2');
      await page.clock.runFor(2000); expect(moves.calls).toHaveLength(8);
      held.response.resolve(); await expect(revision).toHaveText('Move 3'); await paused(8, 3);
      expect(moves.calls.map(c => c.body.revision)).toEqual([0, 0, 0, 0, 0, 0, 1, 2]);
    } finally { await moves.close(); }
  });
}

test('Match ignores a stale authorized response after leaving the active match', async ({ page, request }) => {
  const old = await frozenMatch(page), moves = await controlledMoves(page);
  try {
    await page.locator('#match-autoplay').click(); await page.clock.runFor(300);
    const held = await moves.at(1); held.action.resolve('move');
    expect((await moves.accepted(held)).revision).toBe(1);
    expect((await (await request.get(`${api}/v1/connect4/games/${old.game_id}`)).json()).revision).toBe(1);
    // Keep the document alive so this exercises controller unmount/remount,
    // rather than letting a full navigation discard the stale response for us.
    await page.getByRole('link', { name: 'Back to Play', exact: true }).click();
    await page.getByRole('navigation', { name: 'Main navigation' }).getByRole('link', { name: 'Match Lab', exact: true }).click();
    const fresh = await setupMatch(page, 'alphazero', false);
    // The original route can finish after its page request was aborted on exit.
    held.response.resolve(); await moves.settled(); await page.clock.runFor(10000);
    await expect(page.locator('.board-revision')).toHaveText('Move 0');
    await expect(page.locator('#match-autoplay')).toHaveAttribute('aria-pressed', 'false');
    await expect(page.locator('#match-next')).toBeEnabled(); expect(moves.calls).toHaveLength(1);
    expect((await (await request.get(`${api}/v1/connect4/games/${fresh.game_id}`)).json()).revision).toBe(0);
    await page.locator('#match-next').click(); const next = await moves.at(2);
    expect(next.body.game_id).toBe(fresh.game_id); next.action.resolve('move'); next.response.resolve();
    await expect(page.locator('.board-revision')).toHaveText('Move 1'); await moves.settled();
  } finally { await moves.close(); }
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
