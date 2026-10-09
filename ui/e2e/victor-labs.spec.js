import { test, expect } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { mockTournament, KEY as TOURNAMENT_KEY } from './fixtures/tournament.js';
import { mockSeason, accelerate, finish, KEY as SEASON_KEY } from './fixtures/season.js';
import { WIN, matchFixture } from './fixtures/match.js';
import * as season from '../src/season/model.js';
import * as tournament from '../src/tournament/model.js';
import { verifyEvaluationExport } from '../src/evaluation/verify.js';

const enabled = (process.env.VICTOR_LABS_UI ?? process.env.VICTOR_RELEASE_UI) === 'true';
const victor = { type: 'victor_research' };
const label = 'Victor Research (Experimental)';
async function selectMatch(page, types) {
  await page.goto('/connect4/match-lab');
  for (const [i, type] of types.entries()) await page.locator(`input[name="player-${i + 1}"][value="${type}"]`).check();
  await page.locator('#match-start').click(); await expect(page.locator('.board-revision')).toHaveText('Move 0');
}
const step = async (page, n) => { await page.locator('#match-next').click(); await expect(page.locator('.board-revision')).toHaveText(`Move ${n}`); await expect(page.locator('#game-board')).toHaveAttribute('aria-busy', 'false'); };

for (const width of [1440, 820, 375, 320]) test(`Victor homepage and lab layouts fit at ${width}px`, async ({ page }, info) => {
  await page.setViewportSize({ width, height: 1000 });
  for (const path of ['/', '/connect4/match-lab', '/connect4/tournament', '/connect4/season']) {
    await page.goto(path); expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    if (path === '/') await page.screenshot({ path: info.outputPath(`home-${width}.png`), fullPage: true });
  }
});

test('availability follows flag on homepage and all competition selectors', async ({ page }) => {
  await page.goto('/'); await expect(page.locator('.lab-strip')).toContainText(`${enabled ? 4 : 3} playable agents`);
  await expect(page.locator('#agents').getByRole('heading', { name: label, exact: true })).toHaveCount(enabled ? 1 : 0);
  await expect(page.locator('#research')).toContainText('1,722 selected positions');
  for (const path of ['match-lab', 'tournament', 'season']) {
    await page.goto(`/connect4/${path}`);
    const count = await page.locator('input[value="victor_research"], option').filter({ hasText: label }).count();
    if (path === 'match-lab') await expect(page.locator('input[value="victor_research"]')).toHaveCount(enabled ? 2 : 0);
    else expect(count).toBe(enabled ? 8 : 0);
  }
});

test('saved completed Victor seasons and tournaments retain replay/export with flag disabled', async ({ page }) => {
  let s = season.createSeason(Array(4).fill(victor), 1234, 2, undefined, true);
  while (season.currentFixture(s)) { const plan = season.gamePlan(s); s = season.recordGame(s, season.compactHistory(s, { ...matchFixture(plan.playerConfigs, WIN), rng_seed: plan.gameSeed })); }
  let t = tournament.createTournament(Array(8).fill(victor), 1234, undefined, true);
  while (tournament.nextMatchup(t)) { const m = tournament.nextMatchup(t), plan = tournament.gamePlan(t, m); t = tournament.recordGame(t, m.matchupId, tournament.compactHistory(t, m, { ...matchFixture(plan.playerConfigs, WIN), rng_seed: plan.gameSeed })); }
  await page.addInitScript(({ s, t, sk, tk }) => { localStorage.setItem(sk, JSON.stringify(s)); localStorage.setItem(tk, JSON.stringify(t)); }, { s, t, sk: SEASON_KEY, tk: TOURNAMENT_KEY });
  let posts = 0; page.on('request', r => { if (r.method() === 'POST') posts++; });
  await page.goto('/connect4/season'); await expect(page.locator('.season-status')).toHaveText('Season Complete'); await expect(page.locator('.evaluation-feedback')).toContainText('verified locally');
  await expect(page.getByRole('button', { name: 'Export evaluation JSON' })).toBeEnabled();
  const downloadEvent = page.waitForEvent('download'); await page.getByRole('button', { name: 'Export evaluation JSON' }).click();
  const download = await downloadEvent; const artifact = JSON.parse(await readFile(await download.path(), 'utf8'));
  expect(artifact.schema_version).toBe(2); expect((await verifyEvaluationExport(artifact, { digest: text => createHash('sha256').update(text).digest('hex') })).ok).toBe(true);
  await page.reload(); await expect(page.locator('.season-status')).toHaveText('Season Complete');
  await page.goto('/connect4/tournament'); await expect(page.getByRole('region', { name: 'Tournament Champion' })).toContainText(label);
  await page.locator('#match-previous').click(); await expect(page.locator('.board-revision')).toHaveText('Move 6'); expect(posts).toBe(0);
});

test('disabled flag retains pending field but never starts Victor', async ({ page }) => {
  test.skip(enabled, 'Disabled UI build only');
  const s = season.createSeason(Array(4).fill(victor), 1234, 2, undefined, true);
  await page.addInitScript(({ key, value }) => localStorage.setItem(key, JSON.stringify(value)), { key: SEASON_KEY, value: s });
  let posts = 0; page.on('request', r => { if (r.method() === 'POST') posts++; });
  await page.goto('/connect4/season'); await expect(page.locator('main')).toContainText('execution is disabled'); await expect(page.locator('#season-run-season')).toBeDisabled();
  await page.reload(); await expect(page.locator('.tournament-summary')).toContainText('0 of 12'); expect(posts).toBe(0);
});

test.describe('enabled Victor competition', () => {
  test.skip(!enabled, 'Enabled UI build only');
  for (const types of [['victor_research', 'random'], ['negamax', 'victor_research'], ['victor_research', 'mcts'], ['victor_research', 'victor_research'], ['human', 'victor_research'], ['victor_research', 'human']]) {
    test(`Match Lab selectable colors and legal moves: ${types}`, async ({ page }) => {
      const server = await mockTournament(page); await selectMatch(page, types);
      expect([server.starts[0].player1, server.starts[0].player2].filter(p => p.type === victor.type)).toEqual(Array(types.filter(t => t === victor.type).length).fill(victor));
      for (let n = 0; n < 2; n++) {
        if (types[n] === 'human') { await page.locator(`.cell[data-column="${WIN[n]}"]:enabled`).first().click(); await expect(page.locator('.board-revision')).toHaveText(`Move ${n + 1}`); }
        else await step(page, n + 1);
      }
      expect(server.moves.map(p => p.revision)).toEqual([0, 1]);
      await page.locator('#match-previous').click(); await expect(page.locator('.board-revision')).toHaveText('Move 1'); expect(server.moves).toHaveLength(2);
    });
  }
  for (const kind of ['tournament', 'season']) test(`${kind} Victor advancement and reload preserve sequential evidence`, async ({ page }) => {
    const server = kind === 'season' ? await mockSeason(page) : await mockTournament(page);
    await page.goto(`/connect4/${kind}`);
    if (kind === 'season') await page.locator('#season-size').selectOption('4');
    for (let i = 1; i <= (kind === 'season' ? 4 : 8); i++) await page.locator(`#${kind === 'season' ? 'season-' : ''}entrant-${i}`).selectOption({ label });
    if (kind === 'season') await expect(page.locator('.season-workload').filter({ hasText: 'Free-tier CPU' })).toContainText('Free-tier CPU');
    await page.locator(`#${kind}-seed`).fill('1234'); await page.locator(`#${kind}-create`).click(); await accelerate(page);
    await page.locator(kind === 'season' ? '#season-run-game' : '#run-matchup').click();
    if (kind === 'season') await finish(page); else for (let i = 0; i < 9; i++) { await page.clock.runFor(851); await page.waitForFunction(() => document.querySelector('[aria-label="Tournament execution"]')?.getAttribute('aria-busy') === 'false'); }
    await expect(page.locator('.tournament-summary')).toContainText(kind === 'season' ? '1 of 12' : '1 of 7'); expect(server.moves).toHaveLength(7); expect(server.maxFlight).toBe(1);
    await page.reload(); await expect(page.locator('.tournament-summary')).toContainText(kind === 'season' ? '1 of 12' : '1 of 7'); expect(server.starts).toHaveLength(1);
  });
  test('Human can complete a tournament game against Victor', async ({ page }) => {
    const server = await mockTournament(page); await page.goto('/connect4/tournament');
    for (let i = 1; i <= 8; i++) await page.locator(`#entrant-${i}`).selectOption({ label });
    // Put the Human in the first seeded matchup for this known bracket.
    const t = tournament.createTournament(Array(8).fill(victor), 1234, undefined, true), slot = Number(t.bracketOrder[0].split('-')[1]);
    await page.locator(`#entrant-${slot}`).selectOption({ label: 'You · Human' }); await page.locator('#tournament-seed').fill('1234'); await page.locator('#tournament-create').click(); await page.locator('#play-your-match').click();
    for (let n = 0; n < 7; n++) {
      if (server.active.players[n % 2].type === 'human') await page.getByRole('button', { name: `Drop in column ${WIN[n] + 1}`, exact: true }).click();
      else await page.locator('#match-next').click();
      if (n < 6) { await expect(page.locator('.board-revision')).toHaveText(`Move ${n + 1}`); await expect(page.locator('[aria-label="Tournament execution"]')).toHaveAttribute('aria-busy', 'false'); }
    }
    await expect(page.locator('.tournament-summary')).toContainText('1 of 7'); expect(server.moves).toHaveLength(7);
  });
  for (const mode of ['busy', 'failed', 'disabled', 'lost']) test(`Match Victor ${mode} reconciles without a repeated POST`, async ({ page }) => {
    const server = await mockTournament(page); await selectMatch(page, ['victor_research', 'victor_research']); let attempts = 0;
    await page.route('**/v1/connect4/make_move', async route => {
      attempts++; if (mode === 'lost') { server.moves.push(route.request().postDataJSON()); server.active.columns.push(WIN[0]); await route.abort('failed'); }
      else await route.fulfill({ status: mode === 'disabled' ? 409 : 503, json: { code: mode === 'busy' ? 'agent_busy' : mode === 'failed' ? 'agent_failed' : 'invalid_agent', error: mode } });
    });
    await page.locator('#match-next').click(); await expect(page.locator('#match-start')).toBeEnabled(); await expect(page.locator('#match-message')).toContainText(mode === 'busy' ? 'busy' : mode === 'failed' ? 'could not make a legal move' : mode === 'disabled' ? 'not enabled' : 'could not be confirmed');
    expect(attempts).toBe(1); await expect(page.locator('.board-revision')).toHaveText(`Move ${mode === 'lost' ? 1 : 0}`);
    await page.unroute('**/v1/connect4/make_move'); await page.locator('#match-refresh').click(); await step(page, mode === 'lost' ? 2 : 1); expect(server.moves.map(p => p.revision)).toEqual(mode === 'lost' ? [0, 1] : [0]);
  });
  for (const kind of ['tournament', 'season']) for (const mode of ['busy', 'failed', 'disabled', 'lost', 'lost-read']) test(`${kind} Victor ${mode} preserves progress and recovers explicitly`, async ({ page }) => {
    const server = kind === 'season' ? await mockSeason(page) : await mockTournament(page);
    await page.goto(`/connect4/${kind}`);
    for (let i = 1; i <= 8; i++) await page.locator(`#${kind === 'season' ? 'season-' : ''}entrant-${i}`).selectOption({ label });
    await page.locator(`#${kind}-create`).click(); await page.locator(kind === 'season' ? '#season-watch' : '#tournament-watch').click();
    await expect(page.locator('.board-revision')).toHaveText('Move 0'); let attempts = 0;
    await page.route('**/v1/connect4/make_move', async route => {
      attempts++;
      if (mode.startsWith('lost')) { server.moves.push(route.request().postDataJSON()); server.active.columns.push(WIN[0]); await route.abort('failed'); }
      else await route.fulfill({ status: mode === 'disabled' ? 409 : 503, json: { code: mode === 'busy' ? 'agent_busy' : mode === 'failed' ? 'agent_failed' : 'invalid_agent', error: mode } });
    });
    if (mode === 'lost-read') await page.route('**/history', route => route.abort('failed'));
    await page.locator('#match-next').click(); await expect(page.locator('[aria-label="' + (kind === 'season' ? 'Season' : 'Tournament') + ' execution"]')).toHaveAttribute('aria-busy', 'false');
    await expect(page.getByRole('alert')).toContainText(mode === 'busy' ? 'busy' : mode === 'failed' ? 'could not make a legal move' : mode === 'disabled' ? 'disabled on this game server' : 'lost-read' === mode ? 'Refresh history' : 'Network Error');
    expect(attempts).toBe(1); await expect(page.locator('#match-next')).toBeDisabled();
    await page.unroute('**/v1/connect4/make_move'); if (mode === 'lost-read') await page.unroute('**/history');
    await page.locator(`#${kind}-refresh`).click(); await expect(page.locator('#match-next')).toBeEnabled(); await step(page, mode.startsWith('lost') ? 2 : 1);
    expect(server.moves.map(p => p.revision)).toEqual(mode.startsWith('lost') ? [0, 1] : [0]);
    await page.reload(); await expect(page.locator('.board-revision')).toHaveText(`Move ${mode.startsWith('lost') ? 2 : 1}`); expect(server.starts).toHaveLength(1);
  });
  test('Pause during a Victor autoplay request settles one move', async ({ page }) => {
    const server = await mockTournament(page); await selectMatch(page, ['victor_research', 'victor_research']); let release;
    const gate = new Promise(r => { release = r; });
    await page.route('**/v1/connect4/make_move', async route => { server.moves.push(route.request().postDataJSON()); server.active.columns.push(0); await gate; await route.fulfill({ json: server.fixture().state }); });
    await page.locator('#match-speed').selectOption('fast'); await page.locator('#match-autoplay').click(); await expect(page.locator('#match-message')).toContainText('AI thinking'); await page.locator('#match-autoplay').click(); release();
    await expect(page.locator('.board-revision')).toHaveText('Move 1'); await expect(page.locator('#match-start')).toBeEnabled(); await page.waitForTimeout(700); expect(server.moves).toHaveLength(1);
  });
  for (const kind of ['tournament', 'season']) test(`real local API Victor ${kind} smoke`, async ({ page }) => {
    await page.goto(`/connect4/${kind}`);
    for (let i = 1; i <= 8; i++) await page.locator(`#${kind === 'season' ? 'season-' : ''}entrant-${i}`).selectOption({ label });
    await page.locator(`#${kind}-create`).click(); await page.locator(kind === 'season' ? '#season-watch' : '#tournament-watch').click();
    await expect(page.locator('.board-revision')).toHaveText('Move 0'); await step(page, 1); await step(page, 2);
  });
  test('real local API Victor/Victor match smoke', async ({ page }) => {
    await selectMatch(page, ['victor_research', 'victor_research']); await step(page, 1); await step(page, 2); await expect(page.locator('.move-list')).toContainText(label);
  });
});
