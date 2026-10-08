import { createSeason, defaultField, gamePlan, compactHistory, recordGame } from '../../src/season/model.js';
import { matchFixture, WIN, DRAW } from './match.js';
export { WIN, DRAW };
export const KEY = 'board-game-ai-lab:season:v1';
export function seasonFixture(size = 8, games = 4, count = size * (size - 1) / 2 * games, columns) {
  let s = createSeason(defaultField(size), 1234, games);
  for (let i = 0; i < count; i++) {
    const plan = gamePlan(s), sequence = columns ?? (i % 5 === 0 ? DRAW : i % 3 === 0 ? [0, 1, 0, 1, 2, 1, 2, 1] : WIN);
    s = recordGame(s, compactHistory(s, { ...matchFixture(plan.playerConfigs, sequence), rng_seed: plan.gameSeed }));
  }
  return s;
}
export async function seedSeason(page, season) {
  await page.addInitScript(({ key, season }) => localStorage.setItem(key, JSON.stringify(season)), { key: KEY, season });
}
export async function create(page, size = 8, games = 2) {
  await page.goto('/connect4/season'); await page.locator('#season-size').selectOption(String(size)); await page.locator('#season-games').selectOption(String(games));
  await page.locator('#season-seed').fill('1234'); await page.locator('#season-create').click();
}
export async function mockSeason(page, sequence = WIN) {
  const s = { starts: [], moves: [], active: null, expire: false, maxFlight: 0, inFlight: 0 };
  s.fixture = () => ({ ...matchFixture(s.active.players, s.active.columns, s.active.id), rng_seed: s.active.seed });
  await page.route('**/v1/connect4/**', async route => {
    const url = route.request().url(), mutation = route.request().method() === 'POST';
    if (mutation) s.maxFlight = Math.max(s.maxFlight, ++s.inFlight);
    try {
      if (url.endsWith('/start_game')) {
        const body = route.request().postDataJSON();
        if (s.active && body.replace_game_id !== s.active.id) throw new Error('Missing sequential session replacement');
        s.starts.push(body); s.active = { id: `season-game-${s.starts.length}`, players: [body.player1, body.player2], seed: body.rng_seed, columns: [] };
        await route.fulfill({ status: 201, json: s.fixture().state });
      } else if (url.endsWith('/make_move')) {
        const body = route.request().postDataJSON();
        if (body.revision !== s.active.columns.length || 'column' in body || body.game_id !== s.active.id) throw new Error('Invalid or duplicate AI mutation');
        s.moves.push(body); s.active.columns.push(sequence[s.active.columns.length]); await route.fulfill({ json: s.fixture().state });
      } else if (s.expire) await route.fulfill({ status: 404, json: { error: 'Session expired', code: 'session_not_found' } });
      else await route.fulfill({ json: s.fixture() });
    } finally { if (mutation) s.inFlight--; }
  });
  return s;
}
export async function accelerate(page) { await page.clock.install({ time: new Date('2026-10-07T12:00:00Z') }); await page.clock.pauseAt(new Date('2026-10-07T12:00:01Z')); }
export async function finish(page, limit = 200) {
  for (let i = 0; i < limit; i++) {
    const status = await page.locator('.season-status').textContent(); if (['Paused', 'Season Complete'].includes(status)) return;
    await page.clock.runFor(851); await page.waitForFunction(() => document.querySelector('[aria-label="Season execution"]')?.getAttribute('aria-busy') === 'false');
  }
  throw new Error('Season did not reach its execution boundary');
}
