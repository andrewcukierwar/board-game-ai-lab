import { createTournament, defaultField, nextMatchup, gamePlan, compactHistory, recordGame } from '../../src/tournament/model.js';
import { matchFixture, WIN, DRAW } from './match.js';
export const KEY = 'board-game-ai-lab:tournament:v1';
export function tournamentFixture(size = 8, count = size - 1, columns = WIN) {
  let t = createTournament(defaultField(size), 1234);
  for (let i = 0; i < count; i++) {
    let m = nextMatchup(t);
    do {
      const plan = gamePlan(t, m);
      const history = { ...matchFixture(plan.playerConfigs, columns), rng_seed: plan.gameSeed };
      t = recordGame(t, m.matchupId, compactHistory(t, m, history)); m = t.rounds[m.round][m.index];
    } while (m.status !== 'complete');
  }
  return t;
}
export async function seedTournament(page, tournament) {
  await page.addInitScript(({ key, tournament }) => { localStorage.setItem(key, JSON.stringify(tournament)); }, { key: KEY, tournament });
}
export async function mockTournament(page, sequence = WIN) {
  const s = { starts: [], moves: [], active: null, expire: false, inFlight: 0, maxFlight: 0 };
  s.fixture = () => ({ ...matchFixture(s.active.players, s.active.columns, s.active.id), rng_seed: s.active.seed });
  await page.route('**/v1/connect4/**', async route => {
    const url = route.request().url();
    if (url.endsWith('/start_game')) {
      const body = route.request().postDataJSON(); s.starts.push(body);
      s.active = { id: `game-${s.starts.length}`, seed: body.rng_seed, players: [body.player1, body.player2], columns: [] };
      await route.fulfill({ status: 201, json: s.fixture().state });
    } else if (url.endsWith('/make_move')) {
      s.maxFlight = Math.max(s.maxFlight, ++s.inFlight);
      const body = route.request().postDataJSON(); s.moves.push(body);
      if (body.revision !== s.active.columns.length) throw new Error('Stale revision or duplicate POST');
      s.active.columns.push(sequence[s.active.columns.length]);
      await route.fulfill({ json: s.fixture().state }); s.inFlight--;
    } else if (s.expire) await route.fulfill({ status: 404, json: { code: 'session_not_found', error: 'Session expired' } });
    else await route.fulfill({ json: s.fixture() });
  });
  return s;
}
export async function create(page, size = 8) {
  await page.goto('/connect4/tournament');
  await page.locator('#tournament-size').selectOption(String(size));
  await page.locator('#tournament-seed').fill('1234');
  await page.locator('#tournament-create').click();
}
