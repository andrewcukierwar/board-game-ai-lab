// Fast actual-backend integration: no simulated game results, no provider calls.
import assert from 'node:assert/strict';
import axios from 'axios';
import { TournamentController } from '../src/tournament/controller.js';
import { validateTournament, allMatchups, replayColumns } from '../src/tournament/model.js';
import { STORAGE_KEY } from '../src/tournament/storage.js';
const baseURL = process.env.PLAYWRIGHT_API_URL || 'http://localhost:8006';
assert.ok(['localhost', '127.0.0.1', '[::1]'].includes(new URL(baseURL).hostname), 'Local backend only');
const http = axios.create({ baseURL, timeout: 90000 });
let inFlight = 0, maximum = 0, starts = 0, plies = 0; const ids = [], values = new Map();
const transport = {
  async post(url, body) {
    maximum = Math.max(maximum, ++inFlight);
    try {
      if (url.endsWith('start_game')) {
        if (ids.length) assert.equal(body.replace_game_id, ids.at(-1));
        starts++;
      } else { plies++; assert.equal('column' in body, false); }
      const result = await http.post(url, body);
      if (url.endsWith('start_game')) {
        if (ids.length) assert.equal((await http.get(`/v1/connect4/games/${ids.at(-1)}/history`, { validateStatus: () => true })).status, 404);
        ids.push(result.data.game_id);
      }
      return result;
    } finally { inFlight--; }
  },
  get: url => http.get(url),
};
const c = new TournamentController(transport, { getItem: k => values.get(k), setItem: (k, v) => values.set(k, v) }, fn => setTimeout(fn, 0));
c.attach(); c.create(Array.from({ length: 8 }, (_, i) => i % 3 === 0 ? { type: 'random' } : { type: 'negamax', depth: i % 2 + 1 }), 1234);
c.run('tournament');
const start = Date.now();
while (c.state.mode !== 'paused' && Date.now() - start < 120000) await new Promise(r => setTimeout(r, 10));
assert.equal(c.state.error, ''); assert.equal(c.state.tournament.status, 'complete'); assert.equal(maximum, 1);
const t = validateTournament(JSON.parse(values.get(STORAGE_KEY))); assert.equal(allMatchups(t).filter(m => m.status === 'complete').length, 7);
for (const m of allMatchups(t)) for (const g of m.games) assert.equal(replayColumns(g.columns, g.playerConfigs).game.gameOver, true);
console.log(JSON.stringify({ champion: t.championEntrantId, matchups: 7, games: starts, plies, maxConcurrentPosts: maximum, retiredSessions: ids.length - 1, elapsedMs: Date.now() - start })); c.detach();
