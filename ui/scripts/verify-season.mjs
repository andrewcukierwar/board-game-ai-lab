import assert from 'node:assert/strict';
import axios from 'axios';
import { SeasonController } from '../src/season/controller.js';
import { validateSeason, replayColumns } from '../src/season/model.js';
import { seasonStandings, seasonRatings } from '../src/season/analytics.js';
import { STORAGE_KEY } from '../src/season/storage.js';
const baseURL = process.env.PLAYWRIGHT_API_URL || 'http://localhost:8008';
assert.ok(['localhost', '127.0.0.1', '[::1]'].includes(new URL(baseURL).hostname), 'Local API only');
const http = axios.create({ baseURL, timeout: 90000 }), ids = [], values = new Map();
let inFlight = 0, maxFlight = 0, starts = 0, plies = 0;
const transport = {
  async post(url, body) {
    maxFlight = Math.max(maxFlight, ++inFlight);
    try {
      if (url.endsWith('start_game')) { if (ids.length) assert.equal(body.replace_game_id, ids.at(-1)); starts++; }
      else { assert.equal('column' in body, false); plies++; }
      const result = await http.post(url, body);
      if (url.endsWith('start_game')) { if (ids.length) assert.equal((await http.get(`/v1/connect4/games/${ids.at(-1)}/history`, { validateStatus: () => true })).status, 404); ids.push(result.data.game_id); }
      return result;
    } finally { inFlight--; }
  }, get: url => http.get(url),
};
const storage = { getItem: k => values.get(k), setItem: (k, v) => values.set(k, v) };
const c = new SeasonController(transport, storage, fn => setTimeout(fn, 0));
const field = [{ type: 'random' }, { type: 'negamax', depth: 1 }, { type: 'negamax', depth: 2 }, { type: 'random' }];
c.attach(); c.create(field, 1234, 2); assert.equal(starts, 0); c.run('season');
const began = Date.now();
while (c.state.mode !== 'paused' && Date.now() - began < 120000) await new Promise(r => setTimeout(r, 10));
assert.equal(c.state.error, ''); assert.equal(c.state.season.status, 'complete'); assert.equal(starts, 12); assert.equal(maxFlight, 1);
const s = validateSeason(JSON.parse(values.get(STORAGE_KEY))), pairs = new Map(), raw = Object.fromEntries(s.entrants.map(e => [e.entrantId, { played: 0, wins: 0, draws: 0, losses: 0, points: 0, rating: 1500 }]));
for (const g of s.completedGames) {
  assert.deepEqual(replayColumns(g.columns, g.playerConfigs).result, g.result);
  const key = [g.redEntrantId, g.yellowEntrantId].sort().join(':'); if (!pairs.has(key)) pairs.set(key, []); pairs.get(key).push(g.redEntrantId);
  const a = raw[g.redEntrantId], b = raw[g.yellowEntrantId], score = g.result.status === 'draw' ? .5 : g.result.winnerIndex === 0 ? 1 : 0;
  // Independent direct computation, without production analytics helpers.
  const delta = 24 * (score - 1 / (1 + Math.pow(10, (b.rating - a.rating) / 400))); a.rating += delta; b.rating -= delta;
  for (const [row, outcome] of [[a, score], [b, 1 - score]]) { row.played++; row.points += outcome; row[outcome === 1 ? 'wins' : outcome === .5 ? 'draws' : 'losses']++; }
}
assert.equal(pairs.size, 6); for (const colors of pairs.values()) { assert.equal(colors.length, 2); assert.notEqual(colors[0], colors[1]); }
const standings = seasonStandings(s), ratings = seasonRatings(s);
for (const r of standings) for (const key of ['played', 'wins', 'draws', 'losses', 'points']) assert.equal(r[key], raw[r.entrantId][key]);
for (const r of ratings) assert.ok(Math.abs(r.rating - raw[r.entrantId].rating) < 1e-9);
c.detach(); const reload = new SeasonController(transport, storage); reload.attach(); assert.equal(reload.state.mode, 'paused'); assert.deepEqual(reload.state.season, s); reload.detach();
console.log(JSON.stringify({ seed: 1234, field, games: starts, plies, maxConcurrentPosts: maxFlight, retiredSessions: ids.length - 1, serializedBytes: Buffer.byteLength(values.get(STORAGE_KEY)), elapsedMs: Date.now() - began,
  standings: standings.map(r => ({ entrantId: r.entrantId, played: r.played, wins: r.wins, draws: r.draws, losses: r.losses, points: r.points })), ratings: ratings.map(r => ({ entrantId: r.entrantId, elo: r.rating })) }, null, 2));
