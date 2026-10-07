// Test-only legal Human policy. Production has no policy and never chooses Human moves.
import assert from 'node:assert/strict';
import axios from 'axios';
import { TournamentController } from '../src/tournament/controller.js';
import { validateTournament, allMatchups, replayColumns, nextMatchup, matchupHasHuman, gamePlan } from '../src/tournament/model.js';
import { STORAGE_KEY } from '../src/tournament/storage.js';
const baseURL = process.env.PLAYWRIGHT_API_URL || 'http://localhost:8007';
assert.ok(['localhost', '127.0.0.1', '[::1]'].includes(new URL(baseURL).hostname), 'Local backend only');
const http = axios.create({ baseURL, timeout: 90000 });
let inFlight = 0, maximum = 0, starts = 0, humanPlies = 0, aiPlies = 0, live;
const ids = [], values = new Map();
const transport = {
  async post(url, body) {
    maximum = Math.max(maximum, ++inFlight);
    try {
      if (url.endsWith('start_game')) {
        starts++; if (ids.length) assert.equal(body.replace_game_id, ids.at(-1));
      } else {
        assert.equal(body.revision, live.revision);
        const human = live.players[live.currentPlayer].type === 'human'; assert.equal('column' in body, human);
        if (human) humanPlies++; else aiPlies++;
      }
      const response = await http.post(url, body); live = response.data;
      if (url.endsWith('start_game')) {
        assert.equal(live.revision, 0);
        if (ids.length) assert.equal((await http.get(`/v1/connect4/games/${ids.at(-1)}/history`, { validateStatus: () => true })).status, 404);
        ids.push(live.game_id);
      }
      return response;
    } finally { inFlight--; }
  },
  get: url => http.get(url),
};
const c = new TournamentController(transport, { getItem: k => values.get(k), setItem: (k, v) => values.set(k, v) }, fn => setTimeout(fn, 0));
c.attach(); c.create(Array.from({ length: 8 }, (_, i) => i === 0 ? { type: 'human' } : i % 3 === 0 ? { type: 'random' } : { type: 'negamax', depth: i % 2 + 1 }), 1234);
const start = Date.now(); let humanGames = 0;
while (c.state.tournament.status !== 'complete' && Date.now() - start < 120000) {
  assert.equal(c.state.error, '');
  if (!c.state.busy) {
    if (!c.state.tournament.active && c.state.mode === 'paused') {
      const m = nextMatchup(c.state.tournament);
      c.run('tournament');
      if (matchupHasHuman(c.state.tournament, m)) {
        assert.equal(c.state.waitingForHuman, true); assert.equal(c.state.tournament.active, null);
        const expected = gamePlan(c.state.tournament, m); await c.watch(); humanGames++;
        assert.equal(live.players[0].type, expected.playerConfigs[0].type);
        c.run('game');
      }
    } else if (c.humanTurn()) {
      const column = [3, 2, 4, 1, 5, 0, 6].find(col => live.legalMoves.includes(col));
      await c.humanMove(column);
    }
  }
  await new Promise(resolve => setTimeout(resolve, 5));
}
assert.equal(c.state.error, ''); assert.equal(c.state.tournament.status, 'complete'); assert.equal(maximum, 1); assert.ok(humanPlies > 0 && aiPlies > 0);
const t = validateTournament(JSON.parse(values.get(STORAGE_KEY))); assert.equal(allMatchups(t).filter(m => m.status === 'complete').length, 7);
for (const m of allMatchups(t)) for (const g of m.games) assert.equal(replayColumns(g.columns, g.playerConfigs).game.gameOver, true);
console.log(JSON.stringify({ champion: t.championEntrantId, matchups: 7, games: starts, humanGames, humanPlies, aiPlies,
  maxConcurrentPosts: maximum, retiredSessions: ids.length - 1, compactBytes: JSON.stringify(t).length, elapsedMs: Date.now() - start })); c.detach();
