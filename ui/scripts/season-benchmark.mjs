import { performance } from 'node:perf_hooks';
import assert from 'node:assert/strict';
import { createSeason, defaultField, gamePlan, replayColumns, validateSeason } from '../src/season/model.js';
import { seasonStandings, seasonRatings, pairwiseResults, scoreRateInterval } from '../src/season/analytics.js';
import { DRAW } from '../e2e/fixtures/match.js';
function timing(fn, count = 20) { let result; const start = performance.now(); for (let i = 0; i < count; i++) result = fn(); return { ms: (performance.now() - start) / count, result }; }
const creation = timing(() => createSeason(defaultField(12), 1234, 8), 200), s = creation.result;
// Legal 42-ply draws are the maximum compact history, with no live sessions.
s.completedGames = s.schedule.map((f, i) => ({ ...gamePlan(s, f), columns: [...DRAW], result: { status: 'draw', winnerIndex: null }, moveCount: 42, completedIndex: i }));
for (const f of s.schedule) { f.status = 'complete'; f.result = { status: 'draw', winnerIndex: null }; }
s.currentGameIndex = 528; s.status = 'complete';
const serialization = timing(() => JSON.stringify(s)), json = serialization.result;
const hydration = timing(() => validateSeason(JSON.parse(json)), 5);
const standings = timing(() => seasonStandings(s)), elo = timing(() => seasonRatings(s)), pairwise = timing(() => pairwiseResults(s));
const bootstrap = timing(() => s.entrants.map(e => scoreRateInterval(s, e.entrantId)), 5);
const replay = timing(() => replayColumns(s.completedGames[0].columns, s.completedGames[0].playerConfigs), 200);
assert.ok(Buffer.byteLength(json) < 1024 * 1024); assert.equal(hydration.result.completedGames.length, 528);
console.log(JSON.stringify({ entrants: 12, gamesPerPairing: 8, games: 528, plies: 22176, serializedBytes: Buffer.byteLength(json), creationMs: creation.ms,
  serializationMs: serialization.ms, hydrationMs: hydration.ms, standingsMs: standings.ms, eloHistoryMs: elo.ms,
  allEntrantsBootstrapMs: bootstrap.ms, pairwiseMs: pairwise.ms, single42PlyReplayMs: replay.ms }, null, 2));
