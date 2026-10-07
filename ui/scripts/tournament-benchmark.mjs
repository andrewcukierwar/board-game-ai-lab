import { performance } from 'node:perf_hooks';
import { createTournament, defaultField, nextMatchup, gamePlan, compactHistory, recordGame, validateTournament, replayColumns } from '../src/tournament/model.js';
import { DRAW, matchFixture } from '../e2e/fixtures/match.js';
let start = performance.now();
for (let i = 0; i < 200; i++) createTournament(defaultField(64), i);
const createMs = (performance.now() - start) / 200;
let t = createTournament(defaultField(64), 1234), updates = [];
while (nextMatchup(t)) {
  const m = nextMatchup(t), plan = gamePlan(t, m);
  const game = compactHistory(t, m, { ...matchFixture(plan.playerConfigs, DRAW), rng_seed: plan.gameSeed });
  start = performance.now(); t = recordGame(t, m.matchupId, game); updates.push(performance.now() - start);
}
start = performance.now(); const json = JSON.stringify(t), serializeMs = performance.now() - start;
start = performance.now(); validateTournament(JSON.parse(json)); const hydrateMs = performance.now() - start;
start = performance.now(); replayColumns(t.rounds[0][0].games[0].columns, t.rounds[0][0].games[0].playerConfigs); const replayMs = performance.now() - start;
console.log(JSON.stringify({ entrants: 64, matchups: 63, games: 189, columns: 7938, bytes: Buffer.byteLength(json), createMs,
  meanUpdateMs: updates.reduce((a, b) => a + b) / updates.length, maxUpdateMs: Math.max(...updates), serializeMs, hydrateMs, replayMs }, null, 2));
