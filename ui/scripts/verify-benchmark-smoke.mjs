// Read-only independent check of a tiny runner output; never executes gameplay.
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { verifyFile } from './verify-evaluation-export.mjs';
const directory = process.argv[2];
if (!directory) throw new Error('Usage: node ui/scripts/verify-benchmark-smoke.mjs path/to/smoke-run');
const a = JSON.parse(await readFile(join(directory, 'evaluation.json'), 'utf8')),
  metadata = JSON.parse(await readFile(join(directory, 'run-metadata.json'), 'utf8'));
assert.equal(a.evidence.field_size, 4); assert.equal(a.evidence.state.completed_games, 12); assert.equal(a.evidence.state.status, 'complete');
assert.equal((await verifyFile(join(directory, 'evaluation.json'))).ok, true);
assert.equal(metadata.max_concurrent_posts, 1); assert.equal(metadata.game_wall_times_ms.length, 12);
// Generated labels in this exact smoke config contain no comma/quote/newline.
const table = async filename => {
  const [header, ...rows] = (await readFile(join(directory, filename), 'utf8')).trim().split('\r\n').map(r => r.split(','));
  return rows.map(row => Object.fromEntries(header.map((key, i) => [key, row[i]])));
};
const games = await table('games.csv'), summary = await table('summary.csv');
assert.equal(games.length, 12); assert.equal(summary.length, 4);
const blank = () => ({ played: 0, wins: 0, draws: 0, losses: 0, points: 0 });
const raw = Object.fromEntries(a.evidence.entrants.map(e => [e.entrant_id, { ...blank(), red: blank(), yellow: blank(), elo: 1500, peak: 1500, low: 1500 }]));
const pairs = new Map();
for (const [index, g] of a.evidence.completed_games.entries()) {
  const row = games[index]; assert.equal(Number(row.completion_index), index); assert.equal(row.fixture_id, g.fixture_id);
  assert.equal(Number(row.game_seed), g.game_seed); assert.equal(row.red_entrant_id, g.red_entrant_id); assert.equal(row.yellow_entrant_id, g.yellow_entrant_id);
  assert.equal(row.move_columns, g.move_columns.join('|')); assert.equal(Number(row.move_count), g.move_count); assert.equal(row.result, g.terminal_result.status);
  const score = g.terminal_result.status === 'draw' ? .5 : g.terminal_result.winner_index === 0 ? 1 : 0;
  assert.equal(row.winner_entrant_id, score === .5 ? '' : score === 1 ? g.red_entrant_id : g.yellow_entrant_id);
  const red = raw[g.red_entrant_id], yellow = raw[g.yellow_entrant_id], delta = 24 * (score - 1 / (1 + Math.pow(10, (yellow.elo - red.elo) / 400)));
  red.elo += delta; yellow.elo -= delta;
  for (const [r, side, outcome] of [[red, 'red', score], [yellow, 'yellow', 1 - score]]) {
    for (const target of [r, r[side]]) { target.played++; target.points += outcome; target[outcome === 1 ? 'wins' : outcome === .5 ? 'draws' : 'losses']++; }
    r.peak = Math.max(r.peak, r.elo); r.low = Math.min(r.low, r.elo);
  }
  const key = [g.red_entrant_id, g.yellow_entrant_id].sort().join(':'); if (!pairs.has(key)) pairs.set(key, []); pairs.get(key).push(g.red_entrant_id);
}
assert.equal(pairs.size, 6); for (const colors of pairs.values()) { assert.equal(colors.length, 2); assert.notEqual(colors[0], colors[1]); }
for (const row of summary) {
  const r = raw[row.entrant_id];
  for (const key of ['played', 'wins', 'draws', 'losses', 'points']) assert.equal(Number(row[key]), r[key]);
  assert.equal(Number(row.score_rate), r.points / r.played);
  for (const side of ['red', 'yellow']) { assert.equal(r[side].played, 3); for (const key of ['played', 'wins', 'draws', 'losses']) assert.equal(Number(row[`${side}_${key}`]), r[side][key]); assert.equal(Number(row[`${side}_score_rate`]), r[side].points / 3); }
  for (const [key, expected] of [['final_elo', r.elo], ['elo_change', r.elo - 1500], ['peak_elo', r.peak], ['low_elo', r.low]]) assert.ok(Math.abs(Number(row[key]) - expected) < 1e-9);
  assert.equal(row.bootstrap_lower, ''); assert.equal(row.bootstrap_upper, ''); assert.equal(Number(row.bootstrap_games), 6);
}
console.log(JSON.stringify({ verification: 'PASS', games: 12, plies: a.evidence.completed_games.reduce((n, g) => n + g.move_count, 0),
  digest: a.integrity.evidence_sha256, maxConcurrentPosts: metadata.max_concurrent_posts, mutationRequests: metadata.mutation_requests,
  elapsedMs: metadata.wall_time_ms, gameWallTimesMs: metadata.game_wall_times_ms.map(g => g.wall_time_ms), independentStandingsAndElo: raw }, null, 2));
