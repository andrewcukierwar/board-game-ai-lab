import test from 'node:test';
import assert from 'node:assert/strict';
import { createSeason, defaultField, currentFixture, gamePlan, compactHistory, recordGame, validateSeason, replayColumns, FIELD_SIZES, GAMES_PER_PAIRING, totalGames } from '../src/season/model.js';
import { loadSeason, STORAGE_KEY } from '../src/season/storage.js';
import { seasonStandings, seasonRatings, pairwiseResults, scoreRateInterval, sideSplits, ratingHistory, expectedScore, updateElo, scoreFor } from '../src/season/analytics.js';
import { matchFixture, WIN, DRAW } from '../e2e/fixtures/match.js';
const yellowWin = [0, 1, 0, 1, 2, 1, 2, 1];
function finish(s, count = s.schedule.length, sequences = [WIN, DRAW, yellowWin]) {
  for (let i = 0; i < count; i++) {
    const plan = gamePlan(s); s = recordGame(s, compactHistory(s, { ...matchFixture(plan.playerConfigs, sequences[i % sequences.length]), rng_seed: plan.gameSeed }));
  }
  return s;
}
for (const size of FIELD_SIZES) for (const games of GAMES_PER_PAIRING) test(`${size} entrants × ${games}: circle rounds, counts, exact per-pair and season color balance`, () => {
  const s = createSeason(defaultField(size), 1234, games), pairs = new Map(), sides = new Map();
  assert.equal(s.schedule.length, totalGames(size, games));
  for (let round = 1; round <= (size - 1) * games; round++) {
    const fixtures = s.schedule.filter(f => f.round === round);
    assert.equal(fixtures.length, size / 2);
    assert.equal(new Set(fixtures.flatMap(f => [f.redEntrantId, f.yellowEntrantId])).size, size);
  }
  for (const f of s.schedule) {
    const key = [f.redEntrantId, f.yellowEntrantId].sort().join(':');
    if (!pairs.has(key)) pairs.set(key, []); pairs.get(key).push(f.redEntrantId);
    for (const [id, color] of [[f.redEntrantId, 0], [f.yellowEntrantId, 1]]) { if (!sides.has(id)) sides.set(id, [0, 0]); sides.get(id)[color]++; }
  }
  assert.equal(pairs.size, size * (size - 1) / 2);
  for (const red of pairs.values()) { assert.equal(red.length, games); assert.equal(red.filter(id => id === red[0]).length, games / 2); }
  for (const counts of sides.values()) assert.deepEqual(counts, [(size - 1) * games / 2, (size - 1) * games / 2]);
  assert.deepEqual(createSeason(defaultField(size), 1234, games), s);
  assert.equal(new Set(s.schedule.map(f => f.fixtureId)).size, s.schedule.length);
  assert.equal(new Set(s.schedule.map(f => f.gameSeed)).size, s.schedule.length);
  assert.deepEqual(validateSeason(s), s);
});
test('mirrored halves retain pairings and reverse colors for every cycle', () => {
  const s = createSeason(defaultField(8), 0, 8), half = 28;
  for (let cycle = 0; cycle < 4; cycle++) for (let i = 0; i < half; i++) {
    const a = s.schedule[cycle * half * 2 + i], b = s.schedule[cycle * half * 2 + half + i];
    assert.equal(a.redEntrantId, b.yellowEntrantId); assert.equal(a.yellowEntrantId, b.redEntrantId); assert.notEqual(a.gameSeed, b.gameSeed);
  }
});
test('seed changes schedule order, not pair counts; stable IDs; duplicate configurations have separate identities', () => {
  const configs = Array(4).fill({ type: 'random' }), a = createSeason(configs, 0), b = createSeason(configs, 1);
  assert.notDeepEqual(a.schedule, b.schedule); assert.deepEqual(a.schedule.map(f => f.fixtureId), b.schedule.map(f => f.fixtureId));
  assert.equal(new Set(a.entrants.map(e => e.entrantId)).size, 4);
});
for (const size of [0, 2, 3, 5, 7, 9, 11, 13, 16]) test(`reject unsupported field ${size}`, () => assert.throws(() => createSeason(Array(size).fill({ type: 'random' }), 0)));
for (const games of [0, 1, 3, 6, 9, '2']) test(`reject games per pairing ${games}`, () => assert.throws(() => createSeason(defaultField(4), 0, games)));
for (const seed of [-1, 4294967296, .5, NaN, '0']) test(`reject seed ${seed}`, () => assert.throws(() => createSeason(defaultField(4), seed)));
test('Human/private/invalid configs rejected; uint32 endpoints accepted', () => {
  for (const config of [{ type: 'human' }, { type: 'alphazero' }, { type: 'negamax', depth: 3 }, { type: 'mcts', simulations: 999 }, { type: 'random', depth: 1 }]) assert.throws(() => createSeason([config, ...defaultField(4).slice(1)], 0));
  assert.ok(createSeason(defaultField(4), 4294967295));
});
test('canonical storage/replay rebuild from compact evidence; unknown snapshots discarded; aggregates rejected', () => {
  const s = finish(createSeason(defaultField(4), 1234));
  const json = JSON.stringify(s); assert.ok(!json.includes('board')); assert.deepEqual(validateSeason(JSON.parse(json)), s);
  for (const g of s.completedGames) assert.equal(replayColumns(g.columns, g.playerConfigs).game.gameOver, true);
  assert.deepEqual(validateSeason({ ...s, arbitrary: 'ignored' }), s);
  assert.throws(() => validateSeason({ ...s, ratings: [1500] }));
});
const mutations = {
  version: s => s.version++, scheduleVersion: s => s.scheduleVersion++, identity: s => s.entrants[0].entrantId = 'entrant-x',
  topology: s => s.schedule.pop(), duplicateFixture: s => s.schedule[1].fixtureId = s.schedule[0].fixtureId,
  seed: s => s.schedule[0].gameSeed++, colors: s => [s.schedule[0].redEntrantId, s.schedule[0].yellowEntrantId] = [s.schedule[0].yellowEntrantId, s.schedule[0].redEntrantId],
  result: s => s.completedGames[0].result.status = 'draw', columns: s => s.completedGames[0].columns.push(0),
  order: s => s.completedGames.reverse(), index: s => s.currentGameIndex++, gap: s => s.completedGames.splice(0, 1),
  winner: s => s.schedule[0].result.winnerIndex = 1, gameSeed: s => s.completedGames[0].gameSeed++, status: s => s.status = 'running',
};
for (const [name, mutate] of Object.entries(mutations)) test(`untrusted storage rejects ${name}`, () => {
  const s = finish(createSeason(defaultField(4), 1234), 3); mutate(s); assert.throws(() => validateSeason(s));
});
test('malformed JSON and unsupported schema return recoverable load errors', () => {
  for (const raw of ['{', 'null', JSON.stringify({ version: 99 })]) { const loaded = loadSeason({ getItem: key => { assert.equal(key, STORAGE_KEY); return raw; } }); assert.equal(loaded.season, null); assert.match(loaded.error, /could not be validated/); }
});
test('unfinished and interrupted active fixtures never count; active hydration validates provenance and columns', () => {
  const s = createSeason(defaultField(4), 0), f = currentFixture(s);
  s.active = { fixtureId: f.fixtureId, gameId: 'live', columns: [0, 1], status: 'running' }; s.retainedGameId = 'live'; f.status = 'active';
  assert.deepEqual(validateSeason(s), s); assert.ok(seasonStandings(s).every(r => r.played === 0)); assert.ok(seasonRatings(s).every(r => r.rating === 1500));
  const bad = structuredClone(s); bad.retainedGameId = 'other'; assert.throws(() => validateSeason(bad));
  s.active.status = 'interrupted'; f.status = 'interrupted'; assert.deepEqual(validateSeason(s), s);
});
test('standings and pairwise count every result exactly once with deterministic ordering and side splits', () => {
  const s = finish(createSeason(defaultField(4), 10)), rows = seasonStandings(s), matrix = pairwiseResults(s);
  assert.equal(rows.reduce((sum, r) => sum + r.played, 0), 24); assert.equal(rows.reduce((sum, r) => sum + r.points, 0), 12);
  for (const r of rows) {
    const games = s.completedGames.filter(g => [g.redEntrantId, g.yellowEntrantId].includes(r.entrantId));
    const scores = games.map(g => scoreFor(g, r.entrantId));
    assert.equal(r.points, scores.reduce((a, b) => a + b)); assert.equal(r.wins, scores.filter(x => x === 1).length); assert.equal(r.draws, scores.filter(x => x === .5).length); assert.equal(r.losses, scores.filter(x => x === 0).length);
    assert.equal(r.scoreRate, r.points / 6); assert.equal(r.red.played, 3); assert.equal(r.yellow.played, 3);
    for (const side of ['red', 'yellow']) { const scores = games.filter(g => g[`${side}EntrantId`] === r.entrantId).map(g => scoreFor(g, r.entrantId)); assert.equal(r[side].points, scores.reduce((a, b) => a + b)); }
    const opponents = Object.values(matrix[r.entrantId]); assert.equal(opponents.reduce((sum, p) => sum + p.played, 0), 6); assert.equal(opponents.reduce((sum, p) => sum + p.points, 0), r.points);
  }
  for (let i = 1; i < rows.length; i++) assert.ok(rows[i - 1].points >= rows[i].points);
  const drawn = finish(createSeason(defaultField(4), 0), 12, [DRAW]); assert.deepEqual(seasonStandings(drawn).map(r => r.seedNumber), [1, 2, 3, 4]);
  assert.deepEqual(sideSplits(s), rows.map(({ entrantId, red, yellow }) => ({ entrantId, red, yellow })));
});
test('Elo expected score, win, draw, zero sum and full-precision updates', () => {
  assert.equal(expectedScore(1500, 1500), .5); assert.deepEqual(updateElo(1500, 1500, 1), [1512, 1488]); assert.deepEqual(updateElo(1500, 1500, .5), [1500, 1500]);
  const [a, b] = updateElo(1700, 1300, .5); assert.ok(a < 1700 && b > 1300); assert.ok(Math.abs(a + b - 3000) < 1e-9);
  const s = finish(createSeason(defaultField(4), 1234)); const ratings = seasonRatings(s);
  assert.deepEqual(ratings, seasonRatings(validateSeason(s))); assert.ok(ratings.some(r => r.rating !== Math.round(r.rating)));
  assert.ok(Math.abs(ratings.reduce((sum, r) => sum + r.rating, 0) - 6000) < 1e-9);
  for (const r of ratings) { assert.equal(r.history[0], 1500); assert.equal(r.history.at(-1), r.rating); assert.equal(r.history.length, 13); assert.equal(r.peak, Math.max(...r.history)); assert.equal(r.low, Math.min(...r.history)); }
  assert.deepEqual(ratingHistory(s), ratings.map(({ entrantId, history }) => ({ entrantId, history })));
});
test('bootstrap threshold, deterministic seed, bounds and local RNG independence', () => {
  let s = finish(createSeason(defaultField(4), 5, 8), 6);
  assert.equal(scoreRateInterval(s, 'entrant-1'), null);
  s = finish(s, s.schedule.length - s.currentGameIndex);
  const original = Math.random; Math.random = () => { throw new Error('Global RNG called'); };
  try { for (const r of seasonStandings(s)) { const interval = scoreRateInterval(s, r.entrantId); assert.deepEqual(interval, scoreRateInterval(s, r.entrantId)); assert.equal(interval.samples, 1000); assert.ok(interval.lower >= 0 && interval.upper <= 1 && interval.lower <= r.scoreRate && r.scoreRate <= interval.upper); } } finally { Math.random = original; }
  const wins = { ...s, completedGames: Array.from({ length: 8 }, () => ({ redEntrantId: 'entrant-1', yellowEntrantId: 'entrant-2', result: { status: 'win', winnerIndex: 0 } })) };
  assert.equal(scoreRateInterval(wins, 'entrant-1').lower, 1); assert.equal(scoreRateInterval(wins, 'entrant-1').upper, 1);
});
test('standings break point ties by score rate before seed number', () => {
  const s = createSeason(defaultField(4), 0);
  s.completedGames = [[1, 2], [3, 1], [4, 2]].map(([a, b]) => ({ redEntrantId: `entrant-${a}`, yellowEntrantId: `entrant-${b}`, result: { status: 'win', winnerIndex: 0 } }));
  assert.deepEqual(seasonStandings(s).map(r => r.seedNumber), [3, 4, 1, 2]);
});
test('active evidence without a received session ID cannot include columns', () => {
  const s = createSeason(defaultField(4), 0); s.active = { fixtureId: s.schedule[0].fixtureId, gameId: null, status: 'interrupted', columns: [0] }; s.schedule[0].status = 'interrupted'; assert.throws(() => validateSeason(s));
});
test('bootstrap exact 8-game threshold and homogeneous loss/draw samples remain sensible', () => {
  const s = createSeason(defaultField(4), 0);
  for (const [status, winnerIndex, expected] of [['win', 1, 0], ['draw', null, .5]]) {
    s.completedGames = Array.from({ length: 7 }, () => ({ redEntrantId: 'entrant-1', yellowEntrantId: 'entrant-2', result: { status, winnerIndex } }));
    assert.equal(scoreRateInterval(s, 'entrant-1'), null); s.completedGames.push(s.completedGames[0]); const interval = scoreRateInterval(s, 'entrant-1'); assert.equal(interval.played, 8); assert.equal(interval.lower, expected); assert.equal(interval.upper, expected);
  }
});
