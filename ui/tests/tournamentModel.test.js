import { test } from 'node:test';
import assert from 'node:assert/strict';
import { performance } from 'node:perf_hooks';
import { SIZES, defaultField, createTournament, bracketOrder, allMatchups, nextMatchup, findMatchup, gamePlan, compactHistory,
  recordGame, validateTournament, replayColumns, deriveSeed, validSeed } from '../src/tournament/model.js';
import { loadTournament, saveTournament, STORAGE_KEY } from '../src/tournament/storage.js';
import { DRAW, WIN, matchFixture } from '../e2e/fixtures/match.js';
export function completed(t, m = nextMatchup(t), columns = WIN) {
  const plan = gamePlan(t, m);
  return compactHistory(t, m, { ...matchFixture(plan.playerConfigs, columns), rng_seed: plan.gameSeed });
}
const fresh = size => createTournament(defaultField(size ?? 8), 1234);
export function completeField(size = 8, columns = WIN) {
  let t = fresh(size);
  while (nextMatchup(t)) { const m = nextMatchup(t); t = recordGame(t, m.matchupId, completed(t, m, columns)); }
  return t;
}
export function memoryStorage() { const values = new Map(); return { getItem: key => values.get(key) ?? null, setItem: (key, value) => values.set(key, value) }; }
for (const size of SIZES) test(`${size} entrants: complete prebuilt topology, stable IDs, duplicate configs and champion`, () => {
  let t = createTournament(Array(size).fill({ type: 'random' }), 42);
  assert.equal(allMatchups(t).length, size - 1); assert.equal(t.rounds.length, Math.log2(size));
  assert.equal(new Set(t.entrants.map(e => e.entrantId)).size, size);
  assert.equal(new Set(t.bracketOrder).size, size);
  assert.equal(t.rounds[0].length, size / 2);
  assert.ok(t.rounds.slice(1).flat().every(m => m.entrantAId === null && m.entrantBId === null));
  for (let n = 0; n < size - 1; n++) {
    const m = nextMatchup(t), game = completed(t, m), winner = game.playerEntrantIds[0];
    const old = structuredClone(t); t = recordGame(t, m.matchupId, game);
    assert.deepEqual(old.rounds[m.round][m.index], m); // input remains untouched
    if (m.round < Math.log2(size) - 1) {
      assert.equal(t.rounds[m.round + 1][Math.floor(m.index / 2)][m.index % 2 ? 'entrantBId' : 'entrantAId'], winner);
      assert.equal(t.championEntrantId, null);
    } else assert.equal(t.championEntrantId, winner);
    assert.deepEqual(validateTournament(JSON.parse(JSON.stringify(t))), t);
  }
  assert.equal(t.status, 'complete');
});
for (const size of [0, 1, 2, 4, 7, 9, 24, 128, true]) test(`reject size ${size}`, () => {
  assert.throws(() => createTournament(Array(Number(size)).fill({ type: 'random' }), 1));
});
for (const seed of [-1, 0.5, 4294967296, true, false, '12', null, NaN]) test(`reject seed ${seed}`, () => {
  assert.equal(validSeed(seed), false); assert.throws(() => createTournament(defaultField(8), seed));
});
test('seeded bracket and game provenance reproducible including uint32 edges', () => {
  for (const seed of [0, 1234, 4294967295]) {
    const a = createTournament(defaultField(16), seed), b = createTournament(defaultField(16), seed);
    assert.deepEqual(a, b); assert.deepEqual(a.bracketOrder, bracketOrder(16, seed));
    assert.deepEqual(gamePlan(a, a.rounds[0][0]), gamePlan(b, b.rounds[0][0]));
  }
  assert.notDeepEqual(bracketOrder(64, 1), bracketOrder(64, 2));
  const a = fresh(), colors = allMatchups(a).slice(0, 4).map(m => gamePlan(a, m).playerEntrantIds[0] === m.entrantAId);
  assert.ok(colors.includes(true) && colors.includes(false));
  assert.equal(deriveSeed(1234, 'r1-m1:game:1'), 1845687796);
});
test('reject unsupported public presets, missing budget and extra config fields', () => {
  for (const config of [{ type: 'human', depth: 2 }, { type: 'negamax', depth: 3 }, { type: 'mcts', simulations: 50 },
    { type: 'mcts' }, { type: 'random', depth: 2 }, null]) {
    assert.throws(() => createTournament([config, ...defaultField(8).slice(1)], 1));
  }
});
test('impossible advancement, provenance and incomplete history rejected', () => {
  const t = fresh(), m = nextMatchup(t), game = completed(t, m);
  assert.throws(() => recordGame(t, t.rounds[1][0].matchupId, game));
  assert.throws(() => recordGame(t, t.rounds[0][1].matchupId, game));
  for (const patch of [{ gameSeed: game.gameSeed ^ 1 }, { result: { status: 'draw', winnerIndex: null } }, { moveCount: 8 }, { columns: [0] }])
    assert.throws(() => recordGame(t, m.matchupId, { ...game, ...patch }));
  assert.throws(() => compactHistory(t, m, { ...matchFixture(game.playerConfigs, []), rng_seed: game.gameSeed }));
  const done = recordGame(t, m.matchupId, game); assert.throws(() => recordGame(done, m.matchupId, game));
});
for (const decisive of [1, 2, 3, 0]) test(`draw policy decisive game ${decisive || 'none: seeded tiebreak'}`, () => {
  let t = fresh(); const m = nextMatchup(t), id = m.matchupId, first = gamePlan(t, m);
  for (let number = 1; number <= (decisive || 3); number++) {
    const match = findMatchup(t, id), plan = gamePlan(t, match);
    assert.equal(plan.gameNumber, number); assert.equal(plan.gameSeed, deriveSeed(1234, `${id}:game:${number}`));
    if (number === 2) assert.deepEqual(plan.playerEntrantIds, [...first.playerEntrantIds].reverse());
    if (number === 3) {
      const reversed = deriveSeed(1234, `${id}:color:3`) & 1;
      assert.deepEqual(plan.playerEntrantIds, reversed ? [m.entrantBId, m.entrantAId] : [m.entrantAId, m.entrantBId]);
    }
    t = recordGame(t, id, completed(t, match, number === decisive ? WIN : DRAW));
    assert.equal(findMatchup(t, id).status, number === (decisive || 3) ? 'complete' : 'pending');
  }
  const done = findMatchup(t, id);
  assert.equal(done.resolution, decisive ? 'game_win' : 'seeded_draw_tiebreak');
  assert.equal(done.winnerEntrantId, decisive ? done.games.at(-1).playerEntrantIds[0] : deriveSeed(1234, `${id}:draw-tiebreak`) & 1 ? m.entrantBId : m.entrantAId);
  if (!decisive) assert.ok(done.games.every(g => g.result.status === 'draw'));
  assert.deepEqual(validateTournament(t), t);
});
test('reset/new tournament clears results and active state', () => {
  const old = completeField(); const next = createTournament(old.entrants.map(e => e.config), old.tournamentSeed);
  assert.equal(next.championEntrantId, null); assert.equal(next.active, null);
  assert.ok(allMatchups(next).every(m => !m.games.length));
});
test('persistence roundtrip includes completed local replay with no boards', () => {
  const t = completeField(), storage = memoryStorage(); assert.equal(saveTournament(storage, t), true);
  assert.deepEqual(loadTournament(storage).tournament, t);
  assert.ok(!JSON.stringify(t).includes('board_'));
  const g = t.rounds[0][0].games[0], replay = replayColumns(g.columns, g.playerConfigs);
  assert.equal(replay.game.gameOver, true); assert.equal(replay.moves.length, g.moveCount);
  assert.deepEqual(replay.game.board, matchFixture(g.playerConfigs, g.columns).state.board);
});
test('safe storage errors, malformed JSON/version/relationships/topology/history reject whole record', () => {
  const storage = memoryStorage(); storage.setItem(STORAGE_KEY, '{'); assert.equal(loadTournament(storage).tournament, null);
  const t = completeField();
  const mutations = [a => { a.version = 99; }, a => { a.size = 7; }, a => { a.rounds[1][0].entrantAId = 'entrant-999'; },
    a => { a.rounds[0][0].winnerEntrantId = 'entrant-999'; }, a => { a.bracketOrder.reverse(); },
    a => { a.rounds[0][0].games[0].columns.push(0); }, a => { a.championEntrantId = null; },
    a => { a.rounds[0][0].games[0].gameSeed = 1; }, a => { a.rounds[0][0].games = []; }];
  for (const mutate of mutations) { const a = structuredClone(t); mutate(a); storage.setItem(STORAGE_KEY, JSON.stringify(a));
    assert.equal(loadTournament(storage).tournament, null); assert.match(loadTournament(storage).error, /validated/); }
  assert.equal(saveTournament({ setItem() { throw new Error('quota'); } }, t), false);
  assert.deepEqual(loadTournament(null), { tournament: null, error: '', available: false });
  const blocked = { getItem() { throw Object.assign(new Error('blocked'), { name: 'SecurityError' }); } };
  assert.equal(loadTournament(blocked).available, false);
});
for (const columns of [[7], [true], [0.5], Array(7).fill(0), [...WIN, 2], Array(43).fill(0)]) test(`reject replay ${JSON.stringify(columns).slice(0, 30)}`, () => {
  assert.throws(() => replayColumns(columns, [{ type: 'random' }, { type: 'random' }]));
});
test('64-player worst bounded draw campaign storage/performance', () => {
  const start = performance.now(); const initial = fresh(64); const creation = performance.now() - start;
  const updateStart = performance.now(); const t = completeField(64, DRAW); const update = performance.now() - updateStart;
  const serialized = JSON.stringify(t), bytes = Buffer.byteLength(serialized);
  const hydrateStart = performance.now(); assert.deepEqual(validateTournament(JSON.parse(serialized)), t); const hydrate = performance.now() - hydrateStart;
  const replayStart = performance.now(); replayColumns(t.rounds[0][0].games[0].columns, t.rounds[0][0].games[0].playerConfigs); const replay = performance.now() - replayStart;
  assert.equal(allMatchups(initial).length, 63); assert.equal(allMatchups(t).flatMap(m => m.games).length, 189);
  assert.ok(bytes < 200000); assert.equal(t.status, 'complete');
  console.log(JSON.stringify({ size: 64, games: 189, moves: 7938, bytes, creationMs: creation, totalUpdatesMs: update, hydrateMs: hydrate, replayMs: replay }));
});
test('active storage validates game identity, columns, bracket owner and retained session relationship', () => {
  const t = fresh(); t.active = { matchupId: 'r1-m1', gameNumber: 1, gameId: 'known', status: 'running', columns: [0, 1] };
  t.retainedGameId = 'known'; t.rounds[0][0].status = 'active'; assert.deepEqual(validateTournament(t), t);
  for (const patch of [{ matchupId: 'r1-m2' }, { gameNumber: 2 }, { status: 'bogus' }, { gameId: null }, { columns: [true] }]) {
    assert.throws(() => validateTournament({ ...t, active: { ...t.active, ...patch } }));
  }
  assert.throws(() => validateTournament({ ...t, retainedGameId: 'wrong' }));
});
