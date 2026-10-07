import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createTournament, defaultField, bracketOrder, gamePlan, nextMatchup, recordGame, compactHistory,
  humanEntrant, matchupHasHuman, humanTournamentStatus, validateTournament, allMatchups, replayColumns } from '../src/tournament/model.js';
import { saveTournament, loadTournament } from '../src/tournament/storage.js';
import { WIN, DRAW, matchFixture } from '../e2e/fixtures/match.js';
const YELLOW_WIN = [0, 1, 0, 1, 2, 1, 2, 1];
const field = (size = 8, slot = 0) => defaultField(size).map((c, i) => i === slot ? { type: 'human' } : c);
function complete(t, columns) {
  const m = nextMatchup(t), plan = gamePlan(t, m);
  return recordGame(t, m.matchupId, compactHistory(t, m, { ...matchFixture(plan.playerConfigs, columns), rng_seed: plan.gameSeed }));
}
test('version-1 zero/one Human fields hydrate; a second or malformed Human rejects', () => {
  for (const f of [defaultField(8), field()]) {
    const t = createTournament(f, 1234), values = new Map();
    const storage = { setItem: (k, v) => values.set(k, v), getItem: k => values.get(k) };
    saveTournament(storage, t); assert.deepEqual(loadTournament(storage).tournament, t);
    assert.equal(t.version, 1);
  }
  assert.throws(() => createTournament([{ type: 'human' }, ...field().slice(0, 7)], 1), /at most one/);
  const t = createTournament(field(), 1); t.entrants[1].config = { type: 'human' };
  assert.throws(() => validateTournament(t));
  t.entrants[1].config = { type: 'random' }; t.entrants[0].config.extra = true;
  assert.throws(() => validateTournament(t));
  assert.equal(humanTournamentStatus(createTournament(defaultField(8), 1)), null);
});
for (const size of [8, 16, 32, 64]) test(`Human stable slot identity and seeded placement in every ${size}-player slot`, () => {
  for (let slot = 0; slot < size; slot++) {
    const t = createTournament(field(size, slot), 1234), h = humanEntrant(t);
    assert.equal(h.entrantId, `entrant-${slot + 1}`); assert.equal(h.seedNumber, slot + 1);
    assert.deepEqual(t.bracketOrder, bracketOrder(size, 1234));
    assert.equal(t.rounds[0].filter(m => matchupHasHuman(t, m)).length, 1);
    assert.deepEqual(validateTournament(t), t);
  }
});
test('Human can be either color; field type does not change seeds or color algorithm', () => {
  const colors = new Set();
  for (let seed = 0; seed < 24; seed++) {
    const ai = createTournament(defaultField(8), seed), t = createTournament(field(), seed);
    const m = t.rounds[0].find(m => matchupHasHuman(t, m)), p = gamePlan(t, m), old = gamePlan(ai, ai.rounds[0][m.index]);
    assert.equal(p.gameSeed, old.gameSeed); assert.deepEqual(p.playerEntrantIds, old.playerEntrantIds);
    colors.add(p.playerConfigs.findIndex(c => c.type === 'human'));
  }
  assert.deepEqual([...colors].sort(), [0, 1]);
});
for (const win of [true, false]) test(`64-player Human ${win ? 'six-round champion' : 'elimination then AI champion'} remains compact/local`, () => {
  let t = createTournament(field(64, 41), 7); let humanRounds = 0;
  while (nextMatchup(t)) {
    const m = nextMatchup(t), p = gamePlan(t, m), human = matchupHasHuman(t, m);
    const humanColor = p.playerConfigs.findIndex(c => c.type === 'human');
    if (human) humanRounds++;
    const columns = human && (win ? humanColor === 1 : humanColor === 0) ? YELLOW_WIN : WIN;
    t = complete(t, columns);
    assert.deepEqual(validateTournament(JSON.parse(JSON.stringify(t))), t);
  }
  assert.equal(allMatchups(t).length, 63); assert.equal(humanRounds, win ? 6 : 1);
  assert.equal(humanTournamentStatus(t).kind, win ? 'champion' : 'eliminated');
  assert.equal(t.championEntrantId === 'entrant-42', win);
  for (const m of allMatchups(t)) for (const g of m.games) {
    assert.equal(replayColumns(g.columns, g.playerConfigs).game.gameOver, true);
    assert.equal(g.playerConfigs.some(p => p.type === 'human'), matchupHasHuman(t, m));
  }
  assert.ok(JSON.stringify(t).length < 35000); assert.ok(!JSON.stringify(t).includes('board_'));
});
test('Human ready/active/advanced/waiting are derived without extra persisted fields', () => {
  let t = createTournament(field(), 1234);
  assert.equal(humanTournamentStatus(t).kind, matchupHasHuman(t, nextMatchup(t)) ? 'ready' : 'waiting');
  while (!matchupHasHuman(t, nextMatchup(t))) t = complete(t, WIN);
  assert.equal(humanTournamentStatus(t).kind, 'ready');
  const m = nextMatchup(t), p = gamePlan(t, m);
  t.active = { matchupId: m.matchupId, gameNumber: 1, gameId: 'human-live', status: 'running', columns: [] }; m.status = 'active'; t.retainedGameId = 'human-live';
  assert.equal(humanTournamentStatus(t).kind, 'active');
  t = complete(t, p.playerConfigs[0].type === 'human' ? WIN : YELLOW_WIN);
  assert.match(humanTournamentStatus(t).message, /You advanced/);
});
for (const advance of [true, false]) test(`three draws seeded tiebreak can ${advance ? 'advance' : 'eliminate'} Human without a game loss`, () => {
  let found = false;
  for (let seed = 0; seed < 100 && !found; seed++) {
    let t = createTournament(field(), seed);
    while (!matchupHasHuman(t, nextMatchup(t))) t = complete(t, WIN);
    const m = nextMatchup(t), first = gamePlan(t, m);
    t = complete(t, DRAW); const second = gamePlan(t, nextMatchup(t));
    assert.deepEqual(second.playerEntrantIds, [...first.playerEntrantIds].reverse());
    t = complete(t, DRAW); assert.equal(gamePlan(t, nextMatchup(t)).gameNumber, 3);
    t = complete(t, DRAW); const done = t.rounds[m.round][m.index];
    if ((done.winnerEntrantId === humanEntrant(t).entrantId) !== advance) continue;
    found = true; assert.equal(done.resolution, 'seeded_draw_tiebreak');
    assert.ok(done.games.every(g => g.result.status === 'draw' && g.result.winnerIndex === null));
    assert.match(humanTournamentStatus(t).message, /Advanced by seeded tiebreak after three draws/);
    assert.deepEqual(validateTournament(t), t);
  }
  assert.ok(found);
});
