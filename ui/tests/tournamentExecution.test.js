import { test } from 'node:test';
import assert from 'node:assert/strict';
import { TournamentController } from '../src/tournament/controller.js';
import { createTournament, defaultField, allMatchups, validateTournament, gamePlan, nextMatchup, matchupHasHuman, humanTournamentStatus } from '../src/tournament/model.js';
import { STORAGE_KEY, loadTournament } from '../src/tournament/storage.js';
import { matchFixture, WIN, DRAW } from '../e2e/fixtures/match.js';
const fail = (status, code) => Object.assign(new Error(code), { response: { status, data: { code, error: code } } });
function harness(sequence = WIN, storageOverride) {
  const values = new Map(), events = [], timers = new Map(); let timerId = 0;
  const storage = storageOverride ?? { getItem: key => values.get(key) ?? null, setItem: (key, value) => { values.set(key, value); events.push(['persist', JSON.parse(value)]); } };
  const s = { starts: [], posts: [], reads: 0, active: null };
  s.fixture = () => ({ ...matchFixture(s.active.players, s.active.columns, s.active.id), rng_seed: s.active.seed });
  s.post = async (url, body) => {
    if (url.endsWith('start_game')) {
      events.push(['start', body]); s.starts.push(body);
      if (s.active) assert.equal(body.replace_game_id, s.active.id);
      s.active = { id: `game-${s.starts.length}`, seed: body.rng_seed, players: [body.player1, body.player2], columns: [] };
    } else {
      s.posts.push(body); assert.equal(body.game_id, s.active.id); assert.equal(body.revision, s.active.columns.length);
      const human = s.active.players[s.active.columns.length % 2].type === 'human';
      assert.equal('column' in body, human); s.active.columns.push(human ? body.column : sequence[s.active.columns.length]);
    }
    return { data: s.fixture().state };
  };
  s.get = async url => { s.reads++; if (!s.active || !url.includes(s.active.id)) throw fail(404, 'session_not_found'); return { data: s.fixture() }; };
  const c = new TournamentController(s, storage, fn => { timers.set(++timerId, fn); return timerId; }, id => timers.delete(id)); c.attach();
  const tick = async () => { const first = timers.entries().next().value; if (first) { timers.delete(first[0]); first[1](); } await flush(); };
  return { s, c, storage, events, tick, timers };
}
const flush = async () => { for (let i = 0; i < 4; i++) await new Promise(r => setImmediate(r)); };
const field = n => Array(n).fill({ type: 'random' });
async function finish(h, limit = 2000) { for (let i = 0; i < limit && h.c.state.mode !== 'paused'; i++) await h.tick(); assert.equal(h.c.state.mode, 'paused'); }
for (const [mode, count] of [['matchup', 1], ['round', 4], ['tournament', 7]]) test(`Run ${mode} scope is sequential and bounded`, async () => {
  const h = harness(); h.c.create(field(8), 1234); h.c.run(mode); await finish(h);
  assert.equal(allMatchups(h.c.state.tournament).filter(m => m.status === 'complete').length, count);
  assert.equal(h.s.starts.length, count); assert.equal(h.s.posts.length, 7 * count);
  assert.equal(h.c.state.tournament.status, mode === 'tournament' ? 'complete' : 'paused');
  assert.deepEqual(validateTournament(h.c.state.tournament), h.c.state.tournament); h.c.detach();
});
test('creation allocates no sessions; watch paused; Next move serialized one POST', async () => {
  const h = harness(); h.c.create(defaultField(8), 1); assert.equal(h.s.starts.length, 0);
  await h.c.watch(); assert.equal(h.s.posts.length, 0);
  const original = h.s.post; let release;
  h.s.post = (url, body) => new Promise(resolve => { release = async () => resolve(await original(url, body)); });
  const first = h.c.nextMove(); void h.c.nextMove(); assert.equal(h.c.state.busy, true); release(); await first;
  assert.equal(h.s.posts.length, 1); assert.equal(h.c.state.tournament.active.columns.length, 1); h.c.detach();
});
test('in-flight Pause settles/persists without abort or another move', async () => {
  const h = harness(); h.c.create(field(8), 2); await h.c.watch(); const original = h.s.post; let release;
  h.s.post = (url, body, options) => { assert.equal(options?.signal, undefined); return new Promise(resolve => { release = async () => resolve(await original(url, body)); }); };
  h.c.run('tournament'); await h.tick(); assert.equal(h.c.state.busy, true);
  h.c.pause(); release(); await flush(); await h.tick();
  assert.equal(h.s.posts.length, 1); assert.equal(h.c.state.tournament.active.columns.length, 1); assert.equal(h.timers.size, 0); h.c.detach();
});
for (const committed of [true, false]) test(`lost response / agent_busy (${committed}) reconciles and requires explicit continuation`, async () => {
  const h = harness(); h.c.create(field(8), 2); await h.c.watch(); const original = h.s.post; let tries = 0;
  h.s.post = async (url, body) => { tries++; if (committed) await original(url, body); throw committed ? new Error('lost response') : fail(503, 'agent_busy'); };
  h.c.run('tournament'); await h.tick();
  assert.equal(tries, 1); assert.equal(h.c.state.mode, 'paused'); assert.equal(h.c.state.uncertain, false);
  assert.equal(h.c.state.tournament.active.columns.length, Number(committed)); await h.tick(); assert.equal(tries, 1);
  await h.c.refresh(); assert.equal(h.c.state.error, ''); h.s.post = original; await h.c.nextMove();
  assert.equal(h.c.state.tournament.active.columns.length, Number(committed) + 1); h.c.detach();
});
test('lost start persisted interrupted, never repeated automatically', async () => {
  const h = harness(); h.c.create(field(8), 4); const original = h.s.post; let tries = 0;
  h.s.post = async (url, body) => { tries++; await original(url, body); throw new Error('lost'); };
  await h.c.watch(); assert.equal(tries, 1); assert.equal(h.c.state.tournament.active.status, 'interrupted');
  await h.tick(); await h.c.nextMove(); assert.equal(tries, 1);
  assert.equal(loadTournament(h.storage).tournament.active.status, 'interrupted');
  h.s.active = null; h.s.post = original; await h.c.restart(); assert.equal(h.s.starts.length, 2);
  assert.equal(h.s.starts[0].rng_seed, h.s.starts[1].rng_seed); h.c.detach();
});
test('reload reconciles existing active game, stays paused, preserves completed records', async () => {
  const h = harness(); h.c.create(field(8), 5); h.c.run('matchup'); await finish(h);
  await h.c.watch(); await h.c.nextMove(); h.c.detach();
  const c = new TournamentController(h.s, h.storage); c.attach(); await flush();
  assert.equal(c.state.mode, 'paused'); assert.equal(c.state.uncertain, false);
  assert.equal(c.state.tournament.active.columns.length, 1); assert.equal(c.state.tournament.rounds[0][0].games.length, 1);
  assert.ok(!h.storage.getItem(STORAGE_KEY).includes('board_')); c.detach();
});
test('expiry preserves progress; explicit restart uses same seed/config and known replacement ID', async () => {
  const h = harness(); h.c.create(field(8), 6); h.c.run('matchup'); await finish(h); await h.c.watch(); await h.c.nextMove();
  const old = h.s.starts.at(-1), previous = structuredClone(h.c.state.tournament.rounds[0][0]);
  h.s.get = async () => { throw fail(404, 'session_not_found'); }; await h.c.refresh();
  assert.equal(h.c.state.tournament.active.status, 'interrupted'); assert.deepEqual(h.c.state.tournament.rounds[0][0], previous);
  const starts = h.s.starts.length; h.c.run('tournament'); await h.tick(); assert.equal(h.s.starts.length, starts);
  h.s.get = async () => ({ data: h.s.fixture() }); await h.c.restart();
  const restarted = h.s.starts.at(-1); assert.equal(restarted.rng_seed, old.rng_seed);
  assert.deepEqual([restarted.player1, restarted.player2], [old.player1, old.player2]);
  assert.equal(h.c.state.tournament.active.columns.length, 0); assert.deepEqual(h.c.state.tournament.rounds[0][0], previous); h.c.detach();
});
test('wrong seed locks execution until explicit history GET succeeds', async () => {
  const h = harness(); h.c.create(field(8), 3); await h.c.watch(); const original = h.s.get;
  h.s.get = async () => ({ data: { ...h.s.fixture(), rng_seed: 12 } }); await h.c.nextMove();
  assert.equal(h.c.state.uncertain, true); const plies = h.s.posts.length; await h.c.nextMove(); assert.equal(h.s.posts.length, plies);
  h.s.get = original; await h.c.refresh(); assert.equal(h.c.state.tournament.active.columns.length, 1); h.c.detach();
});
test('completion persisted before next start; replacement preserves local replay', async () => {
  const h = harness(); h.c.create(field(8), 7); h.c.run('round'); await finish(h);
  const starts = h.events.map((e, i) => e[0] === 'start' ? i : -1).filter(i => i >= 0);
  starts.slice(1).forEach((index, n) => {
    const saved = h.events.slice(0, index).filter(e => e[0] === 'persist');
    assert.ok(saved.some(e => allMatchups(e[1]).filter(m => m.status === 'complete').length === n + 1));
  });
  assert.equal(loadTournament(h.storage).tournament.rounds[0][0].games[0].columns.length, 7);
  assert.equal(h.s.active.id, 'game-4'); h.c.detach();
});
test('draw orchestration completes three games then explicit seeded tiebreak', async () => {
  const h = harness(DRAW); h.c.create(field(8), 8); h.c.run('matchup'); await finish(h);
  assert.equal(h.s.starts.length, 3); assert.equal(h.s.posts.length, 126);
  const m = h.c.state.tournament.rounds[0][0]; assert.equal(m.resolution, 'seeded_draw_tiebreak');
  assert.ok(m.games.every(g => g.result.status === 'draw')); h.c.detach();
});
test('64-player execution completes 63 matches with cheap synthetic games', async () => {
  const h = harness(); h.c.create(field(64), 9); h.c.run('tournament'); await finish(h);
  assert.equal(h.s.starts.length, 63); assert.equal(h.s.posts.length, 441);
  assert.equal(h.c.state.tournament.rounds.length, 6); assert.ok(h.c.state.tournament.championEntrantId); h.c.detach();
});
test('no storage works in memory; unmount settles in-flight mutation and stops scheduling', async () => {
  const h = harness(WIN, { getItem() { return null; }, setItem() { throw new Error('quota'); } });
  h.c.create(field(8), 9); await h.c.watch(); const original = h.s.post; let release;
  h.s.post = (url, body) => new Promise(resolve => { release = async () => resolve(await original(url, body)); });
  h.c.run('tournament'); await h.tick(); h.c.detach(); release(); await flush();
  assert.equal(h.c.state.tournament.active.columns.length, 1); assert.equal(h.c.state.storageAvailable, false); assert.equal(h.timers.size, 0);
});
test('Autoplay runs only current game, stopping at a draw before rematch', async () => {
  const h = harness(DRAW); h.c.create(field(8), 10); await h.c.watch(); h.c.run('game'); await finish(h);
  assert.equal(h.s.starts.length, 1); assert.equal(h.s.posts.length, 42);
  assert.equal(h.c.state.tournament.rounds[0][0].games.length, 1);
  assert.equal(h.c.state.tournament.rounds[0][0].status, 'pending'); h.c.detach();
});
test('crash after persisting terminal columns resumes by GET and advances without another POST', async () => {
  const h = harness(); h.c.create(field(8), 11); await h.c.watch();
  const t = structuredClone(h.c.state.tournament); t.active.columns = [...WIN]; h.s.active.columns = [...WIN];
  h.storage.setItem(STORAGE_KEY, JSON.stringify(t)); h.c.detach();
  const c = new TournamentController(h.s, h.storage); c.attach(); await flush();
  assert.equal(c.state.tournament.active, null); assert.equal(c.state.tournament.rounds[0][0].status, 'complete');
  assert.equal(h.s.posts.length, 0); assert.equal(c.state.mode, 'paused'); c.detach();
});
test('rehydration of a start with no received ID requires explicit restart without POST', async () => {
  const h = harness(); h.c.create(field(8), 12); const t = structuredClone(h.c.state.tournament);
  t.active = { matchupId: 'r1-m1', gameNumber: 1, gameId: null, status: 'starting', columns: [] }; t.rounds[0][0].status = 'active';
  h.storage.setItem(STORAGE_KEY, JSON.stringify(t)); h.c.detach();
  const c = new TournamentController(h.s, h.storage); c.attach(); await flush();
  assert.equal(c.state.tournament.active.status, 'interrupted'); assert.equal(h.s.starts.length, 0); c.detach();
});

// Human policy below is test-only; production always waits for the user's column.
function humanField(seed = 1234, color = 0, index = 0) {
  const f = field(8);
  // Color assignment is independent of type; choose the appropriate seeded slot.
  const baseline = createTournament(f, seed);
  const id = gamePlan(baseline, baseline.rounds[0][index]).playerEntrantIds[color];
  f[Number(id.split('-')[1]) - 1] = { type: 'human' };
  return f;
}
for (const mode of ['matchup', 'round', 'tournament']) test(`Run ${mode} never starts the next Human pairing or skips it`, async () => {
  const h = harness(); h.c.create(humanField(1234, 0, 2), 1234);
  if (mode === 'matchup') { h.c.run(mode); await finish(h); h.c.run(mode); await finish(h); }
  else { h.c.run(mode); await finish(h); }
  h.c.run(mode); await finish(h);
  assert.equal(h.s.starts.length, 2); assert.equal(h.s.posts.length, 14);
  assert.equal(h.c.state.waitingForHuman, true); assert.equal(h.c.state.tournament.active, null);
  assert.equal(matchupHasHuman(h.c.state.tournament, nextMatchup(h.c.state.tournament)), true); h.c.detach();
});
for (const color of [0, 1]) test(`explicit Human ${color ? 'Yellow' : 'Red'} start, AI opener, waiting autoplay, paused/legal/replay/double-click lifecycle`, async () => {
  const h = harness(); h.c.create(humanField(1234, color), 1234); h.c.run('tournament'); await h.tick();
  assert.equal(h.s.starts.length, 0);
  await Promise.all([h.c.watch(), h.c.watch()]); assert.equal(h.s.starts.length, 1);
  assert.equal(h.c.state.tournament.active.columns.length, 0); assert.equal(h.c.humanTurn(), color === 0);
  if (color === 1) { h.c.run('game'); await h.tick(); assert.equal(h.s.posts.length, 1); assert.equal(h.c.humanTurn(), true); }
  h.c.run('game'); await h.tick(); const before = h.s.posts.length; assert.equal(h.timers.size, 0);
  await h.c.nextMove(); await h.tick(); assert.equal(h.s.posts.length, before);
  h.c.review(); await h.c.humanMove(0); assert.equal(h.s.posts.length, before); h.c.returnLive();
  await h.c.humanMove(99); assert.equal(h.s.posts.length, before);
  const column = WIN[h.s.active.columns.length];
  const original = h.s.post; let release;
  h.s.post = (url, body) => new Promise(resolve => { release = async () => resolve(await original(url, body)); });
  const move = h.c.humanMove(column); void h.c.humanMove(column); assert.equal(h.c.state.busy, true);
  release(); await move; h.s.post = original; await h.tick(); // paused: no AI followup
  assert.equal(h.s.posts.length, before + 1); assert.equal(h.s.posts.at(-1).column, column);
  assert.equal(h.c.humanTurn(), false); assert.equal(h.timers.size, 0);
  await h.c.nextMove(); assert.equal(h.s.posts.length, before + 2); assert.equal('column' in h.s.posts.at(-1), false);
  h.c.run('game'); await h.c.humanMove(WIN[h.s.active.columns.length]); await h.tick();
  assert.equal(h.c.humanTurn(), true); assert.equal(h.timers.size, 0); h.c.pause(); h.c.detach();
});
async function humanGame(h, sequence = WIN, mode = 'tournament') {
  h.c.run(mode);
  for (let i = 0; i < 60 && h.c.state.tournament.active; i++) {
    if (h.c.humanTurn()) await h.c.humanMove(sequence[h.s.active.columns.length]);
    else await h.tick();
  }
  assert.equal(h.c.state.error, ''); assert.equal(h.c.state.tournament.active, null);
  assert.equal(h.c.state.mode, 'paused');
}
for (const color of [0, 1]) test(`Human ${color ? 'loss' : 'win'} pauses broader scope; elimination allows tournament finish`, async () => {
  const h = harness(); h.c.create(humanField(1234, color), 1234); await h.c.watch(); await humanGame(h);
  assert.equal(h.s.starts.length, 1); assert.equal(humanTournamentStatus(h.c.state.tournament).kind, color ? 'eliminated' : 'advanced');
  assert.deepEqual(loadTournament(h.storage).tournament, h.c.state.tournament);
  if (color) { h.c.run('tournament'); await finish(h); assert.equal(h.s.starts.length, 7); assert.equal(h.c.state.tournament.status, 'complete'); }
  h.c.detach();
});
test('Human draws persist and each rematch requires explicit start, with swapped/separate colors and tiebreak', async () => {
  const h = harness(DRAW); h.c.create(humanField(), 1234);
  for (let number = 1; number <= 3; number++) {
    h.c.run('tournament'); await h.tick(); assert.equal(h.s.starts.length, number - 1);
    await h.c.watch(); await humanGame(h, DRAW);
    assert.equal(h.s.starts.length, number); assert.equal(h.c.state.tournament.rounds[0][0].games.length, number);
    assert.ok(h.c.state.tournament.rounds[0][0].games.every(g => g.result.status === 'draw'));
  }
  assert.deepEqual([h.s.starts[1].player1, h.s.starts[1].player2], [h.s.starts[0].player2, h.s.starts[0].player1]);
  assert.equal(h.c.state.tournament.rounds[0][0].resolution, 'seeded_draw_tiebreak'); h.c.detach();
});
for (const human of [true, false]) test(`lost ${human ? 'Human' : 'AI'} response during Human matchup reconciles once without resending`, async () => {
  const h = harness(); h.c.create(humanField(), 1234); await h.c.watch();
  if (!human) await h.c.humanMove(0);
  const original = h.s.post; let tries = 0;
  h.s.post = async (url, body) => { tries++; await original(url, body); throw new Error('lost response after commit'); };
  if (human) await h.c.humanMove(0); else await h.c.nextMove();
  assert.equal(tries, 1); assert.equal(h.c.state.tournament.active.columns.length, human ? 1 : 2);
  assert.equal(h.c.state.mode, 'paused'); assert.equal(h.c.state.uncertain, false); await h.tick(); assert.equal(tries, 1);
  h.s.post = original; await h.c.refresh(); assert.equal(h.c.state.error, ''); h.c.detach();
});
for (const revision of [0, 1, 2]) test(`Human reload at revision ${revision} restores correct paused turn`, async () => {
  const h = harness(); h.c.create(humanField(), 1234); await h.c.watch();
  if (revision > 0) await h.c.humanMove(0); if (revision > 1) await h.c.nextMove(); h.c.detach();
  const c = new TournamentController(h.s, h.storage); c.attach(); await flush();
  assert.equal(c.state.mode, 'paused'); assert.equal(c.state.uncertain, false);
  assert.equal(c.humanTurn(), revision % 2 === 0); assert.equal(c.state.tournament.active.columns.length, revision); c.detach();
});
for (const number of [1, 2, 3]) test(`expired Human Game ${number} preserves completed draws and restarts only current game from zero`, async () => {
  const h = harness(DRAW); h.c.create(humanField(), 1234);
  for (let i = 1; i < number; i++) { await h.c.watch(); await humanGame(h, DRAW); }
  await h.c.watch(); const plan = h.s.starts.at(-1), saved = structuredClone(h.c.state.tournament.rounds[0][0].games);
  if (h.c.humanTurn()) await h.c.humanMove(DRAW[0]); else await h.c.nextMove();
  const get = h.s.get; h.s.get = async () => { throw fail(404, 'session_not_found'); }; await h.c.refresh();
  assert.equal(h.c.state.tournament.active.status, 'interrupted'); assert.match(h.c.state.error, /restart from the beginning/);
  h.c.detach(); const reloaded = new TournamentController(h.s, h.storage); reloaded.attach(); await flush();
  assert.equal(reloaded.state.tournament.active.status, 'interrupted'); assert.equal(reloaded.state.mode, 'paused');
  h.s.get = get; await reloaded.restart();
  assert.equal(reloaded.state.tournament.active.columns.length, 0); assert.equal(reloaded.state.tournament.active.gameNumber, number);
  assert.equal(h.s.starts.at(-1).rng_seed, plan.rng_seed);
  assert.deepEqual([h.s.starts.at(-1).player1, h.s.starts.at(-1).player2], [plan.player1, plan.player2]);
  assert.deepEqual(reloaded.state.tournament.rounds[0][0].games, saved); reloaded.detach();
});
test('full Human column is rejected locally while other columns remain legal', async () => {
  const h = harness(DRAW); h.c.create(humanField(), 1234); await h.c.watch(); h.c.run('game');
  while (h.s.active.columns.length < 8) {
    if (h.c.humanTurn()) await h.c.humanMove(DRAW[h.s.active.columns.length]); else await h.tick();
  }
  assert.equal(h.c.humanTurn(), true); const before = h.s.posts.length;
  await h.c.humanMove(0); assert.equal(h.s.posts.length, before);
  await h.c.humanMove(1); assert.equal(h.s.posts.length, before + 1); h.c.detach();
});
