import test from 'node:test';
import assert from 'node:assert/strict';
import { SeasonController } from '../src/season/controller.js';
import { defaultField, validateSeason } from '../src/season/model.js';
import { STORAGE_KEY, loadSeason } from '../src/season/storage.js';
import { matchFixture, WIN, DRAW } from '../e2e/fixtures/match.js';
const failure = (status, code) => Object.assign(new Error(code), { response: { status, data: { code, error: code } } });
const flush = async () => { for (let i = 0; i < 4; i++) await new Promise(r => setImmediate(r)); };
function harness(sequence = WIN, override) {
  const values = new Map(), timers = new Map(), events = []; let timer = 0;
  const storage = override ?? { getItem: k => values.get(k) ?? null, setItem: (k, v) => { values.set(k, v); events.push(['persist', JSON.parse(v)]); } };
  const http = { starts: [], posts: [], active: null, inFlight: 0, maxFlight: 0 };
  http.fixture = () => ({ ...matchFixture(http.active.players, http.active.columns, http.active.id), rng_seed: http.active.seed });
  http.post = async (url, body) => {
    http.maxFlight = Math.max(http.maxFlight, ++http.inFlight);
    try {
      if (url.endsWith('start_game')) {
        if (http.active) assert.equal(body.replace_game_id, http.active.id);
        events.push(['start', body]); http.starts.push(body); http.active = { id: `game-${http.starts.length}`, players: [body.player1, body.player2], columns: [], seed: body.rng_seed };
      } else { assert.equal(body.game_id, http.active.id); assert.equal(body.revision, http.active.columns.length); assert.equal('column' in body, false); http.posts.push(body); http.active.columns.push(sequence[http.active.columns.length]); }
      return { data: http.fixture().state };
    } finally { http.inFlight--; }
  };
  http.get = async () => ({ data: http.fixture() });
  const c = new SeasonController(http, storage, fn => { timers.set(++timer, fn); return timer; }, id => timers.delete(id)); c.attach(); c.create(defaultField(4), 1234, 2);
  const tick = async () => { const first = timers.entries().next().value; if (first) { timers.delete(first[0]); first[1](); } await flush(); };
  return { c, http, timers, tick, storage, events };
}
async function finish(h, limit = 1000) { for (let i = 0; i < limit && h.c.state.mode !== 'paused'; i++) await h.tick(); assert.equal(h.c.state.mode, 'paused'); assert.equal(h.c.state.error, ''); }
for (const [mode, count] of [['game', 1], ['round', 2], ['season', 12]]) test(`Run ${mode} stops at exact boundary and uses sequential session replacement`, async () => {
  const h = harness(); assert.equal(h.http.starts.length, 0); h.c.run(mode); await finish(h);
  assert.equal(h.c.state.season.completedGames.length, count); assert.equal(h.http.starts.length, count); assert.equal(h.http.posts.length, count * 7); assert.equal(h.http.maxFlight, 1);
  assert.deepEqual(validateSeason(h.c.state.season), h.c.state.season); h.c.detach();
});
test('watch creates one session; Next move one ply; double click locked; Pause settles in flight', async () => {
  const h = harness(); await h.c.watch(); assert.equal(h.http.starts.length, 1); assert.equal(h.http.posts.length, 0);
  const original = h.http.post; let release;
  h.http.post = (url, body, options) => { assert.equal(options?.signal, undefined); return new Promise(r => { release = async () => r(await original(url, body)); }); };
  const p = h.c.nextMove(); void h.c.nextMove(); assert.equal(h.c.state.busy, true); h.c.pause(); release(); await p;
  assert.equal(h.http.posts.length, 1); assert.deepEqual(h.c.state.season.active.columns, [0]); assert.equal(h.timers.size, 0); h.c.detach();
});
test('Run round from midway completes only remaining games in current round', async () => {
  const h = harness(); h.c.run('game'); await finish(h); h.c.run('round'); await finish(h); assert.equal(h.http.starts.length, 2); assert.equal(h.c.state.season.schedule[2].round, 2); h.c.detach();
});
test('draw counts once and advances immediately without rematch', async () => {
  const h = harness(DRAW); h.c.run('game'); await finish(h); assert.equal(h.http.starts.length, 1); assert.equal(h.http.posts.length, 42); assert.equal(h.c.state.season.currentGameIndex, 1); h.c.run('game'); await finish(h); assert.equal(h.http.starts.length, 2); h.c.detach();
});
for (const committed of [true, false]) test(`lost move / agent_busy (${committed}) reconciles once, pauses, explicit continuation`, async () => {
  const h = harness(); await h.c.watch(); const post = h.http.post; let tries = 0;
  h.http.post = async (u, b) => { tries++; if (committed) await post(u, b); throw committed ? new Error('lost') : failure(503, 'agent_busy'); };
  h.c.run('season'); await h.tick(); assert.equal(tries, 1); assert.equal(h.c.state.mode, 'paused'); assert.equal(h.c.state.season.active.columns.length, Number(committed)); await h.tick(); assert.equal(tries, 1);
  await h.c.refresh(); h.http.post = post; await h.c.nextMove(); assert.equal(h.c.state.season.active.columns.length, Number(committed) + 1); h.c.detach();
});
test('expired session preserves completed results; explicit seeded restart only current fixture', async () => {
  const h = harness(); h.c.run('game'); await finish(h); await h.c.watch(); await h.c.nextMove();
  const results = structuredClone(h.c.state.season.completedGames), old = h.http.starts.at(-1), get = h.http.get;
  h.http.get = async () => { throw failure(404, 'session_not_found'); }; await h.c.refresh(); assert.equal(h.c.state.season.active.status, 'interrupted'); assert.deepEqual(validateSeason(h.c.state.season), h.c.state.season);
  h.c.run('season'); await h.tick(); assert.equal(h.http.starts.length, 2); h.http.get = get; await h.c.restart();
  assert.equal(h.http.starts.at(-1).rng_seed, old.rng_seed); assert.deepEqual([h.http.starts.at(-1).player1, h.http.starts.at(-1).player2], [old.player1, old.player2]); assert.deepEqual(h.c.state.season.completedGames, results); assert.equal(h.c.state.season.active.columns.length, 0); h.c.detach();
});
test('lost start never retries automatically, including reload', async () => {
  const h = harness(); const post = h.http.post; h.http.post = async (u, b) => { await post(u, b); throw new Error('lost start'); }; await h.c.watch();
  assert.equal(h.http.starts.length, 1); assert.equal(h.c.state.season.active.status, 'interrupted'); h.c.detach();
  const c = new SeasonController(h.http, h.storage); c.attach(); await flush(); assert.equal(h.http.starts.length, 1); assert.equal(c.state.mode, 'paused');
  h.http.active = null; h.http.post = post; await c.restart(); assert.equal(h.http.starts[0].rng_seed, h.http.starts[1].rng_seed); c.detach();
});
test('reload reconciles history, stays paused and completed game replay survives replacement', async () => {
  const h = harness(); h.c.run('game'); await finish(h); await h.c.watch(); await h.c.nextMove(); h.c.detach();
  const c = new SeasonController(h.http, h.storage); c.attach(); await flush(); assert.equal(c.state.mode, 'paused'); assert.equal(c.state.uncertain, false); assert.equal(c.state.season.active.columns.length, 1); assert.equal(c.state.season.completedGames.length, 1); assert.ok(!h.storage.getItem(STORAGE_KEY).includes('board')); c.detach();
});
test('terminal columns persisted before crash complete by GET without POST', async () => {
  const h = harness(); await h.c.watch(); const s = structuredClone(h.c.state.season); s.active.columns = [...WIN]; h.http.active.columns = [...WIN]; h.storage.setItem(STORAGE_KEY, JSON.stringify(s)); h.c.detach();
  const c = new SeasonController(h.http, h.storage); c.attach(); await flush(); assert.equal(c.state.season.completedGames.length, 1); assert.equal(h.http.posts.length, 0); c.detach();
});
test('in-flight route detach preserves lock and persistence; no storage runs in memory', async () => {
  const h = harness(WIN, { getItem: () => null, setItem: () => { throw new Error('quota'); } }); await h.c.watch(); const post = h.http.post; let release;
  h.http.post = (u, b) => new Promise(r => { release = async () => r(await post(u, b)); }); h.c.run('season'); await h.tick(); h.c.detach(); release(); await flush(); assert.equal(h.c.state.season.active.columns.length, 1); assert.equal(h.c.state.storageAvailable, false); assert.equal(h.timers.size, 0);
});
test('result is persisted before every following start', async () => {
  const h = harness(); h.c.run('season'); await finish(h);
  let count = 0; for (let i = 0; i < h.events.length; i++) if (h.events[i][0] === 'start') { if (count) assert.ok(h.events.slice(0, i).some(e => e[0] === 'persist' && e[1].completedGames.length === count)); count++; }
  assert.equal(loadSeason(h.storage).season.status, 'complete'); h.c.detach();
});
test('wrong history seed or revised validated columns locks execution until a correct GET', async () => {
  const h = harness(); await h.c.watch(); await h.c.nextMove(); const get = h.http.get;
  h.http.get = async () => ({ data: { ...h.http.fixture(), rng_seed: 999 } }); await h.c.nextMove(); assert.equal(h.c.state.uncertain, true); const posts = h.http.posts.length; await h.c.nextMove(); assert.equal(h.http.posts.length, posts); h.http.get = get; await h.c.refresh(); assert.equal(h.c.state.error, ''); assert.equal(h.c.state.season.active.columns.length, 2); h.c.detach();
});
test('maximum 12-player/8-games season executes 528 mocked games with one owner', async () => {
  const h = harness(); h.c.create(defaultField(12), 1234, 8); h.c.run('season'); await finish(h, 5000);
  assert.equal(h.http.starts.length, 528); assert.equal(h.http.posts.length, 3696); assert.equal(h.http.maxFlight, 1); assert.equal(loadSeason(h.storage).season.status, 'complete'); h.c.detach();
});
test('start persisted before receiving an ID hydrates interrupted without a POST', async () => {
  const h = harness(), s = structuredClone(h.c.state.season), f = s.schedule[0];
  s.active = { fixtureId: f.fixtureId, gameId: null, columns: [], status: 'starting' }; f.status = 'active'; h.storage.setItem(STORAGE_KEY, JSON.stringify(s)); h.c.detach();
  const c = new SeasonController(h.http, h.storage); c.attach(); await flush(); assert.equal(c.state.season.active.status, 'interrupted'); assert.equal(h.http.starts.length, 0); c.detach();
});
test('lost terminal response records once and permits explicit continuation without an active session', async () => {
  const h = harness(); await h.c.watch(); for (let i = 0; i < 6; i++) await h.c.nextMove();
  const post = h.http.post; h.http.post = async (u, b) => { await post(u, b); throw new Error('lost terminal response'); };
  await h.c.nextMove(); assert.equal(h.c.state.season.completedGames.length, 1); assert.equal(h.c.state.season.active, null); assert.equal(h.c.state.uncertain, false); assert.equal(h.c.state.error, 'lost terminal response');
  h.c.run('season'); await h.tick(); assert.equal(h.http.starts.length, 1);
  await h.c.refresh(); assert.equal(h.c.state.error, ''); h.http.post = post; h.c.run('game'); await finish(h); assert.equal(h.http.starts.length, 2); assert.equal(h.c.state.season.completedGames.length, 2); h.c.detach();
});
test('authoritative history cannot rewrite any previously accepted column', async () => {
  const h = harness(); await h.c.watch(); await h.c.nextMove(); const get = h.http.get;
  h.http.get = async () => ({ data: { ...matchFixture(h.http.active.players, [2], h.http.active.id), rng_seed: h.http.active.seed } });
  await h.c.refresh(); assert.equal(h.c.state.uncertain, true); assert.deepEqual(h.c.state.season.active.columns, [0]); assert.match(h.c.state.error, /Previously validated moves changed/);
  await h.c.nextMove(); assert.equal(h.http.posts.length, 1); h.http.get = get; await h.c.refresh(); assert.equal(h.c.state.uncertain, false); h.c.detach();
});
