import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { alphaZeroDiscovery, validateAlphaZeroStatus, validateAlphaZeroProvenance, requireSameAlphaZero } from '../src/connect4/alphazero.js';
import { withBusyRetry } from '../src/connect4/busyRetry.js';
import { toPlayerPayload, competitorLabel } from '../src/connect4/competitorConfig.js';
import { samePlayers, validateMatchHistory } from '../src/connect4/matchRecord.js';
import { createSeason, compactHistory, recordGame, validateSeason, gamePlan } from '../src/season/model.js';
import { createTournament, validateTournament } from '../src/tournament/model.js';
import { createEvaluationExport, canonicalPayload } from '../src/evaluation/export.js';
import { verifyEvaluationExport } from '../src/evaluation/verify.js';
import { canonicalJSON } from '../src/evaluation/canonical.js';
import { gamesCSV, summaryCSV } from '../src/evaluation/csv.js';
import { SeasonController } from '../src/season/controller.js';
import { TournamentController } from '../src/tournament/controller.js';
import { matchFixture, WIN } from '../e2e/fixtures/match.js';
import { status, provenance, az } from './fixtures/alphazero.js';
const digest = async text => createHash('sha256').update(text).digest('hex');
const busy = code => Object.assign(new Error(code), { response: { status: 503, data: { code }, headers: {} } });

test('only complete, consistent, explicitly available AlphaZero status is accepted', () => {
  assert.deepEqual(validateAlphaZeroStatus(status), provenance);
  for (const mutate of [s => s.available = 'true', s => s.enabled = false, s => delete s.provenance,
    s => delete s.provenance.tie_break, s => s.model = { ...s.model, id: 'other' },
    s => s.provenance.search.simulations = 512, s => s.provenance.model.artifact_sha256 = 'bad',
    s => s.provenance.extra = true, s => s.options.tactical_guard.default = true]) {
    const s = structuredClone(status); mutate(s); assert.throws(() => validateAlphaZeroStatus(s));
  }
  assert.equal(validateAlphaZeroStatus({ type: 'alphazero', enabled: false, available: false }), null);
  assert.equal(validateAlphaZeroStatus({ type: 'alphazero', enabled: true, available: false }), null);
});
for (const response of [null, {}, '<html>', { type: 'alphazero', enabled: true, available: false }, { type: 'alphazero', enabled: false, available: false }]) {
  test(`discovery fails closed: ${JSON.stringify(response)}`, async () => {
    const store = alphaZeroDiscovery({ get: async () => ({ data: response }) });
    assert.equal((await store.refresh()).available, false);
  });
}
for (const error of [new Error('timeout'), { response: { status: 404 } }]) test('404/network failures are quiet; legacy execution is independent', async () => {
  const store = alphaZeroDiscovery({ get: async () => { throw error; } });
  assert.equal((await store.refresh()).available, false); assert.deepEqual(toPlayerPayload({ type: 'random' }), { type: 'random' });
});
test('cache avoids polling; refresh supersedes old responses and detects status changes', async () => {
  let calls = 0, release;
  const http = { get: async () => ++calls === 1 ? new Promise(resolve => { release = resolve; }) : { data: status } };
  const store = alphaZeroDiscovery(http), older = store.refresh();
  assert.equal((await store.refresh(true)).available, true);
  release({ data: { type: 'alphazero', enabled: false, available: false } }); await older;
  assert.equal(store.snapshot().available, true); await store.refresh(); assert.equal(calls, 2);
  http.get = async () => ({ data: { type: 'alphazero', enabled: true, available: false } });
  await store.refresh(true); assert.equal(store.snapshot().available, false);
});
test('configurations are boolean, distinct and carry no tunable search controls', () => {
  for (const guard of [false, true]) assert.deepEqual(toPlayerPayload(az(guard)), az(guard));
  for (const tactical_guard of ['true', 1, null]) assert.throws(() => toPlayerPayload({ type: 'alphazero', tactical_guard }));
  assert.throws(() => toPlayerPayload({ ...az(false), simulations: 256 }));
  assert.notEqual(competitorLabel(az(false)), competitorLabel(az(true)));
  assert.equal(samePlayers([az(false), az(true)], [az(true), az(false)]), false);
});
test('optional move evidence is accepted and guard/model/visit corruption is rejected', () => {
  const h = matchFixture([az(false), az(true)], [3, 2]); h.alphazero_provenance = provenance;
  h.moves.forEach((m, i) => { m.agent = h.players[i % 2]; m.alphazero = { model_id: provenance.model.id, simulations: 256, tactical_guard: Boolean(i), original_column: m.column, root_visits: [0, 0, 0, 256, 0, 0, 0], guard_reason: i ? 'UNCHANGED' : null }; });
  assert.equal(validateMatchHistory(h).moves.length, 2);
  for (const mutate of [h => h.moves[0].alphazero.root_visits[3]--, h => h.moves[0].agent.tactical_guard = true, h => h.moves[1].alphazero.model_id = 'other']) {
    const copy = structuredClone(h); mutate(copy); assert.throws(() => validateMatchHistory(copy));
  }
});
test('v3 partial and completed AlphaZero evidence self-verifies; identity/guard tampering cannot be rehashed into validity', async () => {
  let s = createSeason([az(false), az(true), { type: 'random' }, { type: 'negamax', depth: 1 }], 42, 2, undefined, false, provenance);
  const partial = await createEvaluationExport(s, { digest }); assert.equal(partial.schema_version, 3); assert.equal((await verifyEvaluationExport(partial, { digest })).ok, true);
  while (s.status !== 'complete') {
    const plan = gamePlan(s), h = matchFixture(plan.playerConfigs, WIN); h.rng_seed = plan.gameSeed; h.moves.forEach((m, i) => { if (m.agent.type === 'alphazero') m.agent = plan.playerConfigs[i % 2]; });
    if (plan.playerConfigs.some(p => p.type === 'alphazero')) h.alphazero_provenance = provenance;
    s = recordGame(s, compactHistory(s, h));
  }
  assert.equal(validateSeason(JSON.parse(JSON.stringify(s))).completedGames.length, 12);
  const artifact = await createEvaluationExport(s, { digest, exportedAt: '2026-10-14T16:00:00Z' });
  assert.equal((await verifyEvaluationExport(artifact, { digest })).ok, true);
  assert.equal(artifact.integrity.evidence_sha256, (await createEvaluationExport(s, { digest })).integrity.evidence_sha256);
  assert.match(gamesCSV(artifact), /alphazero_provenance_v1,red_tactical_guard,yellow_tactical_guard/);
  assert.match(summaryCSV(artifact), /alphazero_provenance_v1,tactical_guard/);
  for (const mutate of [a => a.schema_version = 1, a => delete a.evidence.alphazero_provenance,
    a => a.evidence.completed_games.find(g => g.alphazero_provenance).alphazero_provenance.model.id = 'other',
    a => a.evidence.entrants[0].config.tactical_guard = true]) {
    const copy = structuredClone(artifact); mutate(copy);
    copy.integrity.evidence_sha256 = await digest(canonicalJSON(canonicalPayload(copy)));
    assert.equal((await verifyEvaluationExport(copy, { digest })).ok, false);
  }
});
test('legacy canonical JSON/digests remain unchanged and verify', async () => {
  for (const name of ['evaluation-complete', 'evaluation-partial']) {
    const bytes = readFileSync(new URL(`./fixtures/${name}.json`, import.meta.url), 'utf8');
    const artifact = JSON.parse(bytes); assert.equal((await verifyEvaluationExport(artifact, { digest })).ok, true);
    assert.equal(readFileSync(new URL(`./fixtures/${name}.json`, import.meta.url), 'utf8'), bytes);
  }
});
test('versioned tournament duplicates retain guard identity; downgraded storage is rejected', () => {
  const t = createTournament(Array.from({ length: 8 }, (_, i) => az(Boolean(i % 2))), 42, undefined, false, provenance);
  assert.deepEqual(validateTournament(t), t); assert.equal(t.version, 2);
  assert.throws(() => validateTournament({ ...t, version: 1 }));
  assert.throws(() => createTournament(Array(8).fill(az(false)), 42));
  const changed = structuredClone(provenance); changed.model.artifact_sha256 = 'a'.repeat(64);
  assert.throws(() => requireSameAlphaZero(provenance, changed), /changed/);
});
test('bounded busy retries respect Retry-After, cancellation, and never repeat ambiguous POSTs', async () => {
  let calls = 0; const delays = [];
  assert.equal(await withBusyRetry(async () => { if (++calls < 3) { const e = busy('agent_busy'); e.response.headers['retry-after'] = '1'; throw e; } return 7; }, { wait: async d => delays.push(d) }), 7);
  assert.deepEqual(delays, [1000, 1000]);
  for (const error of [new Error('timeout'), busy('agent_failed'), { response: { status: 409, data: { code: 'stale_revision' } } }]) {
    calls = 0; await assert.rejects(withBusyRetry(async () => { calls++; throw error; })); assert.equal(calls, 1);
  }
  calls = 0; await assert.rejects(withBusyRetry(async () => { calls++; throw busy('agent_busy'); }, { canRetry: () => false, wait: async () => {} }), e => e.retryCancelled); assert.equal(calls, 1);
  calls = 0; await assert.rejects(withBusyRetry(async () => { calls++; throw busy('game_busy'); }, { wait: async () => {} })); assert.equal(calls, 3);
});
for (const Controller of [SeasonController, TournamentController]) test(`${Controller.name} blocks changed model before POST and preserves historical pinned state`, async () => {
  let posts = 0; const changed = structuredClone(status); changed.provenance.model.id = 'changed'; changed.model = changed.provenance.model;
  let currentStatus = status;
  const http = { get: async () => ({ data: currentStatus }), post: async () => { posts++; throw new Error('must not send'); } };
  await alphaZeroDiscovery(http).refresh(); const c = new Controller(http, null); c.attach();
  if (Controller === SeasonController) c.create([az(false), az(true), { type: 'random' }, { type: 'random' }], 42, 2);
  else c.create(Array(8).fill(az(false)), 42);
  currentStatus = changed; await c.watch(); assert.equal(posts, 0); assert.match(c.state.error, /changed/);
  const saved = c.state.season ?? c.state.tournament; assert.equal(saved.alphazeroProvenance.model.id, provenance.model.id); c.detach();
});

for (const Controller of [SeasonController, TournamentController]) test(`${Controller.name} mismatched start history permits explicit original-model restart without losing completed results`, async () => {
  let starts = 0, players, badHistory = true;
  const http = {
    post: async (url, body) => { assert.ok(url.endsWith('start_game')); starts++; players = [body.player1, body.player2]; return { data: matchFixture(players, [], `game-${starts}`).state }; },
    get: async url => {
      if (url.endsWith('agents/alphazero')) return { data: status };
      const h = matchFixture(players, [], `game-${starts}`); h.alphazero_provenance = structuredClone(provenance);
      if (badHistory) h.alphazero_provenance.model.id = 'different-at-start';
      const plan = Controller === SeasonController ? (await import('../src/season/model.js')).gamePlan(c.state.season)
        : (await import('../src/tournament/model.js')).gamePlan(c.state.tournament, (await import('../src/tournament/model.js')).nextMatchup(c.state.tournament));
      h.rng_seed = plan.gameSeed; return { data: h };
    },
  };
  await alphaZeroDiscovery(http).refresh(); const c = new Controller(http, null); c.attach();
  if (Controller === SeasonController) c.create(Array.from({ length: 4 }, (_, i) => az(Boolean(i % 2))), 42, 2);
  else c.create(Array(8).fill(az(false)), 42);
  await c.watch(); const interrupted = c.state.season ?? c.state.tournament;
  assert.equal(interrupted.active.status, 'interrupted'); assert.equal(c.state.uncertain, true); assert.match(c.state.error, /explicitly restart/);
  badHistory = false; await c.restart(); const restored = c.state.season ?? c.state.tournament;
  assert.equal(starts, 2); assert.equal(restored.active.status, 'running'); assert.equal(c.state.uncertain, false);
  assert.deepEqual(restored.alphazeroProvenance, provenance); c.detach();
});
