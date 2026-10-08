import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { webcrypto } from 'node:crypto';
import { canonicalJSON, browserSHA256, evidenceDigest } from '../src/evaluation/canonical.js';
import { createEvaluationExport, canonicalPayload, seasonEvidence } from '../src/evaluation/export.js';
import { verifyEvaluationExport } from '../src/evaluation/verify.js';
import { gamesCSV, summaryCSV, csvCell } from '../src/evaluation/csv.js';
import { downloadText } from '../src/evaluation/download.js';
import { nodeSHA256 } from '../scripts/evaluation-node.mjs';
import { evaluationSeason, DRAW, WIN, backend } from './fixtures/evaluation.js';
import { compactHistory, gamePlan } from '../src/season/model.js';
import { matchFixture } from '../e2e/fixtures/match.js';
import { seasonStandings, seasonRatings, pairwiseResults } from '../src/season/analytics.js';
const options = { digest: nodeSHA256, exportedAt: '2026-10-07T12:00:00.000Z' };
const reverseKeys = value => Array.isArray(value) ? value.map(reverseKeys) : value && typeof value === 'object' ? Object.fromEntries(Object.entries(value).reverse().map(([k, v]) => [k, reverseKeys(v)])) : value;

test('canonical keys sorted recursively; arrays and Unicode preserved; full precision number round trip', async () => {
  const a = { z: ['🟡', 'é', 'e\u0301', '\n"\\', 4294967295, 1512.1234567890123, 1e-7], a: { b: -0, a: true }, empty: null };
  assert.equal(canonicalJSON(a), canonicalJSON(reverseKeys(a)));
  assert.deepEqual(JSON.parse(canonicalJSON(a)).z, a.z);
  assert.equal(canonicalJSON({ z: 1, a: 2 }), '{"a":2,"z":1}');
  assert.equal(await browserSHA256('abc', webcrypto), 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad');
  assert.equal(await evidenceDigest(a), nodeSHA256(canonicalJSON(a)));
});
test('canonicalization rejects unsupported values, sparse arrays, cycles, symbols and accessors', () => {
  const cyclic = {}; cyclic.self = cyclic;
  const getter = Object.defineProperty({}, 'a', { enumerable: true, get() { throw new Error('Must not invoke getter'); } });
  let invoked = false;
  const arrayGetter = Object.defineProperty([1], '0', { enumerable: true, get() { invoked = true; return 1; } });
  const hidden = Object.defineProperty({}, 'a', { value: 1 });
  class ArraySubclass extends Array {}
  for (const value of [undefined, { a: undefined }, NaN, Infinity, 1n, new Date(), new Map(), new Set(), () => 1, Symbol('x'), [undefined], Array(2), cyclic, getter, arrayGetter, hidden, new ArraySubclass(1, 2), { [Symbol('x')]: 1 }]) assert.throws(() => canonicalJSON(value));
  assert.equal(invoked, false);
});
for (const [name, setting] of Object.entries({ partial: { count: 8 }, complete: {}, draw: { sequences: [DRAW] }, duplicates: { configs: Array(4).fill({ type: 'random' }) }, stochastic: { configs: [{ type: 'mcts', simulations: 100 }, { type: 'random' }, { type: 'mcts', simulations: 400 }, { type: 'random' }] }, maximum: { size: 12, games: 8, sequences: [DRAW] } })) test(`${name} deterministic export fixture validates and recomputes`, async () => {
  const s = evaluationSeason(setting), a = await createEvaluationExport(s, options), b = await createEvaluationExport(s, { ...options, exportedAt: '2026-10-08T12:00:00.000Z' });
  assert.equal(a.integrity.evidence_sha256, b.integrity.evidence_sha256); assert.notEqual(a.exported_at, b.exported_at);
  assert.equal(a.evidence.state.completed_games, s.completedGames.length); assert.equal(a.evidence.state.scheduled_games, s.schedule.length);
  assert.equal(a.evidence.state.status, s.status === 'complete' ? 'complete' : 'partial');
  assert.equal((await verifyEvaluationExport(reverseKeys(a), options)).ok, true);
  assert.equal(JSON.stringify(a.evidence).includes('board'), false); assert.equal(JSON.stringify(a).includes('seasonId'), false);
  assert.equal(gamesCSV(a).trim().split('\r\n').length, s.completedGames.length + 1); assert.equal(summaryCSV(a).trim().split('\r\n').length, s.fieldSize + 1);
  if (name === 'draw') { assert.ok(a.derived.ratings.every(r => r.final_elo === 1500)); assert.ok(Object.values(a.derived.bootstrap_intervals).every(i => i.lower === .5 && i.upper === .5)); }
});
test('stored partial/complete JSON golden fixtures retain fixed evidence digests', async () => {
  for (const name of ['partial', 'complete']) {
    const fixture = JSON.parse(await readFile(new URL(`./fixtures/evaluation-${name}.json`, import.meta.url), 'utf8'));
    assert.equal((await verifyEvaluationExport(fixture, options)).ok, true);
    const s = evaluationSeason({ count: name === 'partial' ? 8 : 24 });
    const actual = await createEvaluationExport(s, options); assert.deepEqual(actual, fixture);
  }
});
const mutations = {
  move: a => a.evidence.completed_games[0].move_columns[0] = 6,
  seed: a => a.evidence.completed_games[0].game_seed++, config: a => a.evidence.entrants[1].config.depth = 4,
  winner: a => a.evidence.completed_games[0].terminal_result.winner_index = 1,
  schedule: a => a.evidence.schedule.reverse(), completion: a => a.evidence.completed_games.reverse(),
  elo: a => a.derived.ratings[0].final_elo++, history: a => a.derived.ratings[0].history[0]++,
  bootstrap: a => Object.values(a.derived.bootstrap_intervals).find(Boolean).lower = .123456,
  standings: a => a.derived.standings[0].wins++, splits: a => a.derived.side_splits[0].red.wins++,
  pairwise: a => Object.values(Object.values(a.derived.pairwise)[0])[0].played++,
  digest: a => a.integrity.evidence_sha256 = '0'.repeat(64), methodology: a => a.methodology.elo.k_factor = 25,
  provenance: a => a.provenance.agents.negamax++, gameProvenance: a => a.evidence.completed_games[0].backend_provenance.agents.random++,
  count: a => a.evidence.state.completed_games++, version: a => a.schema_version++, unknown: a => a.evidence.ui_state = {},
};
for (const [name, mutate] of Object.entries(mutations)) test(`verifier detects ${name} tampering with specific failure`, async () => {
  const a = await createEvaluationExport(evaluationSeason(), options); mutate(a);
  const result = await verifyEvaluationExport(a, options); assert.equal(result.ok, false); assert.ok(result.errors[0].stage); assert.ok(result.errors[0].message);
});
test('digest detects evidence changes/order; metadata timestamp and cosmetic UI fields excluded', async () => {
  const s = evaluationSeason(), a = await createEvaluationExport(s, options), hash = a.integrity.evidence_sha256;
  a.exported_at = '2026-10-09T00:00:00.000Z'; assert.equal((await verifyEvaluationExport(a, options)).ok, true);
  const payload = canonicalPayload(a); payload.evidence.completed_games.reverse(); assert.notEqual(await evidenceDigest(payload), hash);
  s.seasonId = 'cosmetic-session'; s.selectedTab = 'ratings';
  assert.equal((await createEvaluationExport(s, options)).integrity.evidence_sha256, hash);
  const legalEdit = await createEvaluationExport(evaluationSeason({ count: 1, sequences: [WIN] }), options);
  legalEdit.evidence.completed_games[0].move_columns = [2, 1, 2, 1, 2, 1, 2]; // Same legal terminal result and analytics.
  const checked = await verifyEvaluationExport(legalEdit, options);
  assert.equal(checked.ok, false); assert.equal(checked.errors[0].stage, 'integrity');
});
test('invalid evidence and failed crypto cannot produce exports; unknown historical provenance stays explicit', async () => {
  const s = evaluationSeason(); s.completedGames[0].gameSeed++; await assert.rejects(createEvaluationExport(s, options));
  await assert.rejects(browserSHA256('abc', {}), /Web Crypto/);
  await assert.rejects(createEvaluationExport(evaluationSeason(), { digest: () => { throw new Error('digest failed'); } }), /digest failed/);
  await assert.rejects(createEvaluationExport(evaluationSeason(), { digest: () => 'wrong' }), /lowercase/);
  const a = await createEvaluationExport(evaluationSeason({ captured: false }), options), result = await verifyEvaluationExport(a, options);
  assert.equal(result.ok, true); assert.equal(result.warnings.length, 1); assert.equal(a.provenance.execution_capture.unrecorded_games, 24);
});
test('per-game backend metadata survives compact history and canonical evidence projection', () => {
  const s = evaluationSeason({ count: 0 }), p = gamePlan(s);
  const g = compactHistory(s, { ...matchFixture(p.playerConfigs, WIN), rng_seed: p.gameSeed, provenance: backend() });
  assert.deepEqual(g.backendProvenance, backend());
  assert.equal(seasonEvidence(s).state.completed_games, 0);
});
test('independent one-win fixture: standings, Elo and pairwise have known exact answers', async () => {
  const s = evaluationSeason({ count: 1, sequences: [WIN] }), g = s.completedGames[0], a = await createEvaluationExport(s, options);
  const winner = a.derived.standings.find(r => r.entrant_id === g.redEntrantId), loser = a.derived.standings.find(r => r.entrant_id === g.yellowEntrantId);
  assert.equal(winner.points, 1); assert.equal(winner.wins, 1); assert.equal(winner.red.score_rate, 1); assert.equal(loser.points, 0); assert.equal(loser.losses, 1);
  assert.equal(a.derived.ratings.find(r => r.entrant_id === g.redEntrantId).final_elo, 1512);
  assert.equal(a.derived.ratings.find(r => r.entrant_id === g.yellowEntrantId).final_elo, 1488);
  assert.deepEqual(a.derived.pairwise[g.redEntrantId][g.yellowEntrantId], { played: 1, wins: 1, draws: 0, losses: 0, points: 1, score_rate: 1 });
  assert.ok(Object.values(a.derived.bootstrap_intervals).every(i => i === null));
});
test('independent repeated-opponent win/draw fixture checks second Elo update, sorting and pairwise perspectives', () => {
  const s = evaluationSeason({ count: 0 });
  s.completedGames = [
    { redEntrantId: 'entrant-1', yellowEntrantId: 'entrant-2', result: { status: 'win', winnerIndex: 0 } },
    { redEntrantId: 'entrant-2', yellowEntrantId: 'entrant-1', result: { status: 'draw', winnerIndex: null } },
  ];
  assert.deepEqual(seasonStandings(s).map(r => [r.entrantId, r.points]), [['entrant-1', 1.5], ['entrant-2', .5], ['entrant-3', 0], ['entrant-4', 0]]);
  const delta = 24 * (.5 - 1 / (1 + Math.pow(10, (1512 - 1488) / 400)));
  const ratings = seasonRatings(s); assert.equal(ratings.find(r => r.entrantId === 'entrant-1').rating, 1512 - delta);
  assert.equal(pairwiseResults(s)['entrant-2']['entrant-1'].scoreRate, .25);
});
test('CSV escaping, CRLF and formula injection protection; numeric negative Elo remains numeric', async () => {
  assert.equal(csvCell('a,"b"\nc'), '"a,""b""\nc"');
  for (const cell of ['=SUM(A1)', '+x', '-x', '@x', '  =x', '\t=x', '\n@x']) assert.ok(csvCell(cell).replace(/^"/, '').startsWith("'"));
  assert.equal(csvCell(-12), '-12'); assert.equal(csvCell('é🟡'), 'é🟡');
  const a = await createEvaluationExport(evaluationSeason({ count: 1 }), options), rows = summaryCSV(a).trim().split('\r\n');
  for (const row of rows.slice(1)) { const fields = row.split(','); assert.deepEqual(fields.slice(3, 6), ['partial', '1', '24']); assert.equal(fields.at(-3), ''); assert.equal(fields.at(-2), ''); }
  const games = gamesCSV(a).trim().split('\r\n'); assert.equal(games[1].split(',').at(-1), WIN.join('|'));
});
test('native download clicks once, removes anchor and releases URL even when click fails', () => {
  for (const fail of [false, true]) {
    const events = [], anchor = { click() { events.push('click'); if (fail) throw new Error('click failed'); }, remove() { events.push('remove'); } };
    const env = { document: { createElement: () => anchor, body: { appendChild: () => events.push('append') } }, URL: { createObjectURL: () => 'blob:test', revokeObjectURL: u => events.push(u) }, schedule: fn => fn() };
    if (fail) assert.throws(() => downloadText('data', 'test.json', 'application/json', env)); else downloadText('data', 'test.json', 'application/json', env);
    assert.deepEqual(events, ['append', 'click', 'remove', 'blob:test']); assert.equal(anchor.download, 'test.json');
  }
});
