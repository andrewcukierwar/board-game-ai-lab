import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, readdir, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { spawnSync } from 'node:child_process';
import { benchmarkPlan, benchmarkConfig, safeAPI, parseArgs, executeBenchmark } from '../scripts/run-connect4-benchmark.mjs';
import { verifyFile } from '../scripts/verify-evaluation-export.mjs';
import { matchFixture, WIN } from '../e2e/fixtures/match.js';
import { backend } from './fixtures/evaluation.js';
const config = JSON.parse(await readFile(new URL('../../benchmarks/connect4/smoke-season-v1.json', import.meta.url), 'utf8'));
test('canonical benchmark field, schedule and zero-request dry-run', async () => {
  const canonical = JSON.parse(await readFile(new URL('../../benchmarks/connect4/canonical-season-v1.json', import.meta.url), 'utf8'));
  const plan = benchmarkPlan(canonical); assert.equal(plan.total_games, 112); assert.equal(plan.games_per_entrant, 28); assert.equal(plan.season_seed, 20261008); assert.equal(plan.api_base, 'http://127.0.0.1:8000');
  assert.deepEqual(canonical.entrants, [{ type: 'random' }, ...[2, 4, 6, 8].map(depth => ({ type: 'negamax', depth })), ...[100, 400, 800].map(simulations => ({ type: 'mcts', simulations }))]);
  assert.equal(plan.mcts_searches_upper_bound, 1764); assert.equal(plan.estimated_stochastic_decisions_upper_bound, 2352);
  const result = spawnSync(process.execPath, ['scripts/run-connect4-benchmark.mjs', '--api', 'http://127.0.0.1:1'], { cwd: new URL('../', import.meta.url), encoding: 'utf8' });
  assert.equal(result.status, 0, result.stderr); assert.match(result.stdout, /DRY-RUN/); assert.match(result.stdout, /112/);
  await assert.rejects(executeBenchmark(config), /explicit/);
});
test('nonlocal, credentials and unexpected CLI arguments refused', () => {
  for (const target of ['https://board-game-ai-lab.onrender.com', 'http://127.0.0.1.evil.test', 'http://192.168.1.2']) assert.throws(() => safeAPI(target), /Non-local/);
  for (const target of ['file:///tmp/test', 'http://user:pass@localhost', 'http://localhost/v1', 'http://localhost?x=1', 'http://localhost#x']) assert.throws(() => safeAPI(target, true));
  assert.equal(safeAPI('http://[::1]:8000'), 'http://[::1]:8000');
  assert.equal(safeAPI('https://example.com', true), 'https://example.com');
  for (const args of [['--execute', '--execute'], ['--config'], ['--resume'], ['--config', '--execute']]) assert.throws(() => parseArgs(args));
  assert.deepEqual(parseArgs(['--config', 'tiny.json', '--execute']), { config: 'tiny.json', execute: true });
  assert.throws(() => benchmarkConfig({ ...config, field_size: 8 }));
});
function mockAPI(failAt = Infinity) {
  let active, posts = 0, flight = 0, maxFlight = 0, retired = 0;
  const history = () => ({ ...matchFixture(active.players, active.columns, active.id), rng_seed: active.seed, provenance: backend() });
  return { get maxFlight() { return maxFlight; }, get retired() { return retired; }, http: {
    async get(path) { return { data: path.endsWith('/provenance') ? backend() : history() }; },
    async post(path, body) {
      maxFlight = Math.max(maxFlight, ++flight); posts++;
      try {
        if (posts === failAt) throw new Error('injected interruption');
        if (path.endsWith('/start_game')) {
          if (active) { assert.equal(body.replace_game_id, active.id); retired++; }
          active = { id: `mock-${posts}`, players: [body.player1, body.player2], seed: body.rng_seed, columns: [] };
        } else { assert.equal(body.revision, active.columns.length); assert.equal(body.game_id, active.id); active.columns.push(WIN[active.columns.length]); }
        await new Promise(r => setImmediate(r)); return { data: history().state };
      } finally { flight--; }
    },
  } };
}
test('cheap mock runner serializes mutations, checkpoints, creates new directories and verifies all outputs', async () => {
  const output = await mkdtemp(join(tmpdir(), 'evaluation-runner-'));
  try {
    const mock = mockAPI(), result = await executeBenchmark(config, { execute: true, http: mock.http, output });
    assert.equal(result.metadata.max_concurrent_posts, 1); assert.equal(mock.maxFlight, 1); assert.equal(mock.retired, 11);
    assert.equal(result.metadata.game_wall_times_ms.length, 12); assert.equal(result.artifact.evidence.state.status, 'complete');
    assert.equal((await verifyFile(join(result.directory, 'evaluation.json'))).ok, true);
    assert.deepEqual((await readdir(result.directory)).sort(), ['evaluation.json', 'games.csv', 'run-metadata.json', 'summary.csv']);
    assert.equal((await readFile(join(result.directory, 'games.csv'), 'utf8')).trim().split('\r\n').length, 13);
    assert.equal((await readFile(join(result.directory, 'summary.csv'), 'utf8')).trim().split('\r\n').length, 5);
    const other = await executeBenchmark(config, { execute: true, http: mockAPI().http, output }); assert.notEqual(result.directory, other.directory);
    const verified = spawnSync(process.execPath, ['scripts/verify-evaluation-export.mjs', join(result.directory, 'evaluation.json')], { cwd: new URL('../', import.meta.url), encoding: 'utf8' });
    assert.equal(verified.status, 0, verified.stderr); assert.match(verified.stdout, /Verification PASS/);
    const bad = spawnSync(process.execPath, ['scripts/verify-evaluation-export.mjs', join(result.directory, 'missing.json')], { cwd: new URL('../', import.meta.url), encoding: 'utf8' }); assert.notEqual(bad.status, 0);
  } finally { await rm(output, { recursive: true, force: true }); }
});
test('interruption preserves validated partial results, CSV and metadata; no blind retries', async () => {
  const output = await mkdtemp(join(tmpdir(), 'evaluation-interrupted-'));
  try {
    await assert.rejects(executeBenchmark(config, { execute: true, http: mockAPI(12).http, output }), /partial evidence retained/);
    const [directory] = await readdir(output), a = JSON.parse(await readFile(join(output, directory, 'evaluation.json'), 'utf8'));
    assert.equal(a.evidence.state.completed_games, 1); assert.equal(a.evidence.state.status, 'partial');
    assert.equal((await verifyFile(join(output, directory, 'evaluation.json'))).ok, true);
    assert.equal(JSON.parse(await readFile(join(output, directory, 'run-metadata.json'), 'utf8')).status, 'interrupted');
  } finally { await rm(output, { recursive: true, force: true }); }
});
