import { mkdir, readFile, writeFile, rename } from 'node:fs/promises';
import { resolve, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { randomUUID } from 'node:crypto';
import os from 'node:os';
import { performance } from 'node:perf_hooks';
import { createSeason, validateSeason } from '../src/season/model.js';
import { SeasonController } from '../src/season/controller.js';
import { competitorLabel } from '../src/connect4/competitorConfig.js';
import { canonicalJSON } from '../src/evaluation/canonical.js';
import { validateBackendProvenance } from '../src/evaluation/provenance.js';
import { createEvaluationExport } from '../src/evaluation/export.js';
import { verifyEvaluationExport } from '../src/evaluation/verify.js';
import { gamesCSV, summaryCSV } from '../src/evaluation/csv.js';
import { nodeSHA256 } from './evaluation-node.mjs';

const root = fileURLToPath(new URL('../../', import.meta.url));
export const DEFAULT_API = 'http://127.0.0.1:8000';
export function benchmarkConfig(config) {
  const keys = ['format', 'version', 'name', 'field_size', 'season_seed', 'games_per_pairing', 'entrants'].sort();
  if (!config || canonicalJSON(Object.keys(config).sort()) !== canonicalJSON(keys) || config.format !== 'connect4-season-benchmark' || config.version !== 1 ||
      typeof config.name !== 'string' || !/^[a-z0-9][a-z0-9-]{0,79}$/.test(config.name)) throw new Error('Invalid benchmark configuration contract/name.');
  const s = createSeason(config.entrants, config.season_seed, config.games_per_pairing);
  if (s.fieldSize !== config.field_size) throw new Error('Benchmark field_size does not match entrants.');
  return s;
}
export function safeAPI(value = DEFAULT_API, allowNonlocal = false) {
  let url; try { url = new URL(value); } catch { throw new Error('Invalid benchmark API origin.'); }
  if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password || url.pathname !== '/' || url.search || url.hash) throw new Error('Benchmark API must be an HTTP(S) origin without credentials, paths, query or fragment.');
  if (!['127.0.0.1', 'localhost', '[::1]'].includes(url.hostname) && !allowNonlocal) throw new Error('Non-local benchmark target refused. Use --allow-nonlocal only for an intentionally authorized target.');
  return url.origin;
}
export function benchmarkPlan(config, { api = DEFAULT_API, allowNonlocal = false, output = join(root, 'benchmarks/connect4/runs') } = {}) {
  const s = benchmarkConfig(config), apiBase = safeAPI(api, allowNonlocal), turnsPerEntrant = (s.fieldSize - 1) * s.gamesPerPairing * 21;
  return { mode: 'DRY-RUN — no API requests or games', name: config.name, field_size: s.fieldSize, season_seed: s.seasonSeed,
    games_per_pairing: s.gamesPerPairing, total_games: s.schedule.length, games_per_entrant: (s.fieldSize - 1) * s.gamesPerPairing,
    entrants: s.entrants.map(e => ({ entrant_id: e.entrantId, label: competitorLabel(e.config), config: e.config })),
    estimated_stochastic_decisions_upper_bound: s.entrants.filter(e => ['random', 'mcts'].includes(e.config.type)).length * turnsPerEntrant,
    mcts_searches_upper_bound: s.entrants.filter(e => e.config.type === 'mcts').length * turnsPerEntrant,
    mcts_simulations_upper_bound: s.entrants.reduce((n, e) => n + (e.config.simulations ?? 0) * turnsPerEntrant, 0),
    workload_note: 'Upper bounds assume 42-ply games; tactical wins can bypass MCTS searches. No measured strength/runtime estimate.',
    api_base: apiBase, output_base: resolve(output), execution: 'Requires --execute; each run creates a new directory.' };
}
export function fetchTransport(api) {
  async function request(path, body) {
    const response = await fetch(`${api}${path}`, { method: body === undefined ? 'GET' : 'POST', redirect: 'error',
      signal: AbortSignal.timeout(90000), ...(body === undefined ? {} : { headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }) });
    const data = await response.json();
    if (!response.ok) throw Object.assign(new Error(data.error || `HTTP ${response.status}`), { response: { status: response.status, data } });
    return { data };
  }
  return { get: path => request(path), post: (path, body) => request(path, body) };
}
export async function executeBenchmark(config, options = {}) {
  if (options.execute !== true) throw new Error('Benchmark execution requires explicit execute: true / --execute.');
  const plan = benchmarkPlan(config, options), http = options.http ?? fetchTransport(plan.api_base);
  const backend = validateBackendProvenance((await http.get('/v1/connect4/provenance')).data);
  const runId = `${config.name}-${new Date().toISOString().replace(/[:.]/g, '-')}-${randomUUID().slice(0, 8)}`;
  await mkdir(plan.output_base, { recursive: true });
  const directory = join(plan.output_base, runId); await mkdir(directory); // exclusive, never reuse/overwrite a previous run.
  const memory = new Map(), storage = { getItem: k => memory.get(k), setItem: (k, v) => memory.set(k, v) };
  let inFlight = 0, maxConcurrentPosts = 0, posts = 0;
  const transport = { get: path => http.get(path), async post(path, body) {
    if (inFlight) throw new Error('Concurrent benchmark mutation refused.');
    maxConcurrentPosts = Math.max(maxConcurrentPosts, ++inFlight); posts++;
    try { return await http.post(path, body); } finally { inFlight--; }
  } };
  const controller = new SeasonController(transport, storage); // manual stepping; no autoplay timer or second evaluation engine.
  controller.create(config.entrants, config.season_seed, config.games_per_pairing);
  const started = performance.now(), metadata = { run_id: runId, status: 'running', started_at: new Date().toISOString(), finished_at: null,
    wall_time_ms: null, host_platform: os.platform(), host_release: os.release(), host_arch: os.arch(), cpu: os.cpus()[0]?.model ?? null,
    cpu_count: os.cpus().length, node_version: process.version, backend_runtime: null,
    api_base: plan.api_base, source_commit: backend.source_commit, backend_provenance: backend, game_wall_times_ms: [], max_concurrent_posts: 0, mutation_requests: 0 };
  const atomic = async (name, contents) => { const path = join(directory, name); await writeFile(`${path}.tmp`, contents, 'utf8'); await rename(`${path}.tmp`, path); };
  let artifact;
  async function checkpoint(csv = false) {
    artifact = await createEvaluationExport(validateSeason(controller.state.season), { digest: nodeSHA256 });
    const verified = await verifyEvaluationExport(artifact, { digest: nodeSHA256 });
    if (!verified.ok) throw new Error(`Benchmark self-verification failed: ${JSON.stringify(verified.errors)}`);
    await atomic('evaluation.json', JSON.stringify(artifact, null, 2) + '\n');
    if (csv) { await atomic('games.csv', gamesCSV(artifact)); await atomic('summary.csv', summaryCSV(artifact)); }
    metadata.wall_time_ms = performance.now() - started; metadata.max_concurrent_posts = maxConcurrentPosts; metadata.mutation_requests = posts;
    await atomic('run-metadata.json', JSON.stringify(metadata, null, 2) + '\n');
  }
  await checkpoint();
  try {
    while (controller.state.season.status !== 'complete') {
      const began = performance.now(), index = controller.state.season.currentGameIndex;
      // Check the public manifest before every fixture; known implementation drift stops execution.
      if (canonicalJSON(validateBackendProvenance((await http.get('/v1/connect4/provenance')).data)) !== canonicalJSON(backend)) throw new Error('Backend provenance changed during the benchmark.');
      await controller.watch();
      if (controller.state.error || controller.state.uncertain) throw new Error(controller.state.error || 'Unconfirmed game start.');
      while (controller.state.season.currentGameIndex === index) {
        await controller.nextMove();
        if (controller.state.error || controller.state.uncertain) throw new Error(controller.state.error || 'Unconfirmed game move.');
      }
      const g = controller.state.season.completedGames.at(-1);
      if (!g.backendProvenance || canonicalJSON(g.backendProvenance) !== canonicalJSON(backend)) throw new Error('Game history implementation provenance differs from benchmark preflight.');
      metadata.game_wall_times_ms.push({ completion_index: index, fixture_id: g.fixtureId, wall_time_ms: performance.now() - began });
      await checkpoint(); options.onProgress?.(`${index + 1}/${plan.total_games} games complete`);
    }
    metadata.status = 'complete'; metadata.finished_at = new Date().toISOString(); await checkpoint(true);
    return { directory, artifact, metadata };
  } catch (error) {
    controller.detach(); metadata.status = 'interrupted'; metadata.error = error.message; metadata.finished_at = new Date().toISOString();
    await checkpoint(true); throw new Error(`${error.message} Validated partial evidence retained in ${directory}. No automatic POST retry or resume.`);
  }
}
export function parseArgs(args) {
  const options = {}, flags = new Map([['--execute', 'execute'], ['--allow-nonlocal', 'allowNonlocal']]), values = new Map([['--config', 'config'], ['--api', 'api'], ['--output', 'output']]);
  for (let i = 0; i < args.length; i++) {
    const key = flags.get(args[i]) ?? values.get(args[i]);
    if (!key || Object.hasOwn(options, key)) throw new Error(`Unknown or repeated argument: ${args[i]}`);
    if (flags.has(args[i])) options[key] = true;
    else { if (!args[i + 1] || args[i + 1].startsWith('--')) throw new Error(`Missing value for ${args[i]}`); options[key] = args[++i]; }
  }
  return options;
}
async function main() {
  const options = parseArgs(process.argv.slice(2)), config = JSON.parse(await readFile(options.config ?? join(root, 'benchmarks/connect4/canonical-season-v1.json'), 'utf8'));
  const plan = benchmarkPlan(config, options); console.log(JSON.stringify(plan, null, 2));
  if (!options.execute) return;
  console.log('Explicit execution requested. Playing sequentially.');
  const result = await executeBenchmark(config, { ...options, onProgress: console.log });
  console.log(`Verification PASS — ${result.artifact.evidence.state.completed_games} games\nOutputs: ${result.directory}`);
}
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) main().catch(e => { console.error(`Benchmark failed: ${e.message}`); process.exitCode = 1; });
