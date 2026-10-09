import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdtemp, writeFile, rm, readFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { execFileSync } from 'node:child_process';
import { competitorTypes, competitorLabel, toPlayerPayload, matchStartPayload } from '../src/connect4/competitorConfig.js';
import { opponentPayload, RESEARCH_DISABLED } from '../src/connect4/researchAgent.js';
import { validateMatchHistory } from '../src/connect4/matchRecord.js';
import * as tournament from '../src/tournament/model.js';
import * as season from '../src/season/model.js';
import { seasonAnalytics } from '../src/season/analytics.js';
import { TournamentController } from '../src/tournament/controller.js';
import { SeasonController } from '../src/season/controller.js';
import { createEvaluationExport, canonicalPayload } from '../src/evaluation/export.js';
import { verifyEvaluationExport } from '../src/evaluation/verify.js';
import { BACKEND_VERSIONS } from '../src/evaluation/provenance.js';
import { canonicalJSON } from '../src/evaluation/canonical.js';
import { WIN, DRAW, matchFixture } from '../e2e/fixtures/match.js';

const victor = { type: 'victor_research' }, random = { type: 'random' };
const field = size => Array.from({ length: size }, (_, i) => i < 2 ? { ...victor } : { ...random });
const digest = text => createHash('sha256').update(text).digest('hex');
const failure = code => Object.assign(new Error(code), { response: { status: 503, data: { code, error: code } } });
const flush = async () => { for (let i = 0; i < 6; i++) await new Promise(r => setImmediate(r)); };

function harness(kind, enabled = true, storage) {
  const values = new Map(); storage ??= { getItem: k => values.get(k) ?? null, setItem: (k, v) => values.set(k, v) };
  const timers = new Map(); let id = 0;
  const h = { posts: [], starts: [], active: null, flight: 0, maxFlight: 0 };
  h.fixture = () => ({ ...matchFixture(h.active.players, h.active.columns, h.active.id), rng_seed: h.active.seed });
  h.post = async (url, body) => {
    h.maxFlight = Math.max(h.maxFlight, ++h.flight);
    try {
      if (url.endsWith('start_game')) {
        if (h.active) assert.equal(body.replace_game_id, h.active.id);
        h.starts.push(body); h.active = { id: `victor-${h.starts.length}`, players: [body.player1, body.player2], seed: body.rng_seed, columns: [] };
      } else {
        assert.equal(body.revision, h.active.columns.length); assert.equal(body.game_id, h.active.id);
        const human = h.active.players[body.revision % 2].type === 'human'; assert.equal('column' in body, human);
        h.posts.push(body); h.active.columns.push(human ? body.column : WIN[body.revision]);
      }
      return { data: h.fixture().state };
    } finally { h.flight--; }
  };
  h.get = async () => ({ data: h.fixture() });
  const C = kind === 'season' ? SeasonController : TournamentController;
  const c = new C(h, storage, fn => { timers.set(++id, fn); return id; }, id => timers.delete(id), enabled);
  const tick = async () => { const entry = timers.entries().next().value; if (entry) { timers.delete(entry[0]); entry[1](); } await flush(); };
  const finish = async () => { for (let i = 0; i < 130 && c.state.mode !== 'paused'; i++) await tick(); assert.equal(c.state.mode, 'paused'); assert.equal(c.state.error, ''); };
  c.attach(); return { h, c, timers, tick, finish, storage };
}

test('Victor config is exact, feature gated, and remains opt-in in every default field', () => {
  assert.equal(competitorTypes(false).includes(victor.type), false); assert.ok(competitorTypes(true).includes(victor.type));
  assert.equal(competitorLabel(victor), 'Victor Research (Experimental)');
  assert.deepEqual(toPlayerPayload(victor, true), victor); assert.throws(() => opponentPayload({ ...victor, node_budget: 3 }, true)); assert.throws(() => toPlayerPayload(victor, false));
  for (const config of [{ ...victor, depth: 1 }, { ...victor, simulations: 100 }, { ...victor, node_budget: 100 }, { type: 'future_agent' }]) assert.throws(() => toPlayerPayload(config, true));
  for (const size of [...tournament.SIZES, ...season.FIELD_SIZES]) assert.ok(!tournament.defaultField(size).some(c => c.type === victor.type));
  assert.throws(() => tournament.createTournament(field(8), 1)); assert.throws(() => season.createSeason(field(4), 1));
  assert.throws(() => season.createSeason([{ type: 'human' }, ...field(4).slice(1)], 1, 2, undefined, true));
  assert.throws(() => tournament.createTournament([{ type: 'human' }, { type: 'human' }, ...field(8).slice(2)], 1, undefined, true));
});
for (const opponent of [random, { type: 'negamax', depth: 2 }, { type: 'mcts', simulations: 100 }, victor, { type: 'human' }]) {
  for (const red of [true, false]) test(`Victor Match history against ${opponent.type}, Victor Red=${red}`, () => {
    const pair = red ? [victor, opponent] : [opponent, victor], payload = matchStartPayload(...pair, undefined, true);
    const history = matchFixture([payload.player1, payload.player2], WIN);
    assert.equal(validateMatchHistory(history).game.gameOver, true);
    history.moves.find(m => m.agent.type === victor.type).agent.depth = 1; assert.throws(() => validateMatchHistory(history));
  });
}
test('all supported tournament sizes retain strict Victor schemas without changing bracket construction', () => {
  for (const size of tournament.SIZES) {
    const t = tournament.createTournament(field(size), 1234, undefined, true);
    assert.deepEqual(t.bracketOrder, tournament.bracketOrder(size, 1234));
    assert.deepEqual(tournament.validateTournament(JSON.parse(JSON.stringify(t))), t);
    const bad = structuredClone(t); bad.entrants[0].config.extra = 1; assert.throws(() => tournament.validateTournament(bad));
  }
});
for (const kind of ['tournament', 'season']) {
  test(`${kind} Victor/Victor advances, round trips, and disabled flag preserves evidence while stopping execution`, async () => {
    const x = harness(kind), { c, h } = x; c.create(field(kind === 'season' ? 4 : 8), 1234, 2);
    c.run(kind === 'season' ? 'season' : 'tournament'); await x.finish();
    const record = c.state[kind]; assert.equal(record.status, 'complete'); assert.equal(h.maxFlight, 1);
    const validate = kind === 'season' ? season.validateSeason : tournament.validateTournament;
    assert.deepEqual(validate(JSON.parse(JSON.stringify(record))), record);
    if (kind === 'season') { const a = seasonAnalytics(record); assert.equal(a.standings.reduce((n, r) => n + r.played, 0), 24); assert.equal(a.ratings[0].history.length, 13); }
    c.detach(); const reloaded = harness(kind, false, x.storage); assert.equal(reloaded.c.state[kind].status, 'complete'); assert.equal(reloaded.h.starts.length, 0); reloaded.c.detach();
    // Pending saved records remain loadable but cannot start a Victor fixture.
    const pending = harness(kind); pending.c.create(Array(kind === 'season' ? 4 : 8).fill(victor), 12, 2); pending.c.detach();
    const disabled = harness(kind, false, pending.storage); await disabled.c.watch(); assert.equal(disabled.h.starts.length, 0); assert.equal(disabled.c.state[kind].active, null); assert.equal(disabled.c.state.error, RESEARCH_DISABLED); assert.equal(disabled.c.state.uncertain, false); disabled.c.detach();
  });
  for (const mode of ['busy', 'failed', 'lost', 'lost-read']) test(`${kind} Victor ${mode} pauses and reconciles without duplicate POSTs`, async () => {
    const x = harness(kind), { c, h } = x; c.create(Array(kind === 'season' ? 4 : 8).fill(victor), 123, 2); await c.watch();
    const post = h.post, get = h.get; let attempts = 0;
    h.post = async (u, b) => { attempts++; if (mode.startsWith('lost')) await post(u, b); throw mode.startsWith('lost') ? new Error('lost response') : failure(mode === 'busy' ? 'agent_busy' : 'agent_failed'); };
    if (mode === 'lost-read') h.get = async () => { throw new Error('read unavailable'); };
    c.run(kind); await x.tick(); assert.equal(attempts, 1); assert.equal(c.state.mode, 'paused'); assert.equal(x.timers.size, 0);
    assert.equal(c.state[kind].active.columns.length, mode === 'lost' ? 1 : 0);
    await c.nextMove(); assert.equal(attempts, 1); h.get = get; await c.refresh(); h.post = post; await c.nextMove();
    assert.equal(c.state[kind].active.columns.length, mode.startsWith('lost') ? 2 : 1); c.detach();
  });
  test(`${kind} pause during Victor request settles once, reload reads only, disabling prevents more moves`, async () => {
    const x = harness(kind), { c, h } = x; c.create(Array(kind === 'season' ? 4 : 8).fill(victor), 5, 2); await c.watch();
    const post = h.post; let release; h.post = (u, b) => new Promise(resolve => { release = async () => resolve(await post(u, b)); });
    c.run(kind); await x.tick(); assert.equal(c.state.busy, true); c.pause(); release(); await flush(); assert.equal(h.posts.length, 1); assert.equal(x.timers.size, 0); c.detach();
    const C = kind === 'season' ? SeasonController : TournamentController;
    const reloaded = new C(h, x.storage, undefined, undefined, false); reloaded.attach(); await flush(); assert.equal(reloaded.state.uncertain, false); assert.equal(reloaded.state[kind].active.columns.length, 1);
    await reloaded.nextMove(); assert.equal(h.posts.length, 1); assert.match(reloaded.state.error, /disabled/); reloaded.detach();
  });
}
test('Human tournament participation against Victor works through normal revision-bound moves', async () => {
  const x = harness('tournament'), { c } = x; const configs = Array(8).fill(victor); configs[0] = { type: 'human' }; c.create(configs, 8);
  while (!tournament.matchupHasHuman(c.state.tournament, tournament.nextMatchup(c.state.tournament))) { c.run('matchup'); await x.finish(); }
  await c.watch(); for (const col of WIN) { if (c.humanTurn()) await c.humanMove(col); else await c.nextMove(); }
  assert.equal(c.state.tournament.active, null); assert.ok(tournament.humanTournamentStatus(c.state.tournament)); c.detach();
});
test('Victor completed Season v2 verifies via independent CLI; v1 bytes remain frozen; malformed configs fail', async () => {
  let s = season.createSeason(field(4), 1234, 2, undefined, true);
  while (season.currentFixture(s)) { const plan = season.gamePlan(s); s = season.recordGame(s, season.compactHistory(s, { ...matchFixture(plan.playerConfigs, s.currentGameIndex % 2 ? DRAW : WIN), rng_seed: plan.gameSeed, provenance: { ...BACKEND_VERSIONS, source_commit: null } })); }
  const artifact = await createEvaluationExport(s, { digest }); assert.equal(artifact.schema_version, 2);
  assert.equal(artifact.provenance.agents.victor_research, 1);
  assert.deepEqual(artifact.evidence.completed_games[0].backend_provenance, { ...BACKEND_VERSIONS, source_commit: null });
  assert.equal(artifact.provenance.execution_capture.recorded_games, 12);
  assert.ok((await verifyEvaluationExport(artifact, { digest })).warnings.some(w => w.includes('does not identify'))); assert.match(artifact.methodology.victor_execution.budget, /wall-clock/);
  assert.equal((await verifyEvaluationExport(artifact, { digest })).ok, true);
  const dir = await mkdtemp(join(tmpdir(), 'victor-evaluation-'));
  try { const file = join(dir, 'evaluation.json'); await writeFile(file, JSON.stringify(artifact)); assert.match(execFileSync(process.execPath, ['scripts/verify-evaluation-export.mjs', file], { encoding: 'utf8' }), /Verification PASS/); } finally { await rm(dir, { recursive: true }); }
  for (const change of [a => a.evidence.entrants[0].config.extra = 1, a => a.evidence.entrants[0].config.type = 'future_agent', a => a.schema_version = 1, a => a.provenance.agents.victor_research = 2, a => a.evidence.completed_games.find(g => g.player_configs.some(p => p.type === victor.type)).player_configs.find(p => p.type === victor.type).depth = 2]) {
    const bad = structuredClone(artifact); change(bad); assert.equal((await verifyEvaluationExport(bad, { digest })).ok, false);
  }
  const frozen = JSON.parse(await readFile('../benchmarks/connect4/results/canonical-season-v1/evaluation.json', 'utf8'));
  assert.equal(frozen.schema_version, 1); assert.equal(digest(canonicalJSON(canonicalPayload(frozen))), frozen.integrity.evidence_sha256); assert.equal((await verifyEvaluationExport(frozen, { digest })).ok, true);
});

test('Victor season presets preserve every supported field size, pairing option and balanced colors', async () => {
  for (const size of season.FIELD_SIZES) for (const games of season.GAMES_PER_PAIRING) {
    const s = season.createSeason(field(size), 29, games, undefined, true);
    assert.equal(s.schedule.length, season.totalGames(size, games));
    assert.deepEqual(season.validateSeason(JSON.parse(JSON.stringify(s))), s);
    const pairs = new Map();
    for (const f of s.schedule) { const ids = [f.redEntrantId, f.yellowEntrantId].sort(), key = ids.join(':'); if (!pairs.has(key)) pairs.set(key, [0, 0]); pairs.get(key)[f.redEntrantId === ids[0] ? 0 : 1]++; }
    assert.ok([...pairs.values()].every(counts => counts[0] === games / 2 && counts[1] === games / 2));
  }
  const partial = await createEvaluationExport(season.createSeason(field(4), 1, 2, undefined, true), { digest });
  assert.equal(partial.schema_version, 2); assert.equal((await verifyEvaluationExport(partial, { digest })).ok, true);
});
