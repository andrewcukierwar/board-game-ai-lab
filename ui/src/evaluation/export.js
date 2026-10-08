import { createSeason, validateSeason, SCHEDULE_VERSION, totalGames } from '../season/model.js';
import { seasonAnalytics, INITIAL_ELO, K_FACTOR, BOOTSTRAP_SAMPLES, MIN_INTERVAL_GAMES } from '../season/analytics.js';
import { canonicalJSON, evidenceDigest } from './canonical.js';
import { FORMAT, SCHEMA_VERSION, METHODOLOGY_VERSION, BACKEND_VERSIONS, sourceCommit, validateBackendProvenance } from './provenance.js';

export const METHODOLOGY = Object.freeze({
  version: METHODOLOGY_VERSION, schedule_version: SCHEDULE_VERSION,
  scoring: Object.freeze({ win: 1, draw: .5, loss: 0 }),
  standings_sort: 'points-desc_score-rate-desc_seed-number-asc',
  side_balance: 'each-pair-half-red-half-yellow_completed-season',
  elo: Object.freeze({ initial: INITIAL_ELO, k_factor: K_FACTOR, scale: 400,
    expected_score: '1/(1+10^((opponent-self)/400))', update_order: 'completion-index-ascending', version: 1 }),
  bootstrap: Object.freeze({ samples: BOOTSTRAP_SAMPLES, minimum_games: MIN_INTERVAL_GAMES,
    lower_percentile: .025, upper_percentile: .975, percentile: 'linear-interpolation-(n-1)*p',
    resampling: 'iid-observed-game-scores-with-replacement', prng: 'mulberry32-v1',
    seed_derivation: 'fnv1a-avalanche-v1', seed_domain: 'season:bootstrap:{entrant_id}', version: 1 }),
  schedule: Object.freeze({ pairing: 'seeded-circle-mirrored-halves-v1', seed_derivation: 'fnv1a-avalanche-v1',
    shuffle_domain: 'season:schedule:v1', color_domain: 'season:color:{cycle}:{round-index}:{fixture-index}',
    game_seed_domain: 'season:game:v1:{fixture_id}', move_columns: 'zero-based-0-through-6' }),
});
const equal = (a, b) => canonicalJSON(a) === canonicalJSON(b);
const requireEqual = (a, b, message) => { if (!equal(a, b)) throw new Error(message); };

function exportGame(g) {
  return { fixture_id: g.fixtureId, completion_index: g.completedIndex,
    red_entrant_id: g.redEntrantId, yellow_entrant_id: g.yellowEntrantId, game_seed: g.gameSeed,
    player_configs: g.playerConfigs.map(p => ({ ...p })), move_columns: [...g.columns], move_count: g.moveCount,
    terminal_result: { status: g.result.status, winner_index: g.result.winnerIndex },
    backend_provenance: g.backendProvenance ? validateBackendProvenance(g.backendProvenance) : null };
}
export function seasonEvidence(season) {
  const s = validateSeason(season); // Includes unfinished live evidence validation, before dropping it.
  return { season_seed: s.seasonSeed, field_size: s.fieldSize, games_per_pairing: s.gamesPerPairing,
    state: { status: s.status === 'complete' ? 'complete' : 'partial', scheduled_games: s.schedule.length, completed_games: s.completedGames.length },
    entrants: s.entrants.map(e => ({ entrant_id: e.entrantId, seed_number: e.seedNumber, config: { ...e.config } })),
    schedule: s.schedule.map(f => ({ fixture_id: f.fixtureId, round: f.round, cycle: f.cycle,
      red_entrant_id: f.redEntrantId, yellow_entrant_id: f.yellowEntrantId, game_seed: f.gameSeed })),
    completed_games: s.completedGames.map(exportGame) };
}
function internalConfig(p) {
  if (!p || typeof p !== 'object') throw new Error('Malformed entrant configuration.');
  const ordered = p.type === 'negamax' ? { type: p.type, depth: p.depth } : p.type === 'mcts' ? { type: p.type, simulations: p.simulations } : { type: p.type };
  requireEqual(p, ordered, 'Unknown or missing entrant configuration fields.');
  return ordered;
}
function apiConfig(p) {
  const ordered = p?.type === 'negamax' ? { type: p.type, depth: p.depth } : p?.type === 'mcts' ? { type: p.type, simulation_limit: p.simulation_limit } : { type: p?.type };
  requireEqual(p, ordered, 'Unknown or missing player configuration fields.'); return ordered;
}
export function reconstructEvidence(e) {
  if (!e || !Array.isArray(e.entrants) || !Array.isArray(e.schedule) || !Array.isArray(e.completed_games)) throw new Error('Missing canonical Season evidence arrays.');
  if (e.entrants.length > 12 || e.schedule.length > 528 || e.completed_games.length > 528) throw new Error('Season evidence exceeds supported schedule bounds.');
  let s = createSeason(e.entrants.map(row => internalConfig(row.config)), e.season_seed, e.games_per_pairing);
  for (const [index, g] of e.completed_games.entries()) {
    try {
      s.completedGames.push({ fixtureId: g.fixture_id, completedIndex: g.completion_index,
        redEntrantId: g.red_entrant_id, yellowEntrantId: g.yellow_entrant_id, gameSeed: g.game_seed,
        playerConfigs: g.player_configs.map(apiConfig), columns: g.move_columns, moveCount: g.move_count,
        result: { status: g.terminal_result.status, winnerIndex: g.terminal_result.winner_index },
        ...(g.backend_provenance === null ? {} : { backendProvenance: validateBackendProvenance(g.backend_provenance) }) });
      if (!s.schedule[index]) throw new Error('More completed games than scheduled games.');
      s.schedule[index].status = 'complete'; s.schedule[index].result = { ...s.completedGames[index].result };
    } catch (error) { throw new Error(`Completed game ${index} (${g?.fixture_id ?? 'unknown'}): ${error.message}`); }
  }
  s.currentGameIndex = s.completedGames.length; s.status = s.currentGameIndex === s.schedule.length ? 'complete' : 'paused';
  s = validateSeason(s);
  requireEqual(e, seasonEvidence(s), 'Canonical evidence differs from the reconstructed Season: check entrant IDs, exact schedule, seeds, colors, counts, results or unknown fields.');
  // Explicit count/balance check in addition to exact scheduler reconstruction.
  if (s.schedule.length !== totalGames(s.fieldSize, s.gamesPerPairing)) throw new Error('Incorrect schedule count.');
  const pairs = new Map();
  for (const f of s.schedule) {
    const ids = [f.redEntrantId, f.yellowEntrantId].sort(), key = ids.join(':');
    if (!pairs.has(key)) pairs.set(key, [0, 0]); pairs.get(key)[f.redEntrantId === ids[0] ? 0 : 1]++;
  }
  if (pairs.size !== s.fieldSize * (s.fieldSize - 1) / 2 || [...pairs.values()].some(counts => counts.some(n => n !== s.gamesPerPairing / 2))) throw new Error('Schedule is not pairwise Red/Yellow balanced.');
  return s;
}
const stats = r => ({ played: r.played, wins: r.wins, draws: r.draws, losses: r.losses, points: r.points, score_rate: r.scoreRate });
export function derivedAnalytics(s) {
  const a = seasonAnalytics(s);
  return { standings: a.standings.map(r => ({ entrant_id: r.entrantId, seed_number: r.seedNumber, ...stats(r), red: stats(r.red), yellow: stats(r.yellow) })),
    ratings: a.ratings.map(r => ({ entrant_id: r.entrantId, final_elo: r.rating, elo_change: r.rating - INITIAL_ELO,
      peak_elo: r.peak, low_elo: r.low, history: [...r.history] })),
    side_splits: a.standings.map(r => ({ entrant_id: r.entrantId, red: stats(r.red), yellow: stats(r.yellow) })),
    pairwise: Object.fromEntries(Object.entries(a.pairwise).map(([id, opponents]) => [id, Object.fromEntries(Object.entries(opponents).map(([other, r]) => [other, stats(r)]))])),
    bootstrap_intervals: Object.fromEntries(Object.entries(a.intervals).map(([id, r]) => [id, r && { lower: r.lower, upper: r.upper, samples: r.samples, games: r.played }])) };
}
export function evaluationProvenance(evidence, commit = null) {
  const captured = evidence.completed_games.filter(g => g.backend_provenance !== null).length;
  return { ...BACKEND_VERSIONS, agents: { ...BACKEND_VERSIONS.agents }, season_schedule_version: SCHEDULE_VERSION,
    evaluation_methodology_version: METHODOLOGY_VERSION, export_schema_version: SCHEMA_VERSION,
    source_commit: sourceCommit(commit), execution_capture: { recorded_games: captured,
      unrecorded_games: evidence.completed_games.length - captured, scope: 'per-game-history-response',
      declared_versions: 'reference-implementation-identifiers-not-authentication' } };
}
// Precisely this object is hashed. exported_at, derived and integrity are excluded.
export function canonicalPayload(artifact) {
  return { format: artifact.format, schema_version: artifact.schema_version, provenance: artifact.provenance,
    methodology: artifact.methodology, evidence: artifact.evidence };
}
export async function createEvaluationExport(season, { digest, exportedAt = new Date().toISOString(), sourceCommit: commit = null } = {}) {
  const evidence = seasonEvidence(season), s = reconstructEvidence(evidence);
  if (typeof exportedAt !== 'string' || !/^\d{4}-\d{2}-\d{2}T.*Z$/.test(exportedAt) || !Number.isFinite(Date.parse(exportedAt))) throw new Error('Export timestamp must be an ISO-8601 UTC timestamp.');
  const artifact = { format: FORMAT, schema_version: SCHEMA_VERSION, exported_at: exportedAt,
    provenance: evaluationProvenance(evidence, commit), methodology: structuredClone(METHODOLOGY), evidence, derived: derivedAnalytics(s) };
  artifact.integrity = { algorithm: 'SHA-256', canonicalization: 'sorted-json-ecmascript-v1',
    coverage: 'format,schema_version,provenance,methodology,evidence', evidence_sha256: await evidenceDigest(canonicalPayload(artifact), digest) };
  return artifact;
}
