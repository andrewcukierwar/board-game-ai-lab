import { assertResearchExecution, researchAgentEnabled } from '../connect4/researchAgent.js';
import { toPlayerPayload, competitorLabel } from '../connect4/competitorConfig.js';
import { validateMatchHistory } from '../connect4/matchRecord.js';
import { deriveSeed, validSeed, bracketOrder, entrantConfig, defaultField, replayColumns } from '../tournament/model.js';
import { validateBackendProvenance } from '../evaluation/provenance.js';
export { deriveSeed, validSeed, defaultField, replayColumns };
export const FIELD_SIZES = [4, 6, 8, 10, 12];
export const GAMES_PER_PAIRING = [2, 4, 8];
export const VERSION = 1;
export const SCHEDULE_VERSION = 1;
const check = (ok, message = 'Invalid season evidence.') => { if (!ok) throw new Error(message); };
const equal = (a, b) => JSON.stringify(a) === JSON.stringify(b);
export const totalGames = (size, games) => size * (size - 1) / 2 * games;
export const entrant = (s, id) => s.entrants.find(e => e.entrantId === id);
export const entrantLabel = (s, id) => { const e = entrant(s, id); return e ? `#${e.seedNumber} ${competitorLabel(e.config)}` : 'Unknown entrant'; };
export const currentFixture = s => s?.schedule[s.currentGameIndex] ?? null;
export function createSeason(configs, seed, gamesPerPairing = 2, seasonId = `season-${seed}`, researchEnabled = researchAgentEnabled()) {
  check(Array.isArray(configs) && FIELD_SIZES.includes(configs.length), 'Choose 4, 6, 8, 10 or 12 entrants.');
  check(GAMES_PER_PAIRING.includes(gamesPerPairing), 'Choose 2, 4 or 8 games per pairing.');
  check(validSeed(seed), 'Season seed must be a whole number from 0 to 4294967295.');
  check(typeof seasonId === 'string' && seasonId.length > 0 && seasonId.length <= 100);
  assertResearchExecution(configs, researchEnabled);
  const entrants = configs.map((config, i) => {
    check(config?.type !== 'human', 'Season Lab supports AI entrants only.');
    return { entrantId: `entrant-${i + 1}`, seedNumber: i + 1, config: entrantConfig(config) };
  });
  // Circle method: fix the first seeded entrant and rotate the remaining ring.
  const ring = bracketOrder(entrants.length, deriveSeed(seed, 'season:schedule:v1'));
  const halves = [];
  for (let r = 0; r < entrants.length - 1; r++) {
    halves.push(Array.from({ length: entrants.length / 2 }, (_, i) => [ring[i], ring[ring.length - 1 - i]]));
    ring.splice(1, 0, ring.pop());
  }
  const schedule = [];
  for (let cycle = 1; cycle <= gamesPerPairing / 2; cycle++) for (let half = 0; half < 2; half++) {
    for (const [r, pairs] of halves.entries()) for (const [i, pair] of pairs.entries()) {
      const fixtureId = `c${cycle}-h${half + 1}-r${r + 1}-f${i + 1}`;
      const reverse = Boolean(deriveSeed(seed, `season:color:${cycle}:${r}:${i}`) & 1) !== Boolean(half);
      const [redEntrantId, yellowEntrantId] = reverse ? [...pair].reverse() : pair;
      schedule.push({ fixtureId, round: (cycle - 1) * 2 * (entrants.length - 1) + half * (entrants.length - 1) + r + 1,
        cycle, redEntrantId, yellowEntrantId, gameSeed: deriveSeed(seed, `season:game:v1:${fixtureId}`), status: 'pending', result: null });
    }
  }
  return { version: VERSION, scheduleVersion: SCHEDULE_VERSION, seasonId, seasonSeed: seed, fieldSize: entrants.length,
    gamesPerPairing, entrants, schedule, currentGameIndex: 0, status: 'paused', completedGames: [], active: null, retainedGameId: null };
}
export function gamePlan(s, fixture = currentFixture(s)) {
  check(fixture);
  return { fixtureId: fixture.fixtureId, gameSeed: fixture.gameSeed, redEntrantId: fixture.redEntrantId, yellowEntrantId: fixture.yellowEntrantId,
    playerConfigs: [fixture.redEntrantId, fixture.yellowEntrantId].map(id => toPlayerPayload(entrant(s, id).config, true)) };
}
export function compactHistory(s, history) {
  const plan = gamePlan(s);
  const record = validateMatchHistory(history, { players: plan.playerConfigs, rng_seed: plan.gameSeed });
  check(record.game.gameOver, 'Only confirmed terminal games count in a season.');
  const columns = record.moves.map(m => m.column), { result } = replayColumns(columns, plan.playerConfigs);
  return { ...plan, columns, result, moveCount: columns.length, completedIndex: s.currentGameIndex,
    ...(history.provenance ? { backendProvenance: validateBackendProvenance(history.provenance) } : {}) };
}
function validateGame(s, game, index) {
  const plan = gamePlan(s, s.schedule[index]);
  check(game && equal(Object.keys(game).sort(), [...Object.keys(plan), 'columns', 'result', 'moveCount', 'completedIndex',
    ...(Object.hasOwn(game, 'backendProvenance') ? ['backendProvenance'] : [])].sort()));
  if (Object.hasOwn(game, 'backendProvenance')) validateBackendProvenance(game.backendProvenance);
  for (const key of Object.keys(plan)) check(equal(game[key], plan[key]), 'Game seed, colors or configuration do not match the schedule.');
  const { result } = replayColumns(game.columns, plan.playerConfigs);
  check(result.status !== 'ongoing' && equal(result, game.result) && game.moveCount === game.columns.length && game.completedIndex === index, 'Malformed completed result.');
}
function appendGame(s, game) {
  s.completedGames.push(structuredClone(game));
  const f = s.schedule[s.currentGameIndex++]; f.status = 'complete'; f.result = { ...game.result };
  s.active = null; s.status = s.currentGameIndex === s.schedule.length ? 'complete' : 'paused';
}
export function recordGame(s, game) {
  check(currentFixture(s), 'Season is already complete.'); validateGame(s, game, s.currentGameIndex);
  const copy = structuredClone(s); appendGame(copy, game); return copy;
}
const validId = id => id === null || typeof id === 'string' && id.length > 0 && id.length <= 64;
export function validateSeason(value) {
  check(value && value.version === VERSION && value.scheduleVersion === SCHEDULE_VERSION, 'Unsupported season schema.');
  check(Array.isArray(value.entrants) && value.entrants.length === value.fieldSize);
  const rebuilt = createSeason(value.entrants.map(e => e.config), value.seasonSeed, value.gamesPerPairing, value.seasonId, true);
  check(equal(rebuilt.entrants, value.entrants), 'Invalid entrant identities.');
  check(Array.isArray(value.completedGames) && value.completedGames.length <= rebuilt.schedule.length);
  for (const [i, game] of value.completedGames.entries()) { validateGame(rebuilt, game, i); appendGame(rebuilt, game); }
  check(value.currentGameIndex === rebuilt.currentGameIndex && value.status === rebuilt.status, 'Impossible season progress.');
  check(validId(value.retainedGameId));
  if (value.active !== null) {
    const a = value.active, f = currentFixture(rebuilt);
    check(a && f && a.fixtureId === f.fixtureId && ['starting', 'running', 'interrupted'].includes(a.status) && validId(a.gameId));
    check(a.status !== 'running' || a.gameId !== null);
    check(a.gameId !== null || Array.isArray(a.columns) && a.columns.length === 0, 'Unconfirmed starts cannot have moves.');
    check(a.gameId === null || a.gameId === value.retainedGameId, 'Session replacement identity mismatch.');
    check(a.status !== 'starting' || a.gameId === null && Array.isArray(a.columns) && a.columns.length === 0);
    replayColumns(a.columns, gamePlan(rebuilt).playerConfigs);
    rebuilt.active = { fixtureId: a.fixtureId, gameId: a.gameId, status: a.status, columns: [...a.columns] };
    f.status = a.status === 'interrupted' ? 'interrupted' : 'active';
  }
  check(equal(value.schedule, rebuilt.schedule), 'Impossible schedule, fixture identity, seed or color balance.');
  // Derived aggregates are never accepted as canonical evidence.
  check(!['ratings', 'standings', 'ratingHistory'].some(key => Object.hasOwn(value, key)), 'Persist results rather than ratings.');
  rebuilt.retainedGameId = value.retainedGameId;
  return rebuilt;
}
