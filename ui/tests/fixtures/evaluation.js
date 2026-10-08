import { createSeason, defaultField, gamePlan, replayColumns, validateSeason } from '../../src/season/model.js';
import { BACKEND_VERSIONS } from '../../src/evaluation/provenance.js';
import { DRAW, WIN } from '../../e2e/fixtures/match.js';
export { DRAW, WIN };
export const backend = () => ({ ...BACKEND_VERSIONS, agents: { ...BACKEND_VERSIONS.agents }, source_commit: null });
export function evaluationSeason({ size = 4, games = 4, seed = 1234, count, sequences = [WIN, DRAW, [0, 1, 0, 1, 2, 1, 2, 1]], captured = true, configs = defaultField(size) } = {}) {
  const s = createSeason(configs, seed, games);
  count ??= s.schedule.length;
  s.completedGames = s.schedule.slice(0, count).map((f, i) => {
    const plan = gamePlan(s, f), columns = [...sequences[i % sequences.length]], result = replayColumns(columns, plan.playerConfigs).result;
    f.status = 'complete'; f.result = { ...result };
    return { ...plan, columns, result, moveCount: columns.length, completedIndex: i, ...(captured ? { backendProvenance: backend() } : {}) };
  });
  s.currentGameIndex = count; s.status = count === s.schedule.length ? 'complete' : 'paused';
  return validateSeason(s);
}
