import { createTournament, defaultField, gamePlan, nextMatchup, matchupHasHuman, compactHistory, recordGame } from '../../src/tournament/model.js';
import { matchFixture, WIN } from './match.js';
export const YELLOW_WIN = [0, 1, 0, 1, 2, 1, 2, 1];
export function humanFixture(color = 0, matchIndex = 0, size = 8) {
  const field = defaultField(size), baseline = createTournament(field, 1234);
  const id = gamePlan(baseline, baseline.rounds[0][matchIndex]).playerEntrantIds[color];
  field[Number(id.split('-')[1]) - 1] = { type: 'human' };
  return createTournament(field, 1234);
}
export function resultFixture(win = true, size = 8, champion = false) {
  let t = humanFixture(0, 0, size);
  do {
    const m = nextMatchup(t), plan = gamePlan(t, m), human = matchupHasHuman(t, m);
    const color = plan.playerConfigs.findIndex(c => c.type === 'human');
    const columns = human && (win ? color === 1 : color === 0) ? YELLOW_WIN : WIN;
    t = recordGame(t, m.matchupId, compactHistory(t, m, { ...matchFixture(plan.playerConfigs, columns), rng_seed: plan.gameSeed }));
  } while (champion && nextMatchup(t));
  return t;
}
