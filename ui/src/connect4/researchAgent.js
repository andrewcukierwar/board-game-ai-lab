import { ALPHAZERO } from './alphazero.js';
export * from './researchCompetitor.js';
import { VICTOR_RESEARCH } from './researchCompetitor.js';
import { toPlayerPayload } from './competitorConfig.js';

// Play-page selection includes turn order and inactive ordinary-agent controls.
export function opponentPayload(selection, enabled) {
  if (selection.type === ALPHAZERO) return toPlayerPayload({ type: ALPHAZERO, tactical_guard: selection.tactical_guard ?? false });
  if (selection.type === VICTOR_RESEARCH) {
    if (!enabled) throw new Error('Unsupported competitor.');
    if (Object.keys(selection).some(key => !['type', 'first', 'depth', 'simulations', 'tactical_guard'].includes(key))) {
      throw new Error('Unexpected Victor selection fields.');
    }
    return toPlayerPayload({ type: VICTOR_RESEARCH }, enabled);
  }
  return toPlayerPayload(selection);
}
