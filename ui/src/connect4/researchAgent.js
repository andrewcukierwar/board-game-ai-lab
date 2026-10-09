// Opt-in experimental Victor research agent for the Connect 4 play page only.
// Hidden unless the UI build sets VITE_VICTOR_RESEARCH_ENABLED=true AND the API
// sets VICTOR_RESEARCH_ENABLED=true; a server without the flag rejects it and
// the page reports that clearly. Match, tournament and season labs, saved data
// and exports keep the public agent list.
import { PUBLIC_PLAYER_TYPES } from './gameSnapshot.js';
import { toPlayerPayload } from './competitorConfig.js';

export const VICTOR_RESEARCH = 'victor_research';

export const researchAgent = {
  type: VICTOR_RESEARCH,
  name: 'Victor Research (Experimental)',
  label: 'Experimental research solver',
  description: 'An experimental solver: exact opening book, exact endgame search and Allis-style strategy rules, with a heuristic fallback. It is not perfect play and can lose. A move can take about a second.',
};

// Strict like the API flag: only "true" or "1" enables it.
export function researchAgentEnabled(value = import.meta.env?.VITE_VICTOR_RESEARCH_ENABLED) {
  return value === 'true' || value === '1';
}

export function playerTypes(enabled) {
  return enabled ? [...PUBLIC_PLAYER_TYPES, VICTOR_RESEARCH] : PUBLIC_PLAYER_TYPES;
}

export function opponentPayload(selection, enabled) {
  if (selection.type === VICTOR_RESEARCH) {
    if (!enabled) throw new Error('Unsupported competitor.');
    return { type: VICTOR_RESEARCH };
  }
  return toPlayerPayload(selection);
}

// Server responses keep their own wording; only clarify the two research cases.
export function researchErrorMessage(error, selection) {
  const data = error.response?.data;
  if (selection?.type !== VICTOR_RESEARCH || !data) return null;
  if (data.code === 'invalid_agent') return 'Victor Research (Experimental) is not enabled on this game server. Choose another opponent.';
  if (data.code === 'agent_busy') return 'Victor Research is busy with another game. Wait a moment, then retry the AI move.';
  return null;
}
