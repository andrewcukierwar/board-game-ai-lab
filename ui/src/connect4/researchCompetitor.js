// Authoritative optional competitor definition; no payload/validation imports.

export const VICTOR_RESEARCH = 'victor_research';
export const PUBLIC_PLAYER_TYPES = ['human', 'random', 'negamax', 'mcts'];
export const KNOWN_PLAYER_TYPES = [...PUBLIC_PLAYER_TYPES, VICTOR_RESEARCH];
export const RESEARCH_DISABLED = 'Victor Research execution is disabled in this UI. Saved evidence remains available; re-enable the research feature flag to continue.';
export const RESEARCH_CAVEAT = 'Seeds determine schedules, colors and seeded agents’ randomness. Victor uses wall-clock search budgets: CPU contention and hosting performance can change its moves. Recorded moves are historical evidence, not a guarantee of identical future play.';
export function hasResearch(configs) { return configs.some(c => c?.type === VICTOR_RESEARCH); }
export function assertResearchExecution(configs, enabled = researchAgentEnabled()) {
  if (hasResearch(configs) && !enabled) throw new Error(RESEARCH_DISABLED);
}
export function validResearchConfig(config) {
  return config?.type === VICTOR_RESEARCH && !Array.isArray(config) && Object.keys(config).length === 1 && Object.hasOwn(config, 'type');
}

export const researchAgent = {
  type: VICTOR_RESEARCH,
  name: 'Victor Research (Experimental)',
  label: 'Experimental research solver',
  description: 'Experimental bounded exact/strategic analysis with a heuristic fallback. It is not perfect play and can lose. It uses an exact opening book and native bounded proof search.',
};

// Strict like the API flag: only "true" or "1" enables it.
export function researchAgentEnabled(value = import.meta.env?.VITE_VICTOR_RESEARCH_ENABLED) {
  return value === 'true' || value === '1';
}

export function playerTypes(enabled) {
  return enabled ? [...PUBLIC_PLAYER_TYPES, VICTOR_RESEARCH] : PUBLIC_PLAYER_TYPES;
}

// Explain research failures without altering the recovery state machine.
export function researchErrorMessage(error, selection, competition = false) {
  const data = error.response?.data;
  if (selection?.type !== VICTOR_RESEARCH || !data) return null;
  if (data.code === 'invalid_agent') return competition ? 'Victor Research (Experimental) is disabled on this game server. Completed results are safe. Re-enable it on the server, then refresh history and explicitly continue or restart only an unconfirmed game.' : 'Victor Research (Experimental) is not enabled on this game server. Choose another opponent.';
  if (data.code === 'agent_failed') return 'Victor Research could not make a legal move. Refresh history, then explicitly continue or retry. Completed results are safe.';
  if (data.code === 'agent_busy') return 'Victor Research is busy with another game. Wait a moment; refresh history before explicitly continuing or retrying the AI move.';
  return null;
}
