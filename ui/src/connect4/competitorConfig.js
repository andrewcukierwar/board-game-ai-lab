// Public presets; translate the UI model at the API boundary only.
import { PUBLIC_PLAYER_TYPES, VICTOR_RESEARCH, researchAgent, researchAgentEnabled, assertResearchExecution, validResearchConfig } from './researchCompetitor.js';
export const COMPETITOR_TYPES = PUBLIC_PLAYER_TYPES;
export const competitorTypes = (enabled = researchAgentEnabled()) => enabled ? [...COMPETITOR_TYPES, VICTOR_RESEARCH] : COMPETITOR_TYPES;
export const NEGAMAX_DEPTHS = [1, 2, 4, 6, 8];
export const MCTS_SIMULATIONS = [100, 400, 800];
export const DEFAULT_COMPETITOR = { type: 'negamax', depth: 2, simulations: 100 };
export const playerColor = index => index === 0 ? 'Red' : 'Yellow';

export function competitorLabel(config) {
  if (config.type === VICTOR_RESEARCH) return researchAgent.name;
  if (config.type === 'negamax') return `Negamax · depth ${config.depth ?? 2}`;
  if (config.type === 'mcts') return `MCTS · ${config.simulations ?? config.simulation_limit ?? 100} simulations`;
  return config.type === 'human' ? 'Human' : 'Random';
}

export function toPlayerPayload(config, researchEnabled = researchAgentEnabled()) {
  if (config?.type === VICTOR_RESEARCH) {
    if (!validResearchConfig(config)) throw new Error('Victor accepts only its type; no public search settings.');
    assertResearchExecution([config], researchEnabled);
    return { type: VICTOR_RESEARCH };
  }
  if (!COMPETITOR_TYPES.includes(config.type)) throw new Error('Unsupported competitor.');
  if (config.type === 'negamax') {
    const depth = config.depth ?? 2;
    if (!NEGAMAX_DEPTHS.includes(depth)) throw new Error('Choose a public Negamax depth.');
    return { type: config.type, depth };
  }
  if (config.type === 'mcts') {
    const simulations = config.simulations ?? config.simulation_limit ?? 100;
    if (!MCTS_SIMULATIONS.includes(simulations)) throw new Error('Choose a public MCTS simulation budget.');
    return { type: config.type, simulation_limit: simulations };
  }
  return { type: config.type };
}

export function matchStartPayload(player1, player2, replaceId, researchEnabled = researchAgentEnabled()) {
  return { player1: toPlayerPayload(player1, researchEnabled), player2: toPlayerPayload(player2, researchEnabled),
    ...(replaceId ? { replace_game_id: replaceId } : {}) };
}
