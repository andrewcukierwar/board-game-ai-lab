// Public presets; translate the UI model at the API boundary only.
import { PUBLIC_PLAYER_TYPES, VICTOR_RESEARCH, researchAgent, researchAgentEnabled, assertResearchExecution, validResearchConfig } from './researchCompetitor.js';
export const COMPETITOR_TYPES = PUBLIC_PLAYER_TYPES;
export const competitorTypes = (enabled = researchAgentEnabled()) => enabled ? [...COMPETITOR_TYPES, VICTOR_RESEARCH] : COMPETITOR_TYPES;
export const NEGAMAX_DEPTHS = [1, 2, 4, 6, 8, 10, 12];
export const MCTS_SIMULATIONS = [500, 1000, 2000, 5000, 10000, 25000];
export const TACTICAL_MCTS_SIMULATIONS = [200, 400, 800, 2000, 4000, 10000];
// Accept historical game settings even if the current preset menus omit them.
export const SEARCH_LIMITS = Object.freeze({ negamax: 12, mcts: 25000, mcts_tactical: 10000 });
export const validDepth = n => Number.isInteger(n) && n >= 1 && n <= SEARCH_LIMITS.negamax;
export const validSimulations = (n, type) => Number.isInteger(n) && n >= 1 && n <= SEARCH_LIMITS[type];
export const DEFAULT_COMPETITOR = { type: 'negamax', depth: 2, simulations: 2000 };
export const playerColor = index => index === 0 ? 'Red' : 'Yellow';

export function competitorLabel(config) {
  if (config.type === VICTOR_RESEARCH) return researchAgent.name;
  if (config.type === 'negamax') return `Negamax · depth ${config.depth ?? 2}`;
  if (config.type === 'mcts') return `Classic MCTS · ${config.simulations ?? config.simulation_limit ?? 2000} simulations`;
  if (config.type === 'mcts_tactical') return `Tactical MCTS · ${config.simulations ?? config.simulation_limit ?? 800} max simulations`;
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
    if (!validDepth(depth)) throw new Error('Choose an integer Negamax depth from 1 to 12.');
    return { type: config.type, depth };
  }
  if (config.type === 'mcts' || config.type === 'mcts_tactical') {
    const simulations = config.simulations ?? config.simulation_limit ?? (config.type === 'mcts' ? 2000 : 800);
    if (!validSimulations(simulations, config.type)) throw new Error('Choose a valid integer MCTS simulation budget.');
    return { type: config.type, simulation_limit: simulations };
  }
  return { type: config.type };
}

export function matchStartPayload(player1, player2, replaceId, researchEnabled = researchAgentEnabled()) {
  return { player1: toPlayerPayload(player1, researchEnabled), player2: toPlayerPayload(player2, researchEnabled),
    ...(replaceId ? { replace_game_id: replaceId } : {}) };
}
