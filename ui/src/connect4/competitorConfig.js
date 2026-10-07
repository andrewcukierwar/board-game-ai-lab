// Public presets; translate the UI model at the API boundary only.
export const COMPETITOR_TYPES = ['human', 'random', 'negamax', 'mcts'];
export const NEGAMAX_DEPTHS = [1, 2, 4, 6, 8];
export const MCTS_SIMULATIONS = [100, 400, 800];
export const DEFAULT_COMPETITOR = { type: 'negamax', depth: 2, simulations: 100 };
export const playerColor = index => index === 0 ? 'Red' : 'Yellow';

export function competitorLabel(config) {
  if (config.type === 'negamax') return `Negamax · depth ${config.depth ?? 2}`;
  if (config.type === 'mcts') return `MCTS · ${config.simulations ?? config.simulation_limit ?? 100} simulations`;
  return config.type === 'human' ? 'Human' : 'Random';
}

export function toPlayerPayload(config) {
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

export function matchStartPayload(player1, player2, replaceId) {
  return { player1: toPlayerPayload(player1), player2: toPlayerPayload(player2),
    ...(replaceId ? { replace_game_id: replaceId } : {}) };
}
