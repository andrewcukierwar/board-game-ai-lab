import { ALPHAZERO } from './alphazero.js';
import { AI_AGENT_ORDER, VICTOR_RESEARCH } from './researchCompetitor.js';
import { NEGAMAX_DEPTHS, MCTS_SIMULATIONS, TACTICAL_MCTS_SIMULATIONS } from './competitorConfig.js';

const random = () => ({ type: 'random' });
const negamax = depth => ({ type: 'negamax', depth });
const mcts = simulations => ({ type: 'mcts', simulations });
const tactical = simulations => ({ type: 'mcts_tactical', simulations });
const alphaZero = tactical_guard => ({ type: ALPHAZERO, tactical_guard });
const victor = () => ({ type: VICTOR_RESEARCH });

// Tournament and Season menus: lower-cost presets in canonical agent order, Human last.
export function entrantPresets({ victor: victorEnabled = false, alphaZero: alphaZeroAvailable = false, human = false } = {}) {
  return [random(), ...NEGAMAX_DEPTHS.map(negamax), ...MCTS_SIMULATIONS.filter(n => n <= 5000).map(mcts),
    ...TACTICAL_MCTS_SIMULATIONS.filter(n => n <= 2000).map(tactical),
    ...(alphaZeroAvailable ? [alphaZero(false), alphaZero(true)] : []), ...(victorEnabled ? [victor()] : []),
    ...(human ? [{ type: 'human' }] : [])];
}

// Priority order: the first eight cover every agent family; larger fields add further
// presets. Depths 10–12 and the largest MCTS budgets stay manual because of their runtime.
const RECOMMENDED = [negamax(2), mcts(500), tactical(400), alphaZero(false), random(), victor(), negamax(6), alphaZero(true),
  negamax(4), mcts(1000), tactical(800), negamax(8), tactical(200), mcts(2000), negamax(1), tactical(2000)];
// Brackets larger than the unique pool repeat mid-cost entrants, never Random or Victor.
const REPEATED = [negamax(4), mcts(1000), tactical(400), alphaZero(false), negamax(6), mcts(500), tactical(800), alphaZero(true), negamax(2)];
const rank = config => [AI_AGENT_ORDER.indexOf(config.type), config.depth ?? config.simulations ?? Number(config.tactical_guard ?? 0)];
const byRank = (a, b) => { const [x, y] = [rank(a), rank(b)]; return x[0] - y[0] || x[1] - y[1]; };

// New-field convenience only: saved competitions keep their own entrants.
export function recommendedField(size, { victor: victorEnabled = false, alphaZero: alphaZeroAvailable = false } = {}) {
  const available = config => config.type === VICTOR_RESEARCH ? victorEnabled : config.type === ALPHAZERO ? alphaZeroAvailable : true;
  const field = RECOMMENDED.filter(available).slice(0, size), repeated = REPEATED.filter(available);
  for (let i = 0; field.length < size; i++) field.push(repeated[i % repeated.length]);
  return field.map(config => ({ ...config })).sort(byRank);
}
