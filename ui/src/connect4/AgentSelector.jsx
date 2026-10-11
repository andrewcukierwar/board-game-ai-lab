import { researchAgent } from './researchAgent.js';
import { NEGAMAX_DEPTHS, MCTS_SIMULATIONS, TACTICAL_MCTS_SIMULATIONS, validDepth, validSimulations } from './competitorConfig.js';

const publicAgents = [
  { type: 'random', name: 'Random', label: 'Baseline', description: 'Chooses uniformly from available legal moves.' },
  { type: 'negamax', name: 'Negamax', label: 'Adversarial search', description: 'Searches future move sequences and scores resulting positions.' },
  { type: 'mcts', name: 'Classic MCTS', label: 'Monte Carlo Tree Search', description: 'Random-rollout MCTS with UCT exploration and root tactical checks.' },
  { type: 'mcts_tactical', name: 'Tactical MCTS', label: 'Solver-enhanced search', description: 'F1: tactical rollouts, proven win/loss propagation and tuned exploration. Search can stop early when a position is solved.' },
];

export default function AgentSelector({ selection, select, busy, game, researchEnabled = false }) {
  const agents = researchEnabled ? [...publicAgents, researchAgent] : publicAgents;
  const activeConfig = game?.players.find(player => player.type !== 'human');
  const activeOpponent = agents.find(agent => agent.type === activeConfig?.type)?.name;
  return <div className="agent-selection">
    <p className="eyebrow">Choose an approach</p><h2>Opponent</h2>
    <fieldset id="opponent-type" disabled={busy}>
      <legend className="visually-hidden">Opponent:</legend>
      {agents.map(agent => <label className={`agent-option ${selection.type === agent.type ? 'is-selected' : ''}`} key={agent.type}>
        <input type="radio" name="opponent" value={agent.type} checked={selection.type === agent.type}
          onChange={() => select({ type: agent.type, ...(agent.type === 'mcts' ? { simulations: 2000 } : agent.type === 'mcts_tactical' ? { simulations: 800 } : {}) })} />
        <span><strong>{agent.name}</strong><small>{agent.label}</small></span>
      </label>)}
    </fieldset>
    <p className="agent-description">{agents.find(agent => agent.type === selection.type).description}</p>
    <fieldset className="turn-order" disabled={busy}>
      <legend>Who moves first?</legend>
      {['human', 'ai'].map(first => <label className={`turn-option ${selection.first === first ? 'is-selected' : ''}`} key={first}>
        <input id={`first-${first}`} type="radio" name="first-player" value={first} checked={selection.first === first}
          onChange={() => select({ first })} />
        <span>{first === 'human' ? 'You go first' : 'AI goes first'}</span>
      </label>)}
    </fieldset>
    <div id="negamax-options" hidden={selection.type !== 'negamax'} className="agent-setting">
      <label htmlFor="opponent-depth">Search depth:</label>
      <select id="opponent-depth" value={selection.depth} disabled={busy} onChange={event => select({ depth: Number(event.target.value) })}>
        {!NEGAMAX_DEPTHS.includes(selection.depth) && <option value={selection.depth}>Custom · depth {selection.depth}</option>}
        {NEGAMAX_DEPTHS.map(depth => <option key={depth} value={depth}>{depth}{depth >= 10 ? ' — Experimental' : ''}</option>)}
      </select>
      <details className="advanced-search"><summary>Advanced · custom depth</summary>
        <label htmlFor="opponent-custom-depth">Enter a depth (1–12)</label>
        <input id="opponent-custom-depth" type="number" min="1" max="12" step="1" value={selection.depth}
          disabled={busy} aria-invalid={!validDepth(selection.depth)} onChange={event => select({ depth: Number(event.target.value) })} />
      </details>
      {selection.depth >= 9 && <p className="search-warning">Advanced search may be slow. After 20 seconds the server stops without making a move; retry at a lower depth.</p>}
    </div>
    <div id="mcts-options" hidden={!['mcts', 'mcts_tactical'].includes(selection.type)} className="agent-setting">
      <label htmlFor="opponent-simulations">Maximum simulations:</label>
      <select id="opponent-simulations" value={selection.simulations} disabled={busy} onChange={event => select({ simulations: Number(event.target.value) })}>
        {!(selection.type === 'mcts' ? MCTS_SIMULATIONS : TACTICAL_MCTS_SIMULATIONS).includes(selection.simulations) &&
          <option value={selection.simulations}>Custom · {selection.simulations}</option>}
        {(selection.type === 'mcts' ? MCTS_SIMULATIONS : TACTICAL_MCTS_SIMULATIONS).map(n =>
          <option key={n} value={n}>{n.toLocaleString()} simulations</option>)}
      </select>
      <details className="advanced-search"><summary>Advanced · custom simulations</summary>
        <label htmlFor="opponent-custom-simulations">Enter a cap (1–{selection.type === 'mcts' ? '25,000' : '10,000'})</label>
        <input id="opponent-custom-simulations" type="number" min="1" max={selection.type === 'mcts' ? 25000 : 10000} step="1"
          value={selection.simulations} disabled={busy} aria-invalid={!validSimulations(selection.simulations, selection.type)}
          onChange={event => select({ simulations: Number(event.target.value) })} />
      </details>
      {selection.type === 'mcts_tactical' && <p className="search-warning">F1 may stop before its cap when it proves an outcome.</p>}
    </div>
    <p className="settings-note">Settings apply when you start a new game.{activeOpponent && <span>Current game: <strong>{activeOpponent}</strong>{activeConfig.depth ? ` · depth ${activeConfig.depth}` : activeConfig.simulation_limit ? ` · ${activeConfig.simulation_limit} max simulations` : ''} · {game.players[0].type === 'human' ? 'You move first' : 'AI moves first'}.</span>}</p>
  </div>;
}
