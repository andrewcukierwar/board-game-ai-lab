import { researchAgent } from './researchAgent.js';

const publicAgents = [
  { type: 'random', name: 'Random', label: 'Baseline', description: 'Chooses uniformly from available legal moves.' },
  { type: 'negamax', name: 'Negamax', label: 'Adversarial search', description: 'Searches future move sequences and scores resulting positions.' },
  { type: 'mcts', name: 'MCTS', label: 'Monte Carlo Tree Search', description: 'Builds a search tree using simulated games and UCT-style exploration.' },
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
          onChange={() => select({ type: agent.type })} />
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
        <option value="1">1 — Quick</option><option value="2">2 — Balanced (default)</option><option value="4">4 — Stronger</option><option value="6">6 — Deep</option><option value="8">8 — Very deep</option>
      </select>
    </div>
    <div id="mcts-options" hidden={selection.type !== 'mcts'} className="agent-setting">
      <label htmlFor="opponent-simulations">Search simulations:</label>
      <select id="opponent-simulations" value={selection.simulations} disabled={busy} onChange={event => select({ simulations: Number(event.target.value) })}>
        <option value="100">100 — Quick (default)</option><option value="400">400 — Balanced</option><option value="800">800 — Deep</option>
      </select>
    </div>
    <p className="settings-note">Settings apply when you start a new game.{activeOpponent && <span>Current game: <strong>{activeOpponent}</strong>{activeConfig.depth ? ` · depth ${activeConfig.depth}` : activeConfig.simulation_limit ? ` · ${activeConfig.simulation_limit} simulations` : ''} · {game.players[0].type === 'human' ? 'You move first' : 'AI moves first'}.</span>}</p>
  </div>;
}
