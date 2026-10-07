const agents = [
  { type: 'random', name: 'Random', label: 'Baseline', description: 'Chooses uniformly from available legal moves.' },
  { type: 'negamax', name: 'Negamax', label: 'Adversarial search', description: 'Searches future move sequences and scores resulting positions.' },
  { type: 'mcts', name: 'MCTS', label: 'Monte Carlo Tree Search', description: 'Builds a search tree using simulated games and UCT-style exploration.' },
];

export default function AgentSelector({ selection, select, busy, game }) {
  const activeOpponent = agents.find(agent => agent.type === game?.players[1].type)?.name;
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
    <div id="negamax-options" hidden={selection.type !== 'negamax'} className="agent-setting">
      <label htmlFor="opponent-depth">Search depth:</label>
      <select id="opponent-depth" value={selection.depth} disabled={busy} onChange={event => select({ depth: Number(event.target.value) })}>
        <option value="1">1 — Quick</option><option value="2">2 — Balanced (default)</option><option value="3">3</option><option value="4">4 — Deeper</option>
      </select>
    </div>
    <div id="mcts-options" hidden={selection.type !== 'mcts'} className="agent-setting">
      <label htmlFor="opponent-simulations">Search simulations:</label>
      <select id="opponent-simulations" value={selection.simulations} disabled={busy} onChange={event => select({ simulations: Number(event.target.value) })}>
        <option value="50">50 — Quick</option><option value="100">100 — Balanced (default)</option><option value="250">250 — Deeper</option>
      </select>
    </div>
    <p className="settings-note">Settings apply when you start a new game.{activeOpponent && <span>Current game: <strong>{activeOpponent}</strong>{game.players[1].depth ? ` · depth ${game.players[1].depth}` : game.players[1].simulation_limit ? ` · ${game.players[1].simulation_limit} simulations` : ''}.</span>}</p>
  </div>;
}
