import { competitorTypes, NEGAMAX_DEPTHS, MCTS_SIMULATIONS, competitorLabel, playerColor } from './competitorConfig.js';

import { researchAgent, VICTOR_RESEARCH } from './researchAgent.js';

export default function CompetitorSelector({ index, config, select, busy, researchEnabled }) {
  const id = `player-${index + 1}`;
  return <fieldset className={`competitor-selector competitor-selector--${index}`} disabled={busy}>
    <legend><span className={`legend-piece legend-piece--${index ? 'yellow' : 'red'}`} />Player {index + 1} · {playerColor(index)}<small>{index ? 'Moves second' : 'Moves first'}</small></legend>
    <div className="competitor-choices">
      {competitorTypes(researchEnabled).map(type => <label key={type} className={`turn-option ${type === VICTOR_RESEARCH ? 'turn-option--research' : ''} ${config.type === type ? 'is-selected' : ''}`}>
        <input type="radio" name={id} value={type} checked={config.type === type} onChange={() => select(index, { type })} />
        <span>{competitorLabel({ type }).split(' · ')[0]}</span>
      </label>)}
    </div>
    <div className="competitor-budget">
      {config.type === 'negamax' ? <><label htmlFor={`${id}-depth`}>Search depth</label>
        <select id={`${id}-depth`} value={config.depth} onChange={e => select(index, { depth: Number(e.target.value) })}>
          {NEGAMAX_DEPTHS.map(depth => <option key={depth} value={depth}>Depth {depth}</option>)}
        </select></> : config.type === 'mcts' ? <><label htmlFor={`${id}-simulations`}>Search simulations</label>
        <select id={`${id}-simulations`} value={config.simulations} onChange={e => select(index, { simulations: Number(e.target.value) })}>
          {MCTS_SIMULATIONS.map(simulations => <option key={simulations} value={simulations}>{simulations} simulations</option>)}
        </select></> : <p>{config.type === VICTOR_RESEARCH ? researchAgent.description : config.type === 'human' ? 'Choose a legal column on your turn.' : 'A uniformly chosen legal column.'}</p>}
    </div>
  </fieldset>;
}
