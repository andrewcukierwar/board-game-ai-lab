import { ALPHAZERO } from './alphazero.js';
import AlphaZeroSettings from './AlphaZeroSettings.jsx';
import { competitorTypes, NEGAMAX_DEPTHS, MCTS_SIMULATIONS, TACTICAL_MCTS_SIMULATIONS, validDepth, validSimulations, competitorLabel, playerColor } from './competitorConfig.js';

import { researchAgent, VICTOR_RESEARCH } from './researchAgent.js';

export default function CompetitorSelector({ index, config, select, busy, researchEnabled, alphaZero }) {
  const id = `player-${index + 1}`;
  return <fieldset className={`competitor-selector competitor-selector--${index}`} disabled={busy}>
    <legend><span className={`legend-piece legend-piece--${index ? 'yellow' : 'red'}`} />Player {index + 1} · {playerColor(index)}<small>{index ? 'Moves second' : 'Moves first'}</small></legend>
    <div className="competitor-choices">
      {competitorTypes(researchEnabled, alphaZero?.available).map(type => <label key={type} className={`turn-option ${type === VICTOR_RESEARCH ? 'turn-option--research' : ''} ${config.type === type ? 'is-selected' : ''}`}>
        <input type="radio" name={id} value={type} checked={config.type === type} onChange={() => select(index, { type })} />
        <span>{competitorLabel({ type }).split(' · ')[0]}</span>
      </label>)}
    </div>
    <div className="competitor-budget">
      {config.type === ALPHAZERO ? <AlphaZeroSettings id={`${id}-guard`} guard={config.tactical_guard} change={tactical_guard => select(index, { tactical_guard })} /> : config.type === 'negamax' ? <><label htmlFor={`${id}-depth`}>Search depth</label>
        <select id={`${id}-depth`} value={config.depth} onChange={e => select(index, { depth: Number(e.target.value) })}>
          {!NEGAMAX_DEPTHS.includes(config.depth) && <option value={config.depth}>Custom depth {config.depth}</option>}
          {NEGAMAX_DEPTHS.map(depth => <option key={depth} value={depth}>Depth {depth}</option>)}
        </select><details className="advanced-search"><summary>Advanced · custom depth</summary>
          <label htmlFor={`${id}-custom-depth`}>Depth 1–12</label><input id={`${id}-custom-depth`} type="number" min="1" max="12" step="1" value={config.depth}
            aria-invalid={!validDepth(config.depth)} onChange={e => select(index, { depth: Number(e.target.value) })} />
        </details>{config.depth >= 9 && <p className="search-warning">Depth 9–12 uses a 20-second server-side safety limit.</p>}</> : ['mcts', 'mcts_tactical'].includes(config.type) ? <><label htmlFor={`${id}-simulations`}>Maximum simulations</label>
        <select id={`${id}-simulations`} value={config.simulations} onChange={e => select(index, { simulations: Number(e.target.value) })}>
          {!(config.type === 'mcts' ? MCTS_SIMULATIONS : TACTICAL_MCTS_SIMULATIONS).includes(config.simulations) &&
            <option value={config.simulations}>Custom · {config.simulations}</option>}
          {(config.type === 'mcts' ? MCTS_SIMULATIONS : TACTICAL_MCTS_SIMULATIONS).map(simulations =>
            <option key={simulations} value={simulations}>{simulations.toLocaleString()} simulations</option>)}
        </select><details className="advanced-search"><summary>Advanced · custom simulations</summary>
          <label htmlFor={`${id}-custom-simulations`}>1–{config.type === 'mcts' ? '25,000' : '10,000'}</label>
          <input id={`${id}-custom-simulations`} type="number" min="1" max={config.type === 'mcts' ? 25000 : 10000} step="1"
            value={config.simulations} aria-invalid={!validSimulations(config.simulations, config.type)}
            onChange={e => select(index, { simulations: Number(e.target.value) })} />
        </details></> : <p>{config.type === VICTOR_RESEARCH ? researchAgent.description : config.type === 'human' ? 'Choose a legal column on your turn.' : 'A uniformly chosen legal column.'}</p>}
    </div>
  </fieldset>;
}
