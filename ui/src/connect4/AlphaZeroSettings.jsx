import { alphaZeroAgent } from './alphazero.js';
export default function AlphaZeroSettings({ id, guard, change, disabled, showDescription = true }) {
  return <div className="agent-setting">{showDescription && <p className="agent-description">{alphaZeroAgent.description}</p>}
    <label htmlFor={id}>Tactical safeguard</label>
    <select id={id} value={guard === true ? 'on' : 'off'} disabled={disabled} onChange={e => change(e.target.value === 'on')}>
      <option value="off">Off</option><option value="on">On</option>
    </select>
    <p className="analysis-helper">Checks for immediate wins and avoids moves that allow an immediate opponent win when a safe alternative exists.</p>
    <details className="advanced-search"><summary>About the safeguard</summary><p>Same trained CNN, with optional one-ply blunder protection after search. Across two trained CNNs against Negamax3 and Negamax5, controlled evaluations measured a small +1.302 percentage-point match-score effect (95% interval +0.521 to +2.344), below the declared +2-point practical threshold. It does not guarantee stronger play in every position.</p></details>
  </div>;
}
