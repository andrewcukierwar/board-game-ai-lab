import { useState } from 'react';
import { entrantLabel } from './model.js';
const colors = ['#91d5ed', '#f6c967', '#ce8dfa', '#ed948e', '#7cd1a3', '#f3a0d3', '#afc579', '#8baff0', '#c5cad2', '#e7ac72', '#77d4d4', '#d6b4fa'];
export default function RatingHistory({ season: s, ratings }) {
  const [selected, setSelected] = useState(() => s.entrants.slice(0, 4).map(e => e.entrantId));
  const shown = ratings.filter(r => selected.includes(r.entrantId));
  const values = shown.flatMap(r => r.history), low = Math.floor((Math.min(1480, ...values) - 10) / 25) * 25, high = Math.ceil((Math.max(1520, ...values) + 10) / 25) * 25;
  const x = i => 60 + i / Math.max(1, s.completedGames.length) * 700, y = value => 255 - (value - low) / (high - low) * 225;
  return <section className="season-panel season-history" aria-labelledby="elo-history-heading">
    <p className="eyebrow">Starting rating 1500</p><h2 id="elo-history-heading">Elo history</h2>
    <fieldset className="chart-choices"><legend>Entrants to compare</legend>{s.entrants.map((e, i) => <label key={e.entrantId}><input type="checkbox" checked={selected.includes(e.entrantId)} onChange={ev => setSelected(ev.target.checked ? [...selected, e.entrantId] : selected.filter(id => id !== e.entrantId))} /><span style={{ borderColor: colors[i] }}>{entrantLabel(s, e.entrantId)}</span></label>)}</fieldset>
    <svg viewBox="0 0 800 305" role="img" aria-labelledby="elo-chart-title elo-chart-desc">
      <title id="elo-chart-title">Elo rating history by completed season game</title><desc id="elo-chart-desc">{shown.length} selected entrants, {s.completedGames.length} completed games. Ratings begin at 1500. Current, peak and low ratings and full histories are in the table below.</desc>
      {[low, 1500, high].map(v => <g key={v}><line x1="60" x2="760" y1={y(v)} y2={y(v)} stroke="currentColor" opacity={v === 1500 ? .5 : .2} strokeDasharray={v === 1500 ? '5 5' : undefined} /><text x="52" y={y(v) + 4} textAnchor="end">{v}</text></g>)}
      {shown.map(r => { const i = s.entrants.findIndex(e => e.entrantId === r.entrantId); return <polyline key={r.entrantId} fill="none" stroke={colors[i]} strokeWidth="2" strokeDasharray={i % 3 === 1 ? '8 3' : i % 3 === 2 ? '3 3' : undefined} points={r.history.map((v, j) => `${x(j)},${y(v)}`).join(' ')}><title>{entrantLabel(s, r.entrantId)}: {Math.round(r.rating)} Elo</title></polyline>; })}
      <text x="60" y="278">0</text><text x="760" y="278" textAnchor="end">{s.completedGames.length}</text><text x="410" y="300" textAnchor="middle">Completed season games</text><text x="12" y="15">Elo</text>
    </svg>
    <div className="season-table-scroll" role="region" aria-label="Rating history summary" tabIndex={0}><table><caption>Current, peak and low Elo · full precision used internally</caption><thead><tr><th scope="col">Entrant</th><th scope="col">Start</th><th scope="col">Current</th><th scope="col">Peak</th><th scope="col">Low</th></tr></thead><tbody>{ratings.map(r => <tr key={r.entrantId}><th scope="row">{entrantLabel(s, r.entrantId)}</th><td>1500</td><td>{Math.round(r.rating)}</td><td>{Math.round(r.peak)}</td><td>{Math.round(r.low)}</td></tr>)}</tbody></table></div>
    <details><summary>Game-by-game rating values (text equivalent)</summary><div className="season-table-scroll" role="region" tabIndex={0} aria-label="Game by game Elo history"><table><thead><tr><th scope="col">Game</th>{shown.map(r => <th scope="col" key={r.entrantId}>{entrantLabel(s, r.entrantId)}</th>)}</tr></thead><tbody>{Array.from({ length: s.completedGames.length + 1 }, (_, i) => <tr key={i}><th scope="row">{i}</th>{shown.map(r => <td key={r.entrantId}>{r.history[i].toFixed(2)}</td>)}</tr>)}</tbody></table></div></details>
  </section>;
}
