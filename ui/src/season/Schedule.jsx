import { useState } from 'react';
import { entrantLabel, currentFixture } from './model.js';
export const resultLabel = f => f.result ? f.result.status === 'draw' ? 'Draw' : `${f.result.winnerIndex === 0 ? 'Red' : 'Yellow'} win` : f.status;
export default function Schedule({ season: s, selected, select }) {
  const [round, setRound] = useState(null);
  const shown = round ?? currentFixture(s)?.round ?? s.schedule.at(-1).round;
  return <section className="season-panel season-schedule" aria-labelledby="schedule-heading">
    <div className="tournament-section-heading"><div><p className="eyebrow">Balanced return fixtures</p><h2 id="schedule-heading">Schedule progress</h2></div><span>{s.currentGameIndex} / {s.schedule.length} games</span></div>
    <progress aria-label="Completed season games" max={s.schedule.length} value={s.currentGameIndex} />
    <div className="season-round-nav"><label htmlFor="season-round">League round</label><select id="season-round" value={shown} onChange={e => setRound(Number(e.target.value))}>
      {Array.from({ length: s.schedule.at(-1).round }, (_, i) => <option key={i} value={i + 1}>Round {i + 1} · Cycle {Math.floor(i / (2 * (s.fieldSize - 1))) + 1}</option>)}</select>
      <button onClick={() => setRound(null)}>Current round</button></div>
    <div className="season-fixtures">{s.schedule.filter(f => f.round === shown).map(f => <button key={f.fixtureId} className={`season-fixture ${selected === f.fixtureId ? 'is-selected' : ''}`} aria-pressed={selected === f.fixtureId} onClick={() => select(f.fixtureId)}>
      <span className="card-round">Game {s.schedule.indexOf(f) + 1} · {resultLabel(f)}</span>
      <span><i className="piece-dot piece-dot--red" aria-hidden="true" />Red: {entrantLabel(s, f.redEntrantId)}</span>
      <span><i className="piece-dot piece-dot--yellow" aria-hidden="true" />Yellow: {entrantLabel(s, f.yellowEntrantId)}</span>
      <span className="season-fixture-footer">{f.status === 'complete' ? 'Open local replay' : f.status === 'active' ? 'View active fixture' : 'Preview fixture'}</span>
    </button>)}</div>
  </section>;
}
