import { useEffect, useState } from 'react';
import { entrant, roundLabel, matchupHasHuman, humanEntrant } from './model.js';
import { competitorLabel } from '../connect4/competitorConfig.js';
export const entrantLabel = (t, id) => id ? `#${entrant(t, id).seedNumber} · ${entrant(t, id).config.type === 'human' ? 'You' : competitorLabel(entrant(t, id).config)}` : 'Awaiting winner';
function MatchupCard({ tournament: t, matchup: m, selected, select }) {
  const human = matchupHasHuman(t, m), you = humanEntrant(t)?.entrantId;
  return <button className={`bracket-card ${human ? 'has-human' : ''} ${m.status === 'active' ? 'is-active' : ''} ${selected === m.matchupId ? 'is-selected' : ''}`}
    data-matchup={m.matchupId} aria-pressed={selected === m.matchupId} onClick={() => select(m.matchupId)}
    aria-label={`${roundLabel(t.size, m.round)}, matchup ${m.index + 1}: ${entrantLabel(t, m.entrantAId)} versus ${entrantLabel(t, m.entrantBId)}. ${human ? 'Your match. ' : ''}${m.status}. ${m.winnerEntrantId ? `Winner ${entrantLabel(t, m.winnerEntrantId)}. Replay` : 'Watch'}`}>
    <span className="card-round">Match {m.index + 1}{human && ' · Your match'} <span>{m.status === 'complete' ? 'Complete' : m.status === 'active' ? 'Active' : m.entrantAId && m.entrantBId ? 'Ready' : 'Pending'}</span></span>
    {[m.entrantAId, m.entrantBId].map((id, i) => <span key={i} className={`bracket-entrant ${id && id === m.winnerEntrantId ? 'is-winner' : ''}`}>
      <span>{entrantLabel(t, id)}</span>{id && id === you && <span className="you-badge">YOU</span>}{id && id === m.winnerEntrantId && <b aria-label="Advancing entrant">✓</b>}
    </span>)}
    <span className="card-footer">{human && m.status === 'complete' && `${m.winnerEntrantId === you ? 'You advance' : 'Eliminated'} · `}{m.resolution === 'seeded_draw_tiebreak' ? 'Seeded tiebreak · Replay' : m.status === 'complete' ? `${m.games.length} game${m.games.length > 1 ? 's' : ''} · Replay` : m.games.length ? `Game ${m.games.length + 1} · Draw rematch` : 'Watch / Continue'}<span aria-hidden="true">→</span></span>
  </button>;
}
export default function Bracket({ tournament: t, selected, select }) {
  const [round, setRound] = useState(0);
  useEffect(() => { const m = t.rounds.flat().find(m => m.matchupId === selected); if (m) setRound(m.round); }, [selected, t.tournamentId]);
  const mobileRound = Math.min(round, t.rounds.length - 1);
  return <section className="tournament-bracket" aria-labelledby="bracket-title">
    <div className="tournament-section-heading"><div><p className="eyebrow">The road to the final</p><h2 id="bracket-title">The bracket</h2></div>
      <span>{t.rounds.flat().filter(m => m.status === 'complete').length} / {t.size - 1} matchups complete</span></div>
    <p className="analysis-helper">Winners advance in pairs to the next round. Bracket order is separate from Red / Yellow game assignment. Select a matchup to watch or replay.</p>
    <div className="mobile-round-nav"><label htmlFor="tournament-round">View round</label><select id="tournament-round" value={mobileRound} onChange={e => setRound(Number(e.target.value))}>
      {t.rounds.map((_, r) => <option key={r} value={r}>{roundLabel(t.size, r)}{t.rounds[r].some(m => matchupHasHuman(t, m)) ? ' · Your path' : ''}</option>)}
    </select></div>
    <div className="desktop-bracket" tabIndex={0} aria-label="Tournament bracket, scroll horizontally for later rounds">
      <div className="bracket-columns">{t.rounds.map((matches, r) => <section key={r} className="bracket-round" aria-label={roundLabel(t.size, r)}>
        <h3>{roundLabel(t.size, r)}</h3><div className="bracket-round-matches" style={{ '--slots': t.size / 2 }}>
          {matches.map((m, i) => <div key={m.matchupId} className={`bracket-node ${r === t.rounds.length - 1 ? 'is-final' : ''}`}
            style={{ gridRow: `${i * 2 ** r + 1} / span ${2 ** r}` }}>
            <MatchupCard tournament={t} matchup={m} selected={selected} select={select} />
          </div>)}
        </div>
      </section>)}<section className="bracket-destination" aria-label="Champion destination"><h3>Champion</h3><div><span className="eyebrow">Tournament Champion</span><p>{t.championEntrantId ? entrantLabel(t, t.championEntrantId) : 'The final winner arrives here.'}</p></div></section></div>
    </div>
    <section className="mobile-round-list" aria-label={`${roundLabel(t.size, mobileRound)} matchups`}><h3>{roundLabel(t.size, mobileRound)}</h3>
      {t.rounds[mobileRound].map(m => <MatchupCard key={m.matchupId} tournament={t} matchup={m} selected={selected} select={select} />)}
      {mobileRound === t.rounds.length - 1 && <p className="mobile-champion">Tournament Champion · {t.championEntrantId ? entrantLabel(t, t.championEntrantId) : 'Awaiting final'}</p>}
    </section>
  </section>;
}
