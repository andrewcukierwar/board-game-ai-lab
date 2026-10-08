import { useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import axios from 'axios';
import { competitorLabel, NEGAMAX_DEPTHS, MCTS_SIMULATIONS } from '../connect4/competitorConfig.js';
import { FIELD_SIZES, GAMES_PER_PAIRING, defaultField, validSeed, currentFixture, totalGames, entrantLabel } from '../season/model.js';
import { useSeason } from '../season/useSeason.js';
import { seasonAnalytics, K_FACTOR, INITIAL_ELO, BOOTSTRAP_SAMPLES } from '../season/analytics.js';
import Schedule from '../season/Schedule.jsx';
import MatchViewer from '../season/MatchViewer.jsx';
import RatingHistory from '../season/RatingHistory.jsx';
import { Standings, Ratings, Pairwise } from '../season/AnalyticsViews.jsx';
import '../../connect4/connect4.css';
import './match-lab.css';
import './tournament-lab.css';
import './season-lab.css';
const seasonHttp = axios.create({ baseURL: import.meta.env?.VITE_API_BASE, timeout: 90000 });
const presets = [{ type: 'random' }, ...NEGAMAX_DEPTHS.map(depth => ({ type: 'negamax', depth })), ...MCTS_SIMULATIONS.map(simulations => ({ type: 'mcts', simulations }))];
const randomSeed = () => globalThis.crypto.getRandomValues(new Uint32Array(1))[0];
const modes = { paused: 'Paused', game: 'Running current game', round: 'Running current round', season: 'Running season' };
export default function SeasonLabPage({ http = seasonHttp, storage }) {
  const state = useSeason(http, storage), { season: s, controller: c } = state;
  const [size, setSize] = useState(8), [games, setGames] = useState(2), [field, setField] = useState(() => defaultField(8)), [seed, setSeed] = useState(() => String(randomSeed()));
  const [setup, setSetup] = useState(false), [selected, setSelected] = useState(null), [setupError, setSetupError] = useState(''), [epoch, setEpoch] = useState(0);
  const analytics = useMemo(() => s ? seasonAnalytics(s) : null, [s?.completedGames, s?.entrants, s?.seasonSeed]);
  const shown = s && (s.schedule.find(f => f.fixtureId === selected) ?? currentFixture(s) ?? s.schedule.at(-1));
  const blocked = state.busy || state.uncertain || Boolean(state.error) || s?.status === 'complete';
  function create(e) {
    e.preventDefault(); const parsedSeed = /^\d+$/.test(seed) ? Number(seed) : NaN;
    if (!validSeed(parsedSeed)) { setSetupError('Enter a whole number from 0 to 4294967295.'); return; }
    try { c.create(field, parsedSeed, games); } catch (error) { setSetupError(error.message); return; }
    setSetup(false); setSelected(null); setSetupError(''); setEpoch(n => n + 1);
  }
  function run(mode) { setSelected(null); setEpoch(n => n + 1); c.returnLive(); c.run(mode); }
  return <main className="connect4-wrapper match-lab tournament-lab season-lab site-container">
    <header className="game-page-heading"><Link className="game-home-link" to="/connect4/tournament">Back to Tournament Lab</Link><p className="eyebrow">Comparative evaluation · Connect 4</p><h1>Season Lab</h1><p>Run balanced repeated matchups, track standings, and measure how the agents perform over a full season.</p></header>
    <div className="tournament-status season-status" role="status" aria-live="polite" aria-atomic="true">{s?.status === 'complete' ? 'Season Complete' : state.waiting || state.busy && state.uncertain ? 'Recovering' : state.uncertain ? 'Recovery required · Paused' : modes[state.mode]}</div>
    {(state.error || setupError) && <p className="tournament-error" role="alert">{setupError || state.error}</p>}
    {!state.storageAvailable && <p className="analysis-helper">Browser storage is unavailable. Keep this page open to retain season progress.</p>}
    {!s || setup ? <form className="tournament-setup" aria-label="Season setup" onSubmit={create}>
      <div className="tournament-section-heading"><div><p className="eyebrow">Repeated comparisons</p><h2>Season setup</h2></div><span>AI-only · duplicate configurations allowed</span></div>
      <div className="setup-fields season-setup-fields"><div><label htmlFor="season-size">Field size</label><select id="season-size" value={size} onChange={e => { const n = Number(e.target.value); setSize(n); setField(defaultField(n)); }}>{FIELD_SIZES.map(n => <option value={n} key={n}>{n} entrants</option>)}</select></div>
        <div><label htmlFor="season-games">Games per pairing</label><select id="season-games" value={games} onChange={e => setGames(Number(e.target.value))}>{GAMES_PER_PAIRING.map(n => <option value={n} key={n}>{n} games · {n / 2} per color</option>)}</select></div>
        <div><label htmlFor="season-seed">Season seed</label><input id="season-seed" type="text" inputMode="numeric" value={seed} onChange={e => setSeed(e.target.value)} aria-describedby="season-seed-help" /></div></div>
      <p id="season-seed-help" className="analysis-helper">Whole number 0–4294967295. The same field, slot order, seed and season length reproduce the schedule, colors and game seeds.</p>
      <p className="season-workload" role="status">{size * (size - 1) / 2} pairings × {games} = <strong>{totalGames(size, games)} scheduled games.</strong> Higher-depth Negamax and MCTS entrants may make a full season take several minutes or longer.</p>
      <div className="field-heading"><h3>Entrants</h3><button type="button" onClick={() => setField(defaultField(size))}>Reset default field</button></div>
      <div className="entrant-field">{field.map((config, i) => <div className="entrant-row" key={i}><label htmlFor={`season-entrant-${i + 1}`}>#{i + 1}</label><select id={`season-entrant-${i + 1}`} aria-label={`Entrant ${i + 1} configuration`} value={presets.findIndex(p => JSON.stringify(p) === JSON.stringify(config))} onChange={e => setField(field.map((p, j) => j === i ? { ...presets[Number(e.target.value)] } : p))}>{presets.map((p, j) => <option value={j} key={j}>{competitorLabel(p)}</option>)}</select></div>)}</div>
      <div className="setup-submit"><button id="season-create" className="action-link action-link--primary" disabled={state.busy}>Create season</button>{s && <button type="button" onClick={() => setSetup(false)}>Cancel new season</button>}</div>
    </form> : <>
      <section className="tournament-summary" aria-label="Season summary"><div><span className="eyebrow">Exact Red / Yellow balance</span><h2>{s.fieldSize} entrants <span>·</span> {s.gamesPerPairing} games per pairing</h2><p>Seed {s.seasonSeed} · {s.currentGameIndex} of {s.schedule.length} games complete · {s.schedule.at(-1).round} rounds</p></div><button disabled={state.busy} onClick={() => { c.pause(); setSize(s.fieldSize); setGames(s.gamesPerPairing); setField(s.entrants.map(e => ({ ...e.config }))); setSeed(String(randomSeed())); setSetup(true); }}>New season</button></section>
      {s.status === 'complete' && <section className="season-leaders season-panel" aria-label="Season Complete"><h2>Season Complete</h2><div><article><p className="eyebrow">Standings Leader</p><h3>{entrantLabel(s, analytics.standings[0].entrantId)}</h3><p>{analytics.standings[0].points} / {analytics.standings[0].played} points</p></article><article><p className="eyebrow">Elo Leader</p><h3>{entrantLabel(s, analytics.ratings[0].entrantId)}</h3><p>{Math.round(analytics.ratings[0].rating)} Elo</p></article></div></section>}
      <section className="tournament-controls" aria-label="Season execution" aria-busy={state.busy}><p className="eyebrow">One live game · sequential execution</p><h2>Season controls</h2><div className="tournament-actions">{[['game', 'Autoplay game'], ['round', 'Run round'], ['season', 'Run season']].map(([mode, label]) => <button id={`season-run-${mode}`} key={mode} disabled={blocked || state.mode !== 'paused'} onClick={() => run(mode)}>{label}</button>)}<button id="season-pause" disabled={state.mode === 'paused'} onClick={() => c.pause()}>Pause</button>{(s.active || state.error && !state.uncertain) && (state.uncertain || state.error) && <button id="season-refresh" disabled={state.busy || s.active?.status === 'interrupted'} onClick={() => c.refresh()}>{s.active ? 'Refresh history / Continue' : 'Continue after confirmed result'}</button>}</div><p className="analysis-helper">Pause lets the in-flight request settle safely. A draw counts immediately; there are no rematches.</p></section>
      <nav className="season-sections" aria-label="Season dashboard sections"><a href="#schedule-heading">Schedule</a><a href="#standings-heading">Standings</a><a href="#ratings-heading">Ratings</a><a href="#elo-history-heading">Elo history</a><a href="#pairwise-heading">Pairwise</a></nav>
      <Schedule key={`schedule:${s.seasonId}:${s.fieldSize}:${s.gamesPerPairing}`} season={s} selected={shown?.fixtureId} select={id => { c.pause(); c.returnLive(); setSelected(id); setEpoch(n => n + 1); }} />
      {shown && <MatchViewer key={`${shown.fixtureId}:${epoch}`} season={s} fixture={shown} state={state} controller={c} />}
      <Standings season={s} rows={analytics.standings} /><div className="season-rating-group"><Ratings season={s} ratings={analytics.ratings} standings={analytics.standings} intervals={analytics.intervals} />
      <RatingHistory key={`history:${s.seasonId}:${s.fieldSize}:${s.gamesPerPairing}`} season={s} ratings={analytics.ratings} /></div><Pairwise key={`pairwise:${s.seasonId}:${s.fieldSize}`} season={s} matrix={analytics.pairwise} />
      <details className="tournament-provenance"><summary>Reproducibility · season configuration</summary><p>Season seed {s.seasonSeed} · {s.fieldSize} entrants · {s.gamesPerPairing} games per pairing · {s.schedule.length} total games</p><p>K-factor {K_FACTOR} · Initial Elo {INITIAL_ELO} · Bootstrap samples {BOOTSTRAP_SAMPLES} · Schedule version {s.scheduleVersion}</p><ol>{s.entrants.map(e => <li key={e.entrantId}>{entrantLabel(s, e.entrantId)}</li>)}</ol></details>
    </>}
    <p className="tournament-method">Elo ratings are relative to this field and depend on the seeded schedule and chosen K-factor. Balanced Red/Yellow fixtures reduce first-player bias but do not eliminate sampling variation. Ratings describe this season, are order dependent, and are not universal Connect 4 ratings.</p>
  </main>;
}
