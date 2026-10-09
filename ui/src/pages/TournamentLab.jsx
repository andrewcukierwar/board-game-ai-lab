import { researchAgentEnabled, researchAgent, VICTOR_RESEARCH, hasResearch, RESEARCH_DISABLED, RESEARCH_CAVEAT } from '../connect4/researchAgent.js';
import { useState } from 'react';
import { Link } from 'react-router-dom';
import axios from 'axios';
import { competitorLabel, NEGAMAX_DEPTHS, MCTS_SIMULATIONS } from '../connect4/competitorConfig.js';
import { SIZES, defaultField, validSeed, findMatchup, nextMatchup, allMatchups, humanTournamentStatus, humanEntrant } from '../tournament/model.js';
import { useTournament } from '../tournament/useTournament.js';
import Bracket, { entrantLabel } from '../tournament/Bracket.jsx';
import MatchViewer from '../tournament/MatchViewer.jsx';
import '../../connect4/connect4.css';
import './match-lab.css';
import './tournament-lab.css';
const tournamentHttp = axios.create({ baseURL: import.meta.env?.VITE_API_BASE, timeout: 90000 });
const ordinaryPresets = [{ type: 'random' }, ...NEGAMAX_DEPTHS.map(depth => ({ type: 'negamax', depth })), ...MCTS_SIMULATIONS.map(simulations => ({ type: 'mcts', simulations })), { type: 'human' }];
function randomSeed() { return globalThis.crypto?.getRandomValues ? globalThis.crypto.getRandomValues(new Uint32Array(1))[0] : Math.floor(Math.random() * 0x100000000); }
const modeLabel = { paused: 'Paused', game: 'Autoplay · current game', matchup: 'Running current matchup', round: 'Running current round', tournament: 'Running entire tournament' };
export default function TournamentLabPage({ http = tournamentHttp, storage }) {
  const enabled = researchAgentEnabled();
  const presets = enabled ? [...ordinaryPresets, { type: VICTOR_RESEARCH }] : ordinaryPresets;
  const state = useTournament(http, storage), { tournament: t, controller: c } = state;
  const [size, setSize] = useState(8), [field, setField] = useState(() => defaultField(8)), [seed, setSeed] = useState(() => String(randomSeed()));
  const [setup, setSetup] = useState(false), [selected, setSelected] = useState(null), [setupError, setSetupError] = useState(''), [playbackEpoch, setPlaybackEpoch] = useState(0);
  const parsedSeed = /^\d+$/.test(seed) ? Number(seed) : NaN;
  const shown = t && (findMatchup(t, selected) ?? findMatchup(t, t.active?.matchupId) ?? nextMatchup(t) ?? t.rounds.at(-1)[0]);
  function create(e) { e.preventDefault(); if (!validSeed(parsedSeed)) { setSetupError('Enter a whole number from 0 to 4294967295.'); return; }
    try { c.create(field, parsedSeed); } catch (error) { setSetupError(error.message); return; } setSetup(false); setSelected(null); setSetupError(''); }
  function select(id) { c.pause(); setSelected(id); }
  function run(mode) { setSelected(null); setPlaybackEpoch(n => n + 1); c.returnLive(); c.run(mode); }
  const participant = humanTournamentStatus(t);
  const ready = t && nextMatchup(t);
  function playHuman() { setSelected(ready.matchupId); c.watch(); }
  const researchField = hasResearch((!t || setup) ? field : t.entrants.map(e => e.config));
  const researchBlocked = researchField && !enabled;
  const blocked = researchBlocked || state.busy || state.uncertain || Boolean(state.error) || t?.status === 'complete';
  return <main className="connect4-wrapper match-lab tournament-lab site-container">
    <header className="game-page-heading"><Link className="game-home-link" to="/connect4/match-lab">Back to Match Lab</Link>
      <p className="eyebrow">Competition Lab · Connect 4</p><h1>Tournament Lab</h1>
      <p>Build a single-elimination field, enter it yourself, and play your matches on the road to a champion.</p>
    </header>
    <div className="tournament-status" role="status" aria-live="polite" aria-atomic="true">
      {t?.status === 'complete' ? 'Tournament complete. Champion crowned.' : state.waiting || state.busy && state.uncertain ? 'Waiting / recovering authoritative history' : state.uncertain ? 'Recovery required' : state.waitingForHuman ? 'Waiting for Human · Play your match' : humanEntrant(t) && state.mode === 'tournament' ? 'Running until your next match or tournament completion' : modeLabel[state.mode]}
    </div>
    {(state.error || setupError) && <p className="tournament-error" role="alert">{setupError || state.error}</p>}
    {researchField && <p className="analysis-helper">{researchAgent.name}: {researchAgent.description}</p>}
    {researchBlocked && <p className="analysis-helper" role="status">{RESEARCH_DISABLED}</p>}
    {!state.storageAvailable && <p className="analysis-helper">Browser storage is unavailable. This tournament works in memory; keep this page open to retain progress.</p>}
    {(!t || setup) ? <form className="tournament-setup" onSubmit={create} aria-label="Tournament setup">
      <div className="tournament-section-heading"><div><p className="eyebrow">Configure the field</p><h2>Tournament setup</h2></div><span>AI field · one optional Human</span></div>
      <div className="setup-fields"><div><label htmlFor="tournament-size">Tournament size</label><select id="tournament-size" value={size} onChange={e => { const n = Number(e.target.value); setSize(n); setField(defaultField(n)); }}>
        {SIZES.map(n => <option key={n} value={n}>{n} entrants</option>)}</select></div>
        <div><label htmlFor="tournament-seed">Tournament seed</label><input id="tournament-seed" type="text" inputMode="numeric" value={seed} onChange={e => setSeed(e.target.value)} aria-describedby="seed-help" /></div>
      </div><p id="seed-help" className="analysis-helper">Whole number 0–4294967295. The same field and seed reproduce the bracket, colors and game seeds.</p>
      <div className="field-heading"><h3>Entrants</h3><button type="button" onClick={() => setField(defaultField(size))}>Reset default field</button></div>
      <p className="analysis-helper">Add yourself to one slot, then play your matches when they appear in the bracket. Duplicate AI configurations are valid.</p>
      <div className="entrant-field">{field.map((config, i) => <div className="entrant-row" key={i}><label htmlFor={`entrant-${i + 1}`}>#{i + 1}</label>
        <select id={`entrant-${i + 1}`} aria-label={`Entrant ${i + 1} configuration`} value={presets.findIndex(p => JSON.stringify(p) === JSON.stringify(config))}
          onChange={e => setField(field.map((p, j) => j === i ? { ...presets[Number(e.target.value)] } : p))}>
          {!enabled && config.type === VICTOR_RESEARCH && <option value={-1} disabled>{researchAgent.name} · unavailable</option>}{presets.map((p, j) => <option key={j} value={j} disabled={p.type === 'human' && config.type !== 'human' && field.some(c => c.type === 'human')}>{p.type === 'human' ? 'You · Human' : competitorLabel(p)}</option>)}
        </select></div>)}</div>
      <div className="setup-submit"><button id="tournament-create" className="action-link action-link--primary" disabled={state.busy}>Create tournament</button>{t && <button type="button" onClick={() => setSetup(false)}>Cancel new tournament</button>}</div>
    </form> : <>
      <section className="tournament-summary" aria-label="Tournament summary"><div><span className="eyebrow">Single elimination</span><h2>{t.size} entrants <span>·</span> {t.rounds.length} rounds</h2><p>Seed {t.tournamentSeed} · {allMatchups(t).filter(m => m.status === 'complete').length} of {t.size - 1} matchups complete</p></div>
        <button disabled={state.busy} onClick={() => { c.pause(); setSize(t.size); setField(t.entrants.map(e => ({ ...e.config }))); setSeed(String(randomSeed())); setSetup(true); }}>New tournament</button>
      </section>
      {t.championEntrantId && <section className="champion-banner" aria-label="Tournament Champion"><span className="champion-symbol" aria-hidden="true">✦</span><div><p className="eyebrow">Tournament Champion</p><h2>{t.championEntrantId === humanEntrant(t)?.entrantId ? 'You are the Tournament Champion' : entrantLabel(t, t.championEntrantId)}</h2><p>{t.championEntrantId === humanEntrant(t)?.entrantId && `${entrantLabel(t, t.championEntrantId)} · `}Champion — {t.size}-player tournament · Seed {t.tournamentSeed}</p></div></section>}
      {participant && <section className="human-tournament-banner" aria-label="Your tournament progress">
        <div role="status" aria-live="polite" aria-atomic="true"><p className="eyebrow">{entrantLabel(t, humanEntrant(t).entrantId)} · Your tournament</p><h2>{participant.message}</h2></div>
        {participant.kind === 'ready' && <button id="play-your-match" className="action-link action-link--primary" disabled={blocked} onClick={playHuman}>{ready.games.length ? 'Play rematch' : 'Play your match'}</button>}
      </section>}
      <section className="tournament-controls" aria-label="Tournament execution" aria-busy={state.busy}><div><p className="eyebrow">Sequential execution</p><h2>Tournament controls</h2></div>
        <div className="tournament-actions">{[['matchup', 'Run matchup'], ['round', 'Run round'], ['tournament', 'Run tournament']].map(([mode, label]) => <button id={`run-${mode}`} key={mode} disabled={blocked || state.mode !== 'paused'} onClick={() => run(mode)}>{label}</button>)}
          <button id="tournament-pause" disabled={state.mode === 'paused'} onClick={() => c.pause()}>Pause</button>
          {state.uncertain || state.error ? <button id="tournament-refresh" disabled={state.busy || t.active?.status === 'interrupted'} onClick={() => c.refresh()}>Refresh history / Continue</button> : null}
        </div><p className="analysis-helper">{humanEntrant(t) ? 'Run controls stop at your next match. Start it explicitly, then use the board on your turn. ' : ''}One game at a time. Pause lets the current request finish safely. Draws receive up to two rematches, then a seeded tiebreak.</p>
      </section>
      <Bracket tournament={t} selected={shown?.matchupId} select={select} />
      {shown && <MatchViewer key={`${shown.matchupId}:${playbackEpoch}`} tournament={t} matchup={shown} controller={c} state={state} researchBlocked={researchBlocked} start={() => { setSelected(shown.matchupId); c.watch(); }} />}
      <details className="tournament-provenance"><summary>Reproducibility · field and bracket order</summary><p>Size {t.size} · Seed {t.tournamentSeed}</p><ol>{t.bracketOrder.map(id => <li key={id}>{entrantLabel(t, id)}</li>)}</ol></details>
    </>}
    <p className="tournament-method">{RESEARCH_CAVEAT}</p>
    <p className="tournament-method">Single elimination is sensitive to bracket path and color assignment. One-game matchups are not rigorous strength estimates. Human decisions are not determined by the tournament seed; Season Lab uses repeated, balanced comparisons and pool-relative ratings.</p>
  </main>;
}
