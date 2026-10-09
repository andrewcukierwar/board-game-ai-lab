import { RESEARCH_DISABLED } from '../connect4/researchAgent.js';
import { useEffect, useRef, useState } from 'react';
import { gamePlan, nextMatchup, replayColumns, roundLabel, matchupHasHuman } from './model.js';
import { entrantLabel } from './Bracket.jsx';
import { replaySnapshot } from '../connect4/matchRecord.js';
import GameBoard from '../connect4/GameBoard.jsx';
import MatchControls from '../connect4/MatchControls.jsx';
import MatchTimeline from '../connect4/MatchTimeline.jsx';
export default function MatchViewer({ tournament: t, matchup: m, controller: c, state, researchBlocked = false, start = () => c.watch() }) {
  const viewerRef = useRef(null), previousActive = useRef(null);
  const [number, setNumber] = useState(1), [viewed, setViewed] = useState(null);
  const a = t.active?.matchupId === m.matchupId ? t.active : null;
  const human = matchupHasHuman(t, m);
  useEffect(() => {
    if (human && a?.gameId && previousActive.current !== a.gameId) viewerRef.current?.focus();
    previousActive.current = a?.gameId;
  }, [a?.gameId, human]);
  useEffect(() => { c.returnLive(); }, [c, m.matchupId, a?.gameNumber, m.games.length]);
  const currentNumber = a?.gameNumber ?? m.games.length + 1;
  useEffect(() => { setNumber(a?.gameNumber ?? (human && m.status !== 'complete' ? m.games.length + 1 : m.games.length || 1)); setViewed(null); }, [m.matchupId, a?.gameNumber, m.games.length]);
  const completed = m.games[number - 1];
  const isActive = Boolean(a && number === currentNumber && !completed);
  const plan = completed ?? (m.entrantAId && m.entrantBId ? gamePlan(t, m) : null);
  const replay = completed ? replayColumns(completed.columns, completed.playerConfigs) : isActive && a.gameId ? replayColumns(a.columns, plan.playerConfigs, a.gameId) : null;
  const live = viewed === null;
  const game = replay?.game, moves = replay?.moves ?? [];
  const review = revision => { c.review(); if (revision >= 0 && revision <= moves.length) { setViewed(revision === moves.length ? null : revision); if (revision === moves.length) c.returnLive(); } };
  const returnLive = () => { c.pause(); c.returnLive(); setViewed(null); };
  const labels = plan?.playerEntrantIds.map(id => entrantLabel(t, id));
  const humanIndex = plan?.playerConfigs.findIndex(p => p.type === 'human');
  const yourTurn = Boolean(isActive && game && !game.gameOver && game.players[game.currentPlayer].type === 'human');
  const color = humanIndex === 0 ? 'Red' : 'Yellow';
  const status = completed ? completed.result.status === 'draw' ? 'Draw.' : completed.playerConfigs[completed.result.winnerIndex].type === 'human' ? 'You win!' : `${labels[completed.result.winnerIndex]} wins.` :
    !live ? 'Reviewing earlier moves. Return to live to play.' : state.uncertain || state.error ? 'Recovery required. Refresh history before continuing.' :
    yourTurn ? `Your turn · You are ${color}.${state.mode !== 'paused' ? ' Autoplay waiting for you.' : ''}` :
    isActive && game ? `${labels[game.currentPlayer]}${state.busy || state.mode !== 'paused' ? ' is thinking.' : ' to move.'}` : human ? 'Start your match when you are ready.' : 'Start the matchup when you are ready.';
  const next = nextMatchup(t)?.matchupId === m.matchupId;
  return <section ref={viewerRef} tabIndex={-1} className="tournament-viewer" aria-labelledby="viewer-title">
    <div className="tournament-section-heading"><div><p className="eyebrow">{roundLabel(t.size, m.round)} · Match {m.index + 1}</p><h2 id="viewer-title">Match viewer</h2></div><span>{human ? `Your match · ${m.status}` : m.status === 'complete' ? 'Completed matchup' : isActive ? 'Active matchup' : 'Matchup preview'}</span></div>
    <p className="viewer-pairing">{entrantLabel(t, m.entrantAId)} <span>vs</span> {entrantLabel(t, m.entrantBId)}</p>
    {m.resolution === 'seeded_draw_tiebreak' && <p className="tiebreak-note">Advanced by seeded tiebreak after three draws. {entrantLabel(t, m.winnerEntrantId)} advances.</p>}
    <div className="game-switcher" aria-label="Games in this matchup">
      {Array.from({ length: m.games.length + (m.status !== 'complete' ? 1 : 0) }, (_, i) => <button key={i} aria-pressed={number === i + 1}
        onClick={() => { c.pause(); if (m.games[i]) c.review(); else c.returnLive(); setNumber(i + 1); setViewed(null); }}>Game {i + 1}{m.games[i] ? m.games[i].result.status === 'draw' ? ' — Draw' : ` — ${entrantLabel(t, m.games[i].playerEntrantIds[m.games[i].result.winnerIndex])} wins` : i > 0 ? ' — Draw rematch' : ''}</button>)}
    </div>
    {number > 1 && <p className="analysis-helper">{number === 2 ? 'Game 1 was a draw. This rematch swaps Red and Yellow.' : 'Two draws led to a final game with a separately seeded color assignment.'}</p>}
    <div className="match-workspace">
      <div className="match-field"><div className={`match-view-mode ${live && isActive ? 'is-live' : 'is-review'}`}>
        <span>{completed ? live ? 'LOCAL REPLAY · FINAL POSITION' : `LOCAL REPLAY · MOVE ${viewed} OF ${moves.length}` : game ? live ? 'LIVE' : `REVIEWING MOVE ${viewed} OF ${moves.length}` : 'MATCH PREVIEW'}</span>
        {!live && <button onClick={returnLive}>{completed ? 'Return to end' : 'Return to live'}</button>}
      </div>
      {human && <p className="human-match-status" role="status" aria-live="polite" aria-atomic="true">{status}</p>}
      <GameBoard game={game && !live ? replaySnapshot(game, moves, viewed) : game} busy={state.busy} uncertain={state.uncertain} interactive={!researchBlocked && isActive && live && yourTurn && !state.error && !state.reviewing} move={column => { if (live && isActive) c.humanMove(column); }} labels={labels} columnControls={human} />
      {human && plan && <p className="human-color-assignment">You are {color} · {humanIndex === 0 ? 'You move first.' : `${labels[0]} moves first.`}</p>}
      <p className="game-instructions">{plan ? `Red: ${labels[0]}. Yellow: ${labels[1]}. Game seed: ${plan.gameSeed}.` : 'Entrants arrive after their preceding matchups finish.'}</p>
      </div>
      <aside className="match-sidebar" aria-label="Tournament match playback and replay">
        {!completed && next && !a && <button id="tournament-watch" className="action-link action-link--primary" disabled={researchBlocked || state.busy || state.uncertain || Boolean(state.error)} onClick={start}>{human ? m.games.length ? 'Play rematch' : 'Play your match' : 'Start / Watch matchup'}</button>}
        {isActive && <MatchControls game={game} busy={state.busy} uncertain={state.uncertain} error={state.error || (researchBlocked ? RESEARCH_DISABLED : '')} live={live} humanTurn={yourTurn}
          autoplay={state.mode !== 'paused'} nextMove={() => c.nextMove()} enableAutoplay={() => c.run('game')} pause={() => c.pause()}
          speed={state.speed} setSpeed={s => c.setSpeed(s)} refresh={() => c.refresh()} />}
        {!completed && a?.status === 'interrupted' && <button id="tournament-restart" className="action-link action-link--primary" disabled={researchBlocked || state.busy} onClick={() => c.restart()}>{human ? 'Restart this game' : 'Restart interrupted game'}</button>}
        <MatchTimeline game={game} moves={moves} live={live} viewedRevision={viewed} review={review} returnLive={returnLive} endLabel={completed ? 'Return to end' : 'Return to live'} />
      </aside>
    </div>
  </section>;
}
