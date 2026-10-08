import { useState } from 'react';
import { entrantLabel, gamePlan, currentFixture, replayColumns } from './model.js';
import { replaySnapshot } from '../connect4/matchRecord.js';
import GameBoard from '../connect4/GameBoard.jsx';
import MatchControls from '../connect4/MatchControls.jsx';
import MatchTimeline from '../connect4/MatchTimeline.jsx';
import { resultLabel } from './Schedule.jsx';
export default function MatchViewer({ season: s, fixture: f, state, controller: c }) {
  const [viewed, setViewed] = useState(null);
  const completed = s.completedGames.find(g => g.fixtureId === f.fixtureId), a = s.active?.fixtureId === f.fixtureId ? s.active : null;
  const plan = completed ?? gamePlan(s, f), live = viewed === null;
  const replay = completed ? replayColumns(completed.columns, completed.playerConfigs) : a?.gameId ? replayColumns(a.columns, plan.playerConfigs, a.gameId) : null;
  const game = replay?.game, moves = replay?.moves ?? [], labels = [f.redEntrantId, f.yellowEntrantId].map(id => entrantLabel(s, id));
  const review = revision => { c.review(); setViewed(revision === moves.length ? null : revision); if (revision === moves.length) c.returnLive(); };
  const returnLive = () => { c.pause(); c.returnLive(); setViewed(null); };
  return <section className="season-panel season-viewer" aria-labelledby="season-viewer-title">
    <div className="tournament-section-heading"><div><p className="eyebrow">Round {f.round} · Game {s.schedule.indexOf(f) + 1}</p><h2 id="season-viewer-title">Current match viewer</h2></div><span>{resultLabel(f)}</span></div>
    <p className="viewer-pairing">{labels[0]} <span>vs</span> {labels[1]}</p>
    <div className="match-workspace"><div className="match-field">
      <div className={`match-view-mode ${a && live ? 'is-live' : 'is-review'}`}><span>{completed ? `LOCAL REPLAY · ${live ? 'FINAL POSITION' : `MOVE ${viewed} OF ${moves.length}`}` : a ? live ? 'LIVE' : `REVIEWING MOVE ${viewed} OF ${moves.length}` : 'FIXTURE PREVIEW'}</span></div>
      {a?.status === 'running' && game && <p className="season-turn" role="status">{!live ? 'Reviewing earlier moves. Return to live to continue.' : `${game.currentPlayer === 0 ? 'Red' : 'Yellow'} to move · ${labels[game.currentPlayer]}${state.busy ? ' · Thinking' : ''}`}</p>}
      <GameBoard game={game && !live ? replaySnapshot(game, moves, viewed) : game} busy={state.busy} uncertain={state.uncertain} interactive={false} labels={labels} />
      <p className="game-instructions">Red: {labels[0]}. Yellow: {labels[1]}. Game seed: {f.gameSeed}.</p>
    </div><aside className="match-sidebar" aria-label="Season game playback and replay">
      {!a && !completed && currentFixture(s)?.fixtureId === f.fixtureId && <button id="season-watch" className="action-link action-link--primary" disabled={state.busy || state.uncertain || Boolean(state.error)} onClick={() => c.watch()}>Start current fixture</button>}
      {a?.status === 'running' && <MatchControls game={game} busy={state.busy} uncertain={state.uncertain} error={state.error} live={live} autoplay={state.mode !== 'paused'}
        nextMove={() => c.nextMove()} enableAutoplay={() => { setViewed(null); c.run('game'); }} pause={() => c.pause()} speed={state.speed} setSpeed={v => c.setSpeed(v)} refresh={() => c.refresh()} />}
      {a?.status === 'interrupted' && <button id="season-restart" disabled={state.busy} onClick={() => c.restart()}>Restart interrupted fixture</button>}
      {!a && !completed && currentFixture(s)?.fixtureId !== f.fixtureId && <p className="analysis-helper">Upcoming fixture. Execution follows the saved schedule in order.</p>}
      <MatchTimeline game={game} moves={moves} live={live} viewedRevision={viewed} review={review} returnLive={returnLive} endLabel={completed ? 'Return to end' : 'Return to live'} />
    </aside></div>
  </section>;
}
