export default function MatchControls({ game, busy, uncertain, error, live, humanTurn, autoplay, nextMove, enableAutoplay, pause, speed, setSpeed, refresh }) {
  const blocked = !game || game.gameOver || busy || uncertain || Boolean(error) || !live;
  return <section className="match-controls" aria-labelledby="match-controls-title">
    <p className="eyebrow">One move at a time</p><h2 id="match-controls-title">Match controls</h2>
    <div className="match-action-row">
      <button id="match-next" className="action-link action-link--primary" disabled={blocked || humanTurn || autoplay} onClick={nextMove}>Next move</button>
      <button id="match-autoplay" className="action-link action-link--secondary" aria-pressed={autoplay}
        disabled={autoplay ? false : blocked || game?.players.every(p => p.type === 'human')}
        onClick={autoplay ? pause : enableAutoplay}>{autoplay ? 'Pause' : 'Autoplay'}</button>
    </div>
    <div className="match-speed"><label htmlFor="match-speed">Playback speed</label>
      <select id="match-speed" value={speed} onChange={e => setSpeed(e.target.value)}>
        <option value="slow">Slow</option><option value="normal">Normal</option><option value="fast">Fast</option>
      </select>
    </div>
    <p className="analysis-helper">Delay between moves. Search time depends on the competitor.</p>
    {(uncertain || error && game) && <button id="match-refresh" className="action-link action-link--secondary" disabled={busy} onClick={refresh}>Refresh match</button>}
  </section>;
}
