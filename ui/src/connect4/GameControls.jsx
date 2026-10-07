export default function GameControls({ game, busy, uncertain, retryAI, start, retry }) {
  return <div className="game-controls">
    <button id="start-button" className="action-link action-link--primary" type="button" hidden={Boolean(game)} disabled={busy} onClick={start}>Start game</button>
    <button id="restart-button" className="action-link action-link--primary" type="button" hidden={!game} disabled={busy} onClick={start}>Start new game</button>
    <button id="retry-button" className="action-link action-link--secondary" type="button" hidden={!(uncertain || retryAI)} disabled={busy} onClick={retry}>{uncertain ? 'Refresh game' : 'Retry AI move'}</button>
  </div>;
}
