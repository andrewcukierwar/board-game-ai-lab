import { competitorLabel, playerColor } from './competitorConfig.js';
import { resultLabel } from './matchRecord.js';

export default function MatchStatus({ game, result, busy, phase, live, viewedRevision, autoplay, humanTurn, uncertain, error, expired }) {
  const turn = game && !game.gameOver ? `${playerColor(game.currentPlayer)} · ${competitorLabel(game.players[game.currentPlayer])}` : '';
  const title = expired ? 'Session expired' : error ? 'Match paused · confirmation needed' : !live ? `Reviewing move ${viewedRevision} of ${game.revision}`
    : result ? resultLabel(result) : !game ? 'Ready to compare' : autoplay ? humanTurn ? `Autoplay waiting for Human · ${playerColor(game.currentPlayer)}` : 'Autoplay running'
      : humanTurn ? `${turn} to move · Match paused` : `Match paused · Move ${game.revision}`;
  return <div className={`game-status match-status ${error || uncertain ? 'game-status--error' : ''}`}>
    <span className="status-marker" aria-hidden="true" /><div>
      {/* Stable playback/recovery/result announcements; do not announce every AI ply. */}
      <strong role="status" aria-live="polite" aria-atomic="true">{title}</strong>
      <p id="match-message">{error ? `${error}${game ? uncertain ? ' Refresh match before continuing.' : ' Review the confirmed board, then refresh to continue.' : ''}`
        : !live ? 'Replay only. The live match stays at its latest move.' : busy ? phase === 'ai-move' ? 'AI thinking. Pause will stop after this move is confirmed.'
          : phase === 'starting' ? 'Connecting to the game server. The first request may take longer while it wakes up.' : 'Confirming the board and move history…'
          : result ? 'The match is complete. Explore its timeline or start a new match.' : game ? `${turn} to move. ${humanTurn ? 'Choose a column.' : autoplay ? 'Playback continues after the viewer delay.' : 'Step forward or enable autoplay.'}`
          : 'Configure both competitors, then start a paused match.'}</p>
    </div>
  </div>;
}
