import { humanTurn } from './useConnect4Game.js';

export default function GameStatus({ game, busy, phase, uncertain, retryAI, notice, message }) {
  const kind = busy ? phase : uncertain ? 'uncertain' : notice || (game?.gameOver ? 'terminal' : game ? 'turn' : 'ready');
  const title = busy ? ({ starting: 'Connecting to the game server', 'human-move': 'Placing your piece', 'ai-move': 'AI thinking', refreshing: 'Checking the authoritative board' })[phase]
    : uncertain ? 'Board confirmation needed' : notice === 'expired' ? 'Session expired' : notice === 'error' ? 'Recoverable request error'
      : game?.gameOver ? (game.winner === 'Draw' ? 'Draw' : game.winner === 'Player 1' ? 'You win' : 'The AI wins')
        : game ? (humanTurn(game) ? 'Your turn' : retryAI ? 'AI move ready to retry' : 'AI turn') : 'Ready to play';
  return <div className={`game-status game-status--${kind}`} role="status" aria-live="polite" aria-atomic="true">
    <span className="status-marker" aria-hidden="true" /><div><strong>{title}</strong>
      <p id="message">{busy && phase === 'ai-move' ? 'The AI is choosing its next move. Your turn follows its reply.' : message}</p>
      <p id="loading" hidden={!busy}>Waiting for the game server… The first request can take longer while the service wakes up. No need to start again.</p>
    </div>
  </div>;
}
