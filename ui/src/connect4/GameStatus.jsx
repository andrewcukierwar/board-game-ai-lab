import { humanTurn, winnerLabel } from './useConnect4Game.js';
import { useEffect, useState } from 'react';

// The timer only describes an in-flight request; the server owns its search cap.
export function SearchElapsed({ active }) {
  const [elapsed, setElapsed] = useState(0);
  useEffect(() => {
    if (!active) { setElapsed(0); return; }
    const began = Date.now();
    const timer = setInterval(() => setElapsed((Date.now() - began) / 1000), 500);
    return () => clearInterval(timer);
  }, [active]);
  return active ? <span className="search-elapsed"> · {elapsed.toFixed(1)} s elapsed</span> : null;
}

export default function GameStatus({ game, busy, phase, uncertain, retryAI, notice, message }) {
  const kind = busy ? phase : uncertain ? 'uncertain' : notice || (game?.gameOver ? 'terminal' : game ? 'turn' : 'ready');
  const title = busy ? ({ starting: 'Connecting to the game server', 'human-move': 'Placing your piece', 'ai-move': 'AI thinking', refreshing: 'Checking the authoritative board' })[phase]
    : uncertain ? 'Board confirmation needed' : notice === 'expired' ? 'Session expired' : notice === 'error' ? 'Recoverable request error'
      : game?.gameOver ? (game.winner === 'Draw' ? 'Draw' : winnerLabel(game))
        : game ? (humanTurn(game) ? 'Your turn' : retryAI ? 'AI move ready to retry' : 'AI turn') : 'Ready to play';
  return <div className={`game-status game-status--${kind}`} role="status" aria-live="polite" aria-atomic="true">
    <span className="status-marker" aria-hidden="true" /><div><strong>{title}</strong>
      <p id="message">{busy && phase === 'ai-move' ? 'The AI is choosing its next move. Your turn follows its reply.' : message}{<SearchElapsed active={busy && phase === 'ai-move'} />}</p>
      <p id="loading" hidden={!busy}>Waiting for the game server… The first request can take longer while the service wakes up. No need to start again.</p>
    </div>
  </div>;
}
