import { useEffect, useRef } from 'react';
import { timelineLabel } from './matchRecord.js';

export default function MatchTimeline({ game, moves, live, viewedRevision, review, returnLive }) {
  const list = useRef(null);
  useEffect(() => {
    if (live && list.current) list.current.scrollTop = list.current.scrollHeight;
  }, [moves.length, live]);
  const revision = live ? game?.revision ?? 0 : viewedRevision;
  const ready = Boolean(game);
  return <section className="match-timeline" aria-labelledby="timeline-title">
    <div className="timeline-heading"><h2 id="timeline-title">Timeline</h2><span>{moves.length} moves</span></div>
    <p className="analysis-helper">Inspect any position without changing the match.</p>
    <div className="replay-controls">
      <button id="match-previous" disabled={!ready || revision === 0} onClick={() => review(revision - 1)}>Previous</button>
      <button id="match-forward" disabled={!ready || revision >= moves.length} onClick={() => review(revision + 1)}>Next</button>
      <button id="match-live" disabled={!ready || live} onClick={returnLive}>Return to live</button>
    </div>
    <ol ref={list} className="move-list" aria-label="Match move history" start="0">
      <li><button disabled={!ready} aria-current={ready && revision === 0 ? 'step' : undefined} onClick={() => review(0)}><span className="move-index">0</span><span>Start · Empty board</span></button></li>
      {moves.map(record => <li key={record.revision}><button aria-current={revision === record.revision ? 'step' : undefined}
        onClick={() => review(record.revision)} aria-label={timelineLabel(record, game.players)}>
        <span className={`move-index move-index--${record.player}`}>{record.move_number}</span>
        <span>{timelineLabel(record, game.players).split(' · ').slice(1).join(' · ')}</span>
      </button></li>)}
    </ol>
  </section>;
}
