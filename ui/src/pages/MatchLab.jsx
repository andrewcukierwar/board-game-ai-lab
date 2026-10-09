import { Link } from 'react-router-dom';
import axios from 'axios';
import { useConnect4Match } from '../connect4/useConnect4Match.js';
import { competitorLabel } from '../connect4/competitorConfig.js';
import GameBoard from '../connect4/GameBoard.jsx';
import CompetitorSelector from '../connect4/CompetitorSelector.jsx';
import MatchControls from '../connect4/MatchControls.jsx';
import MatchTimeline from '../connect4/MatchTimeline.jsx';
import MatchStatus from '../connect4/MatchStatus.jsx';
import '../../connect4/connect4.css';
import './match-lab.css';

const matchHttp = axios.create({ baseURL: import.meta.env?.VITE_API_BASE, timeout: 90000 });
export default function MatchLabPage({ http = matchHttp, researchEnabled }) {
  const match = useConnect4Match(http, researchEnabled);
  const labels = (match.game?.players ?? match.selections).map(competitorLabel);
  return <main className="connect4-wrapper match-lab site-container">
    <header className="game-page-heading">
      <Link className="game-home-link" to="/connect4">Back to Play</Link>
      <p className="eyebrow">Competition Lab · Connect 4</p><h1>Match Lab</h1>
      <p>Pit any two competitors against each other. Step through each move or watch the match play automatically.</p>
    </header>
    <MatchStatus {...match} />
    <section className="match-setup" aria-label="Match setup">
      <div className="competitor-grid">{match.selections.map((config, index) => <CompetitorSelector key={index} index={index} config={config} {...match} />)}</div>
      <div className="match-start-row"><p>Red moves first. Settings apply to the next match.</p>
        <button id="match-start" className="action-link action-link--primary" disabled={match.busy} onClick={match.start}>{match.game ? 'Start new match' : 'Start match'}</button>
      </div>
    </section>
    <div className="match-workspace">
      <div className="match-field">
        <div id="match-view-mode" className={`match-view-mode ${match.live ? 'is-live' : 'is-review'}`}>
          <span>{match.game ? match.live ? 'LIVE' : `REVIEWING MOVE ${match.viewedRevision} OF ${match.game.revision}` : 'MATCH PREVIEW'}</span>
          {!match.live && <button onClick={match.returnLive}>Return to live</button>}
        </div>
        <GameBoard game={match.displayedGame} busy={match.busy} uncertain={match.uncertain} interactive={match.interactive} labels={labels} move={match.move} />
        <p className="game-instructions">Connect four horizontally, vertically, or diagonally. Human turns use the board; AI turns use the match controls.</p>
      </div>
      <aside className="match-sidebar" aria-label="Match playback and replay"><MatchControls {...match} /><MatchTimeline {...match} /></aside>
    </div>
  </main>;
}
