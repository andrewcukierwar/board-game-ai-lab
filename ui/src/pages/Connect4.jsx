import { Link } from 'react-router-dom';
import axios from 'axios';
import { useConnect4Game } from '../connect4/useConnect4Game.js';
import GameBoard from '../connect4/GameBoard.jsx';
import AgentSelector from '../connect4/AgentSelector.jsx';
import GameStatus from '../connect4/GameStatus.jsx';
import GameControls from '../connect4/GameControls.jsx';
import AnalysisPanel from '../connect4/AnalysisPanel.jsx';
import { useConnect4Analysis } from '../connect4/useConnect4Analysis.js';
import '../../connect4/connect4.css';

const gameHttp = axios.create({ baseURL: import.meta.env?.VITE_API_BASE, timeout: 90000 });

export default function Connect4Page({ http = gameHttp }) {
  const gameplay = useConnect4Game(http);
  const analysis = useConnect4Analysis(http, gameplay.game, gameplay.busy || gameplay.uncertain);
  return <main className="connect4-wrapper site-container">
    <header className="game-page-heading">
      <Link className="game-home-link" to="/">Home</Link>
      <p className="eyebrow">Play · compare · understand</p>
      <h1>Connect 4</h1>
      <p>Play against classical game-playing AI. Explore how each approach responds.</p>
    </header>
    <div className="game-workspace">
      <GameStatus {...gameplay} />
      <GameBoard {...gameplay} highlights={analysis.highlights} />
      <aside className="opponent-panel" aria-label="Game setup">
        <AgentSelector {...gameplay} />
        <GameControls {...gameplay} />
      </aside>
    </div>
    <p className="game-instructions">Choose a column to drop a piece. Connect four horizontally, vertically, or diagonally. You move first.</p>
    <AnalysisPanel analysis={analysis} />
  </main>;
}
