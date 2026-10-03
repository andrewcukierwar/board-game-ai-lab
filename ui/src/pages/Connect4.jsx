import { useEffect } from 'react';
import { Link } from 'react-router-dom';
import axios from 'axios';
import { mountConnect4 } from '../../legacy/connect4.js';
import '../../connect4/connect4.css';

export default function Connect4Page() {
  useEffect(() => mountConnect4({ document, http: axios.create({
    baseURL: import.meta.env.VITE_API_BASE,
    timeout: 90000,
  }) }), []);

  return (
    <main className="connect4-wrapper" style={{ padding: '20px' }}>
      <Link to="/">Home</Link>
      <h1>Connect 4</h1>
      <p>Connect four red pieces horizontally, vertically, or diagonally. You move first.</p>
      <div className="agent-selection">
        <label htmlFor="opponent-type">Opponent:</label>
        <select id="opponent-type" defaultValue="negamax">
          <option value="random">Random</option>
          <option value="negamax">Negamax</option>
          <option value="mcts">MCTS</option>
        </select>
        <div id="negamax-options">
          <label htmlFor="opponent-depth">Search depth:</label>
          <select id="opponent-depth" defaultValue="2">
            <option value="1">1 — Quick</option>
            <option value="2">2 — Default</option>
            <option value="3">3</option>
            <option value="4">4 — Deeper</option>
          </select>
        </div>
        <div id="mcts-options" hidden>
          <label htmlFor="opponent-simulations">Search simulations:</label>
          <select id="opponent-simulations" defaultValue="100">
            <option value="50">Quick — 50 simulations</option>
            <option value="100">Standard — 100 simulations (default)</option>
            <option value="250">Deeper — 250 simulations</option>
          </select>
        </div>
        <p>Opponent settings apply when you start a new game.</p>
      </div>
      <button id="start-button" type="button">Start game</button>
      <button id="restart-button" type="button" hidden>Start new game</button>
      <button id="retry-button" type="button" hidden>Retry AI move</button>
      <div id="message" role="status" aria-live="polite" />
      <div id="loading" hidden>Waiting for the game server… The first request may take about a minute while it wakes up.</div>
      <div id="game-board" role="group" aria-label="Connect 4 board" />
      <section id="explanation-panel" aria-label="Grounded explanations">
        <h2>Understand the position</h2>
        <p>Ask about a move or position. Explanations connect verified board facts to Connect 4 strategy.</p>
        <label htmlFor="explanation-question">Optional question (500 characters maximum):</label>
        <textarea id="explanation-question" maxLength={500} rows={2} />
        <div className="explanation-controls">
          <button id="explain-last" type="button">Explain Last Move</button>
          <button id="analyze-position" type="button">Analyze Position</button>
          <label htmlFor="what-if-column">Hypothetical move:</label>
          <select id="what-if-column" />
          <button id="what-if" type="button">What If?</button>
        </div>
        <p id="explanation-status" aria-live="polite" />
        <div id="explanation-result" />
      </section>
    </main>
  );
}
