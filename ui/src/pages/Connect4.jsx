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
        <p>Opponent settings apply when you start a new game.</p>
      </div>
      <button id="start-button" type="button">Start game</button>
      <button id="restart-button" type="button" hidden>Start new game</button>
      <button id="retry-button" type="button" hidden>Retry AI move</button>
      <div id="message" role="status" aria-live="polite" />
      <div id="loading" hidden>Waiting for the game server… The first request may take about a minute while it wakes up.</div>
      <div id="game-board" role="group" aria-label="Connect 4 board" />
    </main>
  );
}
