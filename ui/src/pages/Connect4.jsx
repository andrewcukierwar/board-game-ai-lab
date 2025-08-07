import { useEffect } from "react";
import axios from "axios";
import "../../connect4/connect4.css";

// Dynamically import the legacy JS *once* after the component mounts.
// It attaches its functions (startGame, makeMove, …) to the window object.
export default function Connect4Page() {
  useEffect(() => {
    window.axios = axios;
    import("../../legacy/connect4.js");          // relative to this file
  }, []);

  return (
    <div className="connect4-wrapper" style={{ padding: "20px" }}>
      {/* Paste the body *inside* <div> … </div> of your old HTML,   */}
      {/* but without the <script>/<link> tags — React owns them now */}
      {/* Agent Selection for Player 1 */}
      <div className="agent-selection" id="player1-selection">
        <h3>Player 1 Configuration</h3>
        <label htmlFor="player1-type">Agent Type:</label>
        <select id="player1-type" onChange={() => window.togglePlayer1Options()}>
          <option value="human">Human</option>
          <option value="negamax">Negamax Agent</option>
          <option value="random">Random Agent</option>
          <option value="mcts">MCTS Agent</option>
          <option value="mcts_nn">MCTS NN Agent</option>
          <option value="victor">Victor Agent</option>
          {/* Add more agent types here */}
        </select>
      
        {/* Negamax Depth Selection for Player 1 */}
        <div id="player1-negamax-options" className="hidden">
          <label htmlFor="player1-depth">Depth:</label>
          <select id="player1-depth" defaultValue="2">
            <option value="1">1</option>
            <option value="2">2 (Easy)</option>
            <option value="3">3</option>
            <option value="4">4 (Medium)</option>
            <option value="5">5</option>
            <option value="6">6 (Hard)</option>
            <option value="7">7</option>
            <option value="8">8 (Expert)</option>
            <option value="9">9</option>
            <option value="10">10 (Master)</option>
          </select>
        </div>
      </div>

      {/* Agent Selection for Player 2 */}
      <div className="agent-selection" id="player2-selection">
        <h3>Player 2 Configuration</h3>
        <label htmlFor="player2-type">Agent Type:</label>
        <select id="player2-type" onChange={() => window.togglePlayer2Options()}>
          <option value="human">Human</option>
          <option value="negamax">Negamax Agent</option>
          <option value="random">Random Agent</option>
          <option value="mcts">MCTS Agent</option>
          <option value="mcts_nn">MCTS NN Agent</option>
          <option value="victor">Victor Agent</option>
          {/* Add more agent types here */}
        </select>
    
        {/* Negamax Depth Selection for Player 2 */}
        <div id="player2-negamax-options" className="hidden">
          <label htmlFor="player2-depth">Depth:</label>
          <select id="player2-depth" defaultValue="2">
            <option value="1">1</option>
            <option value="2">2 (Easy)</option>
            <option value="3">3</option>
            <option value="4">4 (Medium)</option>
            <option value="5">5</option>
            <option value="6">6 (Hard)</option>
            <option value="7">7</option>
            <option value="8">8 (Expert)</option>
            <option value="9">9</option>
            <option value="10">10 (Master)</option>
          </select>
        </div>
      </div>

      {/* (include the rest of connect4.html exactly as-is) */}
      <button
        id="start-button"
        onClick={() => window.startGame()}>
        Start Game
      </button>
      <button
        id="restart-button"
        style={{ display: "none" }}
        onClick={() => window.restartGame()}>
        Restart Game
      </button>

      <div id="message"></div>
      <div id="loading" className="hidden">
        AI is thinking…
      </div>
      <div id="game-board"></div>
    </div>
  );
}
