// React owns the page; this small controller owns its board and request lifecycle.
// No window globals: mounting twice or navigating away cannot retain an old game.
import { mountExplanations } from './explanations.js';

export function mountConnect4({ document, http }) {
  const el = id => document.getElementById(id);
  const boardElement = el('game-board');
  let game = null;
  let busy = false;
  let uncertain = false;
  let retryAI = false;
  let active = true;
  let message = 'Choose an opponent and start a game. You play red and move first.';
  const abort = new AbortController();
  const options = { signal: abort.signal };
  const listeners = [];
  const explanations = mountExplanations({ document, http });
  const humanTurn = () => game?.players[game.currentPlayer].type === 'human';

  function render() {
    if (!active) return;
    el('message').textContent = message;
    el('loading').hidden = !busy;
    el('opponent-type').disabled = busy;
    el('opponent-depth').disabled = busy;
    el('opponent-simulations').disabled = busy;
    el('negamax-options').hidden = el('opponent-type').value !== 'negamax';
    el('mcts-options').hidden = el('opponent-type').value !== 'mcts';
    el('start-button').hidden = Boolean(game);
    el('restart-button').hidden = !game;
    el('start-button').disabled = busy;
    el('restart-button').disabled = busy;
    el('retry-button').hidden = !(uncertain || retryAI);
    el('retry-button').disabled = busy;
    el('retry-button').textContent = uncertain ? 'Refresh game' : 'Retry AI move';
    explanations.update(game, busy || uncertain);
    const board = el('game-board');
    board.replaceChildren();
    if (!game) return;
    board.setAttribute('aria-busy', String(busy));
    game.board.forEach((row, rowIndex) => row.forEach((piece, column) => {
      const cell = document.createElement('button');
      cell.type = 'button';
      cell.className = 'cell';
      cell.disabled = busy || uncertain || game.gameOver || !humanTurn() || !game.legalMoves.includes(column);
      cell.setAttribute('aria-label', `Column ${column + 1}, row ${rowIndex + 1}: ${piece === 'X' ? 'red' : piece === 'O' ? 'yellow' : 'empty'}`);
      cell.dataset.column = String(column);
      cell.dataset.square = `${String.fromCharCode(97 + column)}${6 - rowIndex}`;
      const circle = document.createElement('span');
      circle.className = `circle ${piece === 'X' ? 'x' : piece === 'O' ? 'o' : 'empty'}`;
      cell.appendChild(circle);
      cell.addEventListener('click', () => move(column));
      board.appendChild(cell);
    }));
  }

  function accept(data) {
    // A waking service/proxy can return an HTML page with HTTP 200. Keep the
    // last valid game until a real API snapshot arrives, so recovery still works.
    if (!data || typeof data.game_id !== 'string' || !Number.isInteger(data.revision) ||
        !Array.isArray(data.board) || data.board.length !== 6 ||
        !data.board.every(row => Array.isArray(row) && row.length === 7) ||
        !Array.isArray(data.players) || data.players.length !== 2 ||
        !data.players.every(player => player && typeof player.type === 'string') ||
        ![0, 1].includes(data.currentPlayer) || typeof data.gameOver !== 'boolean' ||
        !Array.isArray(data.legalMoves)) {
      throw new Error('The game server did not return a game snapshot.');
    }
    game = data;
    uncertain = false;
    retryAI = false;
    message = game.gameOver
      ? game.winner === 'Draw' ? "It's a draw! Start a new game to play again."
        : `${game.winner === 'Player 1' ? 'You win' : 'The AI wins'}! Start a new game to play again.`
      : humanTurn() ? 'Your turn — choose a column.' : 'AI turn.';
  }

  async function recover(error) {
    if (!active) return;
    const reason = error.response?.data?.error ||
      'The game server could not be reached. It may be waking up; wait a moment and try again.';
    if (game) {
      // A response can be lost after the server commits a move. Read the board
      // before offering another move; never blindly replay an uncertain POST.
      uncertain = true;
      try {
        const response = await http.get(`/v1/connect4/games/${game.game_id}`, options);
        if (!active) return;
        accept(response.data);
        retryAI = !game.gameOver && !humanTurn();
      } catch (refreshError) {
        if (!active) return;
        if (refreshError.response?.status === 404) {
          game = null;
          uncertain = false;
          retryAI = false;
          message = 'This game expired or the server restarted. Start a new game.';
          return;
        }
        retryAI = false;
      }
    }
    message = reason + (uncertain ? ' Refresh the game before continuing.' : retryAI ? ' Retry the AI move or start a new game.' : '');
  }

  async function run(action) {
    if (!active || busy) return;
    busy = true;
    render();
    try {
      await action();
    } catch (error) {
      await recover(error);
    } finally {
      busy = false;
      render();
    }
  }

  async function botMove() {
    if (!active || !game || game.gameOver || humanTurn()) return;
    const response = await http.post('/v1/connect4/make_move', {
      game_id: game.game_id, revision: game.revision,
    }, options);
    if (active) accept(response.data);
  }

  async function start() {
    return run(async () => {
      const type = el('opponent-type').value;
      const opponent = { type };
      if (type === 'negamax') opponent.depth = Number(el('opponent-depth').value);
      if (type === 'mcts') opponent.simulation_limit = Number(el('opponent-simulations').value);
      const body = { player1: { type: 'human' }, player2: opponent };
      if (game) body.replace_game_id = game.game_id;
      const response = await http.post('/v1/connect4/start_game', body, options);
      if (active) accept(response.data);
    });
  }

  async function move(column) {
    if (!game || game.gameOver || uncertain || !humanTurn() || !game.legalMoves.includes(column)) return;
    return run(async () => {
      const response = await http.post('/v1/connect4/make_move', {
        game_id: game.game_id, revision: game.revision, column,
      }, options);
      if (!active) return;
      accept(response.data);
      render();
      await botMove();
    });
  }

  async function retry() {
    return run(async () => {
      if (!game) return;
      const response = await http.get(`/v1/connect4/games/${game.game_id}`, options);
      if (!active) return;
      accept(response.data);
      await botMove();
    });
  }

  for (const [id, event, handler] of [
    ['start-button', 'click', start], ['restart-button', 'click', start],
    ['retry-button', 'click', retry], ['opponent-type', 'change', render],
  ]) {
    const node = el(id);
    node.addEventListener(event, handler);
    listeners.push(() => node.removeEventListener(event, handler));
  }
  render();
  return () => {
    active = false;
    abort.abort();
    explanations.cleanup();
    listeners.forEach(remove => remove());
    boardElement.replaceChildren();
  };
}
