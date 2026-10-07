import { useEffect, useReducer, useRef } from 'react';

export const humanTurn = game => game?.players[game.currentPlayer].type === 'human';

function validateSnapshot(data) {
  // Waking proxies can return HTML with HTTP 200. Never discard a valid board
  // until the response passes the same snapshot contract as the old controller.
  if (!data || typeof data.game_id !== 'string' || !Number.isInteger(data.revision) ||
      !Array.isArray(data.board) || data.board.length !== 6 ||
      !data.board.every(row => Array.isArray(row) && row.length === 7) ||
      !Array.isArray(data.players) || data.players.length !== 2 ||
      !data.players.every(player => player && typeof player.type === 'string') ||
      ![0, 1].includes(data.currentPlayer) || typeof data.gameOver !== 'boolean' ||
      !Array.isArray(data.legalMoves)) {
    throw new Error('The game server did not return a game snapshot.');
  }
  return data;
}

const initialState = {
  game: null, phase: 'idle', uncertain: false, retryAI: false, notice: null,
  message: 'Choose an opponent and start a game. You play red and move first.',
  selection: { type: 'negamax', depth: 2, simulations: 100 },
};

function reducer(state, action) {
  if (action.type === 'select') return { ...state, selection: { ...state.selection, ...action.value } };
  if (action.type === 'accept') {
    const game = action.game;
    return { ...state, game, uncertain: false, retryAI: false, notice: null,
      message: game.gameOver
        ? game.winner === 'Draw' ? "It's a draw! Start a new game to play again."
          : `${game.winner === 'Player 1' ? 'You win' : 'The AI wins'}! Start a new game to play again.`
        : humanTurn(game) ? 'Your turn — choose a column.' : 'AI turn.' };
  }
  return { ...state, ...action.value };
}

export function useConnect4Game(http) {
  const [state, dispatch] = useReducer(reducer, initialState);
  // A synchronous mirror of reducer transitions protects revisions and the busy
  // lock even before React commits a render (including two events in one tick).
  const current = useRef(state);
  const session = useRef(null);
  useEffect(() => {
    const lifetime = { active: true, abort: new AbortController() };
    session.current = lifetime;
    return () => { lifetime.active = false; lifetime.abort.abort(); };
  }, [http]);

  function send(action, lifetime = session.current) {
    if (!lifetime?.active || session.current !== lifetime) return;
    current.current = reducer(current.current, action);
    dispatch(action);
  }
  const patch = (value, lifetime) => send({ type: 'patch', value }, lifetime);
  const accept = (data, lifetime) => send({ type: 'accept', game: validateSnapshot(data) }, lifetime);

  async function recover(error, lifetime, options) {
    if (!lifetime.active) return;
    const reason = error.response?.data?.error ||
      'The game server could not be reached. It may be waking up; wait a moment and try again.';
    const game = current.current.game;
    if (game) {
      // Never replay an uncertain POST. Reconcile first, then offer an explicit
      // AI retry only if the authoritative snapshot still belongs to the AI.
      patch({ uncertain: true, phase: 'refreshing' }, lifetime);
      try {
        const response = await http.get(`/v1/connect4/games/${game.game_id}`, options);
        if (!lifetime.active) return;
        accept(response.data, lifetime);
        const latest = current.current.game;
        patch({ retryAI: !latest.gameOver && !humanTurn(latest) }, lifetime);
      } catch (refreshError) {
        if (!lifetime.active) return;
        if (refreshError.response?.status === 404) {
          patch({ game: null, uncertain: false, retryAI: false, notice: 'expired',
            message: 'This game expired or the server restarted. Start a new game.' }, lifetime);
          return;
        }
        patch({ retryAI: false }, lifetime);
      }
    }
    patch({ notice: 'error', message: reason + (current.current.uncertain
      ? ' Refresh the game before continuing.' : current.current.retryAI
        ? ' Retry the AI move or start a new game.' : '') }, lifetime);
  }

  async function run(phase, action) {
    const lifetime = session.current;
    if (!lifetime?.active || current.current.phase !== 'idle') return;
    patch({ phase }, lifetime);
    const options = { signal: lifetime.abort.signal };
    try { await action(lifetime, options); }
    catch (error) { await recover(error, lifetime, options); }
    finally { patch({ phase: 'idle' }, lifetime); }
  }

  async function botMove(lifetime, options) {
    const game = current.current.game;
    if (!lifetime.active || !game || game.gameOver || humanTurn(game)) return;
    patch({ phase: 'ai-move' }, lifetime);
    const response = await http.post('/v1/connect4/make_move', {
      game_id: game.game_id, revision: game.revision,
    }, options);
    if (lifetime.active) accept(response.data, lifetime);
  }

  function start() {
    return run('starting', async (lifetime, options) => {
      const { selection, game } = current.current;
      const opponent = { type: selection.type };
      if (selection.type === 'negamax') opponent.depth = selection.depth;
      if (selection.type === 'mcts') opponent.simulation_limit = selection.simulations;
      const body = { player1: { type: 'human' }, player2: opponent };
      if (game) body.replace_game_id = game.game_id;
      const response = await http.post('/v1/connect4/start_game', body, options);
      if (lifetime.active) accept(response.data, lifetime);
    });
  }

  function move(column) {
    const { game, uncertain } = current.current;
    if (!game || game.gameOver || uncertain || !humanTurn(game) || !game.legalMoves.includes(column)) return;
    return run('human-move', async (lifetime, options) => {
      const response = await http.post('/v1/connect4/make_move', {
        game_id: game.game_id, revision: game.revision, column,
      }, options);
      if (!lifetime.active) return;
      accept(response.data, lifetime);
      // Accept the human snapshot before initiating a separate AI request.
      await botMove(lifetime, options);
    });
  }

  function retry() {
    return run('refreshing', async (lifetime, options) => {
      const game = current.current.game;
      if (!game) return;
      const response = await http.get(`/v1/connect4/games/${game.game_id}`, options);
      if (!lifetime.active) return;
      accept(response.data, lifetime);
      await botMove(lifetime, options);
    });
  }

  function select(value) {
    if (current.current.phase === 'idle') send({ type: 'select', value });
  }
  return { ...state, busy: state.phase !== 'idle', start, move, retry, select };
}
