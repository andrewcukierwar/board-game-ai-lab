export function validateSnapshot(data, expected = {}) {
  // Waking proxies can return HTML with HTTP 200. Never discard a valid board
  // until the response passes the game snapshot contract.
  if (!data || typeof data.game_id !== 'string' || !data.game_id || !Number.isInteger(data.revision) || data.revision < 0 ||
      !Array.isArray(data.board) || data.board.length !== 6 ||
      !data.board.every(row => Array.isArray(row) && row.length === 7 && row.every(piece => ['X', 'O', ' '].includes(piece))) ||
      !Array.isArray(data.players) || data.players.length !== 2 ||
      !data.players.every(player => player && ['human', 'random', 'negamax', 'mcts'].includes(player.type)) ||
      ![0, 1].includes(data.currentPlayer) || data.currentPlayer !== data.revision % 2 || typeof data.gameOver !== 'boolean' ||
      !Array.isArray(data.legalMoves) || !data.legalMoves.every(col => Number.isInteger(col) && col >= 0 && col < 7) ||
      (expected.game_id !== undefined && data.game_id !== expected.game_id) ||
      (expected.revision !== undefined && data.revision !== expected.revision) ||
      (expected.minRevision !== undefined && data.revision < expected.minRevision) ||
      (data.revision === 0 && (data.currentPlayer !== 0 || data.gameOver || data.board.some(row => row.some(piece => piece !== ' '))))) {
    throw new Error('The game server did not return a game snapshot.');
  }
  return data;
}

