// Deterministic game evidence only. No provider or search internals.
export const DRAW = [2, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
  3, 3, 3, 3, 3, 3, 6, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6];
export const WIN = [0, 1, 0, 1, 0, 1, 0];
function result(board) {
  for (let r = 0; r < 6; r++) for (let c = 0; c < 7; c++) {
    const piece = board[r][c];
    if (piece === ' ') continue;
    for (const [dr, dc] of [[0, 1], [1, 0], [1, 1], [1, -1]]) {
      let count = 1;
      while (count < 4 && board[r + dr * count]?.[c + dc * count] === piece) count++;
      if (count === 4) return { status: 'win', winner: piece === 'X' ? 0 : 1 };
    }
  }
  return { status: board.flat().includes(' ') ? 'ongoing' : 'draw', winner: null };
}
export function matchFixture(players = [{ type: 'random' }, { type: 'random' }], sequence = [], game_id = 'match-a') {
  let board = Array.from({ length: 6 }, () => Array(7).fill(' '));
  const moves = [];
  for (const [i, column] of sequence.entries()) {
    const before = structuredClone(board), player = i % 2;
    const row = board.findLastIndex(r => r[column] === ' ');
    board[row][column] = player ? 'O' : 'X';
    moves.push({ move_number: i + 1, revision_before: i, revision: i + 1, player,
      agent: { type: players[player].type, ...(players[player].depth ? { depth: players[player].depth } : {}) },
      column, board_before: before, board_after: structuredClone(board), outcome: result(board) });
  }
  const outcome = result(board), gameOver = outcome.status !== 'ongoing', revision = sequence.length;
  const state = { game_id, revision, board, players, currentPlayer: revision % 2, gameOver,
    winner: outcome.status === 'win' ? `Player ${outcome.winner + 1}` : gameOver ? 'Draw' : null,
    legalMoves: gameOver ? [] : board[0].flatMap((p, c) => p === ' ' ? [c] : []) };
  return { game_id, revision, players, state, moves };
}
