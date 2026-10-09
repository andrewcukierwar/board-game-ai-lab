import { KNOWN_PLAYER_TYPES, VICTOR_RESEARCH, validResearchConfig } from './researchAgent.js';
import { validateSnapshot } from './gameSnapshot.js';
import { competitorLabel, playerColor } from './competitorConfig.js';

export const emptyBoard = () => Array.from({ length: 6 }, () => Array(7).fill(' '));
const equal = (a, b) => JSON.stringify(a) === JSON.stringify(b);
export const samePlayers = (a, b) => a?.length === 2 && b?.length === 2 && a.every((p, i) =>
  (p.type !== VICTOR_RESEARCH || validResearchConfig(p) && validResearchConfig(b[i])) &&
  p.type === b[i].type && p.depth === b[i].depth && p.simulation_limit === b[i].simulation_limit);

function outcome(board) {
  for (let r = 0; r < 6; r++) for (let c = 0; c < 7; c++) {
    if (board[r][c] === ' ') continue;
    for (const [dr, dc] of [[0, 1], [1, 0], [1, 1], [1, -1]]) {
      if (Array.from({ length: 4 }, (_, i) => board[r + dr * i]?.[c + dc * i]).every(p => p === board[r][c])) {
        return { status: 'win', winner: board[r][c] === 'X' ? 0 : 1 };
      }
    }
  }
  return { status: board.flat().includes(' ') ? 'ongoing' : 'draw', winner: null };
}

// Reconstruct independently from columns, then verify every serialized board,
// revision and outcome. A partial/mismatched history never unlocks execution.
export function validateMatchHistory(data, expected = {}) {
  const invalid = () => { throw new Error('The match history does not agree with the authoritative board.'); };
  const game = validateSnapshot(data?.state, expected, KNOWN_PLAYER_TYPES);
  if ((data.rng_seed !== undefined && (!Number.isInteger(data.rng_seed) || data.rng_seed < 0 || data.rng_seed > 4294967295)) ||
      (expected.rng_seed !== undefined && data.rng_seed !== expected.rng_seed)) invalid();
  if (data.game_id !== game.game_id || data.revision !== game.revision ||
      !samePlayers(data.players, game.players) || !Array.isArray(data.moves) ||
      data.moves.length !== game.revision || game.revision > 42 ||
      (expected.players && !samePlayers(game.players, expected.players))) invalid();
  if (!game.players.every(p => (p.type !== 'negamax' || Number.isInteger(p.depth) && p.depth >= 1 && p.depth <= 8) &&
      (p.type !== 'mcts' || [50, 100, 250, 400, 800].includes(p.simulation_limit)))) invalid();
  let board = emptyBoard(), result = outcome(board);
  for (const [i, record] of data.moves.entries()) {
    if (!record || result.status !== 'ongoing' || record.revision_before !== i || record.revision !== i + 1 ||
        record.move_number !== i + 1 || record.player !== i % 2 || !Number.isInteger(record.column) ||
        record.column < 0 || record.column > 6 || record.agent?.type !== game.players[i % 2].type ||
        (record.agent.type === VICTOR_RESEARCH && !validResearchConfig(record.agent)) ||
        record.agent.depth !== game.players[i % 2].depth || !equal(record.board_before, board)) invalid();
    const row = board.findLastIndex(r => r[record.column] === ' ');
    if (row === -1) invalid();
    board = board.map(r => [...r]);
    board[row][record.column] = i % 2 ? 'O' : 'X';
    result = outcome(board);
    if (!equal(record.board_after, board) || record.outcome?.status !== result.status || record.outcome?.winner !== result.winner) invalid();
  }
  const legal = result.status === 'ongoing' ? board[0].flatMap((p, c) => p === ' ' ? [c] : []) : [];
  const winner = result.status === 'win' ? `Player ${result.winner + 1}` : result.status === 'draw' ? 'Draw' : null;
  if (!equal(game.board, board) || game.gameOver !== (result.status !== 'ongoing') || game.winner !== winner ||
      !equal([...game.legalMoves].sort(), legal)) invalid();
  return { game, moves: data.moves };
}

export function replaySnapshot(game, moves, revision) {
  if (revision === game.revision) return game;
  return { ...game, revision, board: revision === 0 ? emptyBoard() : moves[revision - 1].board_after,
    currentPlayer: revision % 2, gameOver: false, legalMoves: [], winner: null };
}

export function matchResult(game) {
  if (!game?.gameOver) return null;
  const winnerIndex = game.winner === 'Draw' ? null : game.winner === 'Player 1' ? 0 : 1;
  return { status: winnerIndex === null ? 'draw' : 'win', winnerIndex,
    competitor: winnerIndex === null ? null : game.players[winnerIndex],
    color: winnerIndex === null ? null : playerColor(winnerIndex), moveCount: game.revision };
}
export function resultLabel(result) {
  return result.status === 'draw' ? `Draw after ${result.moveCount} moves.`
    : `${competitorLabel(result.competitor)} wins as ${result.color} in ${result.moveCount} moves.`;
}
export const timelineLabel = (record, players) => `${record.move_number} · ${playerColor(record.player)} · ${competitorLabel(players[record.player])} · Column ${record.column + 1}`;
