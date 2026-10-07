import { toPlayerPayload, NEGAMAX_DEPTHS, MCTS_SIMULATIONS } from '../connect4/competitorConfig.js';
import { emptyBoard, validateMatchHistory } from '../connect4/matchRecord.js';

export const SIZES = [8, 16, 32, 64];
export const VERSION = 1;
export const DEFAULT_POOL = [{ type: 'random' }, ...[2, 4, 6, 8].map(depth => ({ type: 'negamax', depth })),
  ...MCTS_SIMULATIONS.map(simulations => ({ type: 'mcts', simulations }))];
export const defaultField = size => Array.from({ length: size }, (_, i) => ({ ...DEFAULT_POOL[i % DEFAULT_POOL.length] }));
const check = (condition, message = 'Invalid tournament record.') => { if (!condition) throw new Error(message); };
const equal = (a, b) => JSON.stringify(a) === JSON.stringify(b);
export function validSeed(seed) { return Number.isInteger(seed) && seed >= 0 && seed <= 0xffffffff; }
export function aiConfig(config) {
  check(config && ['random', 'negamax', 'mcts'].includes(config.type), 'Choose an AI entrant.');
  check(config.type !== 'negamax' || NEGAMAX_DEPTHS.includes(config.depth));
  check(config.type !== 'mcts' || MCTS_SIMULATIONS.includes(config.simulations));
  check(equal(Object.keys(config).sort(), (config.type === 'random' ? ['type'] : config.type === 'negamax' ? ['depth', 'type'] : ['simulations', 'type'])));
  toPlayerPayload(config);
  return { ...config };
}
// Domain-separated FNV-1a over ASCII, followed by an unsigned 32-bit avalanche.
export function deriveSeed(seed, domain) {
  check(validSeed(seed), 'Tournament seed must be an unsigned 32-bit integer.');
  let h = (2166136261 ^ seed) >>> 0;
  for (const char of domain) h = Math.imul(h ^ char.charCodeAt(0), 16777619) >>> 0;
  h = Math.imul(h ^ (h >>> 16), 0x7feb352d) >>> 0;
  h = Math.imul(h ^ (h >>> 15), 0x846ca68b) >>> 0;
  return (h ^ (h >>> 16)) >>> 0;
}
export function bracketOrder(size, seed) {
  const order = Array.from({ length: size }, (_, i) => `entrant-${i + 1}`);
  let counter = 0;
  for (let i = size - 1; i > 0; i--) {
    // Rejection sampling avoids modulo bias; values are deterministic, not cryptographic.
    const limit = Math.floor(0x100000000 / (i + 1)) * (i + 1);
    let value;
    do { value = deriveSeed(seed, `shuffle:${counter++}`); } while (value >= limit);
    const j = value % (i + 1); [order[i], order[j]] = [order[j], order[i]];
  }
  return order;
}
export function roundLabel(size, round) {
  const remaining = size / 2 ** round;
  return ({ 2: 'Final', 4: 'Semifinals', 8: 'Quarterfinals' })[remaining] ?? `Round of ${remaining}`;
}
export function createTournament(configs, seed, tournamentId = `tournament-${seed}`) {
  check(Array.isArray(configs) && SIZES.includes(configs.length), 'Choose 8, 16, 32 or 64 entrants.');
  check(validSeed(seed), 'Tournament seed must be an unsigned 32-bit integer.');
  check(typeof tournamentId === 'string' && tournamentId.length > 0 && tournamentId.length <= 100);
  const size = configs.length, order = bracketOrder(size, seed);
  const entrants = configs.map((config, i) => ({ entrantId: `entrant-${i + 1}`, seedNumber: i + 1, config: aiConfig(config) }));
  const rounds = Array.from({ length: Math.log2(size) }, (_, round) =>
    Array.from({ length: size / 2 ** (round + 1) }, (_, index) => ({ matchupId: `r${round + 1}-m${index + 1}`, round, index,
      entrantAId: round ? null : order[index * 2], entrantBId: round ? null : order[index * 2 + 1],
      status: 'pending', games: [], winnerEntrantId: null, resolution: null })));
  return { version: VERSION, tournamentId, size, tournamentSeed: seed, status: 'paused', entrants, bracketOrder: order,
    rounds, active: null, retainedGameId: null, championEntrantId: null };
}
export const allMatchups = t => t.rounds.flat();
export const findMatchup = (t, id) => allMatchups(t).find(m => m.matchupId === id);
export const entrant = (t, id) => t.entrants.find(e => e.entrantId === id);
export const nextMatchup = t => allMatchups(t).find(m => m.status !== 'complete' && m.entrantAId && m.entrantBId);
export function gamePlan(t, matchup, number = matchup.games.length + 1) {
  check(matchup.entrantAId && matchup.entrantBId && number >= 1 && number <= 3);
  const gameSeed = deriveSeed(t.tournamentSeed, `${matchup.matchupId}:game:${number}`);
  const colorSeed = deriveSeed(t.tournamentSeed, `${matchup.matchupId}:color:${number === 2 ? 1 : number}`);
  const reversed = Boolean(colorSeed & 1) !== (number === 2);
  const playerEntrantIds = reversed ? [matchup.entrantBId, matchup.entrantAId] : [matchup.entrantAId, matchup.entrantBId];
  return { gameNumber: number, gameSeed, playerEntrantIds,
    playerConfigs: playerEntrantIds.map(id => toPlayerPayload(entrant(t, id).config)) };
}
function boardOutcome(board) {
  for (let r = 0; r < 6; r++) for (let c = 0; c < 7; c++) if (board[r][c] !== ' ') {
    for (const [dr, dc] of [[0, 1], [1, 0], [1, 1], [1, -1]]) {
      if ([0, 1, 2, 3].every(i => board[r + dr * i]?.[c + dc * i] === board[r][c]))
        return { status: 'win', winnerIndex: board[r][c] === 'X' ? 0 : 1 };
    }
  }
  return { status: board.flat().includes(' ') ? 'ongoing' : 'draw', winnerIndex: null };
}
// Boards are derived in memory, never saved. Also rejects moves after terminal play.
export function replayColumns(columns, players, gameId = 'local-replay') {
  check(Array.isArray(columns) && columns.length <= 42);
  let board = emptyBoard(), result = boardOutcome(board);
  const moves = [];
  for (const [i, column] of columns.entries()) {
    check(Number.isInteger(column) && column >= 0 && column <= 6 && result.status === 'ongoing', 'Malformed replay columns.');
    const row = board.findLastIndex(r => r[column] === ' '); check(row >= 0, 'Replay column is full.');
    const before = board; board = board.map(r => [...r]); board[row][column] = i % 2 ? 'O' : 'X';
    result = boardOutcome(board);
    moves.push({ column, player: i % 2, revision_before: i, revision: i + 1, move_number: i + 1,
      board_before: before, board_after: board, agent: players[i % 2], outcome: { status: result.status, winner: result.winnerIndex } });
  }
  const game = { game_id: gameId, revision: columns.length, board, players, currentPlayer: columns.length % 2,
    gameOver: result.status !== 'ongoing', winner: result.status === 'win' ? `Player ${result.winnerIndex + 1}` : result.status === 'draw' ? 'Draw' : null,
    legalMoves: result.status !== 'ongoing' ? [] : board[0].flatMap((p, c) => p === ' ' ? [c] : []) };
  return { game, moves, result };
}
export function compactHistory(t, matchup, history) {
  const plan = gamePlan(t, matchup);
  const record = validateMatchHistory(history, { players: plan.playerConfigs, rng_seed: plan.gameSeed });
  check(record.game.gameOver, 'Only confirmed completed games may be compacted.');
  const columns = record.moves.map(m => m.column);
  const { result } = replayColumns(columns, plan.playerConfigs);
  return { ...plan, columns, result, moveCount: columns.length };
}
export function recordGame(t, matchupId, game) {
  const copy = structuredClone(t), m = findMatchup(copy, matchupId);
  check(m && m.status !== 'complete' && m.entrantAId && m.entrantBId, 'Impossible advancement.');
  check(nextMatchup(t)?.matchupId === matchupId, 'Complete matchups in bracket order.');
  validateCompletedGame(copy, m, game, m.games.length + 1);
  m.games.push(structuredClone(game));
  if (game.result.status === 'win') {
    m.winnerEntrantId = game.playerEntrantIds[game.result.winnerIndex]; m.resolution = 'game_win';
  } else if (m.games.length === 3) {
    const pick = deriveSeed(copy.tournamentSeed, `${m.matchupId}:draw-tiebreak`) & 1;
    m.winnerEntrantId = pick ? m.entrantBId : m.entrantAId; m.resolution = 'seeded_draw_tiebreak';
  }
  m.status = m.winnerEntrantId ? 'complete' : 'pending';
  copy.active = null;
  if (m.winnerEntrantId) {
    const next = copy.rounds[m.round + 1]?.[Math.floor(m.index / 2)];
    if (next) { const slot = m.index % 2 ? 'entrantBId' : 'entrantAId'; check(next[slot] === null); next[slot] = m.winnerEntrantId; }
    else { copy.championEntrantId = m.winnerEntrantId; copy.status = 'complete'; }
  }
  return copy;
}
function validateCompletedGame(t, m, game, number) {
  const plan = gamePlan(t, m, number);
  check(game && equal(Object.keys(game).sort(), ['columns', 'gameNumber', 'gameSeed', 'moveCount', 'playerConfigs', 'playerEntrantIds', 'result'].sort()));
  for (const key of Object.keys(plan)) check(equal(game[key], plan[key]), 'Game provenance does not match the bracket.');
  const { result } = replayColumns(game.columns, plan.playerConfigs);
  check(result.status !== 'ongoing' && equal(result, game.result) && game.moveCount === game.columns.length, 'Invalid completed result.');
}
export function validateTournament(value) {
  check(value && value.version === VERSION, 'Unsupported or malformed tournament storage.');
  check(SIZES.includes(value.size) && Array.isArray(value.entrants) && value.entrants.length === value.size);
  const rebuilt = createTournament(value.entrants.map(e => e.config), value.tournamentSeed, value.tournamentId);
  check(equal(rebuilt.entrants, value.entrants) && equal(rebuilt.bracketOrder, value.bracketOrder), 'Invalid entrant field or bracket order.');
  check(Array.isArray(value.rounds) && value.rounds.length === rebuilt.rounds.length);
  let gap = false;
  for (let r = 0; r < rebuilt.rounds.length; r++) {
    check(Array.isArray(value.rounds[r]) && value.rounds[r].length === rebuilt.rounds[r].length);
    for (let i = 0; i < rebuilt.rounds[r].length; i++) {
      const stored = value.rounds[r][i]; check(stored && Array.isArray(stored.games) && stored.games.length <= 3);
      if (stored.games.length) check(!gap, 'Impossible round advancement.');
      for (const game of stored.games) Object.assign(rebuilt, recordGame(rebuilt, stored.matchupId, game));
      const expected = rebuilt.rounds[r][i];
      check(equal(expected, { ...stored, status: stored.status === 'active' ? 'pending' : stored.status }), 'Invalid bracket topology or winner.');
      if (expected.status !== 'complete') gap = true;
    }
  }
  check(value.status === rebuilt.status && value.championEntrantId === rebuilt.championEntrantId);
  check(value.retainedGameId === null || typeof value.retainedGameId === 'string' && value.retainedGameId.length > 0 && value.retainedGameId.length <= 64);
  if (value.active !== null) {
    const a = value.active, m = nextMatchup(rebuilt);
    check(a && m && a.matchupId === m.matchupId && a.gameNumber === m.games.length + 1);
    check(['starting', 'running', 'interrupted'].includes(a.status));
    check(a.gameId === null || typeof a.gameId === 'string' && a.gameId.length > 0 && a.gameId.length <= 64);
    check(a.status !== 'running' || a.gameId !== null);
    check(a.gameId === null || value.retainedGameId === a.gameId, 'Active session identity does not match replacement provenance.');
    check(a.status !== 'starting' || a.gameId === null && a.columns.length === 0);
    replayColumns(a.columns, gamePlan(rebuilt, m).playerConfigs);
    check(value.rounds[m.round][m.index].status === 'active');
  } else check(!allMatchups(value).some(m => m.status === 'active'));
  // Canonical reconstruction drops unknown fields/derived snapshots from storage.
  rebuilt.active = value.active === null ? null : { matchupId: value.active.matchupId, gameNumber: value.active.gameNumber,
    gameId: value.active.gameId, status: value.active.status, columns: [...value.active.columns] };
  rebuilt.retainedGameId = value.retainedGameId;
  if (rebuilt.active) findMatchup(rebuilt, rebuilt.active.matchupId).status = 'active';
  return rebuilt;
}
