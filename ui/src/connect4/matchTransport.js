import { validateSnapshot } from './gameSnapshot.js';
import { samePlayers } from './matchRecord.js';

// One revision-bound ply. The owning match controller supplies serialization;
// this transport never retries or chains another POST.
export async function requestMatchPly(http, game, column, options) {
  const human = game.players[game.currentPlayer].type === 'human';
  if (game.gameOver || (human ? !game.legalMoves.includes(column) : column !== undefined)) {
    throw new Error('This move is unavailable in the live position.');
  }
  const response = await http.post('/v1/connect4/make_move', {
    game_id: game.game_id, revision: game.revision, ...(human ? { column } : {}),
  }, options);
  const accepted = validateSnapshot(response.data, { game_id: game.game_id, revision: game.revision + 1 });
  if (!samePlayers(accepted.players, game.players)) throw new Error('The match players changed.');
  return accepted;
}
