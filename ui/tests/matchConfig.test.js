import { test } from 'node:test';
import assert from 'node:assert/strict';
import { competitorLabel, matchStartPayload, toPlayerPayload } from '../src/connect4/competitorConfig.js';
import { matchResult, resultLabel, replaySnapshot, timelineLabel, validateMatchHistory } from '../src/connect4/matchRecord.js';
import { DRAW, WIN, matchFixture } from '../e2e/fixtures/match.js';

const configs = [{ type: 'human' }, { type: 'random' }, ...[1, 2, 4, 6, 8].map(depth => ({ type: 'negamax', depth })),
  ...[100, 400, 800].map(simulations => ({ type: 'mcts', simulations }))];
for (const config of configs) test(`exact public payload and identity: ${competitorLabel(config)}`, () => {
  const payload = config.type === 'mcts' ? { type: 'mcts', simulation_limit: config.simulations } : config;
  assert.deepEqual(toPlayerPayload({ depth: 8, simulations: 800, ...config }), payload);
  assert.equal(competitorLabel(payload), competitorLabel(config));
});
test('any independent competitor pairing builds a detached replacement payload', () => {
  for (const a of configs) for (const b of configs) {
    const before = structuredClone([a, b]);
    assert.deepEqual(matchStartPayload(a, b, 'old'), { player1: toPlayerPayload(a), player2: toPlayerPayload(b), replace_game_id: 'old' });
    assert.deepEqual([a, b], before);
  }
});
for (const config of [{ type: 'unknown' }, { type: 'negamax', depth: 3 }, { type: 'negamax', depth: true }, { type: 'mcts', simulations: 250 }]) {
  test(`nonpublic config rejected ${JSON.stringify(config)}`, () => assert.throws(() => toPlayerPayload(config)));
}
for (const seq of [[], [3, 2], WIN, DRAW]) test(`validate complete replay and result at ${seq.length} plies`, () => {
  const fixture = matchFixture([{ type: 'negamax', depth: 6 }, { type: 'mcts', simulation_limit: 400 }], seq);
  const before = structuredClone(fixture), { game, moves } = validateMatchHistory(fixture);
  assert.deepEqual(fixture, before);
  for (let i = 0; i <= seq.length; i++) {
    const replay = replaySnapshot(game, moves, i);
    assert.equal(replay.revision, i);
    assert.deepEqual(replay.board, matchFixture(game.players, seq.slice(0, i)).state.board);
  }
  if (seq === WIN) assert.equal(resultLabel(matchResult(game)), 'Negamax · depth 6 wins as Red in 7 moves.');
  if (seq === DRAW) assert.equal(resultLabel(matchResult(game)), 'Draw after 42 moves.');
  if (moves.length > 1) assert.equal(timelineLabel(moves[1], game.players), `2 · Yellow · MCTS · 400 simulations · Column ${seq[1] + 1}`);
});
const corruptions = [
  d => d.revision++, d => d.moves.pop(), d => d.state.game_id = 'other', d => d.players = [{ type: 'human' }, { type: 'human' }],
  d => d.moves.reverse(), d => d.moves[0].revision = 4, d => d.moves[0].move_number = 4,
  d => d.moves[0].player = 1, d => d.moves[0].column = 5, d => d.moves[0].agent.type = 'human',
  d => d.moves[0].board_before[5][0] = 'O', d => d.moves[0].board_after[5][0] = 'O',
  d => d.moves[0].outcome.status = 'win', d => d.state.board[5][0] = 'O', d => d.state.legalMoves = [],
  d => d.state.winner = 'Player 1', d => d.state.currentPlayer = 1, d => d.state.players[0].depth = 99,
];
for (const [i, mutate] of corruptions.entries()) test(`mismatched authoritative history ${i} locks execution`, () => {
  const fixture = matchFixture(undefined, [3, 2]); mutate(fixture);
  assert.throws(() => validateMatchHistory(fixture));
});
test('history accepts object key order and rejects a different requested revision or players', () => {
  const fixture = matchFixture(undefined, [3]);
  fixture.moves[0].outcome = { winner: null, status: 'ongoing' };
  assert.equal(validateMatchHistory(fixture).game.revision, 1);
  assert.throws(() => validateMatchHistory(fixture, { minRevision: 2 }));
  assert.throws(() => validateMatchHistory(fixture, { players: [{ type: 'human' }, { type: 'human' }] }));
});
