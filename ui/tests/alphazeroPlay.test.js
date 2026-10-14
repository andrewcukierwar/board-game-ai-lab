import { test } from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';
import React, { act, StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { useConnect4Game } from '../src/connect4/useConnect4Game.js';
import { opponentPayload } from '../src/connect4/researchAgent.js';
import { matchFixture } from '../e2e/fixtures/match.js';
import { status, provenance, az } from './fixtures/alphazero.js';
globalThis.IS_REACT_ACT_ENVIRONMENT = true;
const flush = () => act(async () => { await new Promise(r => setImmediate(r)); });
function mount(http) {
  const dom = new JSDOM('<div id="root"></div>'); globalThis.window = dom.window; globalThis.document = dom.window.document;
  let game; function Hook() { game = useConnect4Game(http, { researchEnabled: true }); return null; }
  const root = createRoot(document.getElementById('root'));
  act(() => root.render(React.createElement(StrictMode, null, React.createElement(Hook))));
  return { get game() { return game; }, close: () => { act(() => root.unmount()); dom.window.close(); } };
}
for (const first of ['human', 'ai']) test(`Play ${first}-first model mismatch quarantines board until new verified game`, async () => {
  let players, movePosts = 0, mismatch = true;
  const http = {
    post: async (url, body) => {
      if (url.endsWith('start_game')) { players = [body.player1, body.player2]; return { data: matchFixture(players).state }; }
      movePosts++; return { data: matchFixture(players, [3]).state };
    },
    get: async url => {
      if (url.endsWith('agents/alphazero')) return { data: status };
      const h = matchFixture(players); h.alphazero_provenance = structuredClone(provenance);
      if (mismatch) h.alphazero_provenance.model.id = 'other-model';
      return { data: h };
    },
  };
  const ui = mount(http); await flush(); act(() => ui.game.select({ ...az(false), first }));
  await act(() => ui.game.start()); await flush();
  assert.equal(ui.game.uncertain, true); assert.equal(ui.game.retryAI, false); assert.equal(ui.game.integrityBlocked, true);
  assert.match(ui.game.message, /model or implementation changed/); assert.equal(movePosts, 0);
  await act(() => ui.game.move(3)); assert.equal(movePosts, 0);
  mismatch = false; await act(() => ui.game.start()); await flush();
  assert.equal(ui.game.integrityBlocked, false); assert.equal(ui.game.uncertain, false);
  assert.equal(movePosts, first === 'ai' ? 1 : 0); ui.close();
});
test('Play retry status mismatch cannot unlock a game from snapshot-only recovery', async () => {
  let players, changed = false, moves = 0;
  const http = {
    post: async (url, body) => { if (url.endsWith('start_game')) { players = [body.player1, body.player2]; return { data: matchFixture(players).state }; } moves++; throw new Error('lost response'); },
    get: async url => {
      if (url.endsWith('agents/alphazero')) { const s = structuredClone(status); if (changed) s.provenance.model.id = s.model.id = 'other-model'; return { data: s }; }
      return { data: { ...matchFixture(players), alphazero_provenance: provenance } };
    },
  };
  const ui = mount(http); await flush(); act(() => ui.game.select({ ...az(true), first: 'human' })); await act(() => ui.game.start());
  changed = true; await act(() => ui.game.retry());
  assert.equal(ui.game.uncertain, true); assert.equal(ui.game.integrityBlocked, true); assert.equal(moves, 0); ui.close();
});
test('AlphaZero to Victor Play selection preserves inactive controls without leaking guard', () => {
  assert.deepEqual(opponentPayload({ type: 'victor_research', first: 'human', depth: 6, simulations: 2000, tactical_guard: true }, true), { type: 'victor_research' });
});
