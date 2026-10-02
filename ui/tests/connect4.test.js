import { test } from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';
import { mountConnect4 } from '../legacy/connect4.js';

function state(revision = 0, overrides = {}) {
  const board = Array.from({ length: 6 }, () => Array(7).fill(' '));
  if (revision > 0) board[5][3] = 'X';
  if (revision > 1) board[5][2] = 'O';
  return { game_id: 'game-a', revision, board, players: [{ type: 'human' }, { type: 'random' }],
    currentPlayer: revision % 2, gameOver: false, legalMoves: [0, 1, 2, 3, 4, 5, 6], winner: null, ...overrides };
}
const response = data => ({ data });
const failure = (status, error = 'Request failed') => Object.assign(new Error(error), { response: { status, data: { error } } });
const flush = () => new Promise(resolve => setImmediate(resolve));
function setup(http) {
  const dom = new JSDOM(`<select id="opponent-type"><option value="random">Random</option><option value="negamax">Negamax</option></select>
    <select id="opponent-depth"><option value="2">2</option></select><div id="negamax-options"></div>
    <button id="start-button"></button><button id="restart-button"></button><button id="retry-button"></button>
    <div id="message"></div><div id="loading"></div><div id="game-board"></div>`);
  const doc = dom.window.document;
  const cleanup = mountConnect4({ document: doc, http });
  return { doc, cleanup, el: id => doc.getElementById(id),
    click: id => doc.getElementById(id).click(),
    column: () => doc.querySelector('[data-column="3"]') };
}

test('failed initialization re-enables Start and permits a successful retry', async () => {
  let calls = 0;
  const ui = setup({ post: async () => { if (++calls === 1) throw failure(503, 'Server unavailable'); return response(state()); } });
  ui.click('start-button'); await flush();
  assert.equal(ui.el('start-button').disabled, false);
  assert.equal(ui.el('start-button').hidden, false);
  assert.match(ui.el('message').textContent, /Server unavailable/);
  ui.click('start-button'); await flush();
  assert.equal(ui.doc.querySelectorAll('.cell').length, 42);
  assert.equal(ui.el('restart-button').hidden, false);
  ui.cleanup();
});

test('a cold-start HTML success response keeps initialization recoverable', async () => {
  let calls = 0;
  const ui = setup({ post: async () => response(++calls === 1 ? '<html>Waking up</html>' : state()) });
  ui.click('start-button'); await flush();
  assert.equal(ui.el('start-button').disabled, false);
  assert.equal(ui.doc.querySelectorAll('.cell').length, 0);
  assert.match(ui.el('message').textContent, /waking up/);
  ui.click('start-button'); await flush();
  assert.equal(ui.doc.querySelectorAll('.cell').length, 42);
  ui.cleanup();
});

test('a non-JSON move response preserves the game for snapshot recovery', async () => {
  const ui = setup({ get: async () => response(state(1)), post: async url =>
    response(url.endsWith('start_game') ? state() : '<html>Gateway page</html>') });
  ui.click('start-button'); await flush(); ui.column().click(); await flush();
  assert.equal(ui.doc.querySelectorAll('.circle.x').length, 1);
  assert.equal(ui.el('retry-button').textContent, 'Retry AI move');
  assert.equal(ui.column().disabled, true);
  ui.cleanup();
});

test('double click cannot issue concurrent human or AI requests', async () => {
  let release;
  const calls = [];
  const ui = setup({ post: async (url, body) => {
    calls.push({ url, body });
    if (url.endsWith('start_game')) return response(state());
    if ('column' in body) return new Promise(resolve => { release = () => resolve(response(state(1))); });
    return response(state(2));
  } });
  ui.click('start-button'); await flush();
  const oldCell = ui.column(); oldCell.click(); oldCell.click();
  assert.equal(calls.length, 2);
  assert.equal(ui.el('restart-button').disabled, true);
  release(); await flush();
  assert.equal(calls.length, 3);
  assert.deepEqual(calls[2].body, { game_id: 'game-a', revision: 1 });
  assert.equal(ui.column().disabled, false);
  ui.cleanup();
});

test('AI failure stops and offers explicit retry using the latest revision', async () => {
  let aiCalls = 0;
  const ui = setup({ get: async () => response(state(1)), post: async (url, body) => {
    if (url.endsWith('start_game')) return response(state());
    if ('column' in body) return response(state(1));
    if (++aiCalls === 1) throw failure(503, 'AI unavailable');
    assert.equal(body.revision, 1);
    return response(state(2));
  } });
  ui.click('start-button'); await flush(); ui.column().click(); await flush();
  assert.equal(aiCalls, 1);
  assert.equal(ui.el('retry-button').hidden, false);
  assert.equal(ui.column().disabled, true);
  ui.click('retry-button'); await flush();
  assert.equal(aiCalls, 2);
  assert.equal(ui.el('retry-button').hidden, true);
  assert.equal(ui.column().disabled, false);
  ui.cleanup();
});

test('lost move response is reconciled without replaying the human move', async () => {
  let humanCalls = 0, aiCalls = 0;
  const ui = setup({ get: async () => response(state(1)), post: async (url, body) => {
    if (url.endsWith('start_game')) return response(state());
    if ('column' in body) { humanCalls++; throw new Error('Response lost'); }
    aiCalls++; return response(state(2));
  } });
  ui.click('start-button'); await flush(); ui.column().click(); await flush();
  assert.equal(humanCalls, 1); assert.equal(aiCalls, 0);
  ui.click('retry-button'); await flush();
  assert.equal(humanCalls, 1); assert.equal(aiCalls, 1);
  ui.cleanup();
});

test('AI timeout after commit reads the new board and never repeats the AI move', async () => {
  let aiCalls = 0;
  const ui = setup({ get: async () => response(state(2)), post: async (url, body) => {
    if (url.endsWith('start_game')) return response(state());
    if ('column' in body) return response(state(1));
    aiCalls++;
    throw Object.assign(new Error('Timeout'), { code: 'ECONNABORTED' });
  } });
  ui.click('start-button'); await flush(); ui.column().click(); await flush();
  assert.equal(aiCalls, 1);
  assert.equal(ui.doc.querySelectorAll('.circle.x').length, 1);
  assert.equal(ui.doc.querySelectorAll('.circle.o').length, 1);
  assert.equal(ui.el('retry-button').hidden, true);
  assert.equal(ui.column().disabled, false);
  ui.cleanup();
});

test('failed refresh blocks moves until explicit recovery, including terminal recovery', async () => {
  let refreshes = 0;
  const ui = setup({ get: async () => {
    if (++refreshes === 1) throw new Error('Offline');
    return response(state(2, { gameOver: true, winner: 'Player 2', legalMoves: [] }));
  }, post: async url => {
    if (url.endsWith('start_game')) return response(state());
    throw new Error('Offline');
  } });
  ui.click('start-button'); await flush(); ui.column().click(); await flush();
  assert.equal(ui.column().disabled, true);
  assert.equal(ui.el('retry-button').textContent, 'Refresh game');
  ui.click('retry-button'); await flush();
  assert.match(ui.el('message').textContent, /AI wins/);
  assert.equal(ui.column().disabled, true);
  assert.equal(ui.el('retry-button').hidden, true);
  ui.cleanup();
});

test('expired session returns to a usable Start screen', async () => {
  const ui = setup({ get: async () => { throw failure(404); }, post: async url => {
    if (url.endsWith('start_game')) return response(state());
    throw failure(404);
  } });
  ui.click('start-button'); await flush(); ui.column().click(); await flush();
  assert.match(ui.el('message').textContent, /expired/);
  assert.equal(ui.el('start-button').hidden, false);
  assert.equal(ui.el('start-button').disabled, false);
  assert.equal(ui.doc.querySelectorAll('.cell').length, 0);
  ui.click('start-button'); await flush();
  assert.equal(ui.doc.querySelectorAll('.cell').length, 42);
  ui.cleanup();
});

test('restart applies changed opponent and replaces the old session', async () => {
  const bodies = [];
  const ui = setup({ post: async (url, body) => {
    bodies.push(body);
    return response(state(0, { game_id: `game-${bodies.length}` }));
  } });
  ui.click('start-button'); await flush();
  ui.el('opponent-type').value = 'negamax';
  ui.click('restart-button'); await flush();
  assert.deepEqual(bodies[1], { player1: { type: 'human' }, player2: { type: 'negamax', depth: 2 }, replace_game_id: 'game-1' });
  assert.equal(ui.column().disabled, false);
  ui.cleanup();
});

test('failed restart retains the prior game and allows another attempt', async () => {
  let starts = 0;
  const ui = setup({ get: async () => response(state()), post: async () => {
    if (++starts === 2) throw failure(503);
    return response(state());
  } });
  ui.click('start-button'); await flush(); ui.click('restart-button'); await flush();
  assert.equal(ui.el('restart-button').disabled, false);
  assert.equal(ui.column().disabled, false);
  ui.click('restart-button'); await flush(); assert.equal(starts, 3);
  ui.cleanup();
});

test('navigation aborts requests and late responses cannot update an unmounted page', async () => {
  let release, signal;
  const ui = setup({ post: (url, body, options) => {
    signal = options.signal;
    return new Promise(resolve => { release = resolve; });
  } });
  ui.click('start-button'); ui.cleanup(); ui.doc.body.replaceChildren();
  assert.equal(signal.aborted, true);
  release(response(state())); await flush();
  assert.equal(ui.doc.body.childNodes.length, 0);
});
