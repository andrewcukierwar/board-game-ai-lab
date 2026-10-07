import { test } from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';
import React, { act } from 'react';
import { createRoot } from 'react-dom/client';
import { MemoryRouter } from 'react-router-dom';
import Connect4Page from '../src/pages/Connect4.jsx';
import { explained } from '../e2e/fixtures/analysis.js';

globalThis.IS_REACT_ACT_ENVIRONMENT = true;

function state(revision = 0, overrides = {}) {
  const board = Array.from({ length: 6 }, () => Array(7).fill(' '));
  if (revision > 0) board[5][3] = 'X';
  if (revision > 1) board[5][2] = 'O';
  return { game_id: 'game-a', revision, board, players: [{ type: 'human' }, { type: 'random' }],
    currentPlayer: revision % 2, gameOver: false, legalMoves: [0, 1, 2, 3, 4, 5, 6], winner: null, ...overrides };
}
const response = data => ({ data });
const failure = (status, error = 'Request failed') => Object.assign(new Error(error), { response: { status, data: { error } } });
const flush = () => act(async () => { await new Promise(resolve => setImmediate(resolve)); });
function setup(http) {
  const dom = new JSDOM('<div id="root"></div>', { url: 'http://localhost/connect4' });
  globalThis.window = dom.window;
  globalThis.document = dom.window.document;
  const doc = dom.window.document;
  const root = createRoot(doc.getElementById('root'));
  act(() => root.render(React.createElement(MemoryRouter, { future: { v7_startTransition: true, v7_relativeSplatPath: true } }, React.createElement(Connect4Page, { http }))));
  const cleanup = () => { act(() => root.unmount()); dom.window.close(); };
  return { doc, cleanup, el: id => doc.getElementById(id),
    click: id => act(() => doc.getElementById(id).click()),
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
  ui.click('start-button'); await flush(); act(() => ui.column().click()); await flush();
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
  const oldCell = ui.column(); act(() => { oldCell.click(); oldCell.click(); });
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
  ui.click('start-button'); await flush(); act(() => ui.column().click()); await flush();
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
  ui.click('start-button'); await flush(); act(() => ui.column().click()); await flush();
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
  ui.click('start-button'); await flush(); act(() => ui.column().click()); await flush();
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
  ui.click('start-button'); await flush(); act(() => ui.column().click()); await flush();
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
  ui.click('start-button'); await flush(); act(() => ui.column().click()); await flush();
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
  select(ui, 'opponent-type', 'random');
  ui.click('start-button'); await flush();
  select(ui, 'opponent-type', 'negamax');
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

function select(ui, id, value) {
  act(() => {
    if (id === 'opponent-type') ui.doc.querySelector(`input[name="opponent"][value="${value}"]`).click();
    else {
      ui.el(id).value = value;
      ui.el(id).dispatchEvent(new ui.doc.defaultView.Event('change', { bubbles: true }));
    }
  });
}

for (const limit of [100, 400, 800]) {
  test(`MCTS selector shows only its options and sends ${limit} simulations`, async () => {
    let body;
    const ui = setup({ post: async (url, data) => { body = data; return response(state()); } });
    assert.equal(ui.el('mcts-options').hidden, true);
    select(ui, 'opponent-type', 'mcts');
    assert.equal(ui.el('mcts-options').hidden, false);
    assert.equal(ui.el('negamax-options').hidden, true);
    assert.equal(ui.el('opponent-simulations').value, '100');
    select(ui, 'opponent-simulations', String(limit));
    ui.click('start-button'); await flush();
    assert.deepEqual(body, { player1: { type: 'human' }, player2: { type: 'mcts', simulation_limit: limit } });
    ui.cleanup();
  });
}

test('switching opponents applies only at restart and never leaks agent settings', async () => {
  const bodies = [];
  const ui = setup({ post: async (url, body) => {
    bodies.push(body);
    return response(state(0, { game_id: `game-${bodies.length}`, players: [body.player1, body.player2] }));
  } });
  for (const [index, type] of ['mcts', 'negamax', 'random', 'mcts'].entries()) {
    select(ui, 'opponent-type', type);
    assert.equal(bodies.length, index);
    assert.equal(ui.el('mcts-options').hidden, type !== 'mcts');
    assert.equal(ui.el('negamax-options').hidden, type !== 'negamax');
    ui.click(index ? 'restart-button' : 'start-button'); await flush();
    assert.deepEqual(bodies[index].player2, type === 'mcts' ? { type, simulation_limit: 100 }
      : type === 'negamax' ? { type, depth: 2 } : { type });
    if (index) assert.equal(bodies[index].replace_game_id, `game-${index}`);
  }
  ui.cleanup();
});

for (const committed of [false, true]) {
  test(`MCTS failed response reconciles ${committed ? 'committed' : 'busy'} state without automatic replay`, async () => {
    let aiCalls = 0;
    const players = [{ type: 'human' }, { type: 'mcts', simulation_limit: 50 }];
    const snapshot = revision => state(revision, { players });
    const ui = setup({ get: async () => response(snapshot(committed ? 2 : 1)), post: async (url, body) => {
      if (url.endsWith('start_game')) return response(snapshot(0));
      if ('column' in body) return response(snapshot(1));
      if (++aiCalls === 1) throw failure(503, committed ? 'Response lost' : 'Another MCTS search is running');
      assert.equal(body.revision, 1);
      return response(snapshot(2));
    } });
    select(ui, 'opponent-type', 'mcts');
    ui.click('start-button'); await flush();
    act(() => ui.column().click()); await flush();
    assert.equal(aiCalls, 1);
    assert.equal(ui.el('retry-button').hidden, committed);
    assert.equal(ui.column().disabled, !committed);
    if (!committed) {
      ui.click('retry-button'); await flush();
      assert.equal(aiCalls, 2);
      assert.equal(ui.column().disabled, false);
    }
    ui.cleanup();
  });
}

test('slow initialization locks every start and opponent control without retrying', async () => {
  let release, calls = 0;
  const ui = setup({ post: () => { calls++; return new Promise(resolve => { release = resolve; }); } });
  ui.click('start-button'); ui.click('start-button');
  assert.equal(calls, 1);
  assert.equal(ui.el('start-button').disabled, true);
  assert.ok([...ui.doc.querySelectorAll('input[name="opponent"]')].every(input => input.matches(':disabled')));
  assert.equal(ui.el('opponent-depth').disabled, true);
  assert.equal(ui.el('opponent-simulations').disabled, true);
  assert.equal(ui.el('analyze-position').disabled, true);
  release(response(state())); await flush();
  assert.equal(ui.el('start-button').hidden, true);
  assert.equal(ui.el('restart-button').disabled, false);
  ui.cleanup();
});

for (const stage of ['human', 'ai', 'refresh']) {
  test(`unmount during ${stage} request aborts and ignores late snapshots`, async () => {
    let release, signal, aiCalls = 0;
    const pending = options => {
      signal = options.signal;
      return new Promise(resolve => { release = resolve; });
    };
    const ui = setup({ get: (url, options) => pending(options), post: async (url, body, options) => {
      if (url.endsWith('start_game')) return response(state());
      if ('column' in body) {
        if (stage === 'human') return pending(options);
        if (stage === 'refresh') throw new Error('Offline');
        return response(state(1));
      }
      aiCalls++;
      return pending(options);
    } });
    ui.click('start-button'); await flush();
    act(() => ui.column().click()); await flush();
    const before = aiCalls;
    ui.cleanup();
    assert.equal(signal.aborted, true);
    release(response(state(stage === 'human' ? 1 : 2))); await flush();
    assert.equal(aiCalls, before);
    assert.equal(ui.doc.querySelectorAll('.cell').length, 0);
  });
}

for (const [winner, wording] of [['Player 1', 'You win'], ['Player 2', 'The AI wins'], ['Draw', "It's a draw"]]) {
  test(`${winner} terminal snapshot stops the move chain and disables the board`, async () => {
    let moves = 0;
    const ui = setup({ post: async url => {
      if (url.endsWith('start_game')) return response(state());
      moves++;
      return response(state(1, { gameOver: true, winner, legalMoves: [] }));
    } });
    ui.click('start-button'); await flush();
    act(() => ui.column().click()); await flush();
    assert.equal(moves, 1);
    assert.ok([...ui.doc.querySelectorAll('.cell')].every(cell => cell.disabled));
    assert.ok(ui.el('message').textContent.includes(wording));
    assert.equal(ui.el('restart-button').disabled, false);
    ui.cleanup();
  });
}

for (const explicit of [false, true]) {
  test(`${explicit ? 'explicit' : 'default'} human-first start has no AI opener`, async () => {
    const calls = [];
    const ui = setup({ post: async (url, body) => { calls.push(body); return response(state()); } });
    assert.equal(ui.el('first-human').checked, true);
    if (explicit) { ui.click('first-ai'); ui.click('first-human'); }
    ui.click('start-button'); await flush();
    assert.deepEqual(calls, [{ player1: { type: 'human' }, player2: { type: 'negamax', depth: 2 } }]);
    ui.cleanup();
  });
}

for (const opponent of [{ type: 'random' }, { type: 'negamax', depth: 8 }, { type: 'mcts', simulation_limit: 800 }]) {
  test(`AI-first ${opponent.type} accepts start then issues exactly one locked revision-0 opener`, async () => {
    const calls = []; let release;
    const players = [opponent, { type: 'human' }];
    const ui = setup({ post: async (url, body) => {
      calls.push({ url, body });
      if (url.endsWith('start_game')) return response(state(0, { players }));
      return new Promise(resolve => { release = () => resolve(response(state(1, { players }))); });
    } });
    select(ui, 'opponent-type', opponent.type);
    if (opponent.depth) select(ui, 'opponent-depth', String(opponent.depth));
    if (opponent.simulation_limit) select(ui, 'opponent-simulations', String(opponent.simulation_limit));
    ui.click('first-ai');
    act(() => { ui.el('start-button').click(); ui.el('start-button').click(); });
    await flush();
    assert.deepEqual(calls.map(c => c.body), [{ player1: opponent, player2: { type: 'human' } }, { game_id: 'game-a', revision: 0 }]);
    assert.equal(ui.doc.querySelectorAll('.cell').length, 42);
    assert.equal(ui.doc.querySelectorAll('.circle.x').length, 0);
    assert.equal(ui.el('restart-button').disabled, true);
    assert.equal(ui.el('first-human').matches(':disabled'), true);
    assert.equal(ui.el('analyze-position').disabled, true);
    assert.match(ui.el('message').textContent, /AI is choosing/);
    release(); await flush();
    assert.equal(calls.length, 2);
    assert.equal(ui.doc.querySelectorAll('.circle.x').length, 1);
    assert.equal(ui.column().disabled, false);
    assert.match(ui.doc.querySelector('.board-legend').textContent, /AI · redYou · yellow/);
    assert.match(ui.doc.querySelector('.settings-note').textContent, /AI moves first/);
    assert.equal(ui.el('explain-last').getAttribute('aria-label'), 'Analyze Last AI Move');
    ui.cleanup();
  });
}

for (const committed of [false, true]) {
  test(`AI opening ${committed ? 'lost response after commit' : 'failure'} reconciles before explicit retry`, async () => {
    const players = [{ type: 'random' }, { type: 'human' }];
    let moves = 0; const events = [];
    const ui = setup({ get: async () => { events.push('get'); return response(state(committed ? 1 : 0, { players })); },
      post: async (url, body) => {
        if (url.endsWith('start_game')) return response(state(0, { players }));
        events.push('post');
        if (++moves === 1) throw failure(503, committed ? 'Response lost' : 'AI unavailable');
        assert.equal(body.revision, 0);
        return response(state(1, { players }));
      } });
    ui.click('first-ai'); ui.click('start-button'); await flush();
    assert.deepEqual(events, ['post', 'get']);
    assert.equal(moves, 1);
    assert.equal(ui.column().disabled, !committed);
    assert.equal(ui.el('retry-button').hidden, committed);
    if (!committed) {
      assert.equal(ui.el('retry-button').textContent, 'Retry AI move');
      ui.click('retry-button'); await flush();
      assert.deepEqual(events, ['post', 'get', 'get', 'post']);
      assert.equal(moves, 2);
    }
    assert.equal(ui.doc.querySelectorAll('.circle.x').length, 1);
    ui.cleanup();
  });
}

test('lost opening response and failed GET keep board locked until authoritative recovery', async () => {
  const players = [{ type: 'random' }, { type: 'human' }];
  let reads = 0, moves = 0;
  const ui = setup({ get: async () => {
    if (++reads === 1) throw new Error('Offline');
    return response(state(1, { players }));
  }, post: async url => {
    if (url.endsWith('start_game')) return response(state(0, { players }));
    moves++; throw new Error('Response lost');
  } });
  ui.click('first-ai'); ui.click('start-button'); await flush();
  assert.equal(ui.el('retry-button').textContent, 'Refresh game');
  assert.equal(ui.column().disabled, true);
  assert.equal(ui.el('analyze-position').disabled, true);
  ui.click('retry-button'); await flush();
  assert.equal(moves, 1);
  assert.equal(ui.column().disabled, false);
  assert.equal(ui.el('retry-button').hidden, true);
  ui.cleanup();
});

for (const invalid of ['<html>Gateway</html>', state(2), state(1, { currentPlayer: 0 }), state(1, { game_id: 'wrong-game' })]) {
  test(`invalid opening snapshot (${typeof invalid === 'string' ? 'HTML' : invalid.game_id + ':' + invalid.revision}) uses GET without replay`, async () => {
    const players = [{ type: 'random' }, { type: 'human' }]; let moves = 0;
    const ui = setup({ get: async () => response(state(1, { players })), post: async url => {
      if (url.endsWith('start_game')) return response(state(0, { players }));
      moves++; return response(invalid);
    } });
    ui.click('first-ai'); ui.click('start-button'); await flush();
    assert.equal(moves, 1);
    assert.equal(ui.doc.querySelectorAll('.circle.x').length, 1);
    assert.equal(ui.column().disabled, false);
    ui.cleanup();
  });
}

test('nonzero start revision is rejected before requesting any AI opener', async () => {
  let calls = 0;
  const ui = setup({ post: async () => { calls++; return response(state(1, { players: [{ type: 'random' }, { type: 'human' }] })); } });
  ui.click('first-ai'); ui.click('start-button'); await flush();
  assert.equal(calls, 1);
  assert.equal(ui.doc.querySelectorAll('.cell').length, 0);
  assert.equal(ui.el('start-button').disabled, false);
  ui.cleanup();
});

test('side selection stays pending until restart, then replaces the old game and opens once', async () => {
  let players, game_id, moves = 0; const starts = [];
  const ui = setup({ post: async (url, body) => {
    if (url.endsWith('start_game')) {
      starts.push(body); players = [body.player1, body.player2]; game_id = `game-${starts.length}`;
      return response(state(0, { players, game_id }));
    }
    moves++; return response(state(1, { players, game_id }));
  } });
  ui.click('start-button'); await flush();
  ui.click('first-ai');
  assert.match(ui.doc.querySelector('.settings-note').textContent, /You move first/);
  assert.match(ui.doc.querySelector('.board-legend').textContent, /You · redAI · yellow/);
  assert.equal(ui.column().disabled, false);
  ui.click('restart-button'); await flush();
  assert.deepEqual(starts[1], { player1: { type: 'negamax', depth: 2 }, player2: { type: 'human' }, replace_game_id: 'game-1' });
  assert.equal(moves, 1);
  assert.match(ui.doc.querySelector('.settings-note').textContent, /Negamax · depth 2 · AI moves first/);
  ui.click('first-human'); ui.click('restart-button'); await flush();
  assert.equal(moves, 1);
  assert.equal(starts[2].player1.type, 'human');
  ui.cleanup();
});

for (const [winner, wording] of [['Player 1', 'The AI wins'], ['Player 2', 'You win'], ['Draw', "It's a draw"]]) {
  test(`${winner} result copy when human is Player 2`, async () => {
    const players = [{ type: 'random' }, { type: 'human' }]; let moves = 0;
    const ui = setup({ post: async url => {
      if (url.endsWith('start_game')) return response(state(0, { players }));
      moves++;
      return response(state(moves, moves === 1 ? { players } : { players, gameOver: true, winner, legalMoves: [] }));
    } });
    ui.click('first-ai'); ui.click('start-button'); await flush();
    act(() => ui.column().click()); await flush();
    assert.equal(moves, 2);
    assert.ok(ui.el('message').textContent.includes(wording));
    assert.ok([...ui.doc.querySelectorAll('.cell')].every(cell => cell.disabled));
    ui.cleanup();
  });
}

for (const stage of ['opening', 'opening-recovery']) {
  test(`navigation during ${stage} aborts and cannot replay or accept a late opening`, async () => {
    const players = [{ type: 'random' }, { type: 'human' }]; let release, signal, moves = 0;
    const pending = options => { signal = options.signal; return new Promise(resolve => { release = resolve; }); };
    const ui = setup({ get: (url, options) => pending(options), post: async (url, body, options) => {
      if (url.endsWith('start_game')) return response(state(0, { players }));
      moves++;
      if (stage === 'opening-recovery') throw new Error('Response lost');
      return pending(options);
    } });
    ui.click('first-ai'); ui.click('start-button'); await flush();
    ui.cleanup();
    assert.equal(signal.aborted, true);
    release(response(state(1, { players }))); await flush();
    assert.equal(moves, 1);
    assert.equal(ui.doc.querySelectorAll('.cell').length, 0);
  });
}

test('all grounded analysis actions after AI opener use revision 1 and never move the board', async () => {
  const players = [{ type: 'random' }, { type: 'human' }]; const analysis = []; let moves = 0;
  const ui = setup({ post: async (url, body) => {
    if (url.endsWith('start_game')) return response(state(0, { players }));
    if (url.endsWith('explain')) { analysis.push(body); return response(explained(body)); }
    moves++; return response(state(1, { players }));
  } });
  ui.click('first-ai'); ui.click('start-button'); await flush();
  assert.equal(ui.el('explain-last').getAttribute('aria-label'), 'Analyze Last AI Move');
  for (const id of ['explain-last', 'analyze-position', 'what-if']) { ui.click(id); await flush(); }
  assert.deepEqual(analysis.map(body => body.mode), ['last_move', 'position', 'what_if']);
  assert.ok(analysis.every(body => body.game_id === 'game-a' && body.revision === 1));
  assert.equal(analysis[2].column, 0);
  assert.equal(moves, 1);
  assert.equal(ui.doc.querySelectorAll('.circle.x').length, 1);
  assert.equal(ui.doc.querySelectorAll('.circle.o').length, 0);
  ui.cleanup();
});

for (const depth of [1, 2, 4, 6, 8]) {
  test(`Negamax UI preset ${depth} stays bounded and sends the selected depth`, async () => {
    let body;
    const ui = setup({ post: async (url, value) => { body = value; return response(state()); } });
    assert.deepEqual([...ui.el('opponent-depth').options].map(option => Number(option.value)), [1, 2, 4, 6, 8]);
    select(ui, 'opponent-depth', String(depth)); ui.click('start-button'); await flush();
    assert.equal(body.player2.depth, depth);
    ui.cleanup();
  });
}
