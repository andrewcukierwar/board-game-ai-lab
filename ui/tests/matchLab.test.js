import { test } from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';
import React, { act } from 'react';
import { createRoot } from 'react-dom/client';
import { MemoryRouter } from 'react-router-dom';
import MatchLabPage from '../src/pages/MatchLab.jsx';
import { DRAW, WIN, matchFixture } from '../e2e/fixtures/match.js';

globalThis.IS_REACT_ACT_ENVIRONMENT = true;
const flush = () => act(async () => { await new Promise(resolve => setImmediate(resolve)); });
const fail = (status, code) => Object.assign(new Error(code), { response: { status, data: { code, error: code } } });
function server(sequence = DRAW) {
  const s = { players: null, sequence, plies: [], calls: [], reads: 0, starts: 0 };
  s.fixture = () => matchFixture(s.players, s.plies, `match-${s.starts}`);
  s.post = async (url, body) => {
    s.calls.push({ url, body });
    if (url.endsWith('start_game')) { s.starts++; s.players = [body.player1, body.player2]; s.plies = []; }
    else { assert.equal(body.revision, s.plies.length); s.plies.push(body.column ?? s.sequence[s.plies.length]); }
    return { data: s.fixture().state };
  };
  s.get = async () => { s.reads++; return { data: s.fixture() }; };
  return s;
}
function setup(http) {
  const dom = new JSDOM('<div id="root"></div>');
  globalThis.window = dom.window; globalThis.document = dom.window.document;
  const doc = dom.window.document, root = createRoot(doc.getElementById('root'));
  act(() => root.render(React.createElement(MemoryRouter, { future: { v7_startTransition: true, v7_relativeSplatPath: true } }, React.createElement(MatchLabPage, { http }))));
  const el = id => doc.getElementById(id);
  return { doc, el, click: id => act(() => el(id).click()),
    choose: (index, type) => act(() => doc.querySelector(`input[name="player-${index + 1}"][value="${type}"]`).click()),
    set: (id, value) => act(() => { el(id).value = value; el(id).dispatchEvent(new dom.window.Event('change', { bubbles: true })); }),
    column: c => act(() => doc.querySelector(`.cell[data-column="${c}"]`).click()),
    cleanup: () => { act(() => root.unmount()); dom.window.close(); } };
}
const tick = async (t, ms = 850) => { await act(async () => { t.mock.timers.tick(ms); }); await flush(); };
const revision = ui => ui.doc.querySelector('.board-revision').textContent;
async function start(ui, types = ['random', 'random']) {
  types.forEach((type, i) => ui.choose(i, type)); ui.click('match-start'); await flush();
}
for (const types of [['human', 'human'], ['human', 'random'], ['random', 'human'], ['negamax', 'mcts'], ['negamax', 'negamax']]) {
  test(`independent selectors start paused at zero: ${types}`, async () => {
    const s = server(), ui = setup(s); await start(ui, types);
    assert.deepEqual(s.players.map(p => p.type), types); assert.equal(s.calls.length, 1); assert.equal(revision(ui), 'Move 0');
    assert.equal(ui.el('match-autoplay').getAttribute('aria-pressed'), 'false'); assert.equal(ui.doc.querySelectorAll('.circle.x').length, 0);
    assert.equal(ui.el('match-next').disabled, types[0] === 'human');
    if (types.every(t => t === 'human')) assert.equal(ui.el('match-autoplay').disabled, true);
    ui.cleanup();
  });
}
test('opposite budgets independent; restart resets replay, timers and applies replacement id', async t => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const s = server(), ui = setup(s); ui.choose(0, 'negamax'); ui.choose(1, 'negamax');
  ui.set('player-1-depth', '1'); ui.set('player-2-depth', '8'); ui.click('match-start'); await flush();
  assert.deepEqual(s.players, [{ type: 'negamax', depth: 1 }, { type: 'negamax', depth: 8 }]);
  ui.click('match-next'); await flush(); ui.click('match-previous');
  assert.match(ui.el('match-view-mode').textContent, /REVIEWING MOVE 0 OF 1/);
  ui.choose(0, 'mcts'); ui.set('player-1-simulations', '400'); ui.choose(1, 'mcts'); ui.set('player-2-simulations', '800');
  ui.click('match-start'); await flush();
  assert.deepEqual(s.calls.at(-1).body, { player1: { type: 'mcts', simulation_limit: 400 }, player2: { type: 'mcts', simulation_limit: 800 }, replace_game_id: 'match-1' });
  assert.equal(revision(ui), 'Move 0'); assert.equal(ui.doc.querySelectorAll('.move-list li').length, 1);
  ui.click('match-autoplay'); ui.click('match-start'); await flush(); await tick(t, 5000);
  assert.equal(s.plies.length, 0); assert.equal(ui.el('match-view-mode').textContent, 'LIVE'); ui.cleanup();
});
test('manual step is one locked ply even with two synchronous clicks', async () => {
  const s = server(), original = s.post; let release, requests = 0;
  s.post = (url, body) => url.endsWith('make_move') ? (requests++, new Promise(resolve => { release = async () => resolve(await original(url, body)); })) : original(url, body);
  const ui = setup(s); await start(ui);
  act(() => { ui.el('match-next').click(); ui.el('match-next').click(); }); assert.equal(requests, 1);
  assert.equal(ui.el('match-next').disabled, true); assert.equal(ui.el('match-start').disabled, true);
  assert.equal(ui.doc.querySelectorAll('.cell:enabled').length, 0);
  release(); await flush(); assert.equal(s.plies.length, 1);
  assert.deepEqual(s.calls[1].body, { game_id: 'match-1', revision: 0 }); assert.equal(revision(ui), 'Move 1'); ui.cleanup();
});
test('human click is one ply; paused match does not automatically follow with AI', async () => {
  const s = server(), ui = setup(s); await start(ui, ['human', 'random']); assert.equal(ui.el('match-next').disabled, true);
  act(() => { const cell = ui.doc.querySelector('.cell[data-column="3"]'); cell.click(); cell.click(); }); await flush();
  assert.equal(s.plies.length, 1); assert.equal(s.calls.length, 2); assert.equal(s.calls[1].body.column, 3);
  assert.equal(ui.doc.querySelectorAll('.cell:enabled').length, 0); assert.equal(ui.el('match-next').disabled, false); ui.cleanup();
});
test('autoplay has intentional scheduling boundaries, pauses and stops terminally', async t => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const s = server(WIN), ui = setup(s); await start(ui); ui.click('match-autoplay');
  await tick(t, 849); assert.equal(s.plies.length, 0); await tick(t, 1); assert.equal(s.plies.length, 1);
  ui.click('match-autoplay'); await tick(t, 5000); assert.equal(s.plies.length, 1); ui.click('match-autoplay');
  for (let i = 0; i < 6; i++) await tick(t);
  assert.equal(s.plies.length, 7); assert.match(ui.doc.querySelector('[role="status"]').textContent, /Random wins as Red in 7 moves/);
  assert.equal(ui.el('match-autoplay').getAttribute('aria-pressed'), 'false'); await tick(t, 5000); assert.equal(s.plies.length, 7); ui.cleanup();
});
test('pause during POST permits safe completion and never overlaps or cancels mutation', async t => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const s = server(), original = s.post; let release, signal, inFlight = 0, maximum = 0;
  s.post = (url, body, options) => {
    if (url.endsWith('start_game')) return original(url, body);
    maximum = Math.max(maximum, ++inFlight); signal = options.signal;
    return new Promise(resolve => { release = async () => { const result = await original(url, body); inFlight--; resolve(result); }; });
  };
  const ui = setup(s); await start(ui); ui.click('match-autoplay'); await tick(t); await tick(t, 10000); assert.equal(maximum, 1);
  ui.click('match-autoplay'); assert.equal(signal.aborted, false); release(); await flush(); await tick(t, 10000);
  assert.equal(s.plies.length, 1); assert.equal(revision(ui), 'Move 1'); assert.equal(maximum, 1); ui.cleanup();
});
for (const types of [['human', 'random'], ['random', 'human']]) test(`autoplay waits for humans and resumes: ${types}`, async t => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const s = server(), ui = setup(s); await start(ui, types); ui.click('match-autoplay');
  if (types[0] !== 'human') await tick(t);
  const before = s.plies.length; await tick(t, 5000); assert.equal(s.plies.length, before);
  assert.match(ui.doc.querySelector('[role="status"]').textContent, /waiting for Human/);
  ui.column(6); await flush(); assert.equal(s.plies.length, before + 1); await tick(t); assert.equal(s.plies.length, before + 2);
  assert.equal(ui.el('match-autoplay').getAttribute('aria-pressed'), 'true'); ui.cleanup();
});
for (const committed of [false, true]) test(`lost AI response (${committed ? 'committed' : 'agent_busy'}) requires explicit continuation`, async t => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const s = server(), original = s.post; let attempts = 0;
  s.post = async (url, body) => {
    if (url.endsWith('start_game')) return original(url, body);
    if (++attempts === 1) { if (committed) await original(url, body); throw committed ? new Error('lost') : fail(503, 'agent_busy'); }
    return original(url, body);
  };
  const ui = setup(s); await start(ui); ui.click('match-autoplay'); await tick(t);
  assert.equal(attempts, 1); assert.equal(s.reads, 1); assert.equal(revision(ui), `Move ${Number(committed)}`);
  assert.equal(ui.el('match-autoplay').getAttribute('aria-pressed'), 'false'); await tick(t, 5000); assert.equal(attempts, 1);
  assert.equal(ui.el('match-next').disabled, true); ui.click('match-refresh'); await flush(); assert.equal(attempts, 1);
  ui.click('match-next'); await flush(); assert.equal(attempts, 2); assert.equal(s.calls.at(-1).body.revision, Number(committed)); ui.cleanup();
});
test('failed GET stays locked until explicit refresh rebuilds missing history', async () => {
  const s = server(), original = s.post; let reads = 0;
  s.get = async () => { if (++reads === 1) throw new Error('offline'); return { data: s.fixture() }; };
  s.post = async (url, body) => { const result = await original(url, body); if (url.endsWith('make_move')) throw new Error('lost'); return result; };
  const ui = setup(s); await start(ui); ui.click('match-next'); await flush(); assert.equal(ui.el('match-next').disabled, true);
  ui.click('match-refresh'); await flush(); assert.equal(s.plies.length, 1); assert.equal(ui.doc.querySelectorAll('.move-list li').length, 2);
  assert.equal(ui.el('match-next').disabled, false); ui.cleanup();
});
test('history mismatch cannot unlock or replay an accepted POST', async () => {
  const s = server(); s.get = async () => { const data = s.fixture(); data.moves = []; return { data }; };
  const ui = setup(s); await start(ui); ui.click('match-next'); await flush();
  assert.equal(revision(ui), 'Move 1'); assert.equal(ui.el('match-next').disabled, true); assert.equal(s.plies.length, 1);
  assert.equal(ui.el('match-refresh').disabled, false); ui.cleanup();
});
test('expired match clears state and offers a new paused start', async () => {
  const s = server(), original = s.post;
  s.post = (url, body) => url.endsWith('start_game') ? original(url, body) : Promise.reject(fail(404, 'session_not_found'));
  s.get = async () => { throw fail(404, 'session_not_found'); };
  const ui = setup(s); await start(ui); ui.click('match-next'); await flush();
  assert.match(ui.el('match-message').textContent, /expired/); assert.equal(ui.el('match-start').textContent, 'Start match');
  assert.equal(ui.doc.querySelectorAll('.cell').length, 0); ui.click('match-start'); await flush(); assert.equal(revision(ui), 'Move 0'); ui.cleanup();
});
test('rewind changes display only, pauses autoplay, locks moves and returns live', async t => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const s = server(), ui = setup(s); await start(ui); ui.click('match-autoplay'); await tick(t); await tick(t);
  const calls = s.calls.length, reads = s.reads;
  ui.click('match-previous'); assert.equal(revision(ui), 'Move 1'); assert.equal(ui.el('match-autoplay').getAttribute('aria-pressed'), 'false');
  assert.equal(ui.el('match-next').disabled, true); assert.equal(ui.doc.querySelectorAll('.cell:enabled').length, 0);
  ui.click('match-previous'); assert.equal(revision(ui), 'Move 0'); ui.click('match-forward'); assert.equal(revision(ui), 'Move 1'); ui.column(4);
  await tick(t, 10000); assert.equal(s.calls.length, calls); assert.equal(s.reads, reads);
  ui.click('match-live'); assert.equal(revision(ui), 'Move 2'); assert.equal(ui.el('match-next').disabled, false); assert.equal(s.plies.length, 2); ui.cleanup();
});
test('42-move draw and full terminal replay cause no extra network requests', async t => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const s = server(), ui = setup(s); await start(ui); ui.set('match-speed', 'fast'); ui.click('match-autoplay');
  for (let i = 0; i < 42; i++) await tick(t, 300);
  assert.equal(s.plies.length, 42); assert.equal(s.reads, 42); assert.equal(s.calls.length, 43); assert.equal(ui.doc.querySelectorAll('.move-list li').length, 43);
  assert.match(ui.doc.querySelector('[role="status"]').textContent, /Draw after 42/);
  for (let i = 0; i < 42; i++) ui.click('match-previous');
  assert.equal(revision(ui), 'Move 0'); assert.equal(ui.doc.querySelectorAll('.circle.x, .circle.o').length, 0);
  for (let i = 0; i < 42; i++) ui.click('match-forward');
  assert.equal(revision(ui), 'Move 42'); assert.equal(s.reads, 42); assert.equal(s.calls.length, 43); ui.cleanup();
});
for (const stage of ['start', 'move', 'history', 'timer']) test(`navigation cleans up ${stage} and ignores late work`, async t => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const s = server(), original = s.post; let release, signal, pending = 0;
  const gate = options => { signal = options.signal; pending++; return new Promise(resolve => { release = resolve; }); };
  s.post = (url, body, options) => stage === 'start' || stage === 'move' && url.endsWith('make_move') ? gate(options) : original(url, body);
  if (stage === 'history') s.get = (url, options) => gate(options);
  const ui = setup(s);
  if (stage === 'start') { ui.click('match-start'); await flush(); }
  else { await start(ui); ui.click('match-autoplay'); if (stage !== 'timer') await tick(t); }
  const calls = s.calls.length; ui.cleanup(); if (signal) assert.equal(signal.aborted, true);
  if (release) release({ data: stage === 'history' ? s.fixture() : matchFixture().state });
  await tick(t, 10000); assert.equal(s.calls.length, calls); assert.equal(ui.doc.querySelectorAll('.cell').length, 0); assert.equal(pending, stage === 'timer' ? 0 : 1);
});
for (const kind of ['html', 'nonzero', 'players']) test(`invalid ${kind} start never opens an AI match`, async () => {
  const s = server(), original = s.post;
  s.post = async (url, body) => {
    const result = await original(url, body);
    if (kind === 'html') result.data = '<html>Waking up</html>';
    if (kind === 'nonzero') result.data.revision = 1;
    if (kind === 'players') result.data.players = [{ type: 'human' }, { type: 'human' }];
    return result;
  };
  const ui = setup(s); await start(ui); assert.equal(s.calls.length, 1); assert.equal(ui.doc.querySelectorAll('.cell').length, 0);
  assert.equal(ui.el('match-start').disabled, false); ui.cleanup();
});
test('lost human response after commit synchronizes without inventing an AI followup', async () => {
  const s = server(), original = s.post;
  s.post = async (url, body) => { const response = await original(url, body); if (url.endsWith('make_move')) throw new Error('lost human response'); return response; };
  const ui = setup(s); await start(ui, ['human', 'random']); ui.column(4); await flush();
  assert.equal(s.plies.length, 1); assert.equal(revision(ui), 'Move 1'); assert.equal(s.calls.length, 2);
  assert.equal(ui.el('match-next').disabled, true); ui.click('match-refresh'); await flush(); assert.equal(s.calls.length, 2); ui.cleanup();
});
test('rewind during in-flight move keeps viewing the old position after safe acceptance', async t => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const s = server(), original = s.post; let release;
  const ui = setup(s); await start(ui); ui.click('match-next'); await flush();
  s.post = (url, body) => new Promise(resolve => { release = async () => resolve(await original(url, body)); });
  ui.click('match-autoplay'); await tick(t); ui.click('match-previous');
  release(); await flush(); await tick(t, 5000);
  assert.equal(revision(ui), 'Move 0'); assert.equal(s.plies.length, 2);
  assert.match(ui.el('match-view-mode').textContent, /REVIEWING MOVE 0 OF 2/); ui.click('match-live'); assert.equal(revision(ui), 'Move 2'); ui.cleanup();
});
