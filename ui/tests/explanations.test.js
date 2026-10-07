import { test } from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';
import React, { act } from 'react';
import { createRoot } from 'react-dom/client';
import { useConnect4Analysis } from '../src/connect4/useConnect4Analysis.js';
import AnalysisPanel from '../src/connect4/AnalysisPanel.jsx';
import GameBoard from '../src/connect4/GameBoard.jsx';

globalThis.IS_REACT_ACT_ENVIRONMENT = true;

const flush = () => act(async () => { await new Promise(resolve => setImmediate(resolve)); });
const state = (revision = 0, extra = {}) => ({ game_id: 'game-a', revision,
  board: Array.from({ length: 6 }, () => Array(7).fill(' ')),
  currentPlayer: 0, players: [{ type: 'human' }, { type: 'random' }],
  legalMoves: [0, 2, 3], gameOver: false, ...extra });
const response = (request, extra = {}) => ({ data: { ...request, column: request.column ?? null, cached: false, explanation: {
  facts: [{ id: 'position', text: 'Verified facts', classification: 'confirmed_tactical' }],
  summary: { text: 'Concise verified answer.', focus_id: 'quiet', fact_ids: ['position'] },
  key_facts: [{ id: 'position', text: 'Verified facts', classification: 'confirmed_tactical' }],
  relevant_squares: [], additional_context: [],
  strategic_context: [{ connection: 'This connects to the verified position.', title: 'Immediate tactics', text: 'General concept', classification: 'context_only',
    limitations: ['Limited horizon'], source: { references: [
      { chapter: 3, section: '3.4', thesis_pages: [21, 24] }] } }],
  limitations: ['No recorded agent intent'],
}, ...extra } });

function setup(http) {
  const dom = new JSDOM('<div id="root"></div>');
  globalThis.window = dom.window;
  globalThis.document = dom.window.document;
  const doc = dom.window.document;
  const root = createRoot(doc.getElementById('root'));
  let currentGame = null, unavailable = true, analysis;
  function Harness() {
    analysis = useConnect4Analysis(http, currentGame, unavailable);
    return React.createElement(React.Fragment, null,
      React.createElement(GameBoard, { game: currentGame, busy: unavailable, uncertain: false, move() {}, highlights: analysis.highlights }),
      React.createElement(AnalysisPanel, { analysis }));
  }
  const render = () => act(() => root.render(React.createElement(Harness)));
  render();
  const panel = {
    update(game, blocked) { currentGame = game; unavailable = blocked; render(); },
    cleanup() { act(() => root.unmount()); dom.window.close(); },
  };
  const set = (id, value) => act(() => {
    const node = doc.getElementById(id);
    if (id === 'explanation-question') analysis.setQuestion(value);
    else { node.value = value; node.dispatchEvent(new dom.window.Event('change', { bubbles: true })); }
  });
  return { doc, panel, set, analysis: () => analysis, el: id => doc.getElementById(id),
    click: id => act(() => doc.getElementById(id).click()) };
}

test('disabled before start, all modes explicit, legal columns and sources displayed', async () => {
  const calls = [];
  const ui = setup({ post: async (url, body) => { calls.push({ url, body }); return response(body); } });
  assert.equal(ui.el('analyze-position').disabled, true);
  ui.panel.update(state(), false);
  assert.equal(calls.length, 0);
  assert.deepEqual([...ui.el('what-if-column').options].map(o => o.value), ['0', '2', '3']);
  for (const [id, mode] of [['explain-last', 'last_move'], ['analyze-position', 'position'], ['what-if', 'what_if']]) {
    ui.click(id); await flush();
    assert.equal(calls.at(-1).url, '/v1/connect4/explain');
    assert.equal(calls.at(-1).body.mode, mode);
    assert.match(ui.el('explanation-result').textContent, /Verified tactical evidence/);
    assert.match(ui.el('explanation-result').textContent, /not a proven rule application/);
    assert.match(ui.el('explanation-result').textContent, /§3.4, thesis\/PDF pp. 21–24/);
  }
  assert.equal(calls.at(-1).body.column, 0);
  ui.panel.update(state(2), false);
  assert.equal(ui.el('explain-last').getAttribute('aria-label'), 'Analyze Last AI Move');
  ui.panel.cleanup();
});

test('loading prevents duplicate requests and error allows explicit retry', async () => {
  let reject, calls = 0;
  const ui = setup({ post: (url, body) => {
    if (++calls === 1) return new Promise((resolve, rej) => { reject = rej; });
    return Promise.resolve(response(body));
  } });
  ui.panel.update(state(), false);
  ui.click('analyze-position'); ui.click('analyze-position');
  assert.equal(calls, 1);
  assert.equal(ui.el('analyze-position').disabled, true);
  assert.match(ui.el('explanation-status').textContent, /can still play/);
  reject({ response: { data: { error: 'Explanations are unavailable.' } } }); await flush();
  assert.match(ui.el('explanation-status').textContent, /unavailable.*Gameplay remains available/);
  assert.equal(ui.el('analyze-position').disabled, false);
  ui.click('analyze-position'); await flush();
  assert.match(ui.el('explanation-result').textContent, /Verified facts/);
  assert.equal(calls, 2);
  ui.panel.cleanup();
});

for (const change of ['move', 'restart', 'busy', 'navigation']) {
  test(`${change} aborts pending explanation and ignores late success`, async () => {
    let signal, release;
    const ui = setup({ post: (url, body, options) => {
      signal = options.signal;
      return new Promise(resolve => { release = () => resolve(response(body)); });
    } });
    ui.panel.update(state(), false);
    ui.click('analyze-position');
    if (change === 'navigation') ui.panel.cleanup();
    else ui.panel.update(change === 'move' ? state(1) : change === 'restart' ? state(0, { game_id: 'new' }) : state(), change === 'busy');
    assert.equal(signal.aborted, true);
    release(); await flush();
    assert.equal(ui.el('explanation-result')?.textContent || '', '');
    if (change !== 'navigation') ui.panel.cleanup();
  });
}

test('board change invalidates completed explanation, terminal what-if disabled', async () => {
  const ui = setup({ post: async (url, body) => response(body) });
  ui.panel.update(state(), false);
  ui.click('analyze-position'); await flush();
  assert.notEqual(ui.el('explanation-result').textContent, '');
  ui.panel.update(state(7, { gameOver: true, legalMoves: [] }), false);
  assert.equal(ui.el('explanation-result').textContent, '');
  assert.equal(ui.el('what-if').disabled, true);
  assert.equal(ui.el('analyze-position').disabled, false);
  ui.panel.cleanup();
});

test('HTML, malformed and mismatched revision responses are rejected', async () => {
  for (const value of ['<html>gateway</html>', {}, response(state(4, { mode: 'position' })).data]) {
    const ui = setup({ post: async () => ({ data: value }) });
    ui.panel.update(state(), false);
    ui.click('analyze-position'); await flush();
    assert.match(ui.el('explanation-status').textContent, /could not be loaded/);
    assert.equal(ui.el('explanation-result').textContent, '');
    assert.equal(ui.el('analyze-position').disabled, false);
    ui.panel.cleanup();
  }
});

test('question length checked before call and provider text is rendered without HTML', async () => {
  let calls = 0;
  const ui = setup({ post: async (url, body) => {
    calls++;
    const result = response(body);
    result.data.explanation.facts[0].text = '<img src=x onerror=alert(1)>';
    result.data.explanation.key_facts[0].text = result.data.explanation.facts[0].text;
    return result;
  } });
  ui.panel.update(state(), false);
  ui.set('explanation-question', 'x'.repeat(501));
  ui.click('analyze-position'); await flush();
  assert.equal(calls, 0);
  ui.set('explanation-question', 'question');
  ui.click('analyze-position'); await flush();
  assert.equal(calls, 1);
  assert.equal(ui.el('explanation-result').querySelector('img'), null);
  assert.match(ui.el('explanation-result').textContent, /<img/);
  ui.panel.cleanup();
});

test('React unmount safely removes the analysis panel', () => {
  const ui = setup({ post: async () => { throw new Error('unexpected request'); } });
  assert.doesNotThrow(() => ui.panel.cleanup());
  ui.doc.body.replaceChildren();
});

test('a mismatched hypothetical column never renders under the requested column', async () => {
  const ui = setup({ post: async (url, body) => response(body, { column: 3 }) });
  ui.panel.update(state(), false);
  ui.set('what-if-column', '2');
  ui.click('what-if'); await flush();
  assert.equal(ui.el('explanation-result').textContent, '');
  assert.match(ui.el('explanation-status').textContent, /could not be loaded/);
  assert.equal(ui.el('what-if').disabled, false);
  ui.panel.cleanup();
});

test('concise hierarchy, disclosures and verified-square labels clear on update', async () => {
  const ui = setup({ post: async (url, body) => {
    const result = response(body);
    result.data.explanation.relevant_squares = [
      { name: 'b1', column: 1, row_index: 5, row: 1 },
      { name: 'b2', column: 1, row_index: 4, row: 2 },
    ];
    return result;
  } });
  ui.panel.update(state(36, { legalMoves: [1] }), false);
  ui.click('analyze-position'); await flush();
  const root = ui.el('explanation-result');
  assert.match(root.firstChild.textContent, /Analysis.*Concise verified answer/);
  const disclosures = [...root.querySelectorAll('details')];
  assert.deepEqual(disclosures.map(d => d.querySelector('summary').textContent), ['Detailed analysis', 'Methodology and limitations']);
  assert.ok(disclosures.every(d => !d.open));
  disclosures[0].open = true;
  assert.match(disclosures[0].textContent, /Verified facts/);
  disclosures[0].open = false;
  assert.equal(ui.doc.querySelectorAll('.explanation-square').length, 2);
  assert.deepEqual([...ui.doc.querySelectorAll('.square-label')].map(s => s.textContent), ['b2', 'b1']);
  assert.ok([...ui.doc.querySelectorAll('.explanation-square')].every(cell => !cell.disabled));
  ui.panel.update(state(37), false);
  assert.equal(ui.doc.querySelectorAll('.explanation-square, .square-label').length, 0);
  ui.panel.cleanup();
});

test('invalid coordinates and unreferenced summary facts are rejected', async () => {
  for (const patch of [
    { relevant_squares: [{ name: 'b2', column: 1, row_index: 5, row: 1 }] },
    { summary: { text: 'Unsupported answer', focus_id: 'quiet', fact_ids: ['invented'] } },
  ]) {
    const ui = setup({ post: async (url, body) => {
      const result = response(body);
      Object.assign(result.data.explanation, patch);
      return result;
    } });
    ui.panel.update(state(), false);
    ui.click('analyze-position'); await flush();
    assert.equal(ui.el('explanation-result').textContent, '');
    ui.panel.cleanup();
  }
});

test('late older response never replaces an already displayed newer explanation', async () => {
  let release;
  const ui = setup({ post: (url, body) => body.revision === 0
    ? new Promise(resolve => { release = () => resolve(response(body)); })
    : Promise.resolve(response(body)) });
  ui.panel.update(state(0), false);
  ui.click('analyze-position');
  ui.panel.update(state(2), false);
  ui.click('analyze-position'); await flush();
  const content = ui.el('explanation-result').textContent;
  release(); await flush();
  assert.equal(ui.el('explanation-result').textContent, content);
  assert.match(ui.el('explanation-status').textContent, /revision 2/);
  ui.panel.cleanup();
});

for (const [field, value] of [['game_id', 'another-game'], ['revision', 9], ['mode', 'last_move']]) {
  test(`response with wrong ${field} rejects the entire analysis`, async () => {
    const ui = setup({ post: async (url, body) => response(body, { [field]: value }) });
    ui.panel.update(state(), false);
    ui.click('analyze-position'); await flush();
    assert.equal(ui.el('explanation-result').textContent, '');
    assert.equal(ui.doc.querySelectorAll('.explanation-square').length, 0);
    assert.match(ui.el('explanation-status').textContent, /Gameplay remains available/);
    ui.panel.cleanup();
  });
}

for (const reason of ['disabled', 'unavailable', 'provider failure', 'rate limited', 'stale revision', 'aborted']) {
  test(`${reason} permits explicit analysis retry without locking board input`, async () => {
    let calls = 0;
    const ui = setup({ post: async (url, body) => {
      if (++calls === 1) throw { response: { data: { error: `Analysis ${reason}.` } } };
      return response(body);
    } });
    ui.panel.update(state(), false); ui.click('analyze-position'); await flush();
    assert.match(ui.el('explanation-status').textContent, /Gameplay remains available/);
    assert.equal(ui.doc.querySelector('.cell[data-column="3"]').disabled, false);
    ui.click('analyze-position'); await flush();
    assert.equal(calls, 2);
    assert.match(ui.el('explanation-result').textContent, /Concise verified answer/);
    ui.panel.cleanup();
  });
}

for (const change of ['move', 'restart', 'navigation']) {
  test(`${change} clears completed React highlights and results`, async () => {
    const ui = setup({ post: async (url, body) => {
      const result = response(body);
      result.data.explanation.relevant_squares = [{ name: 'd1', column: 3, row: 1, row_index: 5 }];
      return result;
    } });
    ui.panel.update(state(), false); ui.click('analyze-position'); await flush();
    assert.equal(ui.doc.querySelectorAll('.explanation-square').length, 1);
    if (change === 'navigation') ui.panel.cleanup();
    else ui.panel.update(change === 'move' ? state(1) : state(0, { game_id: 'new-game' }), false);
    assert.equal(ui.doc.querySelectorAll('.explanation-square, .square-label').length, 0);
    assert.equal(ui.el('explanation-result')?.textContent || '', '');
    if (change !== 'navigation') ui.panel.cleanup();
  });
}

test('replacement request aborts the old request and ignores its late failure', async () => {
  let reject, signal;
  const ui = setup({ post: (url, body, options) => {
    if (body.mode === 'position') {
      signal = options.signal;
      return new Promise((resolve, fail) => { reject = fail; });
    }
    return Promise.resolve(response(body));
  } });
  ui.panel.update(state(), false); ui.click('analyze-position');
  act(() => { ui.analysis().request('last_move'); }); await flush();
  assert.equal(signal.aborted, true);
  const content = ui.el('explanation-result').textContent;
  reject(new Error('Late provider failure')); await flush();
  assert.equal(ui.el('explanation-result').textContent, content);
  assert.equal(ui.analysis().mode, 'last_move');
  assert.equal(ui.analysis().phase, 'success');
  ui.panel.cleanup();
});

test('question payload, character count and legal column fallback remain React owned', async () => {
  let body;
  const ui = setup({ post: async (url, request) => { body = request; return response(request); } });
  ui.panel.update(state(), false);
  ui.set('explanation-question', 'Is this safe?');
  ui.set('what-if-column', '3'); ui.click('what-if'); await flush();
  assert.deepEqual(body, { game_id: 'game-a', revision: 0, mode: 'what_if', column: 3, question: 'Is this safe?' });
  assert.equal(ui.el('question-count').textContent, '13/500');
  assert.equal(ui.el('explanation-question').maxLength, 500);
  ui.panel.update(state(2, { legalMoves: [2] }), false);
  assert.equal(ui.el('what-if-column').value, '2');
  ui.click('what-if'); await flush();
  assert.equal(body.column, 2);
  ui.panel.update(state(4, { legalMoves: [] }), false);
  assert.equal(ui.el('what-if').disabled, true);
  act(() => { ui.analysis().request('what_if'); }); await flush();
  assert.equal(body.revision, 2);
  ui.panel.cleanup();
});

test('reference-only material, preconditions and a forged source URL stay bounded', async () => {
  const ui = setup({ post: async (url, body) => {
    const result = response(body);
    result.data.explanation.additional_context = [{ ...result.data.explanation.strategic_context[0],
      classification: 'reference_only', title: 'Before rule', preconditions: ['Requires formal coverage.'],
      source: { url: 'https://untrusted.example', references: [{ chapter: 5, section: '5.6', thesis_pages: [33, 34] }] } }];
    return result;
  } });
  ui.panel.update(state(), false); ui.click('analyze-position'); await flush();
  const links = [...ui.el('explanation-result').querySelectorAll('a')];
  assert.ok(links.every(link => link.href === 'https://tromp.github.io/c4/connect4_thesis.pdf' &&
    link.target === '_blank' && link.rel === 'noopener noreferrer'));
  const details = ui.el('explanation-result').querySelectorAll('details');
  assert.match(details[0].textContent, /Reference only.*Before rule/);
  assert.match(details[1].textContent, /Reference precondition: Requires formal coverage/);
  assert.match(details[1].textContent, /private reasoning, and search traces are not exposed/);
  ui.panel.cleanup();
});

test('a failed replacement clears completed highlights immediately and preserves the snapshot', async () => {
  let calls = 0, reject;
  const game = state();
  const original = structuredClone(game);
  const ui = setup({ post: (url, body) => {
    if (++calls === 2) return new Promise((resolve, fail) => { reject = fail; });
    const result = response(body);
    result.data.explanation.relevant_squares = [{ name: 'd1', column: 3, row: 1, row_index: 5 }];
    return Promise.resolve(result);
  } });
  ui.panel.update(game, false); ui.click('analyze-position'); await flush();
  assert.equal(ui.doc.querySelectorAll('.explanation-square').length, 1);
  ui.click('explain-last');
  assert.equal(ui.doc.querySelectorAll('.explanation-square, .square-label').length, 0);
  assert.equal(ui.el('explanation-result').textContent, '');
  reject(new Error('Provider unavailable')); await flush();
  assert.equal(ui.doc.querySelectorAll('.explanation-square').length, 0);
  assert.equal(ui.doc.querySelector('.cell[data-column="3"]').disabled, false);
  assert.deepEqual(game, original);
  ui.panel.cleanup();
});
