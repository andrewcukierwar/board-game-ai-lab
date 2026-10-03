import { test } from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';
import { mountExplanations } from '../legacy/explanations.js';

const flush = () => new Promise(resolve => setImmediate(resolve));
const state = (revision = 0, extra = {}) => ({ game_id: 'game-a', revision,
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
  const doc = new JSDOM(`<section id="explanation-panel">
    <button id="explain-last"></button><button id="analyze-position"></button><button id="what-if"></button>
    <select id="what-if-column"></select><textarea id="explanation-question"></textarea>
    <p id="explanation-status"></p><div id="explanation-result"></div></section>`).window.document;
  const panel = mountExplanations({ document: doc, http });
  return { doc, panel, el: id => doc.getElementById(id), click: id => doc.getElementById(id).click() };
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
    assert.match(ui.el('explanation-result').textContent, /Key tactical evidence/);
    assert.match(ui.el('explanation-result').textContent, /not a proven rule application/);
    assert.match(ui.el('explanation-result').textContent, /§3.4, thesis\/PDF pp. 21–24/);
  }
  assert.equal(calls.at(-1).body.column, 0);
  ui.panel.update(state(2), false);
  assert.equal(ui.el('explain-last').textContent, 'Analyze Last AI Move');
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
    assert.equal(ui.el('explanation-result').textContent, '');
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
  ui.el('explanation-question').value = 'x'.repeat(501);
  ui.click('analyze-position'); await flush();
  assert.equal(calls, 0);
  ui.el('explanation-question').value = 'question';
  ui.click('analyze-position'); await flush();
  assert.equal(calls, 1);
  assert.equal(ui.el('explanation-result').querySelector('img'), null);
  assert.match(ui.el('explanation-result').textContent, /<img/);
  ui.panel.cleanup();
});

test('React cleanup remains safe after the panel DOM has been removed', () => {
  const ui = setup({ post: async () => { throw new Error('unexpected request'); } });
  ui.doc.body.replaceChildren();
  assert.doesNotThrow(() => ui.panel.cleanup());
});

test('a mismatched hypothetical column never renders under the requested column', async () => {
  const ui = setup({ post: async (url, body) => response(body, { column: 3 }) });
  ui.panel.update(state(), false);
  ui.el('what-if-column').value = '2';
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
  const board = ui.doc.createElement('div');
  board.id = 'game-board';
  board.innerHTML = '<button class="cell" data-square="b1"></button><button class="cell" data-square="b2"></button>';
  ui.doc.body.appendChild(board);
  ui.panel.update(state(36), false);
  ui.click('analyze-position'); await flush();
  const root = ui.el('explanation-result');
  assert.match(root.firstChild.textContent, /Primary explanation.*Concise verified answer/);
  const disclosures = [...root.querySelectorAll('details')];
  assert.deepEqual(disclosures.map(d => d.querySelector('summary').textContent), ['Detailed analysis', 'Methodology and limitations']);
  assert.ok(disclosures.every(d => !d.open));
  disclosures[0].open = true;
  assert.match(disclosures[0].textContent, /Verified facts/);
  disclosures[0].open = false;
  assert.equal(ui.doc.querySelectorAll('.explanation-square').length, 2);
  assert.deepEqual([...ui.doc.querySelectorAll('.square-label')].map(s => s.textContent), ['b1', 'b2']);
  assert.ok([...board.children].every(cell => !cell.disabled));
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
