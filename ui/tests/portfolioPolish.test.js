import { test } from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';
import React, { act } from 'react';
import { createRoot } from 'react-dom/client';
import { MemoryRouter } from 'react-router-dom';
import Home from '../src/pages/Home.jsx';
import TournamentLabPage from '../src/pages/TournamentLab.jsx';
import SeasonLabPage from '../src/pages/SeasonLab.jsx';
import AgentSelector from '../src/connect4/AgentSelector.jsx';
import CompetitorSelector from '../src/connect4/CompetitorSelector.jsx';
import { entrantPresets, recommendedField } from '../src/connect4/competitionFields.js';
import { AI_AGENT_ORDER, aiAgentTypes, playerTypes } from '../src/connect4/researchCompetitor.js';
import { competitorLabel, NEGAMAX_DEPTHS, MCTS_SIMULATIONS, TACTICAL_MCTS_SIMULATIONS } from '../src/connect4/competitorConfig.js';
import * as tournament from '../src/tournament/model.js';
import * as season from '../src/season/model.js';
import { STORAGE_KEY as TOURNAMENT_KEY } from '../src/tournament/storage.js';
import { STORAGE_KEY as SEASON_KEY } from '../src/season/storage.js';
import { status, provenance } from './fixtures/alphazero.js';

globalThis.IS_REACT_ACT_ENVIRONMENT = true;
const flush = () => act(async () => { for (let i = 0; i < 4; i++) await new Promise(resolve => setImmediate(resolve)); });
const VICTOR = 'victor_research', AZ = 'alphazero';
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const sizes = sizesList => sizesList.flatMap(size => [[false, false], [true, false], [false, true], [true, true]].map(([alphaZero, victor]) => ({ size, alphaZero, victor })));

// Discovery is cached per HTTP client, so every page gets a fresh client.
function client(discovery = 'available') {
  let resolve;
  const pending = new Promise(r => { resolve = r; });
  const http = {
    get: url => {
      assert.equal(url, '/v1/connect4/agents/alphazero');
      if (discovery === 'deferred') return pending;
      if (discovery === 'available') return Promise.resolve({ data: status });
      return Promise.reject(Object.assign(new Error('absent'), { response: { status: 404 } }));
    },
    post: async () => { throw new Error('No game may start in setup tests.'); },
  };
  return { http, resolve: () => resolve({ data: status }) };
}
function memoryStorage(entries = {}) {
  const values = new Map(Object.entries(entries));
  return { getItem: key => values.get(key) ?? null, setItem: (key, value) => values.set(key, value), values };
}
function render(element, url = 'http://localhost/') {
  const dom = new JSDOM('<div id="root"></div>', { url });
  globalThis.window = dom.window; globalThis.document = dom.window.document;
  const doc = dom.window.document, root = createRoot(doc.getElementById('root'));
  act(() => root.render(React.createElement(MemoryRouter, { future: { v7_startTransition: true, v7_relativeSplatPath: true } }, element)));
  return { doc, el: id => doc.getElementById(id),
    options: id => [...doc.getElementById(id).options].map(o => ({ text: o.textContent, disabled: o.disabled, selected: o.selected })),
    selected: id => doc.getElementById(id).selectedOptions[0].textContent,
    set: (id, value) => act(() => { const node = doc.getElementById(id); node.value = String(value); node.dispatchEvent(new dom.window.Event('change', { bubbles: true })); }),
    setLabel: (id, label) => act(() => { const node = doc.getElementById(id); node.value = [...node.options].find(o => o.textContent === label).value; node.dispatchEvent(new dom.window.Event('change', { bubbles: true })); }),
    click: node => act(() => node.click()),
    cleanup: () => { act(() => root.unmount()); dom.window.close(); } };
}
const fieldLabels = (ui, prefix, size) => Array.from({ length: size }, (_, i) => ui.selected(`${prefix}-${i + 1}`));

test('canonical agent order puts AlphaZero before Victor and leaves Human placement to each surface', () => {
  assert.deepEqual(AI_AGENT_ORDER, ['random', 'negamax', 'mcts', 'mcts_tactical', AZ, VICTOR]);
  assert.deepEqual(aiAgentTypes(true, true), AI_AGENT_ORDER);
  assert.deepEqual(aiAgentTypes(false, false), ['random', 'negamax', 'mcts', 'mcts_tactical']);
  assert.deepEqual(aiAgentTypes(true, false), ['random', 'negamax', 'mcts', 'mcts_tactical', VICTOR]);
  assert.deepEqual(aiAgentTypes(false, true), ['random', 'negamax', 'mcts', 'mcts_tactical', AZ]);
  assert.deepEqual(playerTypes(true, true), ['human', ...AI_AGENT_ORDER]);
});

test('A: Play lists AlphaZero before Victor, and availability/flags control both', () => {
  for (const [researchEnabled, available] of [[true, true], [false, true], [true, false], [false, false]]) {
    const ui = render(React.createElement(AgentSelector, { selection: { type: 'negamax', depth: 2, simulations: 2000, first: 'human' }, select: () => {}, busy: false, game: null, researchEnabled, alphaZero: { available } }));
    assert.deepEqual([...ui.doc.querySelectorAll('input[name="opponent"]')].map(i => i.value), aiAgentTypes(researchEnabled, available));
    ui.cleanup();
  }
});

test('B: Match Lab lists Human first, then the canonical AI order with AlphaZero before Victor', () => {
  for (const [researchEnabled, available] of [[true, true], [false, true], [true, false], [false, false]]) {
    const ui = render(React.createElement(CompetitorSelector, { index: 0, config: { type: 'human' }, select: () => {}, busy: false, researchEnabled, alphaZero: { available } }));
    assert.deepEqual([...ui.doc.querySelectorAll('input[name="player-1"]')].map(i => i.value), ['human', ...aiAgentTypes(researchEnabled, available)]);
    ui.cleanup();
  }
});

test('C/D/E: entrant menus follow the canonical order; Human is the last Tournament option and absent from Season', () => {
  const tournamentMenu = entrantPresets({ victor: true, alphaZero: true, human: true }), seasonMenu = entrantPresets({ victor: true, alphaZero: true });
  const types = menu => [...new Set(menu.map(c => c.type))];
  assert.deepEqual(types(tournamentMenu), [...AI_AGENT_ORDER, 'human']);
  assert.deepEqual(types(seasonMenu), AI_AGENT_ORDER);
  assert.deepEqual(tournamentMenu.at(-1), { type: 'human' });
  assert.deepEqual(tournamentMenu.filter(c => c.type === AZ), [{ type: AZ, tactical_guard: false }, { type: AZ, tactical_guard: true }]);
  assert.equal(entrantPresets({}).some(c => [AZ, VICTOR, 'human'].includes(c.type)), false);
  assert.deepEqual(types(entrantPresets({ victor: true })), ['random', 'negamax', 'mcts', 'mcts_tactical', VICTOR]);
  assert.deepEqual(types(entrantPresets({ alphaZero: true })), ['random', 'negamax', 'mcts', 'mcts_tactical', AZ]);
  // Menus offer only current presets, never the historical 100/400/800 Classic MCTS defaults.
  assert.deepEqual(seasonMenu.filter(c => c.type === 'mcts').map(c => c.simulations), MCTS_SIMULATIONS.filter(n => n <= 5000));
  assert.deepEqual(seasonMenu.filter(c => c.type === 'mcts_tactical').map(c => c.simulations), TACTICAL_MCTS_SIMULATIONS.filter(n => n <= 2000));
  assert.deepEqual(seasonMenu.filter(c => c.type === 'negamax').map(c => c.depth), NEGAMAX_DEPTHS);
});

test('I: eight-entrant default field covers every family when both optional agents are available', () => {
  assert.deepEqual(recommendedField(8, { alphaZero: true, victor: true }), [
    { type: 'random' }, { type: 'negamax', depth: 2 }, { type: 'negamax', depth: 6 }, { type: 'mcts', simulations: 500 },
    { type: 'mcts_tactical', simulations: 400 }, { type: AZ, tactical_guard: false }, { type: AZ, tactical_guard: true }, { type: VICTOR }]);
});

test('J: sixteen-entrant default field is all distinct configurations when both optional agents are available', () => {
  const field = recommendedField(16, { alphaZero: true, victor: true });
  assert.equal(new Set(field.map(c => JSON.stringify(c))).size, 16);
  for (const type of AI_AGENT_ORDER) assert.ok(field.some(c => c.type === type), type);
  assert.equal(field.filter(c => c.type === AZ).length, 2);
  assert.equal(field.filter(c => c.type === VICTOR).length, 1);
});

test('K: without optional agents the default field substitutes valid ordinary configurations', () => {
  assert.deepEqual(recommendedField(8), [
    { type: 'random' }, { type: 'negamax', depth: 2 }, { type: 'negamax', depth: 4 }, { type: 'negamax', depth: 6 },
    { type: 'mcts', simulations: 500 }, { type: 'mcts', simulations: 1000 }, { type: 'mcts_tactical', simulations: 400 }, { type: 'mcts_tactical', simulations: 800 }]);
  for (const size of tournament.SIZES) {
    const field = recommendedField(size);
    assert.equal(field.length, size);
    assert.equal(field.some(c => [AZ, VICTOR, 'human'].includes(c.type)), false);
    assert.ok(field.some(c => c.type === 'mcts_tactical'));
  }
});

test('I/J/K: every Tournament default is valid, selectable, Human-free, ordered and avoids the slowest settings', () => {
  for (const { size, alphaZero, victor } of sizes(tournament.SIZES)) {
    const field = recommendedField(size, { alphaZero, victor }), menu = entrantPresets({ alphaZero, victor, human: true });
    assert.equal(field.length, size);
    for (const config of field) assert.ok(menu.some(p => same(p, config)), `${JSON.stringify(config)} is not a menu preset`);
    assert.equal(field.some(c => c.type === 'human'), false);
    assert.equal(field.some(c => c.type === AZ), alphaZero);
    assert.equal(field.filter(c => c.type === VICTOR).length, Number(victor));
    assert.equal(field.some(c => c.depth >= 10 || c.simulations > 2000), false);
    const ranks = field.map(c => AI_AGENT_ORDER.indexOf(c.type));
    assert.deepEqual(ranks, [...ranks].sort((a, b) => a - b));
    // Large brackets repeat configurations evenly rather than piling onto one.
    const counts = [...field.reduce((m, c) => m.set(JSON.stringify(c), (m.get(JSON.stringify(c)) ?? 0) + 1), new Map()).values()];
    assert.ok(Math.max(...counts) - Math.min(...counts) <= Math.ceil(size / 9));
    assert.ok(tournament.createTournament(field, 1234, 'defaults', victor, alphaZero ? provenance : undefined));
  }
});

test('N/O/P: Season defaults at sizes 4–12 are AI-only, valid, distinct and use current budgets', () => {
  for (const { size, alphaZero, victor } of sizes(season.FIELD_SIZES)) {
    const field = recommendedField(size, { alphaZero, victor }), menu = entrantPresets({ alphaZero, victor });
    assert.equal(field.length, size);
    assert.equal(new Set(field.map(c => JSON.stringify(c))).size, size);
    for (const config of field) assert.ok(menu.some(p => same(p, config)));
    assert.equal(field.some(c => c.type === 'human'), false);
    assert.equal(field.some(c => c.type === 'mcts' && [100, 400, 800].includes(c.simulations)), false);
    assert.equal(field.some(c => c.depth >= 10 || c.simulations > 2000), false);
    for (const type of ['negamax', 'mcts', 'mcts_tactical']) assert.ok(field.some(c => c.type === type), `${size}: ${type}`);
    if (alphaZero) assert.ok(field.some(c => c.type === AZ));
    if (size >= 6) assert.ok(field.some(c => c.type === 'random'));
    if (size >= 6) assert.equal(field.some(c => c.type === VICTOR), victor);
    assert.ok(season.createSeason(field, 1234, 2, 'defaults', victor, alphaZero ? provenance : undefined));
  }
  assert.deepEqual(recommendedField(4, { alphaZero: true, victor: true }).map(competitorLabel),
    ['Negamax · depth 2', 'Classic MCTS · 500 simulations', 'Tactical MCTS · 400 max simulations', 'AlphaZero (Experimental) · safeguard Off']);
  assert.deepEqual(recommendedField(8, { alphaZero: true, victor: true }), recommendedField(8, { alphaZero: true, victor: true }));
});

test('E/F/G: Tournament page puts Human last in every slot and keeps the one-Human rule', async () => {
  const { http } = client('available'), ui = render(React.createElement(TournamentLabPage, { http, storage: memoryStorage() }), 'http://localhost/connect4/tournament');
  await flush();
  for (let i = 1; i <= 8; i++) {
    const options = ui.options(`entrant-${i}`).filter(o => o.text !== '');
    assert.equal(options.at(-1).text, 'You · Human');
    const labels = options.map(o => o.text);
    assert.ok(labels.indexOf('AlphaZero (Experimental) · safeguard Off') < labels.indexOf('AlphaZero (Experimental) · safeguard On'));
    assert.ok(labels.indexOf('AlphaZero (Experimental) · safeguard On') < labels.indexOf('You · Human'));
    assert.ok(labels.indexOf('Tactical MCTS · 2000 max simulations') < labels.indexOf('AlphaZero (Experimental) · safeguard Off'));
  }
  // G: the available backend's AlphaZero joins the untouched default field.
  assert.deepEqual(fieldLabels(ui, 'entrant', 8).filter(l => l.startsWith('AlphaZero')), ['AlphaZero (Experimental) · safeguard Off', 'AlphaZero (Experimental) · safeguard On']);
  ui.setLabel('entrant-3', 'You · Human');
  assert.equal(ui.selected('entrant-3'), 'You · Human');
  for (const i of [1, 2, 4, 8]) assert.equal(ui.options(`entrant-${i}`).at(-1).disabled, true);
  assert.equal(ui.options('entrant-3').at(-1).disabled, false);
  ui.cleanup();
});

test('G/K: unsupported AlphaZero backend leaves AlphaZero out of menus and the default field', async () => {
  const { http } = client('absent'), ui = render(React.createElement(TournamentLabPage, { http, storage: memoryStorage() }), 'http://localhost/connect4/tournament');
  await flush();
  assert.equal(ui.options('entrant-1').some(o => o.text.startsWith('AlphaZero')), false);
  assert.deepEqual(fieldLabels(ui, 'entrant', 8), recommendedField(8).map(competitorLabel));
  ui.cleanup();
});

test('M/L: late AlphaZero discovery fills an untouched field, never an edited one; size change and reset regenerate', async () => {
  const pristine = client('deferred'), untouched = render(React.createElement(TournamentLabPage, { http: pristine.http, storage: memoryStorage() }), 'http://localhost/connect4/tournament');
  assert.deepEqual(fieldLabels(untouched, 'entrant', 8), recommendedField(8).map(competitorLabel));
  pristine.resolve(); await flush();
  assert.deepEqual(fieldLabels(untouched, 'entrant', 8), recommendedField(8, { alphaZero: true }).map(competitorLabel));
  untouched.cleanup();

  const late = client('deferred'), ui = render(React.createElement(TournamentLabPage, { http: late.http, storage: memoryStorage() }), 'http://localhost/connect4/tournament');
  ui.setLabel('entrant-2', 'Negamax · depth 12');
  const edited = fieldLabels(ui, 'entrant', 8);
  late.resolve(); await flush();
  assert.ok(ui.options('entrant-1').some(o => o.text.startsWith('AlphaZero')), 'menus still update');
  assert.deepEqual(fieldLabels(ui, 'entrant', 8), edited, 'edited field is preserved');

  ui.set('tournament-size', 16);
  assert.deepEqual(fieldLabels(ui, 'entrant', 16), recommendedField(16, { alphaZero: true }).map(competitorLabel));
  ui.setLabel('entrant-1', 'Negamax · depth 12');
  ui.click([...ui.doc.querySelectorAll('button')].find(b => b.textContent === 'Reset default field'));
  assert.deepEqual(fieldLabels(ui, 'entrant', 16), recommendedField(16, { alphaZero: true }).map(competitorLabel));
  ui.cleanup();
});

test('D/M/N/O: Season page is AI-only, orders AlphaZero after search agents and protects edits from late discovery', async () => {
  const late = client('deferred'), ui = render(React.createElement(SeasonLabPage, { http: late.http, storage: memoryStorage() }), 'http://localhost/connect4/season');
  assert.deepEqual(fieldLabels(ui, 'season-entrant', 8), recommendedField(8).map(competitorLabel));
  assert.match(ui.doc.querySelector('.tournament-setup').textContent, /evaluates AI agents only/);
  ui.setLabel('season-entrant-1', 'Classic MCTS · 5000 simulations');
  const edited = fieldLabels(ui, 'season-entrant', 8);
  late.resolve(); await flush();
  assert.deepEqual(fieldLabels(ui, 'season-entrant', 8), edited);
  const labels = ui.options('season-entrant-1').map(o => o.text);
  assert.equal(labels.some(l => l.includes('Human')), false);
  assert.equal(labels.at(-1), 'AlphaZero (Experimental) · safeguard On');
  for (const size of season.FIELD_SIZES) {
    ui.set('season-size', size);
    assert.deepEqual(fieldLabels(ui, 'season-entrant', size), recommendedField(size, { alphaZero: true }).map(competitorLabel));
  }
  assert.match(ui.doc.querySelector('.season-workload').textContent, /132 scheduled games.*one at a time/);
  ui.cleanup();
});

test('Q/S: a saved historical Tournament (Classic MCTS 100/400/800) loads unchanged and copies as saved settings', async () => {
  const saved = tournament.createTournament(tournament.defaultField(8), 77);
  assert.ok(saved.entrants.some(e => same(e.config, { type: 'mcts', simulations: 100 })));
  const storage = memoryStorage({ [TOURNAMENT_KEY]: JSON.stringify(saved) });
  const { http } = client('absent'), ui = render(React.createElement(TournamentLabPage, { http, storage }), 'http://localhost/connect4/tournament');
  await flush();
  assert.match(ui.doc.querySelector('.tournament-summary').textContent, /8 entrants/);
  assert.equal(storage.values.get(TOURNAMENT_KEY), JSON.stringify(saved), 'loading does not rewrite the saved record');
  ui.click([...ui.doc.querySelectorAll('button')].find(b => b.textContent === 'New tournament'));
  const labels = fieldLabels(ui, 'entrant', 8);
  assert.deepEqual(labels, saved.entrants.map(e => ['Random', 'Negamax · depth 2', 'Negamax · depth 4', 'Negamax · depth 6', 'Negamax · depth 8'].includes(competitorLabel(e.config))
    ? competitorLabel(e.config) : `${competitorLabel(e.config)} · saved setting`));
  ui.cleanup();
});

test('R/S: a saved historical Season loads unchanged and still validates', async () => {
  const saved = season.createSeason(tournament.defaultField(8), 77, 2);
  const storage = memoryStorage({ [SEASON_KEY]: JSON.stringify(saved) });
  assert.deepEqual(season.validateSeason(JSON.parse(storage.getItem(SEASON_KEY))), saved);
  const { http } = client('absent'), ui = render(React.createElement(SeasonLabPage, { http, storage }), 'http://localhost/connect4/season');
  await flush();
  assert.match(ui.doc.querySelector('.tournament-summary').textContent, /8 entrants/);
  assert.equal(storage.values.get(SEASON_KEY), JSON.stringify(saved));
  ui.click([...ui.doc.querySelectorAll('button')].find(b => b.textContent === 'New season'));
  assert.equal(ui.selected('season-entrant-6'), 'Classic MCTS · 100 simulations · saved setting');
  ui.cleanup();
});

const headings = ui => [...ui.doc.querySelectorAll('#agents h3')].map(h => h.textContent);
test('V/W: homepage shows AlphaZero before Victor and only calls it playable when the backend reports it', async () => {
  const available = render(React.createElement(Home, { http: client('available').http }));
  await flush();
  // Victor's build flag is off in unit tests; the homepage omits its card exactly as before.
  assert.deepEqual(headings(available), ['Random', 'Negamax', 'Classic MCTS', 'Tactical MCTS', 'AlphaZero']);
  const azCard = [...available.doc.querySelectorAll('#agents article')].find(a => a.querySelector('h3').textContent === 'AlphaZero');
  assert.match(azCard.querySelector('.badge').textContent, /^Playable/);
  assert.match(available.doc.querySelector('.research-boundary').textContent, /Playable now: Random, Negamax, Classic MCTS, Tactical MCTS, AlphaZero\./);
  assert.ok([...available.doc.querySelectorAll('a')].some(a => a.textContent.startsWith('Play against AlphaZero')));
  const research = available.doc.getElementById('research').textContent;
  assert.ok(research.indexOf('Train a neural network') < research.indexOf('Victor Research'));
  for (const fact of ['325,896', '256 PUCT', '+1.30', '+2\u2011point']) assert.ok(research.includes(fact), fact);
  assert.doesNotMatch(available.doc.body.textContent, /\b(three|four|five|six) ways\b|\d+ playable agents|100 \/ 400 \/ 800/);
  assert.match(available.doc.getElementById('agents').textContent, /500–25,000 simulation presets/);
  assert.match(available.doc.getElementById('agents').textContent, /200–10,000 max-simulation presets/);
  available.cleanup();

  const absent = render(React.createElement(Home, { http: client('absent').http }));
  await flush();
  const badge = [...absent.doc.querySelectorAll('#agents article')].find(a => a.querySelector('h3').textContent === 'AlphaZero').querySelector('.badge').textContent;
  assert.equal(badge, 'Not available on this server');
  assert.match(absent.doc.querySelector('.research-boundary').textContent, /Playable now: Random, Negamax, Classic MCTS, Tactical MCTS\./);
  assert.equal([...absent.doc.querySelectorAll('a')].some(a => a.textContent.startsWith('Play against AlphaZero')), false);
  assert.equal([...absent.doc.querySelectorAll('.badge')].some(b => b.textContent === 'Playable · Experimental'), false);
  absent.cleanup();

  const checking = render(React.createElement(Home, { http: client('deferred').http }));
  assert.equal([...checking.doc.querySelectorAll('#agents .badge')].at(-1).textContent, 'Checking availability…');
  checking.cleanup();
});

test('M: an availability recheck keeps an untouched field stable; a confirmed loss removes AlphaZero', async () => {
  const responses = [];
  const http = { get: () => new Promise((resolve, reject) => responses.push({ resolve, reject })), post: async () => { throw new Error('No game may start.'); } };
  const ui = render(React.createElement(TournamentLabPage, { http, storage: memoryStorage() }), 'http://localhost/connect4/tournament');
  responses.shift().resolve({ data: status }); await flush();
  const withAlphaZero = recommendedField(8, { alphaZero: true }).map(competitorLabel);
  assert.deepEqual(fieldLabels(ui, 'entrant', 8), withAlphaZero);
  const refresh = () => ui.click([...ui.doc.querySelectorAll('button')].find(b => b.textContent === 'Refresh agent availability'));

  refresh(); await flush();
  // Checking: menus fail closed, but the untouched field neither reshuffles nor drops AlphaZero.
  assert.equal(ui.options('entrant-2').some(o => o.text === 'AlphaZero (Experimental) · safeguard Off'), false);
  assert.deepEqual(fieldLabels(ui, 'entrant', 8), withAlphaZero.map(l => l.startsWith('AlphaZero') ? `${l} · checking availability` : l));
  responses.shift().resolve({ data: status }); await flush();
  assert.deepEqual(fieldLabels(ui, 'entrant', 8), withAlphaZero);

  refresh(); await flush();
  responses.shift().reject(Object.assign(new Error('absent'), { response: { status: 404 } })); await flush();
  assert.deepEqual(fieldLabels(ui, 'entrant', 8), recommendedField(8).map(competitorLabel));
  ui.cleanup();
});

test('V/W: the hero names the neural agent only when the backend reports it available', async () => {
  for (const [discovery, neural] of [['available', true], ['absent', false]]) {
    const ui = render(React.createElement(Home, { http: client(discovery).http }));
    await flush();
    assert.equal(ui.doc.querySelector('.hero-description').textContent.includes('neural network'), neural);
    ui.cleanup();
  }
});
