import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

// Fail closed if CI selected the wrong file, skipped integration, or retried a
// failure. Feature-dependent skips must be exactly the declared Victor case.
const [file, mode] = process.argv.slice(2);
assert(['true', 'false'].includes(mode), 'Pass the frontend Victor setting (true/false).');
const report = JSON.parse(await readFile(file, 'utf8'));
const specs = [];
function visit(suite) {
  specs.push(...(suite.specs ?? []));
  for (const child of suite.suites ?? []) visit(child);
}
for (const suite of report.suites) visit(suite);
const matchups = ['human', 'negamax', 'mcts', 'mcts_tactical', 'victor_research', 'alphazero'];
for (const opponent of matchups) {
  assert.equal(specs.filter(s => s.title === `real Match AlphaZero vs ${opponent}, step/timeline/replay`).length, 1,
    `Missing/duplicate ${opponent} matchup`);
}
for (const timing of ['pause before busy response', 'pause during retry wait', 'split retry clock']) {
  assert.equal(specs.filter(s => s.title === `Match autoplay busy bounded retries and pause cancels old retry: ${timing}`).length, 1,
    `Missing/duplicate autoplay case: ${timing}`);
}
assert.equal(specs.filter(s => s.title === 'Match ignores a stale authorized response after leaving the active match').length, 1);
assert.equal(report.errors.length, 0, 'Playwright reported global errors');
let passed = 0, skipped = 0;
for (const spec of specs) {
  assert.equal(spec.tests.length, 1, `Expected one Chromium execution: ${spec.title}`);
  const execution = spec.tests[0];
  assert.equal(execution.projectName, 'chromium');
  assert.equal(execution.results.length, 1, `No retries allowed: ${spec.title}`);
  const expectedSkip = mode === 'false' && spec.title === 'real Match AlphaZero vs victor_research, step/timeline/replay';
  assert.equal(execution.results[0].status, expectedSkip ? 'skipped' : 'passed', spec.title);
  if (expectedSkip) skipped++; else passed++;
}
console.log(`Verified Victor ${mode}: ${passed} passed, ${skipped} expected skips; all six matchups and cancellation cases accounted for.`);
