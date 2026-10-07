import { test } from 'node:test';
import assert from 'node:assert/strict';
import { validateAnalysisResponse } from '../src/connect4/analysisValidation.js';

const requested = { game_id: 'a', revision: 2, mode: 'position', question: '' };
const fact = { id: 'reply', text: 'An immediate reply is available.', classification: 'confirmed_tactical' };
const concept = { title: 'Winning squares', text: 'Gravity matters.', classification: 'context_only',
  connection: 'The square is reachable.', preconditions: ['Legal position'], limitations: ['No long-term proof'],
  source: { references: [{ chapter: 3, section: '3.1', thesis_pages: [16, 18] }] } };
const fixture = () => structuredClone({ ...requested, column: null, cached: false, explanation: {
  facts: [fact], key_facts: [{ ...fact }], summary: { text: 'A verified consequence.', focus_id: 'reply', fact_ids: ['reply'] },
  relevant_squares: [{ name: 'b1', column: 1, row: 1, row_index: 5 }],
  strategic_context: [concept], additional_context: [{ ...concept, classification: 'reference_only' }],
  limitations: ['Post-hoc analysis'],
} });

test('complete grounded fixture validates unchanged', () => {
  const data = fixture();
  assert.equal(validateAnalysisResponse(data, requested), data);
});

const invalid = [
  ['missing explanation', d => { delete d.explanation; }],
  ['empty facts', d => { d.explanation.facts = []; }],
  ['null fact', d => { d.explanation.facts = [null]; }],
  ['fact ID type', d => { d.explanation.facts[0].id = 5; }],
  ['fact text type', d => { d.explanation.facts[0].text = {}; }],
  ['fact classification', d => { d.explanation.facts[0].classification = 'context_only'; }],
  ['empty key evidence', d => { d.explanation.key_facts = []; }],
  ['too many key facts', d => { d.explanation.key_facts = Array(4).fill(fact); }],
  ['invented key evidence', d => { d.explanation.key_facts[0].id = 'invented'; }],
  ['altered key text', d => { d.explanation.key_facts[0].text = 'Altered claim'; }],
  ['missing summary', d => { delete d.explanation.summary; }],
  ['summary text type', d => { d.explanation.summary.text = 4; }],
  ['focus ID type', d => { d.explanation.summary.focus_id = null; }],
  ['empty summary references', d => { d.explanation.summary.fact_ids = []; }],
  ['unreferenced summary claim', d => { d.explanation.summary.fact_ids = ['invented']; }],
  ['missing squares', d => { delete d.explanation.relevant_squares; }],
  ['too many squares', d => { d.explanation.relevant_squares = Array(43).fill({ name: 'b1', column: 1, row: 1, row_index: 5 }); }],
  ['null square', d => { d.explanation.relevant_squares = [null]; }],
  ['unsafe square name', d => { d.explanation.relevant_squares[0].name = '"] img'; }],
  ['column out of range', d => { d.explanation.relevant_squares[0].column = 7; }],
  ['row index out of range', d => { d.explanation.relevant_squares[0].row_index = 6; }],
  ['bottom-based row mismatch', d => { d.explanation.relevant_squares[0].row = 2; }],
  ['fractional coordinate', d => { d.explanation.relevant_squares[0].column = .5; }],
  ['missing strategic context', d => { delete d.explanation.strategic_context; }],
  ['null concept', d => { d.explanation.strategic_context = [null]; }],
  ['too many strategic concepts', d => { d.explanation.strategic_context.push(concept); }],
  ['reference-only primary concept', d => { d.explanation.strategic_context[0].classification = 'reference_only'; }],
  ['invented classification', d => { d.explanation.additional_context[0].classification = 'proven'; }],
  ['missing connection', d => { delete d.explanation.strategic_context[0].connection; }],
  ['concept title type', d => { d.explanation.strategic_context[0].title = {}; }],
  ['concept text type', d => { d.explanation.additional_context[0].text = []; }],
  ['concept limitations type', d => { d.explanation.strategic_context[0].limitations = [3]; }],
  ['preconditions type', d => { d.explanation.additional_context[0].preconditions = 'unknown'; }],
  ['empty source references', d => { d.explanation.strategic_context[0].source.references = []; }],
  ['null reference', d => { d.explanation.strategic_context[0].source.references = [null]; }],
  ['chapter type', d => { d.explanation.strategic_context[0].source.references[0].chapter = '3'; }],
  ['section type', d => { d.explanation.strategic_context[0].source.references[0].section = 3; }],
  ['page range shape', d => { d.explanation.strategic_context[0].source.references[0].thesis_pages = [16]; }],
  ['page range type', d => { d.explanation.strategic_context[0].source.references[0].thesis_pages = [16, '18']; }],
  ['missing additional context', d => { delete d.explanation.additional_context; }],
  ['missing limitations', d => { delete d.explanation.limitations; }],
  ['limitation type', d => { d.explanation.limitations = [null]; }],
];
for (const [label, mutate] of invalid) {
  test(`rejects ${label} before any partial rendering`, () => {
    const data = fixture(); mutate(data);
    assert.throws(() => validateAnalysisResponse(data, requested), /unusable response/);
  });
}
