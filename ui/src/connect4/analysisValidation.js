// Validate the complete deterministic evidence and curated reference metadata.
// No partial result or unvalidated highlight may render.
export const ALLIS_SOURCE_URL = 'https://tromp.github.io/c4/connect4_thesis.pdf';

export function validExplanation(data) {
  const explanation = data?.explanation;
  const texts = values => Array.isArray(values) && values.every(value => typeof value === 'string');
  const facts = values => Array.isArray(values) && values.length > 0 && values.every(f =>
    f && typeof f.id === 'string' && typeof f.text === 'string' && f.classification === 'confirmed_tactical');
  const concepts = values => Array.isArray(values) && values.every(entry =>
    entry && ['context_only', 'reference_only'].includes(entry.classification) &&
    typeof entry.title === 'string' && typeof entry.text === 'string' && texts(entry.limitations) &&
    (entry.connection === undefined || typeof entry.connection === 'string') &&
    (entry.preconditions === undefined || texts(entry.preconditions)) &&
    Array.isArray(entry.source?.references) && entry.source.references.length > 0 && entry.source.references.every(ref =>
      ref && Number.isInteger(ref.chapter) && typeof ref.section === 'string' &&
      Array.isArray(ref.thesis_pages) && ref.thesis_pages.length === 2 && ref.thesis_pages.every(Number.isInteger)));
  return explanation && facts(explanation.facts) && facts(explanation.key_facts) && explanation.key_facts.length <= 3 &&
    typeof explanation.summary?.text === 'string' && typeof explanation.summary.focus_id === 'string' &&
    texts(explanation.summary.fact_ids) && explanation.summary.fact_ids.length > 0 &&
    explanation.summary.fact_ids.every(id => explanation.facts.some(f => f.id === id)) &&
    explanation.key_facts.every(key => explanation.facts.some(f => f.id === key.id && f.text === key.text)) &&
    Array.isArray(explanation.relevant_squares) && explanation.relevant_squares.length <= 42 &&
    explanation.relevant_squares.every(s => s && Number.isInteger(s.column) && s.column >= 0 && s.column < 7 &&
      Number.isInteger(s.row_index) && s.row_index >= 0 && s.row_index < 6 && s.row === 6 - s.row_index &&
      s.name === `${String.fromCharCode(97 + s.column)}${s.row}`) &&
    concepts(explanation.strategic_context) && explanation.strategic_context.length <= 1 &&
    explanation.strategic_context.every(e => e.classification === 'context_only' && e.connection) &&
    concepts(explanation.additional_context) && texts(explanation.limitations);
}

export function validateAnalysisResponse(data, requested) {
  if (data?.game_id !== requested.game_id || data?.revision !== requested.revision ||
      data?.mode !== requested.mode || data?.column !== (requested.column ?? null) ||
      !validExplanation(data)) {
    throw new Error('The analysis server returned an unusable response.');
  }
  return data;
}
