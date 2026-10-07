// Deterministic mocked analysis; these fixtures never call a provider.
export function explained(request) {
  return { ...request, column: request.column ?? null, cached: false, explanation: {
    facts: [{ id: 'position', text: 'This position was verified.', classification: 'confirmed_tactical' }],
    summary: { text: 'Concise verified answer.', focus_id: 'quiet', fact_ids: ['position'] },
    key_facts: [{ id: 'position', text: 'This position was verified.', classification: 'confirmed_tactical' }],
    relevant_squares: [], additional_context: [],
    strategic_context: [{ connection: 'This connects to the verified position.', title: 'Immediate tactics', classification: 'context_only', text: 'Check immediate replies.',
      limitations: ['No long-term proof.'], source: { references: [{ chapter: 3, section: '3.4', thesis_pages: [21, 24] }] } }],
    limitations: ['Post-hoc analysis; agent intent is unknown.'],
  } };
}

export function representative(request) {
  const result = explained(request);
  const data = result.explanation;
  data.summary.text = 'This move makes d2 reachable under gravity. The resulting position allows an immediate reply there; avoiding that reply does not prove a long-term win.';
  data.facts = [
    { id: 'position', text: 'The highlighted square d2 becomes reachable when d1 is occupied.', classification: 'confirmed_tactical' },
    { id: 'reply', text: 'Immediate replies were checked against legal columns. Geometric completion squares are not automatically playable.', classification: 'confirmed_tactical' },
  ];
  data.key_facts = data.facts;
  data.relevant_squares = [{ name: 'd1', column: 3, row: 1, row_index: 5 }, { name: 'd2', column: 3, row: 2, row_index: 4 }];
  data.strategic_context[0] = { ...data.strategic_context[0], title: 'Threats and winning squares',
    text: 'A winning square must be reachable under gravity.', connection: 'One relevant strategic concept is the distinction between a completion square and a playable winning square.',
    preconditions: ['The reference applies to a legal position with gravity.'] };
  data.additional_context = [{ ...data.strategic_context[0], title: 'Formal coverage', classification: 'reference_only',
    text: 'Formal coverage requires compatible solutions and additional preconditions.', connection: '',
    limitations: ['No formal rule application is established by this analysis.'] }];
  return result;
}
