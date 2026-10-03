// Separate request lifecycle: explanation failures never lock gameplay.
export function mountExplanations({ document, http }) {
  const el = id => document.getElementById(id);
  if (!el('explanation-panel')) return { update() {}, cleanup() {} };
  let game = null;
  let blocked = true;
  let active = true;
  let controller = null;
  let generation = 0;
  let loading = false;
  let status = 'Start a game to request an explanation.';
  const listeners = [];

  function controls() {
    el('explanation-status').textContent = status;
    el('explanation-panel').setAttribute('aria-busy', String(loading));
    for (const id of ['explain-last', 'analyze-position', 'what-if', 'what-if-column', 'explanation-question']) {
      el(id).disabled = !game || blocked || loading ||
        (['what-if', 'what-if-column'].includes(id) && (game.gameOver || !game.legalMoves.length));
    }
    el('explain-last').textContent = game?.revision > 0 &&
      game.players[1 - game.currentPlayer].type !== 'human' ? 'Explain AI Move' : 'Explain Last Move';
  }

  function cancel() {
    generation++;
    controller?.abort();
    controller = null;
    loading = false;
  }

  function paragraph(parent, text) {
    const node = document.createElement('p');
    node.textContent = text;
    parent.appendChild(node);
  }

  function clearHighlights() {
    document.querySelectorAll('#game-board .explanation-square').forEach(cell => {
      cell.classList.remove('explanation-square');
      cell.querySelector('.square-label')?.remove();
    });
  }

  function show(data) {
    const root = el('explanation-result');
    root.replaceChildren();
    clearHighlights();
    const section = title => {
      const group = document.createElement('section');
      const heading = document.createElement('h3');
      heading.textContent = title;
      group.appendChild(heading);
      root.appendChild(group);
      return group;
    };
    const disclosure = title => {
      const group = document.createElement('details');
      const label = document.createElement('summary');
      label.textContent = title;
      group.appendChild(label);
      root.appendChild(group);
      return group;
    };
    const concept = (parent, entry) => {
      paragraph(parent, `${entry.title}: ${entry.text}`);
      if (entry.connection) paragraph(parent, entry.connection);
      const link = document.createElement('a');
      // Only backend catalog references are rendered; model citations are rejected.
      link.href = 'https://tromp.github.io/c4/connect4_thesis.pdf';
      link.target = '_blank';
      link.rel = 'noopener noreferrer';
      link.textContent = `Allis (1988): ${entry.source.references.map(ref =>
        `Chapter ${ref.chapter}, §${ref.section}, thesis/PDF pp. ${ref.thesis_pages[0]}–${ref.thesis_pages[1]}`).join('; ')}`;
      parent.appendChild(link);
    };
    const primary = section('Primary explanation');
    primary.className = 'primary-explanation';
    paragraph(primary, data.summary.text);
    const facts = section('Key tactical evidence');
    data.key_facts.forEach(fact => paragraph(facts, fact.text));
    if (data.strategic_context.length) {
      const strategy = section('Relevant Allis concept');
      data.strategic_context.forEach(entry => concept(strategy, entry));
      paragraph(strategy, 'Conceptual context; not a proven rule application.');
    }
    const details = disclosure('Detailed analysis');
    paragraph(details, 'Complete verified tactical facts');
    data.facts.forEach(fact => paragraph(details, fact.text));
    data.additional_context.forEach(entry => concept(details, entry));
    const limits = disclosure('Methodology and limitations');
    data.limitations.forEach(text => paragraph(limits, text));
    [...data.strategic_context, ...data.additional_context].forEach(entry => {
      (entry.preconditions || []).forEach(text => paragraph(limits, `${entry.title} — reference precondition: ${text}`));
      entry.limitations.forEach(text => paragraph(limits, text));
    });
    data.relevant_squares.forEach(square => {
      const cell = document.querySelector(`#game-board .cell[data-square="${square.name}"]`);
      if (!cell) return;
      cell.classList.add('explanation-square');
      const label = document.createElement('span');
      label.className = 'square-label';
      label.textContent = square.name;
      label.setAttribute('aria-hidden', 'true');
      cell.appendChild(label);
    });
  }

  function validExplanation(data) {
    const explanation = data?.explanation;
    const texts = values => Array.isArray(values) && values.every(value => typeof value === 'string');
    const facts = values => Array.isArray(values) && values.length > 0 && values.every(f =>
      typeof f.id === 'string' && typeof f.text === 'string' && f.classification === 'confirmed_tactical');
    const concepts = values => Array.isArray(values) && values.every(entry =>
      ['context_only', 'reference_only'].includes(entry.classification) &&
      typeof entry.title === 'string' && typeof entry.text === 'string' && texts(entry.limitations) &&
      (entry.connection === undefined || typeof entry.connection === 'string') &&
      (entry.preconditions === undefined || texts(entry.preconditions)) &&
      Array.isArray(entry.source?.references) && entry.source.references.length > 0 && entry.source.references.every(ref =>
        Number.isInteger(ref.chapter) && typeof ref.section === 'string' &&
        Array.isArray(ref.thesis_pages) && ref.thesis_pages.length === 2 && ref.thesis_pages.every(Number.isInteger)));
    return explanation && facts(explanation.facts) && facts(explanation.key_facts) && explanation.key_facts.length <= 3 &&
      typeof explanation.summary?.text === 'string' && typeof explanation.summary.focus_id === 'string' &&
      texts(explanation.summary.fact_ids) && explanation.summary.fact_ids.length > 0 &&
      explanation.summary.fact_ids.every(id => explanation.facts.some(f => f.id === id)) &&
      explanation.key_facts.every(key => explanation.facts.some(f => f.id === key.id && f.text === key.text)) &&
      Array.isArray(explanation.relevant_squares) && explanation.relevant_squares.length <= 42 &&
      explanation.relevant_squares.every(s => Number.isInteger(s.column) && s.column >= 0 && s.column < 7 &&
        Number.isInteger(s.row_index) && s.row_index >= 0 && s.row_index < 6 && s.row === 6 - s.row_index &&
        s.name === `${String.fromCharCode(97 + s.column)}${s.row}`) &&
      concepts(explanation.strategic_context) && explanation.strategic_context.length <= 1 &&
      explanation.strategic_context.every(e => e.classification === 'context_only' && e.connection) &&
      concepts(explanation.additional_context) && texts(explanation.limitations);
  }

  async function request(mode) {
    if (!active || !game || blocked || loading || (mode === 'what_if' && game.gameOver)) return;
    const question = el('explanation-question').value;
    if (question.length > 500) {
      status = 'Keep the question to 500 characters or fewer.';
      controls();
      return;
    }
    cancel();
    const requestGeneration = generation;
    const requested = { game_id: game.game_id, revision: game.revision, mode, question };
    if (mode === 'what_if') requested.column = Number(el('what-if-column').value);
    controller = new AbortController();
    loading = true;
    status = 'Preparing a grounded explanation… You can still play or restart.';
    el('explanation-result').replaceChildren();
    clearHighlights();
    controls();
    try {
      const response = await http.post('/v1/connect4/explain', requested, { signal: controller.signal });
      if (!active || requestGeneration !== generation) return;
      const data = response.data;
      if (data?.game_id !== game.game_id || data?.revision !== game.revision ||
          data?.mode !== mode || data?.column !== (requested.column ?? null) || !validExplanation(data)) {
        throw new Error('The explanation server returned an unusable response.');
      }
      show(data.explanation);
      status = `${mode === 'what_if' ? `Hypothetical Column ${requested.column + 1}; ` : ''}Board revision ${data.revision}${data.cached ? ' (cached)' : ''}.`;
    } catch (error) {
      if (!active || requestGeneration !== generation) return;
      status = (error.response?.data?.error || 'The explanation could not be loaded. Try the explanation button again.') + ' Gameplay remains available.';
    } finally {
      if (active && requestGeneration === generation) {
        loading = false;
        controller = null;
        controls();
      }
    }
  }

  for (const [id, mode] of [['explain-last', 'last_move'], ['analyze-position', 'position'], ['what-if', 'what_if']]) {
    const handler = () => request(mode);
    const node = el(id);
    node.addEventListener('click', handler);
    listeners.push(() => node.removeEventListener('click', handler));
  }
  controls();
  return {
    update(next, unavailable) {
      const changed = game?.game_id !== next?.game_id || game?.revision !== next?.revision;
      if (changed || unavailable && !blocked) {
        cancel();
        el('explanation-result').replaceChildren();
        clearHighlights();
        status = next ? 'Request an explanation for this board.' : 'Start a game to request an explanation.';
      }
      game = next;
      blocked = unavailable;
      const selected = el('what-if-column').value;
      el('what-if-column').replaceChildren();
      (game?.legalMoves || []).forEach(column => {
        const option = document.createElement('option');
        option.value = String(column);
        option.textContent = `Column ${column + 1}`;
        el('what-if-column').appendChild(option);
      });
      if (game?.legalMoves.includes(Number(selected))) el('what-if-column').value = selected;
      controls();
    },
    cleanup() {
      active = false;
      cancel();
      clearHighlights();
      listeners.forEach(remove => remove());
    },
  };
}
