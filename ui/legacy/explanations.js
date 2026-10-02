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

  function show(data) {
    const root = el('explanation-result');
    root.replaceChildren();
    const section = (title) => {
      const group = document.createElement('section');
      const heading = document.createElement('h3');
      heading.textContent = title;
      group.appendChild(heading);
      root.appendChild(group);
      return group;
    };
    const facts = section('Verified tactical facts');
    data.facts.forEach(fact => paragraph(facts, fact.text));
    const strategy = section('Strategic context from Allis — not a proven rule application');
    data.strategic_context.forEach(entry => {
      paragraph(strategy, `${entry.title}: ${entry.text}`);
      (entry.preconditions || []).forEach(text => paragraph(strategy, `Reference precondition: ${text}`));
      entry.limitations.forEach(text => paragraph(strategy, text));
      const link = document.createElement('a');
      // Citations always come from the curated backend, never model prose.
      link.href = 'https://tromp.github.io/c4/connect4_thesis.pdf';
      link.target = '_blank';
      link.rel = 'noopener noreferrer';
      link.textContent = `Allis (1988): ${entry.source.references.map(ref =>
        `Chapter ${ref.chapter}, §${ref.section}, thesis/PDF pp. ${ref.thesis_pages[0]}–${ref.thesis_pages[1]}`).join('; ')}`;
      strategy.appendChild(link);
    });
    const limits = section('What this analysis cannot establish');
    data.limitations.forEach(text => paragraph(limits, text));
  }

  function validExplanation(data) {
    const explanation = data?.explanation;
    const texts = values => Array.isArray(values) && values.every(value => typeof value === 'string');
    return explanation && Array.isArray(explanation.facts) && explanation.facts.length > 0 &&
      explanation.facts.every(f => typeof f.text === 'string' && f.classification === 'confirmed_tactical') &&
      Array.isArray(explanation.strategic_context) && explanation.strategic_context.every(entry =>
        ['context_only', 'reference_only'].includes(entry.classification) &&
        typeof entry.title === 'string' && typeof entry.text === 'string' && texts(entry.limitations) &&
        (entry.preconditions === undefined || texts(entry.preconditions)) &&
        Array.isArray(entry.source?.references) && entry.source.references.every(ref =>
          Number.isInteger(ref.chapter) && typeof ref.section === 'string' &&
          Array.isArray(ref.thesis_pages) && ref.thesis_pages.length === 2 && ref.thesis_pages.every(Number.isInteger))) &&
      texts(explanation.limitations);
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
    controls();
    try {
      const response = await http.post('/v1/connect4/explain', requested, { signal: controller.signal });
      if (!active || requestGeneration !== generation) return;
      const data = response.data;
      if (data?.game_id !== game.game_id || data?.revision !== game.revision ||
          data?.mode !== mode || !validExplanation(data)) {
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
      listeners.forEach(remove => remove());
    },
  };
}
