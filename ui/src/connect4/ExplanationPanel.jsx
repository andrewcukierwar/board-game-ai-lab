import { useLayoutEffect, useRef } from 'react';
import { mountExplanations } from '../../legacy/explanations.js';

// Temporary DOM island: React supplies the scaffold; the legacy controller
// retains all analysis requests, validation, result rendering and highlights.
export default function ExplanationPanel({ http, game, unavailable }) {
  const controller = useRef(null);
  useLayoutEffect(() => {
    const mounted = mountExplanations({ document, http });
    controller.current = mounted;
    return () => { mounted.cleanup(); controller.current = null; };
  }, [http]);
  useLayoutEffect(() => {
    controller.current?.update(game, unavailable);
    // The legacy controller preserves an empty pre-game select value when
    // column 0 becomes legal. Ensure the newly populated select visibly has
    // a legal default, without taking over its options or request lifecycle.
    const columns = document.getElementById('what-if-column');
    if (columns.options.length && columns.selectedIndex < 0) columns.selectedIndex = 0;
  }, [http, game, unavailable]);

  return <section id="explanation-panel" aria-label="Grounded explanations">
    <div className="analysis-heading"><div><p className="eyebrow">Explore the position</p><h2>Understand the position</h2></div><span className="badge">Grounded analysis</span></div>
    <p className="analysis-intro">Ask about a move or position. Explanations connect verified board facts to Connect 4 strategy.</p>
    <label htmlFor="explanation-question">Optional question (500 characters maximum):</label>
    <textarea id="explanation-question" maxLength={500} rows={2} placeholder="What should I look for in this position?" />
    <div className="explanation-controls">
      <button id="explain-last" type="button">Explain Last Move</button>
      <button id="analyze-position" type="button">Analyze Position</button>
      <div className="what-if-controls"><label htmlFor="what-if-column">Hypothetical move:</label><select id="what-if-column" /><button id="what-if" type="button">What If?</button></div>
    </div>
    <p id="explanation-status" aria-live="polite" />
    <div id="explanation-result" />
  </section>;
}
