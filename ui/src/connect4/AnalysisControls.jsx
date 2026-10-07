export default function AnalysisControls({ analysis }) {
  const { mode, request, lastMoveLabel, disabled, legalColumns, column, setColumn, question, setQuestion } = analysis;
  return <div className="analysis-inputs">
    <div className="analysis-modes" role="group" aria-label="Analysis actions">
      {[
        ['explain-last', 'last_move', lastMoveLabel, 'Last move'],
        ['analyze-position', 'position', 'Analyze Position', 'Current position'],
        ['what-if', 'what_if', 'What If?', 'What if?'],
      ].map(([id, value, label, title]) => <button key={value} id={id} type="button"
        aria-label={label} aria-pressed={mode === value}
        disabled={disabled || (value === 'what_if' && !legalColumns.length)}
        onClick={() => request(value)}>
        <span>{title}</span><small>{value === 'last_move' ? lastMoveLabel : value === 'position' ? 'Inspect this board' : 'Test a legal column'}</small>
      </button>)}
    </div>
    <div className="analysis-fields">
      <div className="analysis-question">
        <div className="question-label"><label htmlFor="explanation-question">Optional question</label><span id="question-count">{question.length}/500</span></div>
        <textarea id="explanation-question" maxLength={500} rows={2} value={question}
          onChange={event => setQuestion(event.target.value)} disabled={disabled}
          aria-describedby="question-help question-count" placeholder="What should I look for in this position?" />
        <p id="question-help" className="analysis-helper">Ask about this board. Answers stay within verified facts and reference context.</p>
      </div>
      {mode === 'what_if' && <div className="what-if-controls">
        <label htmlFor="what-if-column">Hypothetical move:</label>
        <select id="what-if-column" value={column ?? ''} onChange={event => setColumn(Number(event.target.value))}
          disabled={disabled || !legalColumns.length} aria-describedby="column-help">
          {!legalColumns.length && <option value="">No legal columns</option>}
          {legalColumns.map(c => <option key={c} value={c}>Column {c + 1}</option>)}
        </select>
        <p id="column-help" className="analysis-helper">Choose a column, then select What if? The game stays unchanged.</p>
      </div>}
    </div>
  </div>;
}
