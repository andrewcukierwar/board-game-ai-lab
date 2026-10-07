import { ALLIS_SOURCE_URL } from './analysisValidation.js';

function Concept({ entry }) {
  return <article className="analysis-concept">
    <span className="evidence-label">{entry.classification === 'context_only' ? 'Conceptual context' : 'Reference only'}</span>
    <h4>{entry.title}</h4>
    <p>{entry.text}</p>
    {entry.connection && <p>{entry.connection}</p>}
    <a className="analysis-source" href={ALLIS_SOURCE_URL} target="_blank" rel="noopener noreferrer">
      Allis (1988): {entry.source.references.map(ref =>
        `Chapter ${ref.chapter}, §${ref.section}, thesis/PDF pp. ${ref.thesis_pages[0]}–${ref.thesis_pages[1]}`).join('; ')}
      <span className="visually-hidden"> (opens in a new tab)</span>
    </a>
  </article>;
}

export function PrimaryExplanation({ summary }) {
  return <section className="primary-explanation"><p className="eyebrow">Position insight</p><h3>Analysis</h3><p>{summary.text}</p></section>;
}
export function TacticalEvidence({ facts }) {
  return <section className="tactical-evidence"><h3>Verified tactical evidence</h3>
    <p className="analysis-helper">Checked against the board, legal moves, and immediate replies.</p>
    <ul>{facts.map(fact => <li key={fact.id}><span className="evidence-label">Verified</span><p>{fact.text}</p></li>)}</ul>
  </section>;
}
export function StrategicContext({ entries }) {
  return entries.length > 0 && <section className="strategic-context"><h3>Strategic context</h3>
    <p className="analysis-helper">Conceptual context; not a proven rule application.</p>
    {entries.map((entry, index) => <Concept key={index} entry={entry} />)}
  </section>;
}
export function AnalysisDetails({ explanation }) {
  return <>
    <details className="analysis-disclosure"><summary>Detailed analysis</summary>
      <div><h4>Complete verified tactical facts</h4>
        <ul>{explanation.facts.map(fact => <li key={fact.id}>{fact.text}</li>)}</ul>
        {explanation.additional_context.length > 0 && <h4>Additional reference context</h4>}
        {explanation.additional_context.map((entry, index) => <Concept key={index} entry={entry} />)}
      </div>
    </details>
    <details className="analysis-disclosure"><summary>Methodology and limitations</summary>
      <div><p>This is post-hoc analysis of verified consequences. Agent intent, private reasoning, and search traces are not exposed.</p>
        <ul>{explanation.limitations.map((text, index) => <li key={index}>{text}</li>)}</ul>
        {[...explanation.strategic_context, ...explanation.additional_context].map((entry, index) =>
          <section key={index}><h4>{entry.title}</h4>
            {(entry.preconditions || []).map((text, i) => <p key={i}>Reference precondition: {text}</p>)}
            {entry.limitations.map((text, i) => <p key={i}>{text}</p>)}
          </section>)}
      </div>
    </details>
  </>;
}
export default function AnalysisResult({ result }) {
  const data = result.explanation;
  return <div id="explanation-result">
    <PrimaryExplanation summary={data.summary} />
    <TacticalEvidence facts={data.key_facts} />
    <StrategicContext entries={data.strategic_context} />
    <AnalysisDetails explanation={data} />
  </div>;
}
