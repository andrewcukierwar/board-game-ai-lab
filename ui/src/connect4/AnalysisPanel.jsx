import AnalysisControls from './AnalysisControls.jsx';
import AnalysisResult from './AnalysisResult.jsx';

export default function AnalysisPanel({ analysis }) {
  return <section id="explanation-panel" className={`analysis-panel analysis-panel--${analysis.phase}`}
    aria-labelledby="analysis-title" aria-busy={analysis.loading}>
    <div className="analysis-heading"><div><p className="eyebrow">Inspect · verify · explore</p><h2 id="analysis-title">AI Analysis</h2></div><span className="badge">Grounded analysis</span></div>
    <p className="analysis-intro">Inspect verified tactical consequences and grounded strategic context for the current position.</p>
    <p className="analysis-boundary">Post-hoc analysis of the board. Strategic concepts provide context; they do not establish a formal proof or reveal agent intent.</p>
    <AnalysisControls analysis={analysis} />
    <p id="explanation-status" className="analysis-status" aria-live="polite" aria-atomic="true">{analysis.status}</p>
    {analysis.result ? <AnalysisResult key={`${analysis.result.game_id}:${analysis.result.revision}:${analysis.result.mode}:${analysis.result.column}`} result={analysis.result} /> : <div id="explanation-result" />}
  </section>;
}
