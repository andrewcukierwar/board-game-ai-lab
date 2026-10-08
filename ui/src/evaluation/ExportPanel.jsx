import { useEffect, useState } from 'react';
import { createEvaluationExport } from './export.js';
import { verifyEvaluationExport } from './verify.js';
import { gamesCSV, summaryCSV } from './csv.js';
import { downloadText } from './download.js';
import { SCHEMA_VERSION, METHODOLOGY_VERSION, sourceCommit } from './provenance.js';
import { SCHEDULE_VERSION } from '../season/model.js';

const configuredCommit = sourceCommit(import.meta.env?.VITE_EVALUATION_SOURCE_COMMIT);
export default function ExportPanel({ season }) {
  const [preview, setPreview] = useState(null), [error, setError] = useState(''), [feedback, setFeedback] = useState(''), [busy, setBusy] = useState(false);
  useEffect(() => {
    let live = true; setPreview(null); setError(''); setFeedback('');
    createEvaluationExport(season, { sourceCommit: configuredCommit }).then(async artifact => {
      const check = await verifyEvaluationExport(artifact);
      if (!check.ok) throw new Error(check.errors.map(e => e.message).join(' '));
      if (live) setPreview(artifact);
    }).catch(e => { if (live) setError(`Export unavailable: ${e.message} Check the saved Season evidence or create a new season.`); });
    return () => { live = false; };
  }, [season.completedGames, season.entrants, season.seasonSeed, season.gamesPerPairing]);
  async function act(kind) {
    if (busy) return; setBusy(true); setFeedback(''); setError('');
    try {
      // Rebuild and self-verify the current evidence for every offered download.
      const artifact = await createEvaluationExport(season, { sourceCommit: configuredCommit }), check = await verifyEvaluationExport(artifact);
      if (!check.ok) throw new Error(check.errors.map(e => `${e.stage}: ${e.message}`).join(' '));
      if (kind === 'copy') {
        if (!navigator.clipboard?.writeText) throw new Error('Clipboard is unavailable. Select and copy the digest text below.');
        await navigator.clipboard.writeText(artifact.integrity.evidence_sha256); setFeedback('Evidence digest copied.');
      } else {
        const base = `connect4-season-${season.seasonSeed}`;
        if (kind === 'json') downloadText(JSON.stringify(artifact, null, 2) + '\n', `${base}-evaluation.json`, 'application/json;charset=utf-8');
        if (kind === 'games') downloadText(gamesCSV(artifact), `${base}-games.csv`, 'text/csv;charset=utf-8');
        if (kind === 'summary') downloadText(summaryCSV(artifact), `${base}-summary.csv`, 'text/csv;charset=utf-8');
        setFeedback('Verified export download started.');
      }
    } catch (e) { setError(`Export failed: ${e.message}`); } finally { setBusy(false); }
  }
  return <section className="season-panel evaluation-export" aria-labelledby="evaluation-export-heading" aria-busy={busy}>
    <p className="eyebrow">Portable evaluation evidence</p><h2 id="evaluation-export-heading">Evaluation export</h2>
    <p className="evaluation-state">{season.status === 'complete' ? 'Complete' : 'Partial'} evaluation — {season.completedGames.length} of {season.schedule.length} games complete.</p>
    {season.status !== 'complete' && <p className="analysis-helper">Standings and Elo describe completed games so far; they are provisional.</p>}
    <dl className="evaluation-versions"><div><dt>Schema</dt><dd>{SCHEMA_VERSION}</dd></div><div><dt>Methodology</dt><dd>{METHODOLOGY_VERSION}</dd></div><div><dt>Schedule</dt><dd>{SCHEDULE_VERSION}</dd></div><div><dt>UI source commit</dt><dd>{configuredCommit ?? 'Not configured'}</dd></div></dl>
    <p id="evidence-digest-label">Evidence SHA-256</p><code className="evidence-digest" aria-labelledby="evidence-digest-label">{preview?.integrity.evidence_sha256 ?? (error ? 'Digest unavailable' : 'Validating evidence…')}</code>
    <div className="tournament-actions evaluation-actions">{[['json', 'Export evaluation JSON'], ['games', 'Export games CSV'], ['summary', 'Export summary CSV'], ['copy', 'Copy evidence digest']].map(([kind, label]) => <button type="button" key={kind} onClick={() => void act(kind)} disabled={!preview || busy}>{label}</button>)}</div>
    <p role="status" aria-live="polite" aria-atomic="true" className="evaluation-feedback">{feedback || (preview ? 'Evidence and recomputed analytics verified locally.' : '')}</p>
    {error && <p className="tournament-error" role="alert">{error}</p>}
    <details className="evaluation-details"><summary>Provenance and integrity details</summary>
      <p>Engine 1 · Random 2 · corrected Negamax 2 · MCTS 2 · API contract 2 · API provenance 1 · stochastic seeding 1.</p>
      <p>Backend versions and optional source commits are captured in each game’s history. {preview ? `${preview.provenance.execution_capture.recorded_games} games captured; ${preview.provenance.execution_capture.unrecorded_games} games unrecorded.` : ''}</p>
      {preview?.provenance.execution_capture.unrecorded_games > 0 && <p>Historical backend provenance was not recorded for some games. Reference version identifiers do not establish which backend played them.</p>}
      {preview && <ul>{[...new Set(preview.evidence.completed_games.map(g => g.backend_provenance?.source_commit).filter(Boolean))].map(commit => <li key={commit}>Backend source commit: <code>{commit}</code></li>)}</ul>}
      <p>SHA-256 covers the schema, provenance, methodology, deterministic schedule and completed move evidence. Timestamp and derived analytics are excluded; analytics are recomputed during verification. The digest detects changes relative to a trusted digest; it is not a signature or proof of agent execution.</p>
    </details>
  </section>;
}
