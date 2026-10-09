import { canonicalJSON, evidenceDigest } from './canonical.js';
import { FORMAT, SCHEMA_VERSION, RESEARCH_SCHEMA_VERSION, sourceCommit } from './provenance.js';
import { evaluationSchema, evaluationMethodology, evidenceHasResearch, canonicalPayload, reconstructEvidence, derivedAnalytics, evaluationProvenance } from './export.js';

function compareDerived(actual, expected, path = 'derived') {
  // Engine-specific exponent rounding may differ at sub-nanopoint precision.
  if (typeof actual === 'number' && typeof expected === 'number' && path.startsWith('derived.ratings') && Math.abs(actual - expected) <= 1e-9) return;
  if (actual === expected) return;
  if (actual === null || expected === null || typeof actual !== typeof expected || typeof actual !== 'object' || Array.isArray(actual) !== Array.isArray(expected)) throw new Error(`${path} does not match recomputed analytics.`);
  const keys = Object.keys(expected).sort();
  if (canonicalJSON(Object.keys(actual).sort()) !== canonicalJSON(keys)) throw new Error(`${path} has missing or unknown fields.`);
  for (const key of keys) compareDerived(actual[key], expected[key], `${path}.${key}`);
}
export async function verifyEvaluationExport(artifact, { digest } = {}) {
  let stage = 'schema';
  try {
    canonicalJSON(artifact); // Reject non-JSON values even outside the hashed payload.
    if (artifact?.format !== FORMAT || ![SCHEMA_VERSION, RESEARCH_SCHEMA_VERSION].includes(artifact.schema_version)) throw new Error('Unsupported evaluation format/schema version.');
    const keys = ['format', 'schema_version', 'exported_at', 'provenance', 'methodology', 'evidence', 'derived', 'integrity'].sort();
    if (canonicalJSON(Object.keys(artifact).sort()) !== canonicalJSON(keys)) throw new Error('Evaluation envelope has missing or unknown fields.');
    if (typeof artifact.exported_at !== 'string' || !/^\d{4}-\d{2}-\d{2}T.*Z$/.test(artifact.exported_at) || !Number.isFinite(Date.parse(artifact.exported_at))) throw new Error('Invalid ISO-8601 UTC exported_at.');
    stage = 'methodology';
    if (canonicalJSON(artifact.methodology) !== canonicalJSON(evaluationMethodology(artifact.evidence))) throw new Error('Unsupported or modified evaluation methodology.');
    stage = 'evidence';
    const s = reconstructEvidence(artifact.evidence);
    if (artifact.schema_version !== evaluationSchema(artifact.evidence)) throw new Error('Victor evidence requires evaluation schema v2; ordinary evidence uses frozen v1.');
    stage = 'provenance';
    const p = artifact.provenance;
    if (!p || !(p.source_commit === null || sourceCommit(p.source_commit) === p.source_commit) || canonicalJSON(p) !== canonicalJSON(evaluationProvenance(artifact.evidence, p.source_commit))) throw new Error('Malformed, unsupported or inconsistent implementation provenance.');
    stage = 'integrity';
    const hash = await evidenceDigest(canonicalPayload(artifact), digest);
    if (canonicalJSON(artifact.integrity) !== canonicalJSON({ algorithm: 'SHA-256', canonicalization: 'sorted-json-ecmascript-v1',
      coverage: 'format,schema_version,provenance,methodology,evidence', evidence_sha256: hash })) throw new Error('SHA-256 evidence digest or integrity contract does not match.');
    stage = 'analytics'; compareDerived(artifact.derived, derivedAnalytics(s));
    return { ok: true, errors: [], warnings: [...(p.execution_capture.unrecorded_games ? ['Some games lack captured backend provenance; declared versions cannot identify their historical backend.'] : []), ...(evidenceHasResearch(artifact.evidence) ? ['Victor uses runtime-dependent wall-clock budgets. Core backend provenance does not identify the Victor algorithm; its version is a declared reference, not execution authentication. Recorded moves do not guarantee identical future choices.'] : [])],
      summary: { schema: `${FORMAT}/v${artifact.schema_version}`, digest: hash, field_size: s.fieldSize,
        completed_games: s.completedGames.length, scheduled_games: s.schedule.length, status: artifact.evidence.state.status, provenance: p } };
  } catch (error) { return { ok: false, errors: [{ stage, message: error.message }], warnings: [] }; }
}
