import { canonicalJSON } from './canonical.js';

export const FORMAT = 'connect4-season-evaluation';
export const SCHEMA_VERSION = 1;
export const METHODOLOGY_VERSION = 1;
// v1 remains frozen. v2 identifies the bounded Victor reference policy separately
// from engine 1; the current API captures core provenance, not Victor identity.
export const RESEARCH_SCHEMA_VERSION = 2;
export const RESEARCH_METHODOLOGY_VERSION = 2;
export const VICTOR_REFERENCE_VERSION = 1;
export const BACKEND_VERSIONS = Object.freeze({
  api_provenance_version: 1, api_contract_version: 2, connect4_engine_version: 1,
  agents: Object.freeze({ random: 2, negamax: 2, mcts: 2 }), stochastic_seed_version: 1,
});
export function sourceCommit(value) {
  return typeof value === 'string' && /^(?:[0-9a-f]{40}|[0-9a-f]{64})$/.test(value) ? value : null;
}
export function validateBackendProvenance(value) {
  if (!value || canonicalJSON({ ...value, source_commit: null }) !== canonicalJSON({ ...BACKEND_VERSIONS, source_commit: null }) ||
      !(value.source_commit === null || sourceCommit(value.source_commit) === value.source_commit)) {
    throw new Error('Unsupported or malformed backend implementation provenance. Update the evaluation verifier for this implementation.');
  }
  return { ...BACKEND_VERSIONS, agents: { ...BACKEND_VERSIONS.agents }, source_commit: value.source_commit };
}
