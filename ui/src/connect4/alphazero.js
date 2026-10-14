import { canonicalJSON } from '../evaluation/canonical.js';

export const ALPHAZERO = 'alphazero';
export const alphaZeroAgent = { type: ALPHAZERO, name: 'AlphaZero (Neural MCTS)', label: 'Experimental',
  description: 'Experimental neural-network agent trained through self-play. Uses 256 PUCT simulations per move. It is not an optimal Connect 4 solver.' };
export const ALPHAZERO_CAVEAT = 'AlphaZero uses the same CNN with safeguard off or on. Its tie-breaking depends only on the exact position; game seeds do not change its choices and do not reproduce historical research RNG streams.';
export const hasAlphaZero = configs => configs.some(c => c?.type === ALPHAZERO);
export const validAlphaZeroConfig = c => c?.type === ALPHAZERO && typeof c.tactical_guard === 'boolean' &&
  canonicalJSON(Object.keys(c).sort()) === canonicalJSON(['tactical_guard', 'type']);
const hash = s => typeof s === 'string' && /^[0-9a-f]{64}$/.test(s);
const text = s => typeof s === 'string' && s.length > 0 && s.length <= 200;
export function validateAlphaZeroProvenance(p) {
  if (!p || p.version !== 1 || p.type !== ALPHAZERO || p.agent_version !== 1 ||
      !text(p.model?.id) || !text(p.model?.architecture) || p.model.parameters !== 325896 ||
      !hash(p.model.artifact_sha256) || !hash(p.model.source_checkpoint_sha256) ||
      p.search?.algorithm !== 'puct' || p.search.simulations !== 256 || p.search.exploration !== 1.41 ||
      p.search.root_noise !== false || p.search.final_selection !== 'maximum-visit' || p.search.deterministic !== true ||
      p.tie_break !== 'connect4-alphazero-production-tie-break-v1' || p.rng_seed !== 'ignored' ||
      p.tactical_guard?.implementation !== 'connect4-tactical-root-guard-v1' || !hash(p.tactical_guard.source_sha256)) {
    throw new Error('Unsupported or incomplete AlphaZero model/search provenance. Update the verifier before continuing.');
  }
  const normalized = { version: 1, type: ALPHAZERO, agent_version: p.agent_version,
    model: { id: p.model.id, architecture: p.model.architecture, parameters: p.model.parameters,
      artifact_sha256: p.model.artifact_sha256, source_checkpoint_sha256: p.model.source_checkpoint_sha256 },
    search: { algorithm: 'puct', simulations: 256, exploration: 1.41, root_noise: false, final_selection: 'maximum-visit', deterministic: true },
    tie_break: p.tie_break, rng_seed: 'ignored',
    tactical_guard: { implementation: p.tactical_guard.implementation, source_sha256: p.tactical_guard.source_sha256 } };
  if (canonicalJSON(p) !== canonicalJSON(normalized)) throw new Error('Unknown AlphaZero provenance fields.');
  return normalized;
}
export function validateAlphaZeroStatus(s) {
  if (!s || s.type !== ALPHAZERO || typeof s.enabled !== 'boolean' || typeof s.available !== 'boolean' || s.available && !s.enabled) throw new Error('Malformed AlphaZero availability response.');
  if (!s.available) return null;
  const p = validateAlphaZeroProvenance(s.provenance);
  if (s.agent_version !== p.agent_version || !s.model ||
      Object.keys(p.model).some(k => s.model[k] !== p.model[k]) || canonicalJSON(s.search) !== canonicalJSON(p.search) ||
      s.options?.tactical_guard?.type !== 'boolean' || s.options.tactical_guard.default !== false) throw new Error('Inconsistent AlphaZero availability response.');
  return p;
}
export function sameAlphaZero(a, b) { return canonicalJSON(validateAlphaZeroProvenance(a)) === canonicalJSON(validateAlphaZeroProvenance(b)); }
export function requireSameAlphaZero(expected, actual) {
  try {
    if (!sameAlphaZero(expected, actual)) throw new Error('AlphaZero model or implementation changed. Completed results are safe. Restore the original backend model, or create a new competition; historical results remain reviewable.');
  } catch (error) { error.alphaZeroIntegrity = true; throw error; }
}

// Shared discovery per HTTP client. No polling; mount, focus (60s cache), explicit
// refresh, and AlphaZero starts. A newer refresh supersedes older responses.
// `settled` keeps the last confirmed availability through a recheck, for display only.
const discoveries = new WeakMap();
export function alphaZeroDiscovery(http) {
  if (discoveries.has(http)) return discoveries.get(http);
  let state = { available: false, checking: true, settled: false, provenance: null, reason: 'Checking AlphaZero availability…', checkedAt: 0 }, sequence = 0, pending = null;
  const listeners = new Set();
  const publish = next => { state = next; for (const fn of listeners) fn(); };
  const store = { snapshot: () => state, subscribe: fn => { listeners.add(fn); return () => listeners.delete(fn); },
    invalidate: () => { sequence++; pending = null; publish({ available: false, checking: false, settled: false, provenance: null, reason: 'AlphaZero availability must be checked again.', checkedAt: 0 }); },
    async refresh(force = false) {
      if (!force && pending) return pending;
      if (!force && Date.now() - state.checkedAt < 60000) return state;
      const id = ++sequence;
      // Fail closed while rechecking, retaining no selectable stale model.
      publish({ ...state, available: false, checking: true, reason: 'Checking AlphaZero availability…' });
      const task = (async () => {
        let next;
        try {
          const { data } = await http.get('/v1/connect4/agents/alphazero', { timeout: 8000 });
          const provenance = validateAlphaZeroStatus(data);
          next = { available: Boolean(provenance), provenance, reason: provenance ? '' : data.enabled ? 'AlphaZero is unavailable on this backend.' : 'AlphaZero is disabled on this backend.' };
        } catch (error) { next = { available: false, provenance: null, reason: error.response?.status === 404 ? 'AlphaZero is not supported by this backend.' : 'AlphaZero availability could not be confirmed.' }; }
        if (id === sequence) { publish({ ...next, checking: false, settled: next.available, checkedAt: Date.now() }); pending = null; }
        return state;
      })();
      pending = task; return task;
    } };
  discoveries.set(http, store); return store;
}
export async function requireAlphaZero(http, configs, expected) {
  if (!hasAlphaZero(configs)) return null;
  const status = await alphaZeroDiscovery(http).refresh(true);
  if (!status.available) throw Object.assign(new Error(status.reason + ' Completed results are safe. Refresh availability, then explicitly continue with the original model or start a new competition.'), { alphaZeroAvailability: true });
  if (expected) requireSameAlphaZero(expected, status.provenance);
  return status.provenance;
}
export function capturedAlphaZero(http, configs) {
  if (!hasAlphaZero(configs)) return undefined;
  const status = alphaZeroDiscovery(http).snapshot();
  if (!status.available) throw new Error(status.reason || 'AlphaZero is unavailable. Refresh availability before creating the competition.');
  return validateAlphaZeroProvenance(status.provenance);
}
export function alphaZeroErrorMessage(error, config) {
  if (config?.type !== ALPHAZERO) return null;
  const code = error.response?.data?.code;
  if (['invalid_agent', 'agent_unavailable'].includes(code)) return 'AlphaZero is unavailable on this backend. Completed results are safe. Refresh availability and history before continuing; no competitor was replaced.';
  if (code === 'agent_failed') return 'AlphaZero could not complete its search. Refresh history before explicitly retrying; completed results are safe.';
  if (code === 'agent_busy') return 'AlphaZero is busy. Bounded retries ended; refresh history before explicitly continuing.';
  return null;
}
