export const provenance = {
  version: 1, type: 'alphazero', agent_version: 1,
  model: { id: 'az2-hc-g6-cnn-s42-v1', architecture: 'connect4-conv64-128-128-policy7-scalar-tanh-v2', parameters: 325896,
    artifact_sha256: '4c3cf26aa58495f43094436f231486969658d70b94b246a912a5bb54756dc528',
    source_checkpoint_sha256: 'ba2bde2ec71b3ac7486046d1cc26ab591bb59d6b764193ece08c6d6a6f24f58b' },
  search: { algorithm: 'puct', simulations: 256, exploration: 1.41, root_noise: false, final_selection: 'maximum-visit', deterministic: true },
  tie_break: 'connect4-alphazero-production-tie-break-v1', rng_seed: 'ignored',
  tactical_guard: { implementation: 'connect4-tactical-root-guard-v1', source_sha256: 'd2ee54ba1fe9ddbcc4641538e2f5dd89a2435234ff194d6909bed72096632d04' },
};
export const status = { type: 'alphazero', enabled: true, available: true, provenance,
  agent_version: 1, model: provenance.model, search: provenance.search,
  options: { tactical_guard: { type: 'boolean', default: false } } };
export const az = tactical_guard => ({ type: 'alphazero', tactical_guard });
