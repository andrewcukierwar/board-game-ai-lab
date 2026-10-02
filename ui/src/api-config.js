// An empty base uses the local /v1 proxy. Deployment values are origins,
// not endpoint paths: the controller already includes /v1/connect4.
export function apiBase(value = '', { development = false, render = false } = {}) {
  if (development) return '';
  const base = value.trim().replace(/\/$/, '');
  if (!base && !render) return '';
  let url;
  try { url = new URL(base); } catch { /* handled below */ }
  if (!url || !['http:', 'https:'].includes(url.protocol) || url.origin !== base ||
      (render && url.protocol !== 'https:')) {
    throw new Error('VITE_API_BASE must be an API origin (e.g. https://your-api.onrender.com), without /v1, credentials, query or fragment. Render builds require HTTPS.');
  }
  return base;
}
