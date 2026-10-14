// Only an explicit busy response proves no move committed. All other failures
// propagate to the owning controller's read-only revision reconciliation.
export async function withBusyRetry(operation, { signal, canRetry = () => true, wait = delay => new Promise(resolve => setTimeout(resolve, delay)) } = {}) {
  for (let attempt = 0; ; attempt++) {
    try { return await operation(); }
    catch (error) {
      const r = error.response;
      if (r?.status !== 503 || !['agent_busy', 'game_busy'].includes(r.data?.code) || attempt >= 2) throw error;
      const header = r.headers?.get?.('retry-after') ?? r.headers?.['retry-after'];
      const seconds = header === undefined ? 0 : /^\d+(?:\.\d+)?$/.test(String(header)) ? Number(header) : Math.max(0, (Date.parse(header) - Date.now()) / 1000);
      // A long/invalid server delay requires explicit continuation instead.
      if (!Number.isFinite(seconds) || seconds > 5) throw error;
      const delay = Math.max([250, 750][attempt], seconds * 1000);
      await wait(delay);
      if (signal?.aborted || !canRetry()) throw Object.assign(new Error('Busy retry cancelled.'), { retryCancelled: true });
    }
  }
}
