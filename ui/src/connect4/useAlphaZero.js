import { useEffect, useSyncExternalStore } from 'react';
import { alphaZeroDiscovery } from './alphazero.js';
export function useAlphaZero(http) {
  const store = alphaZeroDiscovery(http);
  const status = useSyncExternalStore(store.subscribe, store.snapshot);
  useEffect(() => {
    void store.refresh();
    const focus = () => { void store.refresh(); };
    globalThis.window?.addEventListener('focus', focus);
    return () => globalThis.window?.removeEventListener('focus', focus);
  }, [store]);
  return { ...status, refresh: () => store.refresh(true) };
}
