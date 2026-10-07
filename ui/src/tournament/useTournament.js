import { useEffect, useState, useSyncExternalStore } from 'react';
import { TournamentController } from './controller.js';
import { browserStorage } from './storage.js';

// Keep the sole mutation owner across SPA route remounts. A departing page may
// still be settling a POST; a new page must observe its lock instead of racing it.
const controllers = new WeakMap();
export function useTournament(http, storage) {
  const [controller] = useState(() => {
    const adapter = storage === undefined ? browserStorage() : storage;
    let adapters = controllers.get(http);
    if (!adapters) { adapters = new Map(); controllers.set(http, adapters); }
    if (!adapters.has(adapter)) adapters.set(adapter, new TournamentController(http, adapter));
    return adapters.get(adapter);
  });
  const state = useSyncExternalStore(controller.subscribe, controller.snapshot);
  useEffect(() => { controller.attach(); return () => controller.detach(); }, [controller]);
  return { ...state, controller };
}
