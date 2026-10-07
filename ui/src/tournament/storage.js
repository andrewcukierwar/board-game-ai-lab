import { validateTournament } from './model.js';
export const STORAGE_KEY = 'board-game-ai-lab:tournament:v1';
export function browserStorage() { try { return globalThis.localStorage ?? globalThis.window?.localStorage; } catch { return null; } }
export function loadTournament(storage) {
  try {
    const raw = storage?.getItem(STORAGE_KEY);
    return { tournament: raw ? validateTournament(JSON.parse(raw)) : null, error: '', available: Boolean(storage) };
  } catch (error) {
    if (error.name === 'SecurityError') return { tournament: null, error: '', available: false };
    return { tournament: null, error: 'Saved tournament could not be validated. Create a new tournament to reset it.', available: Boolean(storage) };
  }
}
export function saveTournament(storage, tournament) {
  try { if (!storage) return false; storage.setItem(STORAGE_KEY, JSON.stringify(tournament)); return true; } catch { return false; }
}
