import { validateSeason } from './model.js';
export { browserStorage } from '../tournament/storage.js';
export const STORAGE_KEY = 'board-game-ai-lab:season:v1';
export function loadSeason(storage) {
  try {
    const raw = storage?.getItem(STORAGE_KEY);
    return { season: raw ? validateSeason(JSON.parse(raw)) : null, error: '', available: Boolean(storage) };
  } catch (error) {
    return { season: null, error: error.name === 'SecurityError' ? '' : 'Saved season could not be validated. Create a new season to reset it.', available: error.name !== 'SecurityError' && Boolean(storage) };
  }
}
export function saveSeason(storage, season) {
  try { if (!storage) return false; storage.setItem(STORAGE_KEY, JSON.stringify(season)); return true; } catch { return false; }
}
