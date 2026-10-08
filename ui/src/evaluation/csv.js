import { competitorLabel } from '../connect4/competitorConfig.js';
import { reconstructEvidence, derivedAnalytics } from './export.js';

export function csvCell(value) {
  if (value === null || value === undefined) return '';
  if (typeof value === 'number') { if (!Number.isFinite(value)) throw new Error('CSV requires finite numbers.'); return String(value); }
  let cell = String(value);
  // Quote alone does not prevent spreadsheet formulas; prefix hostile text with
  // an apostrophe, including leading whitespace/control-character bypasses.
  if (/^[\s\u0000-\u001f]*[=+\-@]/u.test(cell) || /^[\t\r\n]/u.test(cell)) cell = `'${cell}`;
  return /[",\r\n]/.test(cell) ? `"${cell.replaceAll('"', '""')}"` : cell;
}
export const csvRows = (headers, rows) => [headers, ...rows].map(row => row.map(csvCell).join(',')).join('\r\n') + '\r\n';
export function gamesCSV(artifact) {
  const s = reconstructEvidence(artifact.evidence), byId = new Map(s.entrants.map(e => [e.entrantId, e]));
  const headers = ['completion_index', 'fixture_id', 'round', 'cycle', 'red_entrant_id', 'red_label', 'red_type', 'red_depth', 'red_simulations',
    'yellow_entrant_id', 'yellow_label', 'yellow_type', 'yellow_depth', 'yellow_simulations', 'game_seed', 'result', 'winner_entrant_id', 'move_count', 'move_columns'];
  const player = id => { const e = byId.get(id); return [id, competitorLabel(e.config), e.config.type, e.config.depth ?? '', e.config.simulations ?? '']; };
  return csvRows(headers, s.completedGames.map(g => { const f = s.schedule[g.completedIndex];
    return [g.completedIndex, g.fixtureId, f.round, f.cycle, ...player(g.redEntrantId), ...player(g.yellowEntrantId), g.gameSeed, g.result.status,
      g.result.winnerIndex === null ? '' : g.result.winnerIndex === 0 ? g.redEntrantId : g.yellowEntrantId, g.moveCount, g.columns.join('|')]; }));
}
export function summaryCSV(artifact) {
  const s = reconstructEvidence(artifact.evidence), a = derivedAnalytics(s), ratings = new Map(a.ratings.map(r => [r.entrant_id, r]));
  const metrics = ['played', 'wins', 'draws', 'losses', 'points', 'score_rate'], sides = ['played', 'wins', 'draws', 'losses', 'score_rate'];
  const headers = ['entrant_id', 'seed_number', 'agent_config_label', 'evaluation_status', 'completed_games', 'scheduled_games', ...metrics, ...sides.map(k => `red_${k}`), ...sides.map(k => `yellow_${k}`),
    'final_elo', 'elo_change', 'peak_elo', 'low_elo', 'bootstrap_lower', 'bootstrap_upper', 'bootstrap_games'];
  return csvRows(headers, a.standings.map(r => { const e = s.entrants.find(e => e.entrantId === r.entrant_id), elo = ratings.get(r.entrant_id), interval = a.bootstrap_intervals[r.entrant_id];
    return [r.entrant_id, r.seed_number, competitorLabel(e.config), artifact.evidence.state.status, s.completedGames.length, s.schedule.length, ...metrics.map(k => r[k]), ...sides.map(k => r.red[k]), ...sides.map(k => r.yellow[k]),
      elo.final_elo, elo.elo_change, elo.peak_elo, elo.low_elo, interval?.lower ?? '', interval?.upper ?? '', r.played]; }));
}
