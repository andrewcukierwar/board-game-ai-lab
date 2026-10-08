import { deriveSeed } from './model.js';
export const INITIAL_ELO = 1500, K_FACTOR = 24, BOOTSTRAP_SAMPLES = 1000, MIN_INTERVAL_GAMES = 8;
export const scoreFor = (g, id) => g.result.status === 'draw' ? .5 : (g.result.winnerIndex === 0 ? g.redEntrantId : g.yellowEntrantId) === id ? 1 : 0;
const blank = () => ({ played: 0, wins: 0, draws: 0, losses: 0, points: 0, scoreRate: 0 });
function tally(row, score) { row.played++; row[score === 1 ? 'wins' : score === .5 ? 'draws' : 'losses']++; row.points += score; row.scoreRate = row.points / row.played; }
export function seasonStandings(s) {
  const rows = s.entrants.map(e => ({ ...e, ...blank(), red: blank(), yellow: blank() }));
  const byId = new Map(rows.map(r => [r.entrantId, r]));
  for (const g of s.completedGames) for (const [side, id] of [['red', g.redEntrantId], ['yellow', g.yellowEntrantId]]) {
    const row = byId.get(id), score = scoreFor(g, id); tally(row, score); tally(row[side], score);
  }
  return rows.sort((a, b) => b.points - a.points || b.scoreRate - a.scoreRate || a.seedNumber - b.seedNumber);
}
export const sideSplits = s => seasonStandings(s).map(({ entrantId, red, yellow }) => ({ entrantId, red, yellow }));
export function pairwiseResults(s) {
  const matrix = Object.fromEntries(s.entrants.map(e => [e.entrantId, Object.fromEntries(s.entrants.filter(o => o.entrantId !== e.entrantId).map(o => [o.entrantId, blank()]))]));
  for (const g of s.completedGames) {
    tally(matrix[g.redEntrantId][g.yellowEntrantId], scoreFor(g, g.redEntrantId));
    tally(matrix[g.yellowEntrantId][g.redEntrantId], scoreFor(g, g.yellowEntrantId));
  }
  return matrix;
}
export const expectedScore = (a, b) => 1 / (1 + 10 ** ((b - a) / 400));
export function updateElo(a, b, score) { const delta = K_FACTOR * (score - expectedScore(a, b)); return [a + delta, b - delta]; }
export function seasonRatings(s) {
  const rows = s.entrants.map(e => ({ ...e, rating: INITIAL_ELO, peak: INITIAL_ELO, low: INITIAL_ELO, history: [INITIAL_ELO] }));
  const byId = new Map(rows.map(r => [r.entrantId, r]));
  for (const g of s.completedGames) {
    const a = byId.get(g.redEntrantId), b = byId.get(g.yellowEntrantId);
    [a.rating, b.rating] = updateElo(a.rating, b.rating, scoreFor(g, a.entrantId));
    for (const row of rows) { row.peak = Math.max(row.peak, row.rating); row.low = Math.min(row.low, row.rating); row.history.push(row.rating); }
  }
  return rows.sort((a, b) => b.rating - a.rating || a.seedNumber - b.seedNumber);
}
export const ratingHistory = s => seasonRatings(s).map(({ entrantId, history }) => ({ entrantId, history }));
// Local domain-separated PRNG; bootstrap cannot disturb the game/global RNG.
function rng(seed) { let state = seed; return () => { state = (state + 0x6d2b79f5) >>> 0; let t = Math.imul(state ^ state >>> 15, 1 | state); t ^= t + Math.imul(t ^ t >>> 7, 61 | t); return ((t ^ t >>> 14) >>> 0) / 0x100000000; }; }
export function scoreRateInterval(s, id) {
  const scores = s.completedGames.filter(g => [g.redEntrantId, g.yellowEntrantId].includes(id)).map(g => scoreFor(g, id));
  if (scores.length < MIN_INTERVAL_GAMES) return null;
  const random = rng(deriveSeed(s.seasonSeed, `season:bootstrap:${id}`)), means = [];
  for (let sample = 0; sample < BOOTSTRAP_SAMPLES; sample++) {
    let sum = 0; for (let i = 0; i < scores.length; i++) sum += scores[Math.floor(random() * scores.length)];
    means.push(sum / scores.length);
  }
  means.sort((a, b) => a - b);
  const percentile = p => { const index = (means.length - 1) * p, lo = Math.floor(index); return means[lo] + (means[Math.ceil(index)] - means[lo]) * (index - lo); };
  return { lower: percentile(.025), upper: percentile(.975), samples: BOOTSTRAP_SAMPLES, played: scores.length };
}
export function seasonAnalytics(s) {
  return { standings: seasonStandings(s), ratings: seasonRatings(s), pairwise: pairwiseResults(s),
    intervals: Object.fromEntries(s.entrants.map(e => [e.entrantId, scoreRateInterval(s, e.entrantId)])) };
}
