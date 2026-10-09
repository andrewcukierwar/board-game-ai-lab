import { assertResearchExecution, researchAgentEnabled, researchErrorMessage, VICTOR_RESEARCH } from '../connect4/researchAgent.js';
import { requestMatchPly } from '../connect4/matchTransport.js';
import { validateMatchHistory } from '../connect4/matchRecord.js';
import { PLAYBACK_SPEEDS } from '../connect4/useConnect4Match.js';
import { createSeason, currentFixture, gamePlan, compactHistory, recordGame, replayColumns } from './model.js';
import { loadSeason, saveSeason } from './storage.js';

const activeCopy = s => ({ ...s, active: s.active && { ...s.active }, schedule: s.schedule.map((f, i) => i === s.currentGameIndex ? { ...f } : f) });

// One mutation owner survives route remounts. Never abort or retry a mutation.
export class SeasonController {
  constructor(http, storage, schedule = (fn, delay) => setTimeout(fn, delay), cancel = id => clearTimeout(id), researchEnabled = researchAgentEnabled()) {
    this.researchEnabled = researchEnabled; this.http = http; this.storage = storage; this.schedule = schedule; this.cancel = cancel;
    const saved = loadSeason(storage);
    this.state = { season: saved.season, error: saved.error, storageAvailable: saved.available, busy: false,
      uncertain: Boolean(saved.season?.active), mode: 'paused', speed: 'normal', waiting: false, reviewing: false };
    this.listeners = new Set(); this.timer = null; this.attached = false; this.minRevision = 0;
    this.snapshot = () => this.state;
    this.subscribe = listener => { this.listeners.add(listener); return () => this.listeners.delete(listener); };
  }
  patch(patch) { this.state = { ...this.state, ...patch }; for (const listener of this.listeners) listener(); }
  persist(season) { this.patch({ season, storageAvailable: saveSeason(this.storage, season) }); }
  attach() { this.attached = true; if (!this.state.busy && this.state.season?.active && this.state.season.active.status !== 'interrupted') void this.refresh(); }
  detach() { this.attached = false; this.pause(); }
  clearTimer() { if (this.timer !== null) this.cancel(this.timer); this.timer = null; }
  pause() { this.clearTimer(); this.patch({ mode: 'paused' }); }
  review() { this.pause(); this.patch({ reviewing: true }); }
  returnLive() { this.patch({ reviewing: false }); }
  create(configs, seed, games) {
    if (this.state.busy) return;
    const fresh = createSeason(configs, seed, games, undefined, this.researchEnabled);
    fresh.retainedGameId = this.state.season?.active?.gameId ?? this.state.season?.retainedGameId ?? null;
    this.pause(); this.minRevision = 0; this.persist(fresh);
    this.patch({ error: '', uncertain: false, waiting: false, reviewing: false });
  }
  setSpeed(speed) { if (!Object.hasOwn(PLAYBACK_SPEEDS, speed)) return; this.clearTimer(); this.patch({ speed }); this.queue(); }
  async locked(action) {
    if (this.state.busy) return;
    this.clearTimer(); this.patch({ busy: true, error: '' });
    try { await action(); } catch (error) { await this.recover(error); }
    finally { this.patch({ busy: false, waiting: false }); this.queue(); }
  }
  interrupted(message) {
    const s = activeCopy(this.state.season);
    if (s?.active) { s.active.status = 'interrupted'; currentFixture(s).status = 'interrupted'; this.persist(s); }
    this.patch({ uncertain: true, waiting: false, error: message }); this.pause();
  }
  async recover(error) {
    this.pause();
    const a = this.state.season?.active;
    const configs = a ? gamePlan(this.state.season).playerConfigs : [];
    const selection = a?.gameId ? configs[a.columns.length % 2] : configs.find(p => p.type === VICTOR_RESEARCH);
    const reason = researchErrorMessage(error, selection, true) || error.response?.data?.error || error.message || 'Request could not be confirmed.';
    this.patch({ error: reason, uncertain: true, waiting: true });
    if (!a) { this.patch({ uncertain: false, waiting: false }); return; }
    if (!a?.gameId || a.status === 'starting') {
      this.interrupted(reason + ' Start could not be confirmed. Explicitly restart this seeded fixture; an unknown session may expire normally.'); return;
    }
    try { await this.sync(); this.patch({ error: reason }); }
    catch (readError) {
      if (readError.response?.status === 404) this.interrupted('The active session expired or the server restarted. Completed results are safe. Explicitly restart this seeded fixture.');
      else this.patch({ uncertain: true, error: `${reason} Refresh history before continuing.` });
    }
  }
  async sync() {
    const s = this.state.season, a = s.active;
    if (!a?.gameId) throw new Error('This fixture needs an explicit seeded restart.');
    const plan = gamePlan(s), response = await this.http.get(`/v1/connect4/games/${a.gameId}/history`);
    const validated = validateMatchHistory(response.data, { game_id: a.gameId, minRevision: Math.max(a.columns.length, this.minRevision), players: plan.playerConfigs, rng_seed: plan.gameSeed });
    if (!a.columns.every((col, i) => validated.moves[i]?.column === col)) throw new Error('Previously validated moves changed.');
    const copy = activeCopy(this.state.season);
    copy.active.columns = validated.moves.map(m => m.column); copy.active.status = 'running'; currentFixture(copy).status = 'active';
    this.persist(copy); this.patch({ uncertain: false }); this.minRevision = validated.game.revision;
    if (validated.game.gameOver) {
      const finished = recordGame(copy, compactHistory(copy, response.data)); finished.retainedGameId = a.gameId;
      this.persist(finished); this.minRevision = 0;
      if (finished.status === 'complete' || this.scope?.mode === 'game' || this.scope?.mode === 'round' && currentFixture(finished)?.round !== this.scope.round) this.pause();
    }
  }
  async startGame(restart = false) {
    const s = this.state.season, f = currentFixture(s);
    if (!f || s.active && !restart) return;
    const plan = gamePlan(s), replaceId = s.active?.gameId ?? s.retainedGameId, copy = activeCopy(s);
    assertResearchExecution(plan.playerConfigs, this.researchEnabled);
    copy.active = { fixtureId: f.fixtureId, gameId: null, status: 'starting', columns: [] }; currentFixture(copy).status = 'active';
    this.persist(copy); this.minRevision = 0;
    const response = await this.http.post('/v1/connect4/start_game', { player1: plan.playerConfigs[0], player2: plan.playerConfigs[1], rng_seed: plan.gameSeed, ...(replaceId ? { replace_game_id: replaceId } : {}) });
    if (typeof response.data?.game_id !== 'string' || !response.data.game_id || response.data.game_id.length > 64) throw new Error('Start did not return a game ID.');
    const accepted = activeCopy(this.state.season);
    accepted.active.gameId = response.data.game_id; accepted.active.status = 'running'; accepted.retainedGameId = response.data.game_id;
    this.persist(accepted); this.patch({ uncertain: true });
    validateMatchHistory({ game_id: response.data.game_id, revision: response.data.revision, players: response.data.players, state: response.data, moves: [], rng_seed: plan.gameSeed }, { revision: 0, players: plan.playerConfigs, rng_seed: plan.gameSeed });
    await this.sync();
  }
  watch() {
    if (!this.state.season || this.state.uncertain || this.state.error) return;
    this.pause(); this.returnLive(); return this.locked(() => this.startGame());
  }
  restart() {
    if (this.state.season?.active?.status !== 'interrupted') return;
    this.pause(); this.returnLive(); return this.locked(() => this.startGame(true));
  }
  refresh() {
    if (!this.state.season?.active) {
      // A lost terminal move can reconcile into a completed record. Acknowledge
      // that confirmed result explicitly before starting the next fixture.
      if (!this.state.uncertain) this.patch({ error: '' });
      return;
    }
    this.pause();
    if (this.state.season?.active?.status === 'starting' || this.state.season?.active?.status === 'interrupted' && !this.state.season.active.gameId) {
      this.interrupted('Start was interrupted. Explicitly restart this seeded fixture.'); return;
    }
    return this.locked(async () => { this.patch({ waiting: true }); await this.sync(); });
  }
  async ply() {
    const s = this.state.season, a = s?.active;
    if (!a || a.status !== 'running' || this.state.uncertain || this.state.error) return;
    const { game } = replayColumns(a.columns, gamePlan(s).playerConfigs, a.gameId);
    if (!game.gameOver) { assertResearchExecution(game.players, this.researchEnabled); const accepted = await requestMatchPly(this.http, game); this.minRevision = accepted.revision; this.patch({ uncertain: true }); }
    await this.sync();
  }
  nextMove() {
    if (this.state.reviewing || this.state.mode !== 'paused' || !this.state.season?.active || this.state.uncertain || this.state.error) return;
    return this.locked(() => this.ply());
  }
  run(mode) {
    if (!['game', 'round', 'season'].includes(mode) || this.state.busy || this.state.uncertain || this.state.error || !this.state.season) return;
    const f = currentFixture(this.state.season); if (!f) return;
    this.scope = { mode, round: f.round }; this.patch({ mode, reviewing: false }); this.queue();
  }
  queue() {
    if (!this.attached || this.timer !== null || this.state.mode === 'paused' || this.state.busy || this.state.uncertain || this.state.error) return;
    if (!currentFixture(this.state.season)) { this.pause(); return; }
    this.timer = this.schedule(() => {
      this.timer = null;
      if (this.state.mode !== 'paused') void this.locked(() => this.state.season.active ? this.ply() : this.startGame());
    }, PLAYBACK_SPEEDS[this.state.speed]);
  }
}
