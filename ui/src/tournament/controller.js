import { requestMatchPly } from '../connect4/matchTransport.js';
import { validateMatchHistory } from '../connect4/matchRecord.js';
import { PLAYBACK_SPEEDS } from '../connect4/useConnect4Match.js';
import { createTournament, findMatchup, nextMatchup, gamePlan, compactHistory, recordGame, replayColumns, matchupHasHuman } from './model.js';
import { loadTournament, saveTournament } from './storage.js';

// Sole mutation owner. React is a subscriber; bracket transitions remain pure.
// Pausing/unmounting never aborts a POST. Reconciliation and persistence still finish.
export class TournamentController {
  constructor(http, storage, schedule = (callback, delay) => setTimeout(callback, delay), cancel = timer => clearTimeout(timer)) {
    this.http = http; this.storage = storage; this.schedule = schedule; this.cancel = cancel;
    const saved = loadTournament(storage);
    const interruptedHuman = saved.tournament?.active?.status === 'interrupted' &&
      matchupHasHuman(saved.tournament, findMatchup(saved.tournament, saved.tournament.active.matchupId));
    this.state = { tournament: saved.tournament, error: saved.error || (interruptedHuman ?
      'This game was interrupted. Prior tournament results are safe, but this game must restart from the beginning.' : ''), storageAvailable: saved.available,
      busy: false, uncertain: Boolean(saved.tournament?.active), mode: 'paused', speed: 'normal', waiting: false, waitingForHuman: false, reviewing: false };
    this.listeners = new Set(); this.timer = null; this.attached = false; this.minRevision = 0;
    this.snapshot = () => this.state;
    this.subscribe = listener => { this.listeners.add(listener); return () => this.listeners.delete(listener); };
  }
  patch(patch) { this.state = { ...this.state, ...patch }; for (const listener of this.listeners) listener(); }
  persist(tournament) {
    const available = saveTournament(this.storage, tournament);
    this.patch({ tournament, storageAvailable: available });
  }
  attach() { this.attached = true; if (!this.state.busy && this.state.tournament?.active && this.state.tournament.active.status !== 'interrupted') void this.refresh(); }
  detach() { this.attached = false; this.pause(); }
  clearTimer() { if (this.timer !== null) this.cancel(this.timer); this.timer = null; }
  pause() { this.clearTimer(); this.patch({ mode: 'paused' }); }
  review() { this.pause(); this.patch({ reviewing: true }); }
  returnLive() { this.patch({ reviewing: false }); }
  humanTurn() {
    const t = this.state.tournament, a = t?.active;
    if (!a || a.status !== 'running' || !matchupHasHuman(t, findMatchup(t, a.matchupId))) return false;
    const { game } = replayColumns(a.columns, gamePlan(t, findMatchup(t, a.matchupId)).playerConfigs);
    return !game.gameOver && game.players[game.currentPlayer].type === 'human';
  }
  create(configs, seed) {
    if (this.state.busy) return;
    const fresh = createTournament(configs, seed);
    fresh.retainedGameId = this.state.tournament?.active?.gameId ?? this.state.tournament?.retainedGameId ?? null;
    this.pause(); this.minRevision = 0;
    this.persist(fresh); this.patch({ error: '', uncertain: false, waiting: false, waitingForHuman: false, reviewing: false });
  }
  setSpeed(speed) { if (!Object.hasOwn(PLAYBACK_SPEEDS, speed)) return; this.clearTimer(); this.patch({ speed }); this.queue(); }
  async locked(action) {
    if (this.state.busy) return;
    this.clearTimer(); this.patch({ busy: true, error: '' });
    try { await action(); }
    catch (error) { await this.recover(error); }
    finally { this.patch({ busy: false }); this.queue(); }
  }
  interrupted(message) {
    const t = structuredClone(this.state.tournament);
    if (t?.active) { t.active.status = 'interrupted'; this.persist(t); }
    this.patch({ uncertain: true, waiting: false, error: message }); this.pause();
  }
  async recover(error) {
    this.pause();
    const reason = error.response?.data?.error || error.message || 'Request could not be confirmed.';
    this.patch({ error: reason, uncertain: true, waiting: true });
    const active = this.state.tournament?.active;
    if (!active?.gameId || active.status === 'starting') {
      this.interrupted('Start could not be confirmed. Explicitly restart this seeded game; an unknown session may expire normally.'); return;
    }
    try { await this.sync(); this.patch({ error: reason }); }
    catch (readError) {
      if (readError.response?.status === 404) this.interrupted(matchupHasHuman(this.state.tournament, findMatchup(this.state.tournament, active.matchupId)) ?
        'The server session expired. Prior tournament results are safe, but this game must restart from the beginning.' :
        'The active session expired or the server restarted. Prior results are safe. Explicitly restart this seeded game.');
      else this.patch({ uncertain: true, error: `${reason} Refresh history before continuing.` });
    }
    finally { this.patch({ waiting: false }); }
  }
  async sync() {
    const t = this.state.tournament, a = t.active;
    if (!a?.gameId) throw new Error('This interrupted game needs an explicit restart.');
    const m = findMatchup(t, a.matchupId), plan = gamePlan(t, m);
    const response = await this.http.get(`/v1/connect4/games/${a.gameId}/history`);
    const validated = validateMatchHistory(response.data, { game_id: a.gameId, minRevision: Math.max(a.columns.length, this.minRevision),
      players: plan.playerConfigs, rng_seed: plan.gameSeed });
    // Reconciliation may add plies, but may never change already validated ones.
    if (!a.columns.every((col, i) => validated.moves[i]?.column === col)) throw new Error('Previously validated moves changed.');
    const copy = structuredClone(this.state.tournament);
    copy.active.columns = validated.moves.map(m => m.column); copy.active.status = 'running';
    this.persist(copy); this.patch({ uncertain: false }); this.minRevision = validated.game.revision;
    if (validated.game.gameOver) {
      const compact = compactHistory(copy, findMatchup(copy, a.matchupId), response.data);
      const finished = recordGame(copy, a.matchupId, compact);
      finished.retainedGameId = a.gameId;
      this.persist(finished); this.minRevision = 0; this.patch({ waitingForHuman: false });
      if (matchupHasHuman(copy, findMatchup(copy, a.matchupId)) || finished.status === 'complete' || this.scope?.mode === 'matchup' && findMatchup(finished, this.scope.matchupId).status === 'complete' ||
          this.scope?.mode === 'round' && finished.rounds[this.scope.round].every(m => m.status === 'complete') || this.scope?.mode === 'game') this.pause();
    }
  }
  async startGame(restart = false) {
    const t = this.state.tournament, m = nextMatchup(t);
    if (!m) return;
    if (t.active && !restart) return;
    const plan = gamePlan(t, m), replaceId = t.active?.gameId ?? t.retainedGameId;
    const copy = structuredClone(t);
    copy.active = { matchupId: m.matchupId, gameNumber: plan.gameNumber, gameId: null, status: 'starting', columns: [] };
    findMatchup(copy, m.matchupId).status = 'active'; this.persist(copy); this.minRevision = 0;
    const response = await this.http.post('/v1/connect4/start_game', { player1: plan.playerConfigs[0], player2: plan.playerConfigs[1],
      rng_seed: plan.gameSeed, ...(replaceId ? { replace_game_id: replaceId } : {}) });
    // Preserve the received ID before subsequent reads/validation can fail.
    if (typeof response.data?.game_id !== 'string' || !response.data.game_id || response.data.game_id.length > 64)
      throw new Error('Start did not return a game ID.');
    const accepted = structuredClone(this.state.tournament);
    accepted.active.gameId = response.data.game_id; accepted.active.status = 'running'; accepted.retainedGameId = response.data.game_id;
    this.persist(accepted); this.patch({ uncertain: true });
    validateMatchHistory({ game_id: response.data.game_id, revision: response.data.revision, players: response.data.players,
      state: response.data, moves: [], rng_seed: plan.gameSeed }, { revision: 0, players: plan.playerConfigs, rng_seed: plan.gameSeed });
    await this.sync();
  }
  watch() {
    if (!this.state.tournament || this.state.uncertain || this.state.error) return;
    this.pause(); this.patch({ waitingForHuman: false, reviewing: false }); return this.locked(() => this.startGame());
  }
  restart() {
    if (this.state.tournament?.active?.status !== 'interrupted') return;
    this.pause(); this.patch({ waitingForHuman: false, reviewing: false }); return this.locked(() => this.startGame(true));
  }
  refresh() {
    this.pause();
    if (this.state.tournament?.active?.status === 'starting' || this.state.tournament?.active?.status === 'interrupted' && !this.state.tournament.active.gameId) {
      this.interrupted('Start was interrupted. Explicitly restart this seeded game.'); return;
    }
    return this.locked(async () => { this.patch({ waiting: true }); await this.sync(); this.patch({ waiting: false }); });
  }
  async ply(column) {
    const t = this.state.tournament, a = t?.active;
    if (!a || a.status !== 'running' || this.state.uncertain || this.state.error) return;
    const plan = gamePlan(t, findMatchup(t, a.matchupId));
    const { game } = replayColumns(a.columns, plan.playerConfigs, a.gameId);
    if (game.gameOver) { await this.sync(); return; }
    const human = game.players[game.currentPlayer].type === 'human';
    if (human ? column === undefined || this.state.reviewing || !game.legalMoves.includes(column) : column !== undefined) return;
    const accepted = await requestMatchPly(this.http, game, column);
    this.minRevision = accepted.revision; this.patch({ uncertain: true });
    await this.sync();
  }
  humanMove(column) {
    if (this.state.busy || this.state.uncertain || this.state.error || this.state.reviewing || !this.humanTurn()) return;
    return this.locked(() => this.ply(column));
  }
  nextMove() {
    if (this.humanTurn() || this.state.reviewing || this.state.mode !== 'paused' || !this.state.tournament?.active || this.state.uncertain || this.state.error) return;
    return this.locked(() => this.ply());
  }
  run(mode) {
    if (!['game', 'matchup', 'round', 'tournament'].includes(mode) || this.state.busy || this.state.uncertain || this.state.error) return;
    const m = nextMatchup(this.state.tournament); if (!m) return;
    this.scope = { mode, matchupId: m.matchupId, round: m.round };
    this.patch({ mode, reviewing: false }); this.queue();
  }
  queue() {
    if (!this.attached || this.timer !== null || this.state.mode === 'paused' || this.state.busy || this.state.uncertain || this.state.error) return;
    const next = nextMatchup(this.state.tournament);
    if (!next) { this.pause(); return; }
    if (!this.state.tournament.active && matchupHasHuman(this.state.tournament, next)) {
      this.pause(); this.patch({ waitingForHuman: true }); return;
    }
    // Autoplay stays armed while waiting for a person, but never schedules a Human POST.
    if (this.humanTurn()) return;
    this.timer = this.schedule(() => {
      this.timer = null;
      if (this.state.mode !== 'paused') void this.locked(() => this.state.tournament.active ? this.ply() : this.startGame());
    }, PLAYBACK_SPEEDS[this.state.speed]);
  }
}
