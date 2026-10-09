import { researchAgentEnabled, assertResearchExecution, researchErrorMessage, VICTOR_RESEARCH } from './researchAgent.js';
import { useEffect, useReducer, useRef } from 'react';
import { DEFAULT_COMPETITOR, matchStartPayload } from './competitorConfig.js';
import { requestMatchPly } from './matchTransport.js';
import { replaySnapshot, validateMatchHistory, matchResult } from './matchRecord.js';

export const PLAYBACK_SPEEDS = { slow: 1600, normal: 850, fast: 300 };
const initial = { game: null, moves: [], viewedRevision: null, phase: 'idle', autoplay: false,
  speed: 'normal', uncertain: false, error: '', expired: false,
  selections: [{ ...DEFAULT_COMPETITOR }, { ...DEFAULT_COMPETITOR, type: 'mcts' }] };
const reducer = (state, patch) => ({ ...state, ...patch });
const isHuman = game => game.players[game.currentPlayer].type === 'human';

export function useConnect4Match(http, researchEnabled = researchAgentEnabled()) {
  const [state, dispatch] = useReducer(reducer, initial);
  const current = useRef(state), session = useRef(null), timer = useRef(null);
  const clearTimer = () => { clearTimeout(timer.current); timer.current = null; };
  useEffect(() => {
    const lifetime = { active: true, abort: new AbortController() };
    session.current = lifetime;
    return () => { clearTimer(); lifetime.active = false; lifetime.abort.abort(); };
  }, [http]);
  function patch(value, lifetime = session.current) {
    if (!lifetime?.active || session.current !== lifetime) return;
    current.current = reducer(current.current, value);
    dispatch(value);
  }
  function pause() { clearTimer(); patch({ autoplay: false }); }

  async function sync(lifetime, options, game) {
    const response = await http.get(`/v1/connect4/games/${game.game_id}/history`, options);
    if (!lifetime.active) return;
    const record = validateMatchHistory(response.data, { game_id: game.game_id, minRevision: game.revision, players: game.players });
    patch({ ...record, uncertain: false, expired: false,
      ...(record.game.gameOver ? { autoplay: false } : {}) }, lifetime);
  }
  async function recover(error, lifetime, options) {
    if (!lifetime.active) return;
    clearTimer();
    const reason = researchErrorMessage(error, current.current.game?.players.find(p => p.type === VICTOR_RESEARCH) ?? current.current.selections.find(p => p.type === VICTOR_RESEARCH)) || error.response?.data?.error || 'The request could not be confirmed. The server may be waking up; wait a moment before continuing.';
    patch({ autoplay: false, error: reason }, lifetime);
    const game = current.current.game;
    if (!game) return; // Lost start IDs cannot be recovered; never replay start.
    patch({ phase: 'refreshing', uncertain: true }, lifetime);
    try { await sync(lifetime, options, game); }
    catch (refreshError) {
      if (refreshError.response?.status === 404) {
        patch({ game: null, moves: [], viewedRevision: null, uncertain: false, expired: true,
          error: 'This match expired or the server restarted. Start a new match.' }, lifetime);
      }
      // Any other GET failure leaves execution locked until explicit Refresh.
    }
  }
  async function run(phase, action) {
    const lifetime = session.current;
    if (!lifetime?.active || current.current.phase !== 'idle') return;
    clearTimer();
    patch({ phase, error: '' }, lifetime); // synchronous lock before any await
    const options = { signal: lifetime.abort.signal };
    try { await action(lifetime, options); }
    catch (error) { await recover(error, lifetime, options); }
    finally { patch({ phase: 'idle' }, lifetime); }
  }
  function start() {
    if (current.current.phase !== 'idle') return;
    pause();
    return run('starting', async (lifetime, options) => {
      const { game, selections } = current.current;
      const body = matchStartPayload(...selections, game?.game_id, researchEnabled);
      const response = await http.post('/v1/connect4/start_game', body, options);
      if (!lifetime.active) return;
      const data = response.data;
      const { game: fresh } = validateMatchHistory({ game_id: data?.game_id, revision: data?.revision,
        players: data?.players, state: data, moves: [] }, { revision: 0, players: [body.player1, body.player2] });
      patch({ game: fresh, moves: [], viewedRevision: null, uncertain: false, expired: false }, lifetime);
      // Match Lab deliberately remains paused at revision 0, including AI-first.
    });
  }
  function playable() {
    const s = current.current;
    return s.game && !s.game.gameOver && !s.uncertain && !s.error && s.viewedRevision === null && s.phase === 'idle';
  }
  function ply(column) {
    if (!playable()) return;
    const game = current.current.game;
    if (isHuman(game) ? !game.legalMoves.includes(column) : column !== undefined) return;
    return run(isHuman(game) ? 'human-move' : 'ai-move', async (lifetime, options) => {
      assertResearchExecution(game.players, researchEnabled);
      const accepted = await requestMatchPly(http, game, column, options);
      if (!lifetime.active) return;
      // Retain the accepted revision even if the subsequent history read fails.
      patch({ game: accepted, uncertain: true }, lifetime);
      await sync(lifetime, options, accepted);
    });
  }
  function nextMove() {
    if (current.current.autoplay || !current.current.game || isHuman(current.current.game)) return;
    return ply();
  }
  function autoplay() {
    if (playable()) patch({ autoplay: true });
  }
  function refresh() {
    return run('refreshing', async (lifetime, options) => {
      const game = current.current.game;
      if (game) await sync(lifetime, options, game);
      // Refresh is read-only. Continuing/stepping is a separate explicit action.
    });
  }
  function review(revision) {
    const s = current.current;
    if (!s.game || !Number.isInteger(revision) || revision < 0 || revision > s.moves.length) return;
    pause();
    patch({ viewedRevision: revision === s.game.revision ? null : revision });
  }
  function returnLive() { pause(); patch({ viewedRevision: null }); }
  function select(index, value) {
    if (current.current.phase !== 'idle') return;
    patch({ selections: current.current.selections.map((config, i) => i === index ? (value.type === VICTOR_RESEARCH ? { type: VICTOR_RESEARCH } : { ...DEFAULT_COMPETITOR, ...config, ...value }) : config) });
  }
  function setSpeed(speed) { if (Object.hasOwn(PLAYBACK_SPEEDS, speed) && current.current.speed !== speed) { clearTimer(); patch({ speed }); } }

  useEffect(() => {
    clearTimer();
    if (state.autoplay && playable() && !isHuman(state.game)) {
      // A timer boundary between every accepted ply, never a recursive POST loop.
      timer.current = setTimeout(() => {
        timer.current = null;
        if (current.current.autoplay) void ply();
      }, PLAYBACK_SPEEDS[state.speed]);
    }
    return clearTimer;
  }, [state.autoplay, state.phase, state.game, state.uncertain, state.error, state.viewedRevision, state.speed]);

  const live = state.viewedRevision === null;
  // History may be temporarily older while an accepted ply's GET is pending.
  const displayedGame = state.game && !live ? replaySnapshot(state.game, state.moves, state.viewedRevision) : state.game;
  return { ...state, researchEnabled, live, displayedGame, result: matchResult(state.game), busy: state.phase !== 'idle',
    humanTurn: Boolean(state.game && isHuman(state.game)),
    interactive: Boolean(live && playable() && isHuman(state.game)),
    start, move: ply, nextMove, autoplay: state.autoplay, enableAutoplay: autoplay, pause, refresh,
    review, returnLive, select, setSpeed };
}
