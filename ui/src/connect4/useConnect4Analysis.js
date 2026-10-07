import { useLayoutEffect, useRef, useState } from 'react';
import { validateAnalysisResponse } from './analysisValidation.js';

const readyStatus = game => game
  ? 'Choose an analysis action for this board.'
  : 'Analysis becomes available after you start a game.';

export function useConnect4Analysis(http, game, unavailable) {
  const [mode, setMode] = useState('position');
  const [question, setQuestion] = useState('');
  const [column, setColumn] = useState(null);
  const [state, setState] = useState({ phase: 'idle', result: null, status: readyStatus(game) });
  const lifecycle = useRef({ active: false, generation: 0, controller: null, requested: null });
  const snapshot = useRef({ game, unavailable });
  // Layout cleanup invalidates requests before the updated board is painted.
  // There is no gameplay dispatch, lock, or recovery dependency in this hook.
  const invalidate = () => {
    const life = lifecycle.current;
    life.generation++;
    life.controller?.abort();
    life.controller = null;
    life.requested = null;
  };
  useLayoutEffect(() => {
    lifecycle.current.active = true;
    setState({ phase: 'idle', result: null, status: readyStatus(snapshot.current.game) });
    return () => { lifecycle.current.active = false; invalidate(); };
  }, [http]);
  useLayoutEffect(() => {
    const previous = snapshot.current;
    snapshot.current = { game, unavailable };
    if (previous.game?.game_id !== game?.game_id || previous.game?.revision !== game?.revision ||
        (unavailable && !previous.unavailable)) {
      invalidate();
      setState({ phase: 'idle', result: null, status: readyStatus(game) });
    }
  }, [game, unavailable, http]);

  const legalColumns = game?.gameOver ? [] : (game?.legalMoves || []).filter(c => Number.isInteger(c) && c >= 0 && c < 7);
  const hypotheticalColumn = legalColumns.includes(column) ? column : (legalColumns[0] ?? null);
  const loading = state.phase === 'loading';
  const disabled = !game || unavailable || loading;

  async function request(nextMode) {
    const life = lifecycle.current;
    const { game: currentGame, unavailable: blocked } = snapshot.current;
    if (!life.active || !currentGame || blocked || !['last_move', 'position', 'what_if'].includes(nextMode)) return;
    if (nextMode === 'what_if' && (currentGame.gameOver ||
        !Number.isInteger(hypotheticalColumn) || !currentGame.legalMoves.includes(hypotheticalColumn))) return;
    if (question.length > 500) {
      invalidate();
      setState({ phase: 'error', result: null, status: 'Keep the question to 500 characters or fewer. Gameplay remains available.' });
      return;
    }
    const requested = { game_id: currentGame.game_id, revision: currentGame.revision, mode: nextMode, question };
    if (nextMode === 'what_if') requested.column = hypotheticalColumn;
    // Same-tick duplicate events cannot issue identical requests. A different
    // request replaces the previous one with both abort and generation protection.
    if (life.controller && JSON.stringify(life.requested) === JSON.stringify(requested)) return;
    invalidate();
    const generation = life.generation;
    const controller = new AbortController();
    life.controller = controller;
    life.requested = requested;
    setMode(nextMode);
    setState({ phase: 'loading', result: null,
      status: 'Preparing grounded analysis… You can still play or restart.' });
    const isCurrent = () => life.active && generation === life.generation &&
      !snapshot.current.unavailable && snapshot.current.game?.game_id === requested.game_id &&
      snapshot.current.game?.revision === requested.revision;
    try {
      const response = await http.post('/v1/connect4/explain', requested, { signal: controller.signal });
      if (!isCurrent()) return;
      const result = validateAnalysisResponse(response.data, requested);
      setState({ phase: 'success', result,
        status: `${nextMode === 'what_if' ? `Hypothetical Column ${requested.column + 1}; ` : ''}Board revision ${result.revision}${result.cached ? ' (cached)' : ''}.` });
    } catch (error) {
      if (!isCurrent()) return;
      const message = error.response?.data?.error;
      setState({ phase: 'error', result: null, status:
        (typeof message === 'string' && message ? message : 'The analysis could not be loaded. Try an analysis action again.') +
        ' Gameplay remains available.' });
    } finally {
      if (isCurrent()) { life.controller = null; life.requested = null; }
    }
  }

  // Also mask results during the render that precedes snapshot invalidation.
  const result = !unavailable && state.result?.game_id === game?.game_id &&
    state.result?.revision === game?.revision ? state.result : null;
  return { mode, question, setQuestion, column: hypotheticalColumn, setColumn, legalColumns,
    loading, disabled, phase: state.phase, status: state.status, result,
    highlights: result?.explanation.relevant_squares || [], request,
    lastMoveLabel: game?.revision > 0 && game.players[1 - game.currentPlayer].type !== 'human'
      ? 'Analyze Last AI Move' : 'Analyze Last Move' };
}
