import sys
sys.path.insert(0, sys.argv[0].rsplit('/', 1)[0])
import mutate as M
V2 = 'games/connect4/alphazero_v2/'
M.MUTATIONS = [
    ('tt_always_exact', V2 + 'reference_negamax.py', '        flag = UPPER if best <= alpha_original else LOWER if best >= beta else EXACT', '        flag = EXACT'),
    ('tt_ignore_bounds_type', V2 + 'reference_negamax.py', '            if flag == EXACT:\n                table.hits += 1\n                return value', '            if True:\n                table.hits += 1\n                return value'),
    ('terminal_not_dominant', V2 + 'reference_negamax.py', '        return -(WIN_SCORE + depth)\n    if state.count == WIDTH * HEIGHT:\n        return 0\n    if depth == 0:\n        return state.heuristic()\n    key', '        return -50\n    if state.count == WIDTH * HEIGHT:\n        return 0\n    if depth == 0:\n        return state.heuristic()\n    key'),
    ('draw_not_zero', V2 + 'reference_negamax.py', '    if state.count == WIDTH * HEIGHT:\n        return 0\n    if depth == 0:\n        return state.heuristic()\n    key', '    if state.count == WIDTH * HEIGHT:\n        return state.heuristic()\n    if depth == 0:\n        return state.heuristic()\n    key'),
    ('negamax_first_move_ties', V2 + 'reference_negamax.py', '        return tied[0] if self.rng is None else self.rng.choice(tied)', '        return tied[0]'),
    ('solver_bounds_exact', V2 + 'oracle.py', '        if best <= alpha_original:\n            upper = min(upper, best)\n        elif best >= beta_original:\n            lower = max(lower, best)\n        else:\n            lower = upper = best', '        lower = upper = best'),
    ('solver_last_move_draw_wrong', V2 + 'oracle.py', '            return 0  # the last move does not win, so the board fills: draw', '            return 1'),
    ('solver_forced_ignored', V2 + 'oracle.py', '            if forced & (forced - 1):\n                return -1', '            if False:\n                return -1\n            possible = possible'),
    ('scan_safe_ignores_win', V2 + 'oracle.py', '        if move in winning_moves_scan(moves) or len(after) == 42 or not winning_moves_scan(after):', '        if len(after) == 42 or not winning_moves_scan(after):'),
    ('family_no_reflection', V2 + 'oracle.py', '    return min(repr(key), repr(mirrored))', '    return repr(key)'),
    ('bootstrap_by_game', V2 + 'statistics.py', '    clusters = [tuple(v) for v in by_family.values()]\n    low, high = cluster_bootstrap(clusters, _score, resamples=resamples, seed=seed)',
     '    clusters = [(RESULT_POINTS[g["result"]], 1) for g in games]\n    low, high = cluster_bootstrap(clusters, _score, resamples=resamples, seed=seed)'),
    ('gate_lower_inclusive', V2 + 'statistics.py', '                  lower_bound=arena["interval95"][0] > gate["lower_bound"],', '                  lower_bound=arena["interval95"][0] >= gate["lower_bound"],'),
    ('gate_ignores_complete', V2 + 'statistics.py', '    passed = summary["complete"] and summary["score"] >= rule["score"] and lower_ok and color_ok', '    passed = summary["score"] >= rule["score"] and lower_ok and color_ok'),
    ('tactical_no_family_average', V2 + 'statistics.py', '        result[name] = _mean(_family_means(selected, correct).values())', '        result[name] = _mean(correct(r) for r in selected)'),
    ('no_side_swap', V2 + 'arena.py', '        for game_index, agent_color in ((0, 0), (1, 1)):', '        for game_index, agent_color in ((0, 0), (1, 0)):'),
    ('abandoned_scored', V2 + 'arena.py', '            if not result["abandoned"]:\n                winner = result["winner"]', '            if True:\n                winner = result.get("winner", -1)'),
    ('arena_shares_game', V2 + 'arena.py', '        move, info = agents[owner].choose(Connect4(game.board, game.current_player), rngs[owner])', '        move, info = agents[owner].choose(game, rngs[owner])'),
    ('package_no_exclusion', V2 + 'packages.py', '                if key in self.seen or not accept(prefix):', '                if not accept(prefix):'),
    ('prune_keep_one', V2 + 'campaign.py', '        keep = self.declaration["retention"]["resume_boundaries"]', '        keep = 1'),
    ('skip_pending_checks', V2 + 'campaign.py', '        return [g for g in self.declaration["champion"]["schedule"] if g <= completed and g not in decided]', '        return []'),
    ('budget_no_prior', V2 + 'campaign.py', '        self.prior_training = prior["training_seconds"].get(str(seed), 0.0)', '        self.prior_training = 0.0'),
    ('incomplete_check_decides', V2 + 'campaign.py', '        if not status["complete"]:\n            self._incomplete_check(seed, generation, attempt_dir, champion, status, records, events)\n        arena', '        arena'),
    ('contradiction_sign', V2 + 'diagnostics.py', '            contradictions[key][0] += example.outcome != proven', '            contradictions[key][0] += example.outcome == proven'),
    ('calibration_uses_runner_rng', V2 + 'evaluation.py', '                        search_rng=seeded_rng(f"{CALIBRATION_DOMAIN}:search", seed),', '                        search_rng=random,'),
]
M.main()
