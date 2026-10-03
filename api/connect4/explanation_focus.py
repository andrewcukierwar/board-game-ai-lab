"""Verified relationships for concise explanations; no model-authored claims.

Every focus cites readable fact IDs and analyzer paths on the captured board.
The provider chooses among these relationships, never supplies their wording.
"""
from copy import deepcopy


def build_focus(context, mode, analyzed, move, facts):
    focuses = []
    player = analyzed['player_to_move']
    end = analyzed['outcome']
    by_id = {f['id']: f for f in facts}

    def add(id, text, ids, squares=(), links=None, paths=()):
        focuses.append({'id': id, 'text': text, 'fact_ids': list(dict.fromkeys(ids)),
                        'squares': deepcopy(list(squares)), 'concept_links': links or {},
                        'evidence_paths': list(paths)})

    def verified(id, text, paths):
        facts.append({'id': id, 'classification': 'confirmed_tactical',
                      'text': text, 'evidence_paths': paths})

    prefix = ''
    anchor = 'position' if mode == 'position' else 'move'
    if mode != 'position' and move:
        landing = move['landing_square']
        prefix = (f"If Player {move['player'] + 1} plays" if mode == 'what_if' else
                  f"Player {move['player'] + 1} played")
        prefix += f" Column {move['column'] + 1}, {'the piece lands' if mode == 'what_if' else 'landing'} on {landing['name']}. "
    if mode == 'last_move' and not move:
        add('no_move', by_id['move']['text'], ['move'])
        return focuses
    if end['status'] != 'ongoing':
        add('outcome', prefix + by_id['outcome']['text'], [anchor, 'outcome'],
            [move['landing_square']] if move else [], paths=['analyzed.outcome'])
        return focuses

    # A recorded/simulated move can expose an already existing winning square.
    if move and analyzed['immediate_winning_columns']:
        replies = [s for s in analyzed['winning_squares'][player]['squares'] if s['playable']]
        newly = {s['square']['name'] for s in move['winning_square_changes'][player]['newly_playable']}
        for reply in replies:
            target = reply['square']
            exposed = target['name'] in newly and target['column'] == move['column'] and target['row'] == move['landing_square']['row'] + 1
            effect = (f"This makes {target['name']} accessible to Player {player + 1}, who can immediately play there to complete four in a row."
                      if exposed else f"Player {player + 1} can now win immediately by playing Column {target['column'] + 1} at {target['name']}.")
            rid = f"move_reply_{target['name']}"
            verified(rid, f"Player {player + 1} has an immediately playable completion square at {target['name']}.",
                     ['analyzed.winning_squares', 'move.winning_square_changes'])
            add(rid, prefix + effect, ['move', rid, 'wins'], [move['landing_square'], target],
                {'winning_square': f"{target['name']} is a winning square that {'became playable after this move' if exposed else 'is playable on the resulting board'}."},
                ['move.landing_square', 'analyzed.immediate_winning_columns', 'move.winning_square_changes'])
        return focuses

    if mode == 'position' and len(analyzed['legal_columns']) == 1:
        alternative = analyzed['alternatives'][0]
        for reply in alternative['opponent_winning_replies']:
            landing, target = alternative['landing_square'], reply['square']
            old = next((s for s in analyzed['winning_squares'][1 - player]['squares'] if s['square'] == target), None)
            exposed = old and not old['playable'] and target['column'] == landing['column'] and target['row'] == landing['row'] + 1
            effect = (f"This makes {target['name']} accessible to Player {2 - player}, who can immediately play there to complete four in a row."
                      if exposed else f"Player {2 - player} can then win immediately at {target['name']}.")
            rid = f"forced_reply_{target['name']}"
            verified(rid, f"The drop lands on {landing['name']}; Player {2 - player}'s completion square {target['name']} is playable afterward.",
                     ['analyzed.legal_columns', 'analyzed.alternatives', 'analyzed.winning_squares'])
            add(rid, f"Column {landing['column'] + 1} is Player {player + 1}'s only available move, but playing there places the piece on {landing['name']}. " + effect,
                ['turn', rid, f"reply_{alternative['column']}"], [landing, target],
                {'winning_square': f"{target['name']} {'cannot be played until the square below it is filled' if exposed else 'is playable after the only legal move'}."},
                ['analyzed.legal_columns', 'analyzed.alternatives', 'analyzed.winning_squares'])
        if focuses:
            return focuses

    if analyzed['immediate_winning_columns']:
        for item in analyzed['winning_squares'][player]['squares']:
            if not item['playable']:
                continue
            sq = item['square']
            add(f"win_{sq['name']}", prefix + f"Player {player + 1} can win now: play Column {sq['column'] + 1} at {sq['name']} to complete four in a row.",
                [anchor, 'wins', f"square_{player}_{sq['name']}"], [sq],
                {'winning_square': f"{sq['name']} completes a line and is reachable on this turn."},
                ['analyzed.immediate_winning_columns', 'analyzed.winning_squares'])
        return focuses

    status = analyzed['defense']['status']
    if status == 'mandatory_block':
        column = analyzed['defense']['mandatory_column']
        sq = next(a['landing_square'] for a in analyzed['alternatives'] if a['column'] == column)
        text = f"Player {player + 1} must play Column {column + 1} at {sq['name']} to block the opponent's immediate win. Every other legal move allows a winning reply."
        add('block', prefix + text, [anchor, 'defense', 'threats'], [sq],
            {'tactics': f"The threat at {sq['name']} makes this defense mandatory.",
             'winning_square': f"The opponent could complete four at {sq['name']}; occupying it prevents that immediate win."},
            ['analyzed.defense', 'analyzed.alternatives', 'analyzed.opponent_immediate_winning_squares'])
        return focuses
    if status == 'unavoidable_loss_next_reply':
        targets = [s['square'] for s in analyzed['opponent_immediate_winning_squares']]
        add('forced_loss', prefix + f"Player {player + 1} cannot prevent a loss on the next reply: every legal move leaves an immediate win for Player {2 - player}.",
            [anchor, 'defense', 'threats'], targets,
            {'tactics': 'All legal defenses were checked; none prevents an immediate winning reply.'},
            ['analyzed.defense', 'analyzed.alternatives'])
        return focuses

    if move and move['was_mandatory_block']:
        sq = move['landing_square']
        add('move_block', prefix + "This was the only move that blocked a loss on the opponent's next reply.",
            ['move', 'move_block', 'wins'], [sq],
            {'tactics': f"Playing {sq['name']} answered the immediate threat."}, paths=['move.was_mandatory_block'])
        return focuses

    base = prefix + f"Player {player + 1} moves next. "
    if not analyzed['opponent_immediate_winning_squares']:
        base += 'Neither player has an immediate winning move on this board.'
    else:
        base += 'The opponent has an immediate threat; choose a move that prevents a winning reply.'
    links = {}
    for t in analyzed['winning_squares']:
        for s in t['squares']:
            if not s['playable']:
                sq = s['square']
                add(f"inaccessible_{t['player']}_{sq['name']}", base + f" Player {t['player'] + 1}'s completion square {sq['name']} is not yet reachable because of gravity.",
                    [anchor, 'wins', 'threats', f"square_{t['player']}_{sq['name']}"], [sq],
                    {'winning_square': f"{sq['name']} would complete four, but is not an immediate legal winning move."},
                    ['analyzed.winning_squares', 'analyzed.immediate_winning_columns'])
    for a in analyzed['alternatives']:
        if a['opponent_winning_replies']:
            target = a['opponent_winning_replies'][0]['square']
            add(f"avoid_{a['column']}", base + f" Playing Column {a['column'] + 1} at {a['landing_square']['name']} would let Player {2 - player} win immediately at {target['name']}.",
                [anchor, 'wins', 'threats', f"reply_{a['column']}"], [a['landing_square'], target],
                {'winning_square': f"After that placement, {target['name']} becomes a playable winning reply."},
                ['analyzed.alternatives'])
    # A quiet board supplies no evidence for competing threats, parity ownership,
    # formal rules, or an optimal opening recommendation.
    if not focuses:
        add('quiet', base, [anchor, 'wins', 'threats'],
            [move['landing_square']] if move else [], links,
            ['analyzed.immediate_winning_columns', 'analyzed.opponent_immediate_winning_squares'])
    return focuses
