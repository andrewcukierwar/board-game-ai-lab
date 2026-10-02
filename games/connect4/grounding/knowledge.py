"""Curated paraphrases checked against Allis (1988), not runtime PDF ingestion.

Page numbers follow the thesis contents. In the supplied 91-page PDF those
numbers equal 1-based PDF pages (the rendered body has no visible page folios).
Formal-rule entries are reference-only; retrieval never asserts applicability.
"""
from copy import deepcopy

SOURCE_URL = 'https://tromp.github.io/c4/connect4_thesis.pdf'
SOURCE_TITLE = 'A Knowledge-Based Approach of Connect-Four: The Game Is Solved: White Wins'


def _reference(section, first, last=None):
    return {'chapter': int(section.split('.')[0]), 'section': section,
            'thesis_pages': [first, last or first],
            'pdf_pages_1_based': [first, last or first],
            'pdf_page_indices_0_based': [first - 1, (last or first) - 1]}


def _entry(id, name, explanation, preconditions, limitations, references, evidence,
           kind='concept'):
    return {'id': id, 'name': name, 'kind': kind, 'explanation': explanation,
            'preconditions': preconditions, 'limitations': limitations,
            'references': references, 'source_url': SOURCE_URL,
            'programmatic_evidence': evidence,
            'application_status': 'reference_only' if kind == 'rule' else 'context_only'}


_FRAMEWORK = (
    'For a position-level conclusion, use the opponent-to-move evaluation and board region '
    'specified in chapters 6 and 8, and a mutually compatible set covering every relevant '
    'opponent group (sections 5.4 and 7.4). Black may attempt this coverage without separately '
    'proving Zugzwang control first (8.1); White requires the odd-threat or threat-combination '
    'setup and excludes its reserved column(s) (8.2-8.4, 9.2).'
)
_NO_RULE = (
    'No detector, compatibility checker, coverage proof or Zugzwang evaluator is implemented. '
    'Local geometry alone is not a supported rule application or a game-result proof.'
)


def _rule(id, name, explanation, conditions, limitations, section, first, last=None):
    return _entry(id, name, explanation, [conditions, _FRAMEWORK],
                  [limitations, _NO_RULE],
                  [_reference(section, first, last), _reference('5.4', 34, 35),
                   _reference('7.4', 50), _reference('8.1', 51), _reference('9.2', 58, 59)],
                  {'status': 'not_implemented', 'fields': []}, kind='rule')


_ENTRIES = (
    _entry('coordinates', 'Board nomenclature',
           'Allis names columns a through g and rows from 1 at the bottom to 6 at the top; White moves first.',
           ['Standard 7 by 6 board.'],
           ['API columns are 0 through 6 and matrix rows run from top to bottom. A recorded move is one ply; thesis move lists pair White and Black turns.'],
           [_reference('1.1', 7), _reference('1.2', 8, 9)],
           {'status': 'implemented', 'fields': ['coordinates', 'position', 'move_history']}),
    _entry('winning_square', 'Threats and winning squares',
           'A completion square can finish a four-stone line. Multiple lines needing the same square are not independent winning opportunities. Allis sometimes describes a threat from the threatened opponent’s perspective; this payload labels the player who would complete the line.',
           ['Three stones of the same player and one empty square in a four-square line.',
            'An immediate winning move additionally requires that gravity makes the square playable and the game has not ended.'],
           ['An inaccessible winning square is not an immediate win. Other wins can make higher squares irrelevant; counting patterns does not prove a result.'],
           [_reference('3.1', 16, 18), _reference('3.2', 18, 19), _reference('5', 32)],
           {'status': 'implemented', 'fields': ['confirmed_tactical_facts.winning_squares',
                                              'confirmed_tactical_facts.immediate_winning_columns']}),
    _entry('tactics', 'Immediate tactics and competing threats',
           'Short forcing sequences can decide play before a long-term strategic plan matters. Separate playable winning squares may require incompatible defenses.',
           ['Check the side to move, gravity, terminal outcomes and all legal replies before asserting that a defense is forced.'],
           ['The analyzer proves only current wins and losses on the next reply. A player with an immediate win need not block. Two lines sharing one completion square are not two independently playable threats.'],
           [_reference('3.4', 21, 24)],
           {'status': 'implemented_bounded', 'fields': ['confirmed_tactical_facts.defense',
                                                       'confirmed_tactical_facts.alternatives',
                                                       'last_move_facts']}),
    _entry('parity', 'Odd and even threats',
           'Odd/even describes the bottom-based row of the completion square. Allis uses move parity and the order of filling columns to reason about who can obtain such squares.',
           ['Apply the position assumptions of section 3.3, including other threats and whether the threats share a column, before using its outcome examples.'],
           ['The program reports row parity only. An odd threat does not by itself prove a White win, nor an even threat a Black win; other tactical threats can change the result.'],
           [_reference('3.3', 19, 21)],
           {'status': 'geometry_only', 'fields': ['confirmed_tactical_facts.winning_squares[].squares[].parity']}),
    _entry('zugzwang', 'Control of Zugzwang',
           'The obligation to move can force an unwanted placement. Allis describes control in terms of directing the allocation of odd and even squares; same-column follow-up illustrates it.',
           ['Follow-up requires available responses and must account for wins before the intended filling sequence ends. White’s control examples reserve an odd-threat column.'],
           ['Control alone is not a win or draw proof. Even empty-square counts alone do not justify a successful follow-up strategy; see the losing initial-position example in 4.2.'],
           [_reference('4.1', 25, 26), _reference('4.2', 26, 27),
            _reference('4.3', 27, 28), _reference('4.5', 30, 31)],
           {'status': 'not_implemented', 'fields': []}),
    _entry('rule_framework', 'Rule coverage, compatibility and uncertainty',
           'Allis combines solutions to opponent groups under compatibility constraints. A local solution to one group does not establish that all opponent wins are prevented. Failure to find a covering set leaves the evaluation unresolved.',
           [_FRAMEWORK],
           ['Section 7.4 has pair-specific conditions: disjoint squares, inverse/Claimeven ordering, column-wise equality/disjointness, and inverse column-set restrictions. Some pairs require two conditions. Specialbefore has additional restrictions on its special squares.',
            'No application or compatibility proof is produced in Phase 3A.'],
           [_reference('5.3', 33, 34), _reference('5.4', 34, 35),
            _reference('7.1', 47, 48), _reference('7.2', 49), _reference('7.3', 49),
            _reference('7.4', 50), _reference('8.1', 51), _reference('8.2', 51, 52),
            _reference('8.4', 55, 57), _reference('9.2', 58, 59)],
           {'status': 'not_implemented', 'fields': ['supported_allis_rule_applications',
                                                   'unknown_or_unproven']}),
    _rule('claimeven', 'Claimeven',
          'Reserves the even upper square by responding above the opponent’s lower placement; it solves opponent groups containing that upper square.',
          'Two empty vertically adjacent squares; the upper square has an even bottom-based row.',
          'Requires the Zugzwang-dependent framework; seeing an empty pair does not establish ownership of the upper square.',
          '6.1', 36, 37),
    _rule('baseinverse', 'Baseinverse',
          'A response on one of two available squares prevents the opponent from acquiring both, solving groups that require both.',
          'Two distinct directly playable squares. A useful solution must cover an opponent group containing both.',
          'Zugzwang-independent as a local response rule (7.2), but compatibility and full coverage remain necessary for a position-level conclusion.',
          '6.2', 37, 38),
    _rule('vertical', 'Vertical',
          'A response within an adjacent vertical pair prevents the opponent from owning both squares; it solves groups containing both.',
          'Two empty vertically adjacent squares; the upper square is odd in the standalone rule.',
          'Zugzwang-independent (7.2). The stronger Claimeven handles even upper squares; Before can use vertical pairs with either upper parity.',
          '6.3', 38, 39),
    _rule('aftereven', 'Aftereven',
          'A group completed via Claimevens finishes before every column of its missing squares can be filled above that group. It also inherits those Claimeven solutions.',
          'A group the applying player can complete using only upper even squares from valid Claimevens. To solve another group by the timing effect, that group must contain a higher square in every column with a missing Aftereven square.',
          'The opponent may determine which column is completed last. A group above only some required columns is not solved by the timing effect.',
          '6.4', 39, 40),
    _rule('lowinverse', 'Lowinverse',
          'Couples two vertical pairs so the applying player obtains at least one upper odd square. It also retains each constituent Vertical solution.',
          'Two different columns, each containing two adjacent empty squares whose upper square is odd. The main solved groups contain both upper squares.',
          'Lower squares need not be playable and upper squares need not share a row. Inverse interactions with Claimevens require extra checks.',
          '6.5', 40, 41),
    _rule('highinverse', 'Highinverse',
          'Extends the inverse to three-square segments: it solves groups through both tops, both middles, or the top two squares within either segment.',
          'Two different columns, each with three consecutive empty squares and an even top square. If a segment’s bottom is directly playable, groups through that bottom and the other segment’s top are additionally solved.',
          'The cross-column bottom/top conclusion requires direct playability of that bottom. The two segment tops need not share a row.',
          '6.6', 41, 42),
    _rule('baseclaim', 'Baseclaim',
          'Uses alternative responses combining Baseinverse and Claimeven effects. It solves groups containing the first square plus the square above the second, and groups containing the second plus the third.',
          'Three distinct directly playable squares, assigned roles first, second and third; the square immediately above the second exists and is even (and empty).',
          'The role assignment matters. This is one coordinated response strategy, not permission to combine overlapping Baseinverse and Claimeven instances arbitrarily.',
          '6.7', 42, 43),
    _rule('before', 'Before',
          'A not-yet-completed group supports responses that acquire its missing squares or their successors. Opponent groups needing every successor are prevented, with constituent Vertical and Claimeven side effects.',
          'A group with no opponent stones and no empty square on the top row. Choose the constituent Claimeven/Vertical pairs for its empty squares as described in 6.8.',
          'Successor means the square immediately above. Coverage of every required successor is essential. All-Claimeven cases use the stronger Aftereven; Before vertical pairs can have even upper squares.',
          '6.8', 43, 45),
    _rule('specialbefore', 'Specialbefore',
          'Modifies a Before response using an extra playable square. It solves groups containing all successors plus that extra square, groups containing the two special playable squares, and the retained component solutions.',
          'A Before-type group without opponent stones, with no empty top-row square, with an empty directly playable square in it, and an extra directly playable square in another column.',
          'The special response replaces the ordinary vertical response at the internal playable square. Do not retain that replaced pair as an independent solution. Section 7.4 adds restrictions for overlap involving the two special squares.',
          '6.9', 45, 46),
)


def retrieve_knowledge(concept_ids):
    """Return detached entries in catalog order; unknown IDs fail explicitly."""
    requested = set(concept_ids)
    unknown = requested - {e['id'] for e in _ENTRIES}
    if unknown:
        raise ValueError(f'Unknown Allis concepts: {sorted(unknown)}')
    return deepcopy([e for e in _ENTRIES if e['id'] in requested])


def knowledge_catalog():
    return deepcopy(list(_ENTRIES))
