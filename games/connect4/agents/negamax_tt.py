"""Lossless integer encodings for depth-limited Negamax transposition tables.

These are identities, not hashes. Python's arbitrary-width integers preserve
both every board bit and unrestricted signed, depth-sensitive scores.
"""
FLAGS = ('exact', 'lower', 'upper')
FLAG_CODES = {flag: code for code, flag in enumerate(FLAGS)}
BITS_MASK = (1 << 49) - 1


def pack_key(x, o, mover, depth):
    """Encode valid internal bitboards (<2**49), mover 0/1, depth >=0.

    Search uses the identical expression inline after its state invariants have
    established these preconditions. Validate the diagnostic/public helper.
    """
    if (any(type(v) is not int for v in (x, o, mover, depth)) or
            not 0 <= x <= BITS_MASK or not 0 <= o <= BITS_MASK or
            mover not in (0, 1) or depth < 0):
        raise ValueError('Invalid TT identity fields')
    return x | (o << 49) | (mover << 98) | (depth << 99)


def unpack_key(key):
    if type(key) is not int or key < 0:
        raise ValueError('Invalid TT key')
    return key & BITS_MASK, (key >> 49) & BITS_MASK, (key >> 98) & 1, key >> 99


def pack_entry(flag, score, move):
    """Normal search writes integer scores and legal moves or None.

    Extra hint codes 7..14 are supported for invalid-hint diagnostics. They
    cannot become legal Connect4 moves. Code 15 represents an absent hint.
    """
    if flag not in FLAG_CODES or type(score) is not int or (
            move is not None and (type(move) is not int or not 0 <= move < 15)):
        raise ValueError('Invalid TT entry fields')
    signed = 2 * score if score >= 0 else -2 * score - 1
    return (signed << 6) | ((15 if move is None else move) << 2) | FLAG_CODES[flag]


def unpack_entry(entry):
    if type(entry) is not int or entry < 0 or entry & 3 == 3:
        raise ValueError('Invalid TT entry')
    signed, hint = entry >> 6, (entry >> 2) & 15
    score = signed // 2 if signed & 1 == 0 else -(signed // 2) - 1
    return FLAGS[entry & 3], score, None if hint == 15 else hint


class DirectMappedEntries:
    """Experimental deterministic replacement; full equality guards every hit.

    Parallel lists avoid an extra tuple per occupied slot. Replacement changes
    cache availability only, never search budgets or the meaning of an entry.
    """
    __slots__ = ('capacity', 'keys', 'values', 'occupancy', 'evictions', 'replacements')

    def __init__(self, capacity):
        if type(capacity) is not int or capacity < 1:
            raise ValueError('capacity must be positive')
        self.capacity = capacity
        self.keys = [None] * capacity
        self.values = [None] * capacity
        self.occupancy = self.evictions = self.replacements = 0

    def get(self, key):
        slot = hash(key) % self.capacity
        return self.values[slot] if self.keys[slot] == key else None

    def __setitem__(self, key, value):
        slot = hash(key) % self.capacity
        old = self.keys[slot]
        if old is None:
            self.occupancy += 1
        elif old == key:
            self.replacements += 1
        else:
            self.evictions += 1
        self.keys[slot], self.values[slot] = key, value

    def __len__(self):
        return self.occupancy

    def items(self):
        return ((key, value) for key, value in zip(self.keys, self.values) if key is not None)
