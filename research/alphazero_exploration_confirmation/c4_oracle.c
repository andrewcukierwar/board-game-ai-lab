/* Independent exact Connect 4 win/draw/loss oracle for benchmarks only.
 *
 * Shares no code with games/connect4/victor. Bitboard representation and the
 * non-losing-move/threat-ordering techniques follow Pascal Pons' public solver
 * write-up; values here are WDL only (-1/0/+1 for the player to move).
 * Every value is a completed terminal-only negamax proof. A node limit aborts
 * the whole position and reports "unknown"; no partial value is ever printed.
 *
 * stdin: one position per line as 1-based columns ("4453"), "." = empty board.
 * stdout: "ok <value> <v1..v7>" (x = illegal column) or "unknown"/"error ...",
 *         followed by the node count.
 * argv[1]: per-position node limit (default 50,000,000).
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef uint64_t u64;
enum { W = 7, H = 6, H1 = 7 };
static const u64 BOTTOM = 0x0040810204081ULL; /* lowest cell of each column */
static const u64 BOARD = 0x0040810204081ULL * 63ULL;

static u64 nodes, limit;
static int aborted;

#define TT_SIZE 8388593ULL /* prime */
typedef struct { u64 key; int8_t value; uint8_t flag; } Entry;
static Entry *table;
enum { EMPTY, EXACT, LOWER, UPPER };

static int popcount(u64 m) { return __builtin_popcountll(m); }

static u64 winning(u64 pos, u64 mask) {
    u64 r = (pos << 1) & (pos << 2) & (pos << 3);
    static const int shifts[3] = {H1, H, H + 2};
    for (int i = 0; i < 3; i++) {
        int d = shifts[i];
        u64 p = (pos << d) & (pos << 2 * d);
        r |= p & (pos << 3 * d);
        r |= p & (pos >> d);
        p = (pos >> d) & (pos >> 2 * d);
        r |= p & (pos << d);
        r |= p & (pos >> 3 * d);
    }
    return r & (BOARD ^ mask);
}

static u64 possible(u64 mask) { return (mask + BOTTOM) & BOARD; }

static u64 non_losing(u64 pos, u64 mask) {
    u64 moves = possible(mask);
    u64 threats = winning(pos ^ mask, mask);
    u64 forced = moves & threats;
    if (forced) {
        if (forced & (forced - 1)) return 0;
        moves = forced;
    }
    return moves & ~(threats >> 1);
}

static u64 mirror(u64 m) {
    u64 r = 0;
    for (int c = 0; c < W; c++) r |= ((m >> (c * H1)) & 127ULL) << ((W - 1 - c) * H1);
    return r;
}

static u64 key_of(u64 pos, u64 mask) {
    u64 a = pos + mask, b = mirror(pos) + mirror(mask);
    return a < b ? a : b;
}

static const int ORDER[W] = {3, 2, 4, 1, 5, 0, 6};

/* Precondition: the player to move cannot win immediately. */
static int negamax(u64 pos, u64 mask, int moves, int alpha, int beta) {
    if (aborted) return 0;
    if (++nodes > limit) { aborted = 1; return 0; }
    u64 next = non_losing(pos, mask);
    if (!next) return -1;
    if (moves >= W * H - 2) return 0;
    u64 key = key_of(pos, mask);
    Entry *e = &table[key % TT_SIZE];
    int a0 = alpha, b0 = beta;
    if (e->flag != EMPTY && e->key == key) {
        if (e->flag == EXACT) return e->value;
        if (e->flag == LOWER && e->value > alpha) alpha = e->value;
        if (e->flag == UPPER && e->value < beta) beta = e->value;
        if (alpha >= beta) return e->value;
    }
    u64 cand[W]; int score[W], n = 0;
    for (int i = 0; i < W; i++) {
        u64 m = next & (63ULL << (ORDER[i] * H1));
        if (!m) continue;
        int s = popcount(winning(pos | m, mask));
        int j = n++;
        while (j && score[j - 1] < s) { cand[j] = cand[j - 1]; score[j] = score[j - 1]; j--; }
        cand[j] = m; score[j] = s;
    }
    int best = -2;
    for (int i = 0; i < n; i++) {
        int v = -negamax(pos ^ mask, mask | cand[i], moves + 1, -beta, -alpha);
        if (aborted) return 0;
        if (v > best) best = v;
        if (v > alpha) alpha = v;
        if (alpha >= beta) break;
    }
    e->key = key;
    e->value = (int8_t)best;
    e->flag = best <= a0 ? UPPER : best >= b0 ? LOWER : EXACT;
    return best;
}

static int wdl(u64 pos, u64 mask, int moves) {
    if (winning(pos, mask) & possible(mask)) return 1;
    if (negamax(pos, mask, moves, 0, 1) >= 1) return 1;
    if (aborted) return 0;
    return negamax(pos, mask, moves, -1, 0) <= -1 ? -1 : 0;
}

static int four(u64 p) {
    static const int shifts[4] = {1, H1, H, H + 2};
    for (int i = 0; i < 4; i++) {
        u64 m = p & (p >> shifts[i]);
        if (m & (m >> 2 * shifts[i])) return 1;
    }
    return 0;
}

int main(int argc, char **argv) {
    limit = argc > 1 ? strtoull(argv[1], 0, 10) : 50000000ULL;
    table = calloc(TT_SIZE, sizeof(Entry));
    if (!table) return 2;
    char line[128];
    while (fgets(line, sizeof line, stdin)) {
        u64 pos = 0, mask = 0; int moves = 0, bad = 0;
        for (char *p = line; *p && *p != '\n' && *p != '\r'; p++) {
            if (*p == '.') continue;
            int c = *p - '1';
            if (c < 0 || c >= W || (mask & (1ULL << (c * H1 + H - 1)))) { bad = 1; break; }
            u64 move = (mask + (1ULL << (c * H1))) & (63ULL << (c * H1));
            if (four(pos ^ mask) || four(pos)) { bad = 1; break; }
            pos ^= mask; mask |= move; moves++;
        }
        if (bad || four(pos ^ mask) || moves == W * H) { printf("error invalid_or_terminal 0\n"); fflush(stdout); continue; }
        nodes = 0; aborted = 0;
        int values[W], value = -2;
        for (int c = 0; c < W && !aborted; c++) {
            u64 col = 63ULL << (c * H1);
            if (mask & (1ULL << (c * H1 + H - 1))) { values[c] = 9; continue; }
            u64 move = (mask + (1ULL << (c * H1))) & col;
            u64 mine = pos | move;
            int v;
            if (four(mine)) v = 1;
            else if (moves + 1 == W * H) v = 0;
            else v = -wdl(mine ^ (mask | move), mask | move, moves + 1);
            values[c] = v;
            if (v > value) value = v;
        }
        if (aborted) { printf("unknown %llu\n", (unsigned long long)nodes); fflush(stdout); continue; }
        printf("ok %d", value);
        for (int c = 0; c < W; c++) values[c] == 9 ? printf(" x") : printf(" %d", values[c]);
        printf(" %llu\n", (unsigned long long)nodes);
        fflush(stdout);
    }
    free(table);
    return 0;
}
