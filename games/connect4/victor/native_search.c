/* Bounded move-proof search, ported from Victor's Python terminal-only core.
 * This is runtime code, NOT the independent test oracle. Only the established
 * CL-only sufficient condition below enters proof search; composite covers and
 * heuristic scores never do. A cancelled invocation never stores a bound.
 * Full 49-bit position keys make TT replacement/collisions correctness-neutral.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stdlib.h>
#include <time.h>

typedef uint64_t Bits;
static const Bits bottom = 0x40810204081ULL;
static const Bits board = 0x40810204081ULL * 63;
static const int centre[7] = {3,2,4,1,5,0,6};
typedef struct { Bits key; int8_t lower, upper; } Entry;
typedef struct {
    Entry *tt;
    uint64_t size, nodes, limit, slice, hits;
    double deadline;
    int stopped, timed_out, claimeven;
    uint64_t bound_hits;
} Search;

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}
static int enter(Search *s) {
    if (s->nodes >= s->slice || s->nodes >= s->limit) { s->stopped=1; return 0; }
    if (!(s->nodes & 1023) && s->deadline != 0.0 && now() >= s->deadline) {
        s->stopped=s->timed_out=1; return 0;
    }
    s->nodes++;
    return 1;
}
static Bits wins(Bits p, Bits mask) {
    Bits r = (p<<1) & (p<<2) & (p<<3);
    const int dirs[3] = {7,6,8};
    for (int i=0;i<3;i++) {
        int d=dirs[i];
        Bits q=(p<<d) & (p<<(2*d));
        r |= q & (p<<(3*d)); r |= q & (p>>d);
        q=(p>>d) & (p>>(2*d));
        r |= q & (p<<d); r |= q & (p>>(3*d));
    }
    return r & (board ^ mask);
}
static Bits mirror(Bits p) {
    Bits r=0;
    for (int c=0;c<7;c++) r |= ((p>>(7*c)) & 127) << (7*(6-c));
    return r;
}
static int four(Bits p) {
    const int dirs[4]={1,7,6,8};
    for(int i=0;i<4;i++) { Bits q=p & (p>>dirs[i]); if(q & (q>>(2*dirs[i]))) return 1; }
    return 0;
}
/* A sufficient CL-only instance of the existing Black non-loss theorem.
 * Every empty even cell with an empty lower neighbour has a disjoint Claimeven.
 * If those uppers and Black stones hit every White group, White cannot win.
 * This is an UPPER bound of zero for White, never a draw or a Black win. */
int victor_claimeven(Bits white, Bits mask) {
    Bits uppers=(bottom*42) & ~mask & ~(mask<<1);
    return !four(board & ~((white^mask)|uppers));
}
/* Precondition: mover has no immediate winning move. */
static int visit(Search *s, Bits p, Bits mask, int played, int alpha, int beta) {
    if (!enter(s)) return 0;
    Bits possible=(mask+bottom)&board, threat=wins(p^mask,mask);
    Bits forced=possible & threat;
    if (forced) {
        if (forced & (forced-1)) return -1;
        possible=forced;
    }
    possible &= ~(threat>>1);
    if (!possible) return -1;
    if (played>=40) return 0;
    if(s->claimeven && !(played&1) && alpha>=0 && victor_claimeven(p,mask)) {
        s->bound_hits++; return 0;
    }
    Bits key=p+mask, reflected=mirror(p)+mirror(mask);
    if(reflected<key) key=reflected;
    /* Multiplicative indexing avoids the low-column bias of a power-of-two mask. */
    Entry *e=&s->tt[((key*11400714819323198485ULL)>>32)%s->size];
    int a0=alpha,b0=beta;
    if(e->key==key) {
        s->hits++;
        if(e->lower==e->upper) return e->lower;
        if(e->lower>alpha) alpha=e->lower;
        if(e->upper<beta) beta=e->upper;
        if(alpha>=beta) return e->lower>=beta ? e->lower : e->upper;
    }
    Bits moves[7]; int scores[7], n=0;
    for(int i=0;i<7;i++) {
        Bits m=possible & (63ULL<<(7*centre[i]));
        if(!m) continue;
        int score=__builtin_popcountll(wins(p|m,mask|m)),j=n++;
        while(j && scores[j-1]<score) { moves[j]=moves[j-1]; scores[j]=scores[j-1]; j--; }
        moves[j]=m; scores[j]=score;
    }
    int best=-2;
    for(int i=0;i<n;i++) {
        int value=-visit(s,p^mask,mask|moves[i],played+1,-beta,-alpha);
        if(s->stopped) return 0;
        if(value>best) best=value;
        if(value>alpha) alpha=value;
        if(alpha>=beta) break;
    }
    /* Descendants may replace this slot: recheck the full key before merging. */
    if(e->key!=key) { e->key=key; e->lower=-1; e->upper=1; }
    if(best>a0 && best< b0) e->lower=e->upper=(int8_t)best;
    else if(best<=a0) { if(best<e->upper) e->upper=(int8_t)best; }
    else if(best>e->lower) e->lower=(int8_t)best;
    return best;
}

/* Each returned [lo,hi] encloses that move's ROOT-relative WDL, including after
 * interruption. Illegal columns use [2,2]. Return: 0 budget, 1 proved optimal
 * move, 2 all moves solved, 3 time, -1 allocation failure. No global mutable state.
 * mode bit 0: serial full move values (ablation), otherwise fair thresholds.
 * mode bit 1: enable the established CL-only upper bound.
 */
int victor_prove(Bits p, Bits mask, int played, uint64_t limit, double seconds,
                 uint64_t entries, const int *order, int mode, int *lo, int *hi,
                 uint64_t *stats) {
    double start=now();
    Search s={0};
    s.size=entries; s.limit=limit; s.claimeven=mode&2; mode &= 1;
    s.deadline=seconds<0 ? 0 : start+seconds;
    for(int c=0;c<7;c++) {
        lo[c]=-1; hi[c]=1;
        Bits m=(mask+(1ULL<<(7*c))) & (63ULL<<(7*c));
        if(!m) {lo[c]=hi[c]=2; continue;}
        if(four(p|m)) lo[c]=hi[c]=1;
        else if(played==41) lo[c]=hi[c]=0;
        else if(wins(p^mask,mask|m) & ((mask|m)+bottom) & board) lo[c]=hi[c]=-1;
    }
    s.tt=entries ? calloc(entries,sizeof(Entry)) : NULL;
    if(!s.tt) return -1;
    uint64_t quantum=mode ? limit : 2048;
    int status=0;
    while(s.nodes<limit && !s.timed_out) {
        int unresolved=0, best=-1, upper=-1;
        for(int c=0;c<7;c++) if(lo[c]!=2) {
            if(lo[c]>best) best=lo[c];
            if(hi[c]>upper) upper=hi[c];
            if(lo[c]!=hi[c]) unresolved++;
        }
        if(!unresolved) {status=2; break;}
        if(!mode && best>=upper) {status=1; break;}
        for(int i=0;i<7 && s.nodes<limit && !s.timed_out;i++) {
            int c=order[i];
            if(lo[c]==hi[c]) continue;
            Bits m=(mask+(1ULL<<(7*c))) & (63ULL<<(7*c));
            /* Prove a win first. If it cannot win, prove/refute non-loss. */
            int alpha=hi[c]==1 ? -1 : 0;
            s.slice=s.nodes + (quantum<limit-s.nodes ? quantum : limit-s.nodes);
            s.stopped=0;
            int value=-visit(&s,p^mask,mask|m,played+1,alpha,alpha+1);
            if(s.stopped) continue;
            if(alpha==-1) {if(value==1) lo[c]=1; else hi[c]=0;}
            else {if(value==-1) hi[c]=-1; else lo[c]=0;}
            if(!mode && lo[c]==1) {status=1; goto done;}
            if(mode && lo[c]!=hi[c]) i--; /* complete this move before next */
        }
        if(quantum<limit) quantum=quantum>limit/4 ? limit : quantum*4;
    }
    if(s.timed_out) status=3;
    /* Completion can occur exactly at the node limit. */
    {int all=1; for(int c=0;c<7;c++) if(lo[c]!=hi[c]) all=0; if(all) status=2;}
done:
    stats[0]=s.nodes; stats[1]=s.hits; stats[2]=s.bound_hits;
    free(s.tt);
    return status;
}
