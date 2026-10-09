/* Standalone ASan/UBSan harness. Input: p mask played then seven exact WDLs
 * (2 for illegal). Generate from independent labelled histories; no Python
 * interpreter or sanitizer runtime injection is necessary. */
#include <assert.h>
#include <stdio.h>
#include <inttypes.h>
#ifndef VICTOR_SOURCE
#define VICTOR_SOURCE "../../games/connect4/victor/native_search.c"
#endif
#include VICTOR_SOURCE

int main(void) {
    uint64_t p, mask; int played, truth[7], cases=0;
    const uint64_t budgets[]={0,1,31,2048,100000};
    const uint64_t sizes[]={1,2,3,17,4096};
    while(scanf("%" SCNx64 " %" SCNx64 " %d", &p,&mask,&played)==3) {
        for(int c=0;c<7;c++) assert(scanf("%d",&truth[c])==1);
        for(int mode=0;mode<4;mode++) for(int j=0;j<5;j++) {
            int lo[7],hi[7]; uint64_t stats[3]={0};
            int status=victor_prove(p,mask,played,budgets[j],-1,sizes[j],centre,mode,lo,hi,stats);
            assert(status>=0 && stats[0]<=budgets[j]);
            for(int c=0;c<7;c++) {
                if(truth[c]==2) assert(lo[c]==2 && hi[c]==2);
                else assert(-1<=lo[c] && lo[c]<=truth[c] && truth[c]<=hi[c] && hi[c]<=1);
            }
        }
        cases++;
    }
    printf("Validated %d positions x 20 budget/table/mode combinations\n",cases);
    return cases ? 0 : 1;
}
