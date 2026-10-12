"""Short cost-only feasibility pilot; output contains timings, never endpoint scores."""
import time
from .common import ARMS, RESULTS, load, runtime, write
from .interventions import Intervention
from .evaluate import fixtures, openings, search, matches


def main():
    runtime()
    origin = time.monotonic()
    rows = []
    for arm in ARMS:
        started = time.monotonic()
        runner, model, base, entry = load(arm,6)
        loading = time.monotonic() - started
        wrapper = Intervention(base,'A')
        started = time.monotonic()
        # Model-blind stride sample; not included in measured analyses and no scores retained.
        for suite in ('tactical','solved','deep'):
            for row in fixtures()[suite][::max(1,len(fixtures()[suite])//4)][:4]:
                search(wrapper,suite,row,4404901)
        roots = (time.monotonic()-started)/12
        started = time.monotonic()
        matches(wrapper,dict(match_seed=4404902),None,lambda:None,rows=openings()[:1])
        games = (time.monotonic()-started)/2
        rows.append(dict(arm=arm,load_seconds=loading,seconds_per_root=roots,seconds_per_match=games))
    roots = sum(r['seconds_per_root'] * 6 * 628 for r in rows)
    games = sum(r['seconds_per_match'] * 5 * 80 for r in rows)
    overhead = sum(r['load_seconds'] * 12 for r in rows)
    projection = (roots + games + overhead + 240) / 3
    write(RESULTS/'pilot.json',dict(cost_only=True,rows=rows,seconds=time.monotonic()-origin,
          projected_wall_seconds_three_workers=projection,contingency_seconds=projection*1.8,
          positions=15072,matches=1600))
    print('Cost-only pilot complete')

if __name__ == '__main__':
    main()
