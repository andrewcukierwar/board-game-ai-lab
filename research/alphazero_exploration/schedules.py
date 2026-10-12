"""Prospective Phase 4D.4D schedules and game identities (see DESIGN.md).

Nothing here is a V2Config default. Every schedule is passed explicitly to the
experimental batched collector; ORIGINAL restates the default for the record.
"""
from games.connect4.alphazero_v2.batching import TemperatureSchedule

ORIGINAL = TemperatureSchedule('original', ((0, 1.0), (8, 0.0)))
EXTENDED = TemperatureSchedule('extended', ((0, 1.0), (16, 0.0)))
SMOOTHED = TemperatureSchedule('smoothed', ((0, 1.0), (8, 0.3), (30, 0.0)))
EXTENDED_SMOOTHED = TemperatureSchedule('extended_smoothed', ((0, 1.0), (16, 0.3), (30, 0.0)))
SCHEDULES = (ORIGINAL, EXTENDED, SMOOTHED, EXTENDED_SMOOTHED)
BY_NAME = {s.name: s for s in SCHEDULES}
# Factor coding for the two-factor comparison: (full temperature through ply 15, 0.3 until ply 29).
FACTORS = dict(original=(0, 0), extended=(1, 0), smoothed=(0, 1), extended_smoothed=(1, 1))

GENERATION = 21          # experimental collection id, as in Phase 4D.4C
POOL = 8
SIMULATIONS = 256
PRIMARY = ((42, 20), (314159, 20))
SUPPLEMENT = ((42, 10),)
# Base seeds of the independent-game protocol. Cohorts are independently seeded
# identities, never repeats; every schedule reuses the same identities.
REGRESSION_SEED = 4404003   # Phase 4D.4C identities, compared with its retained records
PILOT_SEED = 4404400
COHORT_SEEDS = {(42, 20): (4404411, 4404412), (314159, 20): (4404421, 4404422),
                (42, 10): (4404431, 4404432)}
STAGES = (('0-7', 0, 8), ('8-15', 8, 16), ('16-23', 16, 24), ('24+', 24, 42))
