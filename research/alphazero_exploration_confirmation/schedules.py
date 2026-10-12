"""Frozen Phase4D.4E identities; no production configuration change."""
from research.alphazero_exploration.schedules import ORIGINAL, EXTENDED, STAGES
SCHEDULES=(ORIGINAL,EXTENDED)
BY_NAME={s.name:s for s in SCHEDULES}
GENERATION=22
POOL=8
SIMULATIONS=256
PRIMARY=((42,20),(314159,20))
COHORT_SEEDS={(42,20):(4404511,4404512),(314159,20):(4404521,4404522)}
# Only the main mode is authorized in this driver.
