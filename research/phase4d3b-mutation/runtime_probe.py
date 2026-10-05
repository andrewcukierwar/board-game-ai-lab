import sys, json, random
import numpy as np, torch
intra, inter, det = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3] == '1'
torch.set_num_threads(intra); torch.set_num_interop_threads(inter); torch.use_deterministic_algorithms(det)
from games.connect4.alphazero_v2.config import V2Config
from games.connect4.alphazero_v2.generation import GenerationRunner
from games.connect4.alphazero_v2.network import weights_sha256
cfg = V2Config(self_play_simulations=16, games_per_generation=8, replay_generations=2, replay_max_games=16, batch_size=64, max_generations=3)
r = GenerationRunner(cfg)
g = (random.getstate(), np.random.get_state()[1].tobytes(), torch.get_rng_state().clone())
out = [r.run_generation()['learner_weights_sha256'][:12] for _ in range(2)]
g2 = (random.getstate(), np.random.get_state()[1].tobytes(), torch.get_rng_state())
print(json.dumps(dict(intra=intra, inter=inter, det=det, hashes=out,
      global_rng_untouched=g[0]==g2[0] and g[1]==g2[1] and torch.equal(g[2],g2[2]))))
