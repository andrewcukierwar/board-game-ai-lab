"""Torch-free v2 contracts and configuration (docs/phase4d3-alphazero-v2-review.md §10)."""
from dataclasses import asdict, dataclass, fields
import math

# Same canonical board encoding as v1 (own +1 / opponent -1 / empty 0, top row
# first, physical columns); the network, value and artifact contracts differ.
ENCODING = "connect4-current-player-1x6x7-v1"
ARCHITECTURE = "connect4-conv64-128-128-policy7-scalar-tanh-v2"
POLICY_TARGET = "normalized-root-visits-v2"
VALUE_TARGET = "completed-game-actor-relative-outcome-v2"
PUCT_EXPLORATION = 1.41

INFERENCE_CONTRACT = {
    "format": "connect4-alphazero-v2-inference",
    "format_version": 2,
    "architecture": ARCHITECTURE,
    "encoding": ENCODING,
    "input": "cpu float32 (N,1,6,7)",
    "action_order": list(range(7)),
    "policy_output": "raw-logits-7",
    "policy_inference": "legal-logit-mask-before-stable-softmax",
    "value_output": "tanh-scalar-(N,1)",
    "value_perspective": "current_player",
    "value_semantics": "expected final actor-relative outcome in [-1,1]",
}

RESUME_CONTRACT = {
    "format": "connect4-alphazero-v2-resume",
    # v2 (Phase 4D.3B): enforced full runtime identity, execution-source content
    # digest and resume lineage. No v1 resume artifact was ever produced outside tests.
    "format_version": 2,
    "inference_contract": INFERENCE_CONTRACT,
    "policy_target": POLICY_TARGET,
    "value_target": VALUE_TARGET,
    "replay_schema": "generation/games/moves+visits+action_temperatures-v1",
    "boundary": "completed generation (collection, replay insertion and training finished)",
    "exact_continuation": "identical runtime_identity and execution-source sha256; else recorded, not claimed",
}


def action_temperature(ply, exploratory_plies):
    """Self-play execution temperature for a zero-based pre-move ply."""
    if type(ply) is not int or not 0 <= ply < 42:
        raise ValueError("ply must be an integer pre-move occupancy 0..41")
    return 1.0 if ply < exploratory_plies else 0.0


def _positive_int(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _finite(value, name, low, high, *, low_open=False):
    if isinstance(value, bool) or type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if value > high or value < low or (low_open and value == low):
        raise ValueError(f"{name} is outside its permitted range")


@dataclass(frozen=True)
class V2Config:
    """Milestone 1 defaults for a future, separately authorized campaign."""
    seed: int = 42
    self_play_simulations: int = 256
    evaluation_simulations: int = 512
    interactive_simulations: int = 512
    root_noise_epsilon: float = 0.25
    root_dirichlet_alpha: float = 1.0
    exploratory_plies: int = 8
    games_per_generation: int = 256
    replay_generations: int = 8
    replay_max_games: int = 2048
    batch_size: int = 128
    samples_per_new_position: int = 4
    learning_rate: float = 3e-4
    adam_betas: tuple = (0.9, 0.999)
    adam_eps: float = 1e-8
    weight_decay: float = 1e-4
    max_grad_norm: float = 5.0
    reflection_probability: float = 0.5
    max_generations: int = 20

    def __post_init__(self):
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer")
        for name in ("self_play_simulations", "evaluation_simulations", "interactive_simulations",
                     "games_per_generation", "replay_generations", "replay_max_games", "batch_size",
                     "samples_per_new_position", "max_generations"):
            _positive_int(getattr(self, name), name)
        if type(self.exploratory_plies) is not int or not 0 <= self.exploratory_plies <= 42:
            raise ValueError("exploratory_plies must be an integer 0..42")
        if self.replay_generations * self.games_per_generation > self.replay_max_games:
            raise ValueError("replay window cannot hold replay_generations full generations")
        _finite(self.root_noise_epsilon, "root_noise_epsilon", 0, 1)
        _finite(self.root_dirichlet_alpha, "root_dirichlet_alpha", 0, math.inf, low_open=True)
        _finite(self.learning_rate, "learning_rate", 0, math.inf, low_open=True)
        _finite(self.adam_eps, "adam_eps", 0, math.inf, low_open=True)
        _finite(self.weight_decay, "weight_decay", 0, math.inf)
        _finite(self.max_grad_norm, "max_grad_norm", 0, math.inf, low_open=True)
        _finite(self.reflection_probability, "reflection_probability", 0, 1)
        betas = tuple(self.adam_betas)
        if len(betas) != 2:
            raise ValueError("adam_betas must contain two values")
        for beta in betas:
            _finite(beta, "adam_betas", 0, 1)
            if beta == 1:
                raise ValueError("adam_betas must be < 1")
        object.__setattr__(self, "adam_betas", tuple(float(b) for b in betas))

    def updates_for(self, new_positions):
        """ceil(samples_per_new_position * new_positions / batch_size)."""
        if type(new_positions) is not int or new_positions < 1:
            raise ValueError("new_positions must be a positive integer")
        return -(-self.samples_per_new_position * new_positions // self.batch_size)

    def to_dict(self):
        result = asdict(self)
        result["adam_betas"] = list(self.adam_betas)
        return result

    @classmethod
    def from_dict(cls, data):
        names = {f.name for f in fields(cls)}
        if not isinstance(data, dict) or set(data) != names:
            raise ValueError("Config fields do not match V2Config")
        return cls(**data)
