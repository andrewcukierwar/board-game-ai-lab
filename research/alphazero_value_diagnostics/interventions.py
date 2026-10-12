"""Inference-only interventions; unchanged PUCT owns all terminal evaluations."""
from games.connect4.alphazero_v2.network import V2Inference, V2Prediction
from .common import VARIANTS


class Intervention(V2Inference):
    def __init__(self, inference, variant):
        if not isinstance(inference, V2Inference) or variant not in VARIANTS:
            raise ValueError('Expected validated v2 inference and declared variant')
        self.base, self.model, self.variant = inference, inference.model, variant
        self.uniform, self.scale = VARIANTS[variant]

    def transform(self, prediction, state):
        legal = tuple(state.get_valid_moves())
        if state.is_game_over() or not legal:
            raise ValueError('Neural terminal evaluation is forbidden')
        policy = (tuple(1.0 / len(legal) if a in legal else 0.0 for a in range(7))
                  if self.uniform else prediction.policy)
        # Uniform logits keep the effective policy/logit contract consistent; raw learned logits
        # remain accessible via base.predict and are recorded independently in evaluations.
        logits = (0.0,) * 7 if self.uniform else prediction.logits
        value = prediction.value * self.scale
        if not self.uniform:
            assert policy == prediction.policy and logits == prediction.logits
        if self.scale == 1:
            assert value == prediction.value
        return V2Prediction(logits, policy, value)

    def predict(self, state):
        return self.transform(self.base.predict(state), state)

    def predict_batch(self, states):
        predictions = self.base.predict_batch(states)  # preserve strict input validation, order and empty behavior
        return tuple(self.transform(p, s) for p, s in zip(predictions, states))
