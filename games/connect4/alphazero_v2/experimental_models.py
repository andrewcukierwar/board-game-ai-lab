"""Small architecture adapter. New architectures require a new explicit adapter.

Collector/lifecycle consume this interface; a future adapter must validate its
own checkpoint, optimizer grouping, inference and training/output contracts.
Only the historical CNN is supported here.
"""
from .config import ARCHITECTURE, INFERENCE_CONTRACT
from .generation import initial_model
from .network import (model_from_state, model_state, weights_sha256, frozen_copy, V2Inference)
from .training import V2Trainer


class CNNFactory:
    architecture = ARCHITECTURE
    output_contract = INFERENCE_CONTRACT
    construct = staticmethod(initial_model)
    restore = staticmethod(model_from_state)
    state = staticmethod(model_state)
    hash = staticmethod(weights_sha256)
    freeze = staticmethod(frozen_copy)
    inference = staticmethod(V2Inference)
    trainer = staticmethod(V2Trainer)

    @staticmethod
    def restore_optimizer(trainer, state, steps):
        from .experimental import validate_optimizer
        validate_optimizer(trainer, state, steps)


def model_factory(architecture):
    if architecture != ARCHITECTURE:
        raise ValueError(f'Unsupported experimental model architecture: {architecture}')
    return CNNFactory()
