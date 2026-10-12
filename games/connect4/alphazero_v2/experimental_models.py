"""Small architecture adapter. New architectures require a new explicit adapter.

Collector/lifecycle consume this interface; every adapter validates its own
checkpoint, optimizer grouping, inference and training/output contracts.
The historical CNN is the default; the registered ResNets live in resnet.py.
"""
from .config import ARCHITECTURE, INFERENCE_CONTRACT
from .generation import initial_model
from .network import (model_from_state, model_state, weights_sha256, frozen_copy, V2Inference)
from .resnet import RESNET_SPECS, ResNetFactory
from .training import V2Trainer

SUPPORTED_ARCHITECTURES = (ARCHITECTURE, *RESNET_SPECS)


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
    if architecture == ARCHITECTURE:
        return CNNFactory()
    if isinstance(architecture, str) and architecture in RESNET_SPECS:
        return ResNetFactory(architecture)
    raise ValueError(f'Unsupported experimental model architecture: {architecture}')
