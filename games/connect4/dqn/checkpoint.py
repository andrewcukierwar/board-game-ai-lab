"""Versioned inference weights + JSON-compatible provenance; no resume state."""

from collections.abc import Mapping
import json

import torch

from .encoding import ACTION_ORDER, ENCODING_VERSION, VALUE_CONVENTION
from .network import ARCHITECTURE, DQN

CONTRACT = {
    'format_version': 1,
    'architecture': ARCHITECTURE,
    'state_encoding': ENCODING_VERSION,
    'action_order': ACTION_ORDER,
    'value_convention': VALUE_CONVENTION,
}


def metadata_copy(metadata):
    """Restrict provenance to portable JSON data (no tensors/custom objects)."""
    def validate(value):
        if value is None or type(value) in (str, bool, int, float):
            return
        if type(value) is list:
            for item in value:
                validate(item)
            return
        if type(value) is dict and all(type(k) is str for k in value):
            for item in value.values():
                validate(item)
            return
        raise ValueError('Training metadata must contain only JSON data')

    if type(metadata) is not dict:
        raise ValueError('Training metadata must be a dict')
    validate(metadata)
    try:
        return json.loads(json.dumps(metadata, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError('Training metadata must be finite JSON data') from exc


def validate_weights(state, model):
    expected = model.state_dict()
    if not isinstance(state, Mapping) or set(state) != set(expected):
        raise ValueError('Incompatible state_dict keys')
    for key, reference in expected.items():
        value = state[key]
        if (type(value) is not torch.Tensor or value.shape != reference.shape
                or value.dtype != torch.float32 or value.layout != torch.strided
                or value.device.type != 'cpu' or not torch.isfinite(value).all().item()):
            raise ValueError(f'Invalid state_dict tensor: {key}')


def save_checkpoint(path, model, *, training_metadata=None):
    if type(model) is not DQN:
        raise ValueError('Checkpoint requires the versioned DQN architecture')
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    with torch.random.fork_rng(devices=[]):
        reference = DQN()
    validate_weights(state, reference)
    payload = dict(CONTRACT, model_state_dict=state,
                   training_metadata=metadata_copy({} if training_metadata is None else training_metadata))
    torch.save(payload, path)


def load_checkpoint(path):
    """Return (CPU eval DQN, provenance), with no unsafe loading fallback."""
    payload = torch.load(path, weights_only=True, map_location='cpu')
    if type(payload) is not dict or set(payload) != set(CONTRACT) | {'model_state_dict', 'training_metadata'}:
        raise ValueError('Incompatible checkpoint schema')
    for key, expected in CONTRACT.items():
        value = payload[key]
        if type(value) is not type(expected) or value != expected:
            raise ValueError(f'Incompatible checkpoint {key}')
        if key == 'action_order' and any(type(v) is not int for v in value):
            raise ValueError('Incompatible checkpoint action_order')
    metadata = metadata_copy(payload['training_metadata'])
    # Loading should not consume the application's model-initialization RNG.
    with torch.random.fork_rng(devices=[]):
        model = DQN()
    validate_weights(payload['model_state_dict'], model)
    model.load_state_dict(payload['model_state_dict'], strict=True)
    return model.eval(), metadata
