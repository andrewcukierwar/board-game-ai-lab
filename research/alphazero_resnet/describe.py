"""Record static architecture specifications; constructs models only, never times or trains them.

    python -B -m research.alphazero_resnet.describe > research/alphazero_resnet/results/architectures.json
"""
import json

import torch
from torch import nn

from games.connect4.alphazero_v2.experimental_models import SUPPORTED_ARCHITECTURES, model_factory
from games.connect4.alphazero_v2.resnet import RESNET_SPECS

CELLS = 6 * 7  # every convolution preserves the 6x7 board


def multiply_accumulates(module):
    if isinstance(module, nn.Conv2d):
        return module.weight.numel() * CELLS
    return module.weight.numel() if isinstance(module, nn.Linear) else 0


def component(name):
    if name.startswith(('policy', 'value')):
        return name.split('_')[0] + '_head'
    return 'trunk'


def describe(architecture):
    factory = model_factory(architecture)
    model = factory.construct(42)
    layers = [(name, module) for name, module in model.named_modules()
              if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d))]
    parameters, operations = {}, {}
    for name, module in layers:
        group = component(name)
        parameters[group] = parameters.get(group, 0) + sum(p.numel() for p in module.parameters())
        operations[group] = operations.get(group, 0) + multiply_accumulates(module)
    state = model.state_dict()
    spec = RESNET_SPECS.get(architecture)
    return dict(
        architecture=architecture,
        specification=None if spec is None else vars(spec),
        output_contract=factory.output_contract,
        parameters=dict(parameters, total=sum(p.numel() for p in model.parameters())),
        multiply_accumulates_per_position=dict(operations, total=sum(operations.values())),
        layers=dict(convolution=sum(isinstance(m, nn.Conv2d) for _, m in layers),
                    linear=sum(isinstance(m, nn.Linear) for _, m in layers),
                    batch_norm=sum(isinstance(m, nn.BatchNorm2d) for _, m in layers),
                    sequential_trunk_convolutions=sum(isinstance(m, nn.Conv2d) and component(n) == 'trunk'
                                                      for n, m in layers)),
        state=dict(tensors=len(state), parameter_tensors=len(list(model.parameters())),
                   buffer_tensors=len(list(model.buffers())),
                   buffer_elements=sum(b.numel() for b in model.buffers()),
                   bytes=sum(t.numel() * t.element_size() for t in state.values())),
        seed_42_initial_weights_sha256=factory.hash(model))


def main():
    print(json.dumps(dict(
        note='Static specification only. Multiply-accumulates count convolution/linear weights applied per '
             'position and exclude batch norm, activations and framework overhead; they are not measured speed.',
        torch=torch.__version__, architectures=[describe(a) for a in SUPPORTED_ARCHITECTURES]), indent=2))


if __name__ == '__main__':
    main()
