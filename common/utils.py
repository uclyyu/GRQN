import torch.nn as nn


def init_parameters(module, gain):
    """Helper funcion for initialising parameters.
    Args:
        module (torch.Module): pytorch module
        gain (float): gain for nn.init.xavier_uniform_
    """
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_uniform_(module.weight.data, gain)
        if module.bias is not None:
            module.bias.data.fill_(0.)


def count_parameters(cls, trainable_only=True):
    if trainable_only:
        filt = filter(lambda p: p.requires_grad, cls.parameters())
    else:
        filt = cls.parameters()

    count = sum(map(lambda p: p.numel(), filt))

    return count
