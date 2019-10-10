import torch
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


def get_params_l2(model):
    l2 = nn.MSELoss(reduction='sum')
    regulariser = 0
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            regulariser += l2(m.weight, torch.zeros_like(m.weight))
    return regulariser


def get_params_l1(model):
    l1 = nn.L1Loss(reduction='sum')
    regulariser = 0
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            regulariser += l1(m.weight, torch.zeros_like(m.weight))
    return regulariser


class DummyModule(nn.Module):
    # Accommodating torch.jit
    def __init__(self):
        super(DummyModule, self).__init__()

    def forward(self, inp):
        ret = torch.zeros(1, device=inp.device)
        return ret


class SkipConnect(nn.Module):
    def __init__(self, layer1, layer2):
        super(SkipConnect, self).__init__()
        self.layer1 = layer1
        self.layer2 = layer2

    def forward(self, inp):
        return self.layer1(inp) + self.layer2(inp)


class BilinearInterpolate(nn.Module):
    def __init__(self, scale):
        super(BilinearInterpolate, self).__init__()

        self.scale = scale

    def forward(self, inp):
        return nn.functional.interpolate(inp, scale_factor=self.scale, mode='bilinear', align_corners=True)


class GroupNorm1d(nn.Module):
    def __init__(self, features, groups, eps=1e-5):
        super(GroupNorm1d, self).__init__()

        self.gamma = nn.Parameter(torch.ones(1, features))
        self.beta = nn.Parameter(torch.zeros(1, features))
        self.num_groups = groups
        self.eps = eps

    def forward(self, x):
        N, C = x.size()
        G = self.num_groups

        x = x.view(N, G, -1)
        mean = x.mean(dim=2, keepdim=True)
        var = (x - mean).pow(2).sum(2, keepdim=True) / x.size(2)

        x = (x - mean) / (var + self.eps).sqrt()
        x = x.view(N, C)

        return x * self.gamma + self.beta


class GroupNorm2d(nn.Module):
    def __init__(self, channels, groups, eps=1e-5):
        super(GroupNorm2d, self).__init__()

        self.gamma = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.num_groups = groups
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.size()
        G = self.num_groups

        x = x.view(N, G, -1)
        mean = x.mean(dim=2, keepdim=True)
        var = (x - mean).pow(2).sum(2, keepdim=True) / x.size(2)

        x = (x - mean) / (var + self.eps).sqrt()
        x = x.view(N, C, H, W)

        return x * self.gamma + self.beta


@torch.jit.script
def _lstm_zoneout(gf, gi, gs, go, hid, cel, mask, go_prev):
    # gates:
    #   gf - forget gate
    #   gi - input gate
    #   gs - state (cell) gate
    #   go - output gate
    # hid - hidden units
    # cel - cell units
    # mask - 1 for keeping previous output gate, 0 otherwise
    gf = torch.sigmoid(gf)
    gi = torch.sigmoid(gi)
    gs = torch.tanh(gs)
    go = torch.sigmoid(go)

    c_next = gf * cel + mask * gi * gs
    h_next = ((1 - mask) * go + mask * go_prev) * torch.tanh(c_next)

    return h_next, c_next, go


@torch.jit.script
def _lstm(gf, gi, gs, go, hid, cel):
    gf = torch.sigmoid(gf)
    gi = torch.sigmoid(gi)
    gs = torch.tanh(gs)
    go = torch.sigmoid(go)

    c_next = gf * cel + gi * gs
    h_next = go * torch.tanh(c_next)

    return h_next, c_next


class ConvLSTMCell(torch.jit.ScriptModule):
    __constants__ = ['padding', 'zoneout_prob']

    def __init__(self, input_size, hidden_size, kernel_size, zoneout=0, bias=True):
        super(ConvLSTMCell, self).__init__()

        # !TODO:
        #   Zoneout, https://arxiv.org/pdf/1606.01305.pdf
        #   Recurrent Dropout, https://arxiv.org/pdf/1603.05118.pdf
        #   Layer Normalisation
        #   Recurrent Batch Normalisation, https://arxiv.org/pdf/1603.09025.pdf
        kH, kW = kernel_size

        self.kernel_size = kernel_size
        self.padding = int(kH // 2)
        self._weight_ih = nn.Parameter(
            torch.empty(hidden_size * 4, input_size, kH, kW))
        self._weight_hh = nn.Parameter(torch.empty(
            hidden_size * 4, hidden_size, kH, kW))
        self.bias = None
        self.zoneout_prob = zoneout

        if bias:
            self.bias = nn.Parameter(torch.zeros(hidden_size * 4))

        # --- weight initialisation
        # Xavier uniform for input-to-hidden,
        nn.init.xavier_uniform_(self._weight_ih.data)
        # Orthogonal for hidden-to-hidden,
        nn.init.orthogonal_(self._weight_hh.data)
        # From "Learning to Forget: Continual Prediction with LSTM"
        if bias:
            self.bias[:hidden_size].data.fill_(1.)

    @torch.jit.script_method
    def forward(self, x, h, c, o):
        # x: input
        # h: hidden units
        # c: state (cell) units
        # o: previous output gate
        inp = torch.cat([x, h], dim=1)
        weight = torch.cat([self._weight_ih, self._weight_hh], dim=1)
        out = nn.functional.conv2d(
            inp, weight, self.bias, stride=1, padding=self.padding)
        gf, gi, gs, go = torch.chunk(out, 4, dim=1)

        if self.training:
            mask = torch.zeros_like(h).bernoulli_(self.zoneout_prob)
            h, c, o = _lstm_zoneout(gf, gi, gs, go, h, c, mask, o)
        else:
            mask = torch.zeros_like(h)
            h, c, o = _lstm_zoneout(gf, gi, gs, go, h, c, mask, o)

        return h, c, o


class LSTMCell(torch.jit.ScriptModule):
    __constants__ = ['zoneout_prob']

    def __init__(self, input_size, hidden_size, zoneout=0, bias=True):
        super(LSTMCell, self).__init__()

        self._weight_ih = nn.Parameter(
            torch.empty(hidden_size * 4, input_size))
        self._weight_hh = nn.Parameter(
            torch.empty(hidden_size * 4, hidden_size))
        self.bias = None
        self.zoneout_prob = zoneout

        if bias:
            self.bias = nn.Parameter(torch.zeros(hidden_size * 4))

        nn.init.xavier_uniform_(self._weight_ih.data)
        nn.init.orthogonal_(self._weight_hh.data)

        if bias:
            self.bias[:hidden_size].data.fill_(1.)

    @torch.jit.script_method
    def forward(self, x, h, c, o):
        inp = torch.cat([x, h], dim=1)
        weight = torch.cat([self._weight_ih, self._weight_hh], dim=1)

        out = nn.functional.linear(inp, weight, self.bias)
        gf, gi, gs, go = torch.chunk(out, 4, dim=1)
        if self.training:
            mask = torch.zeros_like(h).bernoulli(self.zoneout_prob)
            h, c, o = _lstm_zoneout(gf, gi, gs, go, h, c, mask, o)
        else:
            mask = torch.zeros_like(h)
            h, c, o = _lstm_zoneout(gf, gi, gs, go, h, c, mask, o)

        return h, c, o
