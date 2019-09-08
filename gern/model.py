import torch
import torch.nn as nn
from functools import partial
from .utils import DummyModule, ConvLSTMCell, LSTMCell, GroupNorm2d, GroupNorm1d, SkipConnect, count_parameters, init_parameters


class LosRepresentationEncoder(nn.Module):
    def __init__(self):
        super(LosRepresentationEncoder, self).__init__()

        self.features = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, 16, (3, 3), (2, 2), padding=1),
                nn.LeakyReLU(.01),
                GroupNorm2d(16, 4),
                SkipConnect(
                    nn.Conv2d(16, 16, (3, 3), (1, 1), padding=1),
                    DummyModule()),
                nn.LeakyReLU(.01),
                GroupNorm2d(16, 4),
                nn.MaxPool2d((2, 2), stride=(2, 2)),
                nn.Conv2d(16, 32, (3, 3), (1, 1), padding=1),
                nn.LeakyReLU(.01),
                GroupNorm2d(32, 8),
                SkipConnect(
                    nn.Conv2d(32, 32, (3, 3), (1, 1), padding=1),
                    DummyModule()),
                nn.LeakyReLU(.01),
                GroupNorm2d(32, 8),
                nn.MaxPool2d((2, 2), stride=(2, 2)),
                nn.Conv2d(32, 64, (3, 3), (1, 1), padding=1),
                nn.ReLU(.01),
                GroupNorm2d(64, 8),
                SkipConnect(
                    nn.Sequential(
                        nn.Conv2d(64, 64, (3, 3), (1, 1), padding=1),
                        nn.LeakyReLU(.01),
                        GroupNorm2d(64, 8),
                        nn.Conv2d(64, 64, (3, 3), (1, 1), padding=1),
                    ), DummyModule()),
                nn.LeakyReLU(.01),
                GroupNorm2d(64, 8),
                nn.MaxPool2d((2, 2), stride=(2, 2))
            ),
            nn.Conv2d(68, 128, (3, 3), (2, 2), padding=1),
            nn.Conv2d(64, 128, (3, 3), (2, 2), padding=1),
            nn.Sequential(
                nn.LeakyReLU(.01),
                GroupNorm2d(128, 8),
                nn.MaxPool2d((2, 2), stride=(2, 2)),
                nn.Conv2d(128, 128, (3, 3), (1, 1), padding=1),
                nn.LeakyReLU(.01),
                GroupNorm2d(128, 8),
                SkipConnect(
                    nn.Conv2d(128, 128, (1, 1), (1, 1), bias=False),
                    nn.Sequential(
                        nn.Conv2d(128, 128, (3, 3), (1, 1), padding=1),
                        nn.LeakyReLU(.01),
                        GroupNorm2d(128, 8),
                        nn.Conv2d(128, 128, (3, 3), (1, 1), padding=1))
                ),
                nn.LeakyReLU(.01),
                GroupNorm2d(128, 8),
                nn.MaxPool2d((2, 2), stride=(2, 2)),
                nn.Conv2d(128, 128, (3, 3), (2, 2), padding=1)
            )
        ])

        gain = nn.init.calculate_gain('leaky_relu')
        self.features.apply(partial(init_parameters, gain=gain))

        print('{}: {:,} trainable parameters.'.format(
            self.__class__.__name__, count_parameters(self)))

    def forward(self, x, v=None, index=1):
        h = self.features[0](x)
        if index == 1:
            v = v.expand(-1, -1, h.size(2), h.size(3))
            h = torch.cat([h, v], dim=1)
        h = self.features[index](h)
        h = self.features[3](h)
        return h


class DepRepresentationEncoder(nn.Module):
    def __init__(self):
        super(DepRepresentationEncoder, self).__init__()
        gain = nn.init.calculate_gain('leaky_relu')

        self.features = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, 32, (3, 3), (2, 2), padding=1),
                nn.LeakyReLU(.01),
                GroupNorm2d(32, 8),
                SkipConnect(
                    nn.Conv2d(32, 32, (3, 3), (1, 1), padding=1),
                    DummyModule()),
                nn.LeakyReLU(.01),
                GroupNorm2d(32, 8),
                nn.MaxPool2d((2, 2), stride=(2, 2)),
                nn.Conv2d(32, 64, (3, 3), (1, 1), padding=1),
                nn.LeakyReLU(.01),
                GroupNorm2d(64, 8),
                SkipConnect(
                    nn.Conv2d(64, 64, (3, 3), (1, 1), padding=1),
                    DummyModule()),
                nn.LeakyReLU(.01),
                GroupNorm2d(64, 8),
                nn.MaxPool2d((2, 2), stride=(2, 2)),
                nn.Conv2d(64, 128, (3, 3), (1, 1), padding=1),
                nn.ReLU(.01),
                GroupNorm2d(128, 8),
                SkipConnect(
                    nn.Sequential(
                        nn.Conv2d(128, 128, (3, 3), (1, 1), padding=1),
                        nn.LeakyReLU(.01),
                        GroupNorm2d(128, 8),
                        nn.Conv2d(128, 128, (3, 3), (1, 1), padding=1),
                    ), DummyModule()),
                nn.LeakyReLU(.01),
                GroupNorm2d(128, 8),
                nn.MaxPool2d((2, 2), stride=(2, 2))
            ),
            nn.Sequential(
                nn.Conv2d(132, 128, (3, 3), (2, 2), padding=1),
                nn.LeakyReLU(.01),
                GroupNorm2d(128, 8),
                nn.MaxPool2d((2, 2), stride=(2, 2)),
                nn.Conv2d(128, 256, (3, 3), (1, 1), padding=1),
                nn.LeakyReLU(.01),
                GroupNorm2d(256, 8),
                SkipConnect(
                    nn.Conv2d(256, 256, (1, 1), (1, 1), bias=False),
                    nn.Sequential(
                        nn.Conv2d(256, 256, (3, 3), (1, 1), padding=1),
                        nn.LeakyReLU(.01),
                        GroupNorm2d(256, 8),
                        nn.Conv2d(256, 256, (3, 3), (1, 1), padding=1))
                ),
                nn.LeakyReLU(.01),
                GroupNorm2d(256, 8),
                nn.MaxPool2d((2, 2), stride=(2, 2)),
                nn.Conv2d(256, 256, (3, 3), (2, 2), padding=1)
            )
        ])

        self.features.apply(partial(init_parameters, gain=gain))

        print('{}: {:,} trainable parameters.'.format(
            self.__class__.__name__, count_parameters(self)))

    def forward(self, kx, kv):
        kh = self.features[0](kx)
        kv = kv.expand(-1, -1, kh.size(2), kh.size(3))
        kh = self.features[1](torch.cat([kh, kv], dim=1))

        return kh


class GaussianFactor(nn.Module):
    def __init__(self):
        super(GaussianFactor, self).__init__()
        gain = nn.init.calculate_gain('leaky_relu')

        self.layer = nn.Sequential(
            nn.Conv2d(128, 128, (3, 3), (1, 1), padding=1),
            nn.LeakyReLU(.01),
            SkipConnect(
                nn.Conv2d(128, 128, (3, 3), (1, 1), padding=1),
                nn.Conv2d(128, 128, (1, 1), (1, 1), bias=False)),
            nn.LeakyReLU(.01),
            nn.Conv2d(128, 256, (1, 1), (1, 1))  # mean, log-variance
        )

        self.apply(partial(init_parameters, gain=gain))

        print('{}: {:,} trainable parameters.'.format(
            self.__class__.__name__, count_parameters(self)))

    def forward(self, inp):
        mean, logv = torch.chunk(self.layer(inp), 2, dim=1)
        scale = (0.5 * logv).exp()
        z = torch.randn(inp.size(), device=inp.device)
        sample = z * scale + mean
        return sample, mean, logv


class RecurrentCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size=(3, 3), feature_size=None, zoneout=0.15):
        super(RecurrentCell, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.features = ConvLSTMCell(
            input_channels, hidden_channels, kernel_size, zoneout=zoneout)

        print('{}: {:,} trainable parameters.'.format(
            self.__class__.__name__, count_parameters(self)))

    def forward(self, x, hid=None, cel=None, pog=None):
        # type: (Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor]) -> Tuple[Tensor, Tensor, Tensor]
        B, C, H, W = x.size()
        Ch = self.hidden_channels
        device = x.device

        if hid is None:
            hid = torch.zeros(1, device=device).expand(B, Ch, H, W)
        if cel is None:
            cel = torch.zeros(1, device=device).expand(B, Ch, H, W)
        if pog is None:
            pog = torch.zeros(1, device=device).expand(B, Ch, H, W)

        hid, cel, pog = self.features(x, hid, cel, pog)

        return hid, cel, pog


class GeneratorDelta(nn.Module):
    def __init__(self):
        super(GeneratorDelta, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(256, 128, (3, 3), (1, 1), padding=1),
            nn.LeakyReLU(.01),
            GroupNorm2d(128, 8),
            SkipConnect(nn.Conv2d(128, 128, (3, 3), (1, 1), padding=1),
                        DummyModule()),
            nn.LeakyReLU(.01),
            GroupNorm2d(128, 8),
            nn.Conv2d(128, 128, (3, 3), (1, 1), padding=1)
        )

        gain = nn.init.calculate_gain('leaky_relu')
        self.apply(partial(init_parameters, gain=gain))

        print('{}: {:,} trainable parameters.'.format(
            self.__class__.__name__, count_parameters(self)))

    def forward(self, u, h):
        inp = torch.cat([u, h], dim=1)
        return self.layers(inp)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder_base = nn.Sequential(
            nn.ConvTranspose2d(128, 128, (7, 3), (2, 2), padding=(0, 1)),
            nn.LeakyReLU(.01),
            nn.ConvTranspose2d(128, 128, (7, 3), (2, 2), padding=(0, 1)),
            nn.LeakyReLU(.01),
            GroupNorm2d(128, 8),
            SkipConnect(
                nn.Conv2d(128, 256, (3, 3), (1, 1), padding=1),
                nn.Conv2d(128, 256, (1, 1), (1, 1), bias=False)),
            nn.LeakyReLU(.01),
            GroupNorm2d(256, 8),
            nn.ConvTranspose2d(256, 128, (5, 3), (2, 2), padding=(0, 1)),
            nn.LeakyReLU(.01),
            GroupNorm2d(128, 8),
            SkipConnect(
                nn.Conv2d(128, 64, (3, 3), (1, 1), padding=1),
                nn.Conv2d(128, 64, (1, 1), (1, 1), bias=False)),
            nn.LeakyReLU(.01),
            GroupNorm2d(64, 8),
            nn.ConvTranspose2d(64, 64, (3, 3), (2, 2), padding=(1, 1)),
            nn.LeakyReLU(.01),
            GroupNorm2d(64, 8),
            SkipConnect(
                nn.Conv2d(64, 64, (3, 3), (1, 1), padding=1),
                nn.Conv2d(64, 64, (1, 1), (1, 1), bias=False)),
            nn.LeakyReLU(.01),
            GroupNorm2d(64, 8),
            nn.Conv2d(64, 1, (3, 3), (1, 1), padding=(1, 1)),
            nn.Sigmoid()
        )
        gain = nn.init.calculate_gain('leaky_relu')
        self.apply(partial(init_parameters, gain=gain))

        print('{}: {:,} trainable parameters.'.format(
            self.__class__.__name__, count_parameters(self)))

    def forward(self, x):
        y = self.decoder_base(x)
        return y[:, :, 1:, 1:]
