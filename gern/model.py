import torch
import torch.nn as nn
from functools import partial
from .utils import DummyModule, ConvLSTMCell, LSTMCell, SkipConnect, count_parameters, init_parameters


class EncoderJ(nn.Module):
    def __init__(self, p=.1):
        super(EncoderJ, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 8, (3, 3), (2, 2), padding=1),
            nn.ReLU(True),
            nn.Dropout2d(p),
            SkipConnect(
                nn.Conv2d(8, 8, (3, 3), (1, 1), padding=1),
                DummyModule()),
            nn.ReLU(True),
            nn.MaxPool2d((2, 2), stride=(2, 2)),
            nn.Conv2d(8, 16, (3, 3), (2, 2), padding=1),
            nn.ReLU(True),
            nn.Dropout2d(p),
            SkipConnect(
                nn.Conv2d(16, 16, (3, 3), (1, 1), padding=1),
                DummyModule()),
            nn.ReLU(True),
            nn.MaxPool2d((2, 2), stride=(2, 2)),
            nn.Conv2d(16, 32, (3, 3), (2, 2), padding=1),
            nn.ReLU(True),
            nn.Dropout2d(p),
            SkipConnect(
                nn.Conv2d(32, 32, (3, 3), (1, 1), padding=1),
                DummyModule()),
            nn.ReLU(True),
            nn.MaxPool2d((2, 2), stride=(2, 2)),
            nn.Conv2d(32, 32, (3, 3), (2, 2), padding=1),
            nn.MaxPool2d((2, 2), stride=(1, 1))
        )

        gain = nn.init.calculate_gain('relu')
        self.features.apply(partial(init_parameters, gain=gain))

        print('{}: {:,} trainable parameters.'.format(
            self.__class__.__name__, count_parameters(self)))

    def forward(self, x):
        return self.features(x)


class EncoderD(nn.Module):
    def __init__(self, p=.1):
        super(EncoderD, self).__init__()

        self.features = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, 8, (3, 3), (2, 2), padding=1),
                nn.ReLU(True),
                nn.Dropout2d(p),
                SkipConnect(
                    nn.Conv2d(8, 8, (3, 3), (1, 1), padding=1),
                    DummyModule()),
                nn.ReLU(True),
                nn.MaxPool2d((2, 2), stride=(2, 2)),
                nn.Conv2d(8, 16, (3, 3), (2, 2), padding=1),
                nn.ReLU(True),
                nn.Dropout2d(p),
                SkipConnect(
                    nn.Conv2d(16, 16, (3, 3), (1, 1), padding=1),
                    DummyModule()),
                nn.ReLU(True),
                nn.MaxPool2d((2, 2), stride=(2, 2)),
            ),
            nn.Sequential(
                nn.Conv2d(20, 32, (3, 3), (2, 2), padding=1),
                nn.ReLU(True),
                nn.Dropout2d(p),
                SkipConnect(
                    nn.Conv2d(32, 32, (3, 3), (1, 1), padding=1),
                    DummyModule()),
                nn.ReLU(True),
                nn.MaxPool2d((2, 2), stride=(2, 2)),
                nn.Conv2d(32, 32, (3, 3), (2, 2), padding=1),
                nn.MaxPool2d((2, 2), stride=(1, 1))
            )
        ])

        gain = nn.init.calculate_gain('relu')
        self.features.apply(partial(init_parameters, gain=gain))

        print('{}: {:,} trainable parameters.'.format(
            self.__class__.__name__, count_parameters(self)))

    def forward(self, x, v):
        h = self.features[0](x)
        v = v.expand(-1, -1, h.size(2), h.size(3))
        h = torch.cat([h, v], dim=1)
        h = self.features[1](h)
        return h


class RepresentationD(nn.Module):
    def __init__(self):
        super(RepresentationD, self).__init__()
        gain = nn.init.calculate_gain('relu')

        self.features = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, 8, (3, 3), (2, 2), padding=1),
                nn.ReLU(True),
                SkipConnect(
                    nn.Conv2d(8, 8, (3, 3), (1, 1), padding=1),
                    DummyModule()),
                nn.ReLU(True),
                nn.MaxPool2d((2, 2), stride=(2, 2)),
                nn.Conv2d(8, 16, (3, 3), (1, 1), padding=1),
                nn.ReLU(True),
                SkipConnect(
                    nn.Conv2d(16, 16, (3, 3), (1, 1), padding=1),
                    DummyModule()),
                nn.ReLU(True),
                nn.MaxPool2d((2, 2), stride=(2, 2)),
                nn.Conv2d(16, 32, (3, 3), (1, 1), padding=1),
                nn.ReLU(.01),
                SkipConnect(
                    nn.Sequential(
                        nn.Conv2d(32, 32, (3, 3), (1, 1), padding=1),
                        nn.ReLU(True),
                        nn.Conv2d(32, 32, (3, 3), (1, 1), padding=1),
                    ), DummyModule()),
                nn.ReLU(True),
                nn.MaxPool2d((2, 2), stride=(2, 2))
            ),
            nn.Sequential(
                nn.Conv2d(36, 32, (3, 3), (2, 2), padding=1),
                nn.ReLU(True),
                nn.MaxPool2d((2, 2), stride=(2, 2)),
                nn.Conv2d(32, 64, (3, 3), (1, 1), padding=1),
                nn.ReLU(True),
                SkipConnect(
                    nn.Conv2d(64, 64, (1, 1), (1, 1), bias=False),
                    nn.Sequential(
                        nn.Conv2d(64, 64, (3, 3), (1, 1), padding=1),
                        nn.ReLU(True),
                        nn.Conv2d(64, 64, (3, 3), (1, 1), padding=1))
                ),
                nn.ReLU(True),
                nn.MaxPool2d((2, 2), stride=(2, 2)),
                nn.Conv2d(64, 128, (3, 3), (2, 2), padding=1)
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
    def __init__(self, p=.1):
        super(GaussianFactor, self).__init__()
        gain = nn.init.calculate_gain('relu')

        self.layer = nn.Sequential(
            nn.Conv2d(32, 32, (3, 3), (1, 1), padding=1),
            nn.ReLU(True),
            nn.Dropout2d(p),
            SkipConnect(
                nn.Conv2d(32, 32, (3, 3), (1, 1), padding=1),
                DummyModule()),
            nn.ReLU(True),
            nn.Dropout2d(p),
            nn.Conv2d(32, 64, (1, 1), (1, 1))  # mean, log-variance
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
    def __init__(self, p=.1):
        super(GeneratorDelta, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(64, 64, (3, 3), (1, 1), padding=1),
            nn.ReLU(True),
            nn.Dropout2d(p),
            SkipConnect(nn.Conv2d(64, 64, (3, 3), (1, 1), padding=1),
                        DummyModule()),
            nn.ReLU(True),
            nn.Dropout2d(p),
            nn.Conv2d(64, 32, (3, 3), (1, 1), padding=1)
        )

        gain = nn.init.calculate_gain('relu')
        self.apply(partial(init_parameters, gain=gain))

        print('{}: {:,} trainable parameters.'.format(
            self.__class__.__name__, count_parameters(self)))

    def forward(self, u, h):
        inp = torch.cat([u, h], dim=1)
        return self.layers(inp)


class Decoder(nn.Module):
    def __init__(self, p=.1):
        super(Decoder, self).__init__()

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 32, (3, 3), (2, 2), padding=(0, 0)),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, (3, 3), (2, 2), padding=(1, 1)),
            nn.ReLU(True),
            nn.Dropout2d(p),
            SkipConnect(
                nn.Conv2d(32, 32, (3, 3), (1, 1), padding=1),
                DummyModule()),
            nn.ReLU(True),
            nn.Dropout2d(p),
            nn.ConvTranspose2d(32, 16, (3, 3), (2, 2), padding=(1, 1)),
            nn.ReLU(True),
            nn.Dropout2d(p),
            SkipConnect(
                nn.Conv2d(16, 16, (3, 3), (1, 1), padding=1),
                DummyModule()),
            nn.ReLU(True),
            nn.Dropout2d(p),
            nn.ConvTranspose2d(16, 8, (3, 3), (2, 2), padding=(1, 1)),
            nn.ReLU(True),
            nn.Dropout2d(p),
            SkipConnect(
                nn.Conv2d(8, 8, (3, 3), (1, 1), padding=1),
                DummyModule()),
            nn.ReLU(True),
            nn.Dropout2d(p),
            nn.Conv2d(8, 1, (3, 3), (1, 1), padding=(1, 1)),
            nn.Sigmoid()
        )

        gain = nn.init.calculate_gain('relu')
        self.apply(partial(init_parameters, gain=gain))

        print('{}: {:,} trainable parameters.'.format(
            self.__class__.__name__, count_parameters(self)))

    def forward(self, x):
        return self.decoder(x)[:, :, 1:, 1:]


if __name__ == '__main__':
    # k = torch.randn(1, 1, 240, 320)
    # x = torch.randn(1, 1, 256, 256)
    # v = torch.randn(1, 4, 1, 1)
    # u = torch.randn(1, 64, 16, 16)

    # net = RepresentationD()
    # print(net(k, v).size())

    # net = EncoderJ()
    # print(net(x).size())

    # net = EncoderD()
    # print(net(x, v).size())

    # net = Decoder()
    # print(net(u).size())
    pass
