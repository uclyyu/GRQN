import torch
import torch.nn as nn
from torch import Tensor
from functools import partial
from typing import List, Tuple
from common.utils import count_parameters, init_parameters
from common.layers import DummyModule, ConvLSTMCell, SkipConnect


class EncoderJ(nn.Module):

    def __init__(self, p):
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

    def __init__(self, p):
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
        # Stupid workaround for jit + ModuleList
        h = None

        for feature in self.features:

            if h is None:
                h = feature(x)
                v = v.expand(-1, -1, h.size(2), h.size(3))

            else:
                h = torch.cat([h, v], dim=1)
                h = feature(h)

        return h


class RepresentationD(nn.Module):

    def __init__(self):
        super(RepresentationD, self).__init__()
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
                nn.ReLU(True),
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

        gain = nn.init.calculate_gain('relu')
        self.features.apply(partial(init_parameters, gain=gain))

        print('{}: {:,} trainable parameters.'.format(
            self.__class__.__name__, count_parameters(self)))

    def forward(self, kx, kv):
        # Stupid workaround for jit + ModuleList
        # https://github.com/pytorch/pytorch/issues/16123
        kh = None

        for feature in self.features:

            if kh is None:
                kh = feature(kx)
                kv = kv.expand(-1, -1, kh.size(2), kh.size(3))

            else:
                kh = feature(torch.cat([kh, kv], dim=1))

        return kh


class GaussianFactor(nn.Module):
    def __init__(self, p):
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
    def __init__(self, input_channels, hidden_channels, kernel_size=(3, 3)):
        super(RecurrentCell, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.features = ConvLSTMCell(input_channels, hidden_channels, kernel_size)

        print('{}: {:,} trainable parameters.'.format(
            self.__class__.__name__, count_parameters(self)))

    def forward(self, x, hid=None, cel=None):
        # type: (Tensor, Optional[Tensor], Optional[Tensor]) -> Tuple[Tensor, Tensor]
        B, C, H, W = x.size()
        Ch = self.hidden_channels
        device = x.device

        if hid is None:
            hid = torch.zeros(1, device=device).expand(B, Ch, H, W)

        if cel is None:
            cel = torch.zeros(1, device=device).expand(B, Ch, H, W)

        hid, cel = self.features(x, hid, cel)

        return hid, cel


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
    def __init__(self):
        super(Decoder, self).__init__()

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(32, 128, (3, 3), (1, 1), padding=(1, 1)),
            nn.ReLU(True),
            nn.GroupNorm(2, 128),
            SkipConnect(nn.Conv2d(128, 128, (1, 1), (1, 1)),
                        DummyModule()),
            nn.ReLU(True),
            nn.GroupNorm(2, 128),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 32, (3, 3), (1, 1), padding=(1, 1)),
            nn.ReLU(True),
            nn.GroupNorm(2, 32),
            SkipConnect(nn.Conv2d(32, 32, (1, 1), (1, 1)),
                        DummyModule()),
            nn.ReLU(True),
            nn.GroupNorm(2, 32),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(32, 128, (3, 3), (1, 1), padding=(1, 1)),
            nn.ReLU(True),
            nn.GroupNorm(2, 128),
            SkipConnect(nn.Conv2d(128, 128, (1, 1), (1, 1)),
                        DummyModule()),
            nn.ReLU(True),
            nn.GroupNorm(2, 128),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 32, (3, 3), (1, 1), padding=(1, 1)),
            nn.ReLU(True),
            nn.GroupNorm(2, 32),
            SkipConnect(nn.Conv2d(32, 32, (1, 1), (1, 1)),
                        DummyModule()),
            nn.ReLU(True),
            nn.GroupNorm(2, 32),
            nn.Conv2d(32, 1, (3, 3), (1, 1), padding=(1, 1)),
            nn.Sigmoid()
        )

        gain = nn.init.calculate_gain('relu')
        self.apply(partial(init_parameters, gain=gain))

        print('{}: {:,} trainable parameters.'.format(
            self.__class__.__name__, count_parameters(self)))

    def forward(self, x):
        return self.decoder(x)


def Model(xprm):

    class _Model(torch.jit.ScriptModule):

        @xprm.capture
        def __init__(self, sizes, drop_pr):
            super(_Model, self).__init__()

            # --- default sizes
            self.size_r = sizes['model']['repr']  # nr
            self.size_q = sizes['model']['latent']  # nq
            self.size_h = sizes['model']['hidden']  # nh
            self.size_v = sizes['model']['query']  # nv

            ij_inp_size = self.size_r + self.size_q + self.size_h
            id_inp_size = ij_inp_size + self.size_v
            gj_inp_size = self.size_r + self.size_h
            gd_inp_size = gj_inp_size + self.size_v

            # --- representation network
            self.rop_encoder = RepresentationD()

            # --- inference operators
            self.iop_jencoder = EncoderJ(p=drop_pr['iop_j']['encoder'])
            self.iop_jposterior = GaussianFactor(p=drop_pr['iop_j']['posterior'])
            self.iop_jstate = RecurrentCell(ij_inp_size, self.size_h)

            self.iop_dencoder = EncoderD(p=drop_pr['iop_d']['encoder'])
            self.iop_dposterior = GaussianFactor(p=drop_pr['iop_d']['posterior'])
            self.iop_dstate = RecurrentCell(id_inp_size, self.size_h)

            # --- generation operators
            self.gop_jprior = GaussianFactor(p=drop_pr['gop_j']['prior'])
            self.gop_jstate = RecurrentCell(gj_inp_size, self.size_h)

            self.gop_dprior = GaussianFactor(p=drop_pr['gop_d']['prior'])
            self.gop_dstate = RecurrentCell(gd_inp_size, self.size_h)

            self.gop_jdelta = GeneratorDelta(p=drop_pr['gop_j']['delta'])
            self.gop_ddelta = GeneratorDelta(p=drop_pr['gop_d']['delta'])

            # --- decoding operators
            self.dop_jdecoder = Decoder()
            self.dop_ddecoder = Decoder()

            print('{}: {:,} trainable parameters.'.format(
                self.__class__.__name__, count_parameters(self)))

        @torch.jit.script_method
        def _forward_mc_loop(self, ndraw: int, cnd_repr: Tensor, qry_jrep: Tensor, qry_drep: Tensor, qry_v: Tensor,
                             hi_jlos: Tensor, ci_jlos: Tensor,
                             hi_dlos: Tensor, ci_dlos: Tensor,
                             hg_jlos: Tensor, cg_jlos: Tensor,
                             hg_dlos: Tensor, cg_dlos: Tensor,
                             ug_jlos: Tensor, ug_dlos: Tensor,
                             pr_means_jlos: List[Tensor], pr_logvs_jlos: List[Tensor],
                             pr_means_dlos: List[Tensor], pr_logvs_dlos: List[Tensor],
                             po_means_jlos: List[Tensor], po_logvs_jlos: List[Tensor],
                             po_means_dlos: List[Tensor], po_logvs_dlos: List[Tensor]
                             ) -> Tuple[Tensor, Tensor, List[Tensor], List[Tensor], List[Tensor], List[Tensor], List[Tensor], List[Tensor], List[Tensor], List[Tensor]]:

            for ast in range(ndraw):
                _, pr_mean_jlos, pr_logv_jlos = self.gop_jprior(hg_jlos)
                _, pr_mean_dlos, pr_logv_dlos = self.gop_dprior(hg_dlos)

                inp_ij = torch.cat([cnd_repr, qry_jrep, hg_jlos], dim=1)
                inp_id = torch.cat([cnd_repr, qry_drep, hg_dlos, qry_v], dim=1)

                hi_jlos, ci_jlos = self.iop_jstate(
                    inp_ij, hi_jlos, ci_jlos)
                hi_dlos, ci_dlos = self.iop_dstate(
                    inp_id, hi_dlos, ci_dlos)

                po_z_jlos, po_mean_jlos, po_logv_jlos = self.iop_jposterior(hi_jlos)
                po_z_dlos, po_mean_dlos, po_logv_dlos = self.iop_dposterior(hi_dlos)

                inp_gj = torch.cat([cnd_repr, po_z_jlos], dim=1)
                inp_gd = torch.cat([cnd_repr, po_z_dlos, qry_v], dim=1)

                hg_jlos, cg_jlos = self.gop_jstate(
                    inp_gj, hg_jlos, cg_jlos)
                hg_dlos, cg_dlos = self.gop_dstate(
                    inp_gd, hg_dlos, cg_dlos)

                ug_jlos = ug_jlos + self.gop_jdelta(ug_jlos, hg_jlos)
                ug_dlos = ug_dlos + self.gop_ddelta(ug_dlos, hg_dlos)

                # collect means and log variances
                pr_means_jlos.append(pr_mean_jlos)
                pr_logvs_jlos.append(pr_logv_jlos)
                pr_means_dlos.append(pr_mean_dlos)
                pr_logvs_dlos.append(pr_logv_dlos)

                po_means_jlos.append(po_mean_jlos)
                po_logvs_jlos.append(po_logv_jlos)
                po_means_dlos.append(po_mean_dlos)
                po_logvs_dlos.append(po_logv_dlos)

            return ug_jlos, ug_dlos, pr_means_jlos, pr_logvs_jlos, pr_means_dlos, pr_logvs_dlos, po_means_jlos, po_logvs_jlos, po_means_dlos, po_logvs_dlos

        @torch.jit.script_method
        def _predict_mc_loop(self, ndraw: int, cnd_repr: Tensor, qry_v: Tensor,
                             hg_jlos: Tensor, cg_jlos: Tensor,
                             hg_dlos: Tensor, cg_dlos: Tensor,
                             ug_jlos: Tensor, ug_dlos: Tensor,
                             pr_means_jlos: List[Tensor], pr_logvs_jlos: List[Tensor],
                             pr_means_dlos: List[Tensor], pr_logvs_dlos: List[Tensor]
                             ) -> Tuple[Tensor, Tensor, List[Tensor], List[Tensor], List[Tensor], List[Tensor]]:

            for ast in range(ndraw):
                pr_z_jlos, pr_mean_jlos, pr_logv_jlos = self.gop_jprior(hg_jlos)
                pr_z_dlos, pr_mean_dlos, pr_logv_dlos = self.gop_dprior(hg_dlos)

                inp_gj = torch.cat([cnd_repr, pr_z_jlos], dim=1)
                inp_gd = torch.cat([cnd_repr, pr_z_dlos, qry_v], dim=1)

                hg_jlos, cg_jlos = self.gop_jstate(
                    inp_gj, hg_jlos, cg_jlos)
                hg_dlos, cg_dlos = self.gop_dstate(
                    inp_gd, hg_dlos, cg_dlos)

                ug_jlos = ug_jlos + self.gop_jdelta(ug_jlos, hg_jlos)
                ug_dlos = ug_dlos + self.gop_ddelta(ug_dlos, hg_dlos)

                pr_means_jlos.append(pr_mean_jlos)
                pr_logvs_jlos.append(pr_logv_jlos)
                pr_means_dlos.append(pr_mean_dlos)
                pr_logvs_dlos.append(pr_logv_dlos)

            return ug_jlos, ug_dlos, pr_means_jlos, pr_logvs_jlos, pr_means_dlos, pr_logvs_dlos

        def forward(self, cnd_x, cnd_v, qry_jlos, qry_dlos, qry_v, ndraw):
            pr_means_jlos = []
            pr_logvs_jlos = []
            pr_means_dlos = []
            pr_logvs_dlos = []
            po_means_jlos = []
            po_logvs_jlos = []
            po_means_dlos = []
            po_logvs_dlos = []

            kB, kN, kC, kH, kW = cnd_x.size()
            dev = cnd_x.device

            cnd_x = cnd_x.view(kB * kN, kC, kH, kW)
            cnd_v = cnd_v.view(kB * kN, self.size_v, 1, 1)
            cnd_repr = self.rop_encoder(cnd_x, cnd_v)
            cnd_repr = cnd_repr.view(kB, kN, self.size_r, 1, 1).sum(dim=1)
            cnd_repr = cnd_repr.expand(-1, -1, 16, 16)

            qry_jrep = self.iop_jencoder(qry_jlos)
            qry_drep = self.iop_dencoder(qry_dlos, qry_v)
            qry_jrep = qry_jrep.expand(-1, -1, 16, 16)
            qry_drep = qry_drep.expand(-1, -1, 16, 16)
            qry_v = qry_v.expand(-1, -1, 16, 16)

            hi_jlos = torch.zeros(1, device=dev).expand(kB, self.size_h, 16, 16)
            ci_jlos = torch.zeros(1, device=dev).expand(kB, self.size_h, 16, 16)
            hg_jlos = torch.zeros(1, device=dev).expand(kB, self.size_h, 16, 16)
            cg_jlos = torch.zeros(1, device=dev).expand(kB, self.size_h, 16, 16)

            hi_dlos = torch.zeros(1, device=dev).expand(kB, self.size_h, 16, 16)
            ci_dlos = torch.zeros(1, device=dev).expand(kB, self.size_h, 16, 16)
            hg_dlos = torch.zeros(1, device=dev).expand(kB, self.size_h, 16, 16)
            cg_dlos = torch.zeros(1, device=dev).expand(kB, self.size_h, 16, 16)

            ug_jlos = torch.zeros(1, device=dev).expand(kB, self.size_h, 16, 16)
            ug_dlos = torch.zeros(1, device=dev).expand(kB, self.size_h, 16, 16)

            (ug_jlos, ug_dlos,
             pr_means_jlos, pr_logvs_jlos, pr_means_dlos, pr_logvs_dlos,
             po_means_jlos, po_logvs_jlos, po_means_dlos, po_logvs_dlos) = self._forward_mc_loop(
                ndraw,
                cnd_repr, qry_jrep, qry_drep, qry_v,
                hi_jlos, ci_jlos, hi_dlos, ci_dlos,
                hg_jlos, cg_jlos, hg_dlos, cg_dlos,
                ug_jlos, ug_dlos,
                pr_means_jlos, pr_logvs_jlos, pr_means_dlos, pr_logvs_dlos,
                po_means_jlos, po_logvs_jlos, po_means_dlos, po_logvs_dlos)

            dec_jlos = self.dop_jdecoder(ug_jlos)
            dec_dlos = self.dop_ddecoder(ug_dlos)

            return (dec_jlos, dec_dlos,
                    pr_means_jlos, pr_logvs_jlos, pr_means_dlos, pr_logvs_dlos,
                    po_means_jlos, po_logvs_jlos, po_means_dlos, po_logvs_dlos)

        def predict(self, cnd_x, cnd_v, qry_v, ndraw):
            pr_means_jlos = []
            pr_logvs_jlos = []
            pr_means_dlos = []
            pr_logvs_dlos = []

            kB, kN, kC, kH, kW = cnd_x.size()
            dev = cnd_x.device

            cnd_x = cnd_x.view(kB * kN, kC, kH, kW)
            cnd_v = cnd_v.view(kB * kN, self.size_v, 1, 1)
            cnd_repr = self.rop_encoder(cnd_x, cnd_v)
            cnd_repr = cnd_repr.view(kB, kN, self.size_r, 1, 1).sum(dim=1)
            cnd_repr = cnd_repr.expand(-1, -1, 16, 16)
            qry_v = qry_v.expand(-1, -1, 16, 16)

            hg_jlos = torch.zeros(1, device=dev).expand(kB, self.size_h, 16, 16)
            cg_jlos = torch.zeros(1, device=dev).expand(kB, self.size_h, 16, 16)

            hg_dlos = torch.zeros(1, device=dev).expand(kB, self.size_h, 16, 16)
            cg_dlos = torch.zeros(1, device=dev).expand(kB, self.size_h, 16, 16)

            ug_jlos = torch.zeros(1, device=dev).expand(kB, self.size_h, 16, 16)
            ug_dlos = torch.zeros(1, device=dev).expand(kB, self.size_h, 16, 16)

            # --- Inference/generation
            (ug_jlos, ug_dlos,
             pr_means_jlos, pr_logvs_jlos, 
             pr_means_dlos, pr_logvs_dlos) = self._predict_mc_loop(
                ndraw,
                cnd_repr, qry_v,
                hg_jlos, cg_jlos, hg_dlos, cg_dlos,
                ug_jlos, ug_dlos,
                pr_means_jlos, pr_logvs_jlos, pr_means_dlos, pr_logvs_dlos)

            # --- Decoding
            dec_jlos = self.dop_jdecoder(ug_jlos)
            dec_dlos = self.dop_ddecoder(ug_dlos)

            return (dec_jlos, dec_dlos,
                    pr_means_jlos, pr_logvs_jlos, 
                    pr_means_dlos, pr_logvs_dlos)

    return _Model
