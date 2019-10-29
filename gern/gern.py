import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Tuple
from .utils import count_parameters, init_parameters
from .model import *


class GeRN(nn.Module):
    def __init__(self, nr=256, nq=64, nh=64, nv=4):
        super(GeRN, self).__init__()
        gain = nn.init.calculate_gain('leaky_relu')

        # --- default sizes
        self.size_r = nr
        self.size_q = nq
        self.size_h = nh
        self.size_v = nv

        # --- representation network
        self.rop_encoder = RepresentationD()

        # --- inference operators
        self.iop_jencoder = EncoderJ()
        self.iop_jposterior = GaussianFactor()
        self.iop_jstate = RecurrentCell(nr + nq + nh, nh, zoneout=0.3)

        self.iop_dencoder = EncoderD()
        self.iop_dposterior = GaussianFactor()
        self.iop_dstate = RecurrentCell(nr + nq + nh, nh, zoneout=0.2)

        # --- generation operators
        self.gop_jprior = GaussianFactor()
        self.gop_jstate = RecurrentCell(nr + nh, nh, zoneout=0.3)

        self.gop_dprior = GaussianFactor()
        self.gop_dstate = RecurrentCell(nr + nh + nv, nh, zoneout=0.2)

        self.gop_jdelta = GeneratorDelta()
        self.gop_ddelta = GeneratorDelta()

        # --- decoding operators
        self.dop_jdecoder = Decoder()
        self.dop_ddecoder = Decoder()

        print('{}: {:,} trainable parameters.'.format(
            self.__class__.__name__, count_parameters(self)))

    @torch.jit.script_method
    def _forward_mc_loop(self, ndraw: int, cnd_repr: Tensor, qry_jrep: Tensor, qry_drep: Tensor, qry_v: Tensor,
                         hi_jlos: Tensor, ci_jlos: Tensor, oi_jlos: Tensor,
                         hi_dlos: Tensor, ci_dlos: Tensor, oi_dlos: Tensor,
                         hg_jlos: Tensor, cg_jlos: Tensor, og_jlos: Tensor,
                         hg_dlos: Tensor, cg_dlos: Tensor, og_dlos: Tensor,
                         ug_jlos: Tensor, ug_dlos: Tensor,
                         pr_means_jlos: List[Tensor], pr_logvs_jlos: List[Tensor],
                         pr_means_dlos: List[Tensor], pr_logvs_dlos: List[Tensor],
                         po_means_jlos: List[Tensor], po_logvs_jlos: List[Tensor],
                         po_means_dlos: List[Tensor], po_logvs_dlos: List[Tensor]
                         ) -> Tuple[Tensor, Tensor, List[Tensor], List[Tensor], List[Tensor], List[Tensor], List[Tensor], List[Tensor], List[Tensor], List[Tensor]]:
        """
        def _forward_mc_loop(self,
        ndraw, cnd_repr, qry_jrep, qry_drep, qry_v,
        hi_jlos, ci_jlos, oi_jlos, hi_dlos, ci_dlos,
        oi_dlos, hg_jlos, cg_jlos, og_jlos, hg_dlos,
        cg_dlos, og_dlos, ug_jlos, ug_dlos,
        pr_means_jlos, pr_logvs_jlos, pr_means_dlos,
        pr_logvs_dlos, po_means_jlos, po_logvs_jlos,
        po_means_dlos, po_logvs_dlos):
        type: (int, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,
        Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,
        Tensor, Tensor, Tensor, Tensor, Tensor,
        List[Tensor], List[Tensor], List[Tensor], List[Tensor],
        List[Tensor], List[Tensor], List[Tensor], List[Tensor]) ->
        Tuple[Tensor, Tensor, List[Tensor], List[Tensor],
        List[Tensor], List[Tensor], List[Tensor], List[Tensor], List[Tensor], List[Tensor],]
        """

        for ast in range(ndraw):
            _, pr_mean_jlos, pr_logv_jlos = self.gop_jprior(hg_jlos)
            _, pr_mean_dlos, pr_logv_dlos = self.gop_dprior(hg_dlos)

            inp_ij = torch.cat([cnd_repr, qry_jrep, hg_jlos], dim=1)
            inp_id = torch.cat([cnd_repr, qry_drep, hg_dlos], dim=1)

            hi_jlos, ci_jlos, oi_jlos = self.iop_jstate(
                inp_ij, hi_jlos, ci_jlos, oi_jlos)
            hi_dlos, ci_dlos, oi_dlos = self.iop_dstate(
                inp_id, hi_dlos, ci_dlos, oi_dlos)

            po_z_jlos, po_mean_jlos, po_logv_jlos = self.iop_jposterior(
                hi_jlos)
            po_z_dlos, po_mean_dlos, po_logv_dlos = self.iop_dposterior(
                hi_dlos)

            inp_gj = torch.cat([cnd_repr, po_z_jlos], dim=1)
            inp_gd = torch.cat([cnd_repr, po_z_dlos, qry_v], dim=1)

            hg_jlos, cg_jlos, og_jlos = self.gop_jstate(
                inp_gj, hg_jlos, cg_jlos, og_jlos)
            hg_dlos, cg_dlos, og_dlos = self.gop_dstate(
                inp_gd, hg_dlos, cg_dlos, og_dlos)

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

    def _forward_mc_unroll7(self, ndraw: int, cnd_repr: Tensor, qry_jrep: Tensor, qry_drep: Tensor, qry_v: Tensor,
                            hi_jlos: Tensor, ci_jlos: Tensor, oi_jlos: Tensor,
                            hi_dlos: Tensor, ci_dlos: Tensor, oi_dlos: Tensor,
                            hg_jlos: Tensor, cg_jlos: Tensor, og_jlos: Tensor,
                            hg_dlos: Tensor, cg_dlos: Tensor, og_dlos: Tensor,
                            ug_jlos: Tensor, ug_dlos: Tensor,
                            pr_means_jlos: List[Tensor], pr_logvs_jlos: List[Tensor],
                            pr_means_dlos: List[Tensor], pr_logvs_dlos: List[Tensor],
                            po_means_jlos: List[Tensor], po_logvs_jlos: List[Tensor],
                            po_means_dlos: List[Tensor], po_logvs_dlos: List[Tensor]
                            ) -> Tuple[Tensor, Tensor, List[Tensor], List[Tensor], List[Tensor], List[Tensor], List[Tensor], List[Tensor], List[Tensor], List[Tensor]]:

        # 1. ------------------------------------------------------------------
        _, pr_mean_jlos, pr_logv_jlos = self.gop_jprior(hg_jlos)
        _, pr_mean_dlos, pr_logv_dlos = self.gop_dprior(hg_dlos)

        inp_ij = torch.cat([cnd_repr, qry_jrep, hg_jlos], dim=1)
        inp_id = torch.cat([cnd_repr, qry_drep, hg_dlos], dim=1)

        hi_jlos, ci_jlos, oi_jlos = self.iop_jstate(
            inp_ij, hi_jlos, ci_jlos, oi_jlos)
        hi_dlos, ci_dlos, oi_dlos = self.iop_dstate(
            inp_id, hi_dlos, ci_dlos, oi_dlos)

        po_z_jlos, po_mean_jlos, po_logv_jlos = self.iop_jposterior(
            hi_jlos)
        po_z_dlos, po_mean_dlos, po_logv_dlos = self.iop_dposterior(
            hi_dlos)

        inp_gj = torch.cat([cnd_repr, po_z_jlos], dim=1)
        inp_gd = torch.cat([cnd_repr, po_z_dlos, qry_v], dim=1)

        hg_jlos, cg_jlos, og_jlos = self.gop_jstate(
            inp_gj, hg_jlos, cg_jlos, og_jlos)
        hg_dlos, cg_dlos, og_dlos = self.gop_dstate(
            inp_gd, hg_dlos, cg_dlos, og_dlos)

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
        # 2. ------------------------------------------------------------------
        _, pr_mean_jlos, pr_logv_jlos = self.gop_jprior(hg_jlos)
        _, pr_mean_dlos, pr_logv_dlos = self.gop_dprior(hg_dlos)

        inp_ij = torch.cat([cnd_repr, qry_jrep, hg_jlos], dim=1)
        inp_id = torch.cat([cnd_repr, qry_drep, hg_dlos], dim=1)

        hi_jlos, ci_jlos, oi_jlos = self.iop_jstate(
            inp_ij, hi_jlos, ci_jlos, oi_jlos)
        hi_dlos, ci_dlos, oi_dlos = self.iop_dstate(
            inp_id, hi_dlos, ci_dlos, oi_dlos)

        po_z_jlos, po_mean_jlos, po_logv_jlos = self.iop_jposterior(
            hi_jlos)
        po_z_dlos, po_mean_dlos, po_logv_dlos = self.iop_dposterior(
            hi_dlos)

        inp_gj = torch.cat([cnd_repr, po_z_jlos], dim=1)
        inp_gd = torch.cat([cnd_repr, po_z_dlos, qry_v], dim=1)

        hg_jlos, cg_jlos, og_jlos = self.gop_jstate(
            inp_gj, hg_jlos, cg_jlos, og_jlos)
        hg_dlos, cg_dlos, og_dlos = self.gop_dstate(
            inp_gd, hg_dlos, cg_dlos, og_dlos)

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
        # 3. ------------------------------------------------------------------
        _, pr_mean_jlos, pr_logv_jlos = self.gop_jprior(hg_jlos)
        _, pr_mean_dlos, pr_logv_dlos = self.gop_dprior(hg_dlos)

        inp_ij = torch.cat([cnd_repr, qry_jrep, hg_jlos], dim=1)
        inp_id = torch.cat([cnd_repr, qry_drep, hg_dlos], dim=1)

        hi_jlos, ci_jlos, oi_jlos = self.iop_jstate(
            inp_ij, hi_jlos, ci_jlos, oi_jlos)
        hi_dlos, ci_dlos, oi_dlos = self.iop_dstate(
            inp_id, hi_dlos, ci_dlos, oi_dlos)

        po_z_jlos, po_mean_jlos, po_logv_jlos = self.iop_jposterior(
            hi_jlos)
        po_z_dlos, po_mean_dlos, po_logv_dlos = self.iop_dposterior(
            hi_dlos)

        inp_gj = torch.cat([cnd_repr, po_z_jlos], dim=1)
        inp_gd = torch.cat([cnd_repr, po_z_dlos, qry_v], dim=1)

        hg_jlos, cg_jlos, og_jlos = self.gop_jstate(
            inp_gj, hg_jlos, cg_jlos, og_jlos)
        hg_dlos, cg_dlos, og_dlos = self.gop_dstate(
            inp_gd, hg_dlos, cg_dlos, og_dlos)

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
        # 4. ------------------------------------------------------------------
        _, pr_mean_jlos, pr_logv_jlos = self.gop_jprior(hg_jlos)
        _, pr_mean_dlos, pr_logv_dlos = self.gop_dprior(hg_dlos)

        inp_ij = torch.cat([cnd_repr, qry_jrep, hg_jlos], dim=1)
        inp_id = torch.cat([cnd_repr, qry_drep, hg_dlos], dim=1)

        hi_jlos, ci_jlos, oi_jlos = self.iop_jstate(
            inp_ij, hi_jlos, ci_jlos, oi_jlos)
        hi_dlos, ci_dlos, oi_dlos = self.iop_dstate(
            inp_id, hi_dlos, ci_dlos, oi_dlos)

        po_z_jlos, po_mean_jlos, po_logv_jlos = self.iop_jposterior(
            hi_jlos)
        po_z_dlos, po_mean_dlos, po_logv_dlos = self.iop_dposterior(
            hi_dlos)

        inp_gj = torch.cat([cnd_repr, po_z_jlos], dim=1)
        inp_gd = torch.cat([cnd_repr, po_z_dlos, qry_v], dim=1)

        hg_jlos, cg_jlos, og_jlos = self.gop_jstate(
            inp_gj, hg_jlos, cg_jlos, og_jlos)
        hg_dlos, cg_dlos, og_dlos = self.gop_dstate(
            inp_gd, hg_dlos, cg_dlos, og_dlos)

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
        # 5. ------------------------------------------------------------------
        _, pr_mean_jlos, pr_logv_jlos = self.gop_jprior(hg_jlos)
        _, pr_mean_dlos, pr_logv_dlos = self.gop_dprior(hg_dlos)

        inp_ij = torch.cat([cnd_repr, qry_jrep, hg_jlos], dim=1)
        inp_id = torch.cat([cnd_repr, qry_drep, hg_dlos], dim=1)

        hi_jlos, ci_jlos, oi_jlos = self.iop_jstate(
            inp_ij, hi_jlos, ci_jlos, oi_jlos)
        hi_dlos, ci_dlos, oi_dlos = self.iop_dstate(
            inp_id, hi_dlos, ci_dlos, oi_dlos)

        po_z_jlos, po_mean_jlos, po_logv_jlos = self.iop_jposterior(
            hi_jlos)
        po_z_dlos, po_mean_dlos, po_logv_dlos = self.iop_dposterior(
            hi_dlos)

        inp_gj = torch.cat([cnd_repr, po_z_jlos], dim=1)
        inp_gd = torch.cat([cnd_repr, po_z_dlos, qry_v], dim=1)

        hg_jlos, cg_jlos, og_jlos = self.gop_jstate(
            inp_gj, hg_jlos, cg_jlos, og_jlos)
        hg_dlos, cg_dlos, og_dlos = self.gop_dstate(
            inp_gd, hg_dlos, cg_dlos, og_dlos)

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
        # 6. ------------------------------------------------------------------
        _, pr_mean_jlos, pr_logv_jlos = self.gop_jprior(hg_jlos)
        _, pr_mean_dlos, pr_logv_dlos = self.gop_dprior(hg_dlos)

        inp_ij = torch.cat([cnd_repr, qry_jrep, hg_jlos], dim=1)
        inp_id = torch.cat([cnd_repr, qry_drep, hg_dlos], dim=1)

        hi_jlos, ci_jlos, oi_jlos = self.iop_jstate(
            inp_ij, hi_jlos, ci_jlos, oi_jlos)
        hi_dlos, ci_dlos, oi_dlos = self.iop_dstate(
            inp_id, hi_dlos, ci_dlos, oi_dlos)

        po_z_jlos, po_mean_jlos, po_logv_jlos = self.iop_jposterior(
            hi_jlos)
        po_z_dlos, po_mean_dlos, po_logv_dlos = self.iop_dposterior(
            hi_dlos)

        inp_gj = torch.cat([cnd_repr, po_z_jlos], dim=1)
        inp_gd = torch.cat([cnd_repr, po_z_dlos, qry_v], dim=1)

        hg_jlos, cg_jlos, og_jlos = self.gop_jstate(
            inp_gj, hg_jlos, cg_jlos, og_jlos)
        hg_dlos, cg_dlos, og_dlos = self.gop_dstate(
            inp_gd, hg_dlos, cg_dlos, og_dlos)

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
        # 7. ------------------------------------------------------------------
        _, pr_mean_jlos, pr_logv_jlos = self.gop_jprior(hg_jlos)
        _, pr_mean_dlos, pr_logv_dlos = self.gop_dprior(hg_dlos)

        inp_ij = torch.cat([cnd_repr, qry_jrep, hg_jlos], dim=1)
        inp_id = torch.cat([cnd_repr, qry_drep, hg_dlos], dim=1)

        hi_jlos, ci_jlos, oi_jlos = self.iop_jstate(
            inp_ij, hi_jlos, ci_jlos, oi_jlos)
        hi_dlos, ci_dlos, oi_dlos = self.iop_dstate(
            inp_id, hi_dlos, ci_dlos, oi_dlos)

        po_z_jlos, po_mean_jlos, po_logv_jlos = self.iop_jposterior(
            hi_jlos)
        po_z_dlos, po_mean_dlos, po_logv_dlos = self.iop_dposterior(
            hi_dlos)

        inp_gj = torch.cat([cnd_repr, po_z_jlos], dim=1)
        inp_gd = torch.cat([cnd_repr, po_z_dlos, qry_v], dim=1)

        hg_jlos, cg_jlos, og_jlos = self.gop_jstate(
            inp_gj, hg_jlos, cg_jlos, og_jlos)
        hg_dlos, cg_dlos, og_dlos = self.gop_dstate(
            inp_gd, hg_dlos, cg_dlos, og_dlos)

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

    def forward(self, cnd_x, cnd_v, qry_jlos, qry_dlos, qry_v, ndraw=7):
        # Containers to hold outputs.
        pr_means_jlos = []
        pr_logvs_jlos = []
        pr_means_dlos = []
        pr_logvs_dlos = []
        po_means_jlos = []
        po_logvs_jlos = []
        po_means_dlos = []
        po_logvs_dlos = []

        # Size information.
        kB, kN, kC, kH, kW = cnd_x.size()
        dev = cnd_x.device

        # --- Conditional representations
        cnd_x = cnd_x.view(kB * kN, kC, kH, kW)
        cnd_v = cnd_v.view(kB * kN, -1, 1, 1)
        cnd_repr = self.rop_encoder(cnd_x, cnd_v)
        cnd_repr = cnd_repr.view(kB, kN, -1, 1, 1).sum(dim=1)
        cnd_repr = cnd_repr.expand(-1, -1, 16, 16)

        # --- Inputs for DRAW
        qry_jrep = self.iop_jencoder(qry_jlos)
        qry_drep = self.iop_dencoder(qry_dlos, qry_v)
        qry_jrep = qry_jrep.expand(-1, -1, 16, 16)
        qry_drep = qry_drep.expand(-1, -1, 16, 16)
        qry_v = qry_v.expand(-1, -1, 16, 16)

        # --- LSTM hidden/cell/prior output gate for inference/generator operators
        hi_jlos = torch.zeros(1, device=dev).expand(kB, self.size_h, 16, 16)
        ci_jlos = torch.zeros(1, device=dev).expand(kB, self.size_h, 16, 16)
        oi_jlos = torch.zeros(1, device=dev).expand(kB, self.size_h, 16, 16)
        hg_jlos = torch.zeros(1, device=dev).expand(kB, self.size_h, 16, 16)
        cg_jlos = torch.zeros(1, device=dev).expand(kB, self.size_h, 16, 16)
        og_jlos = torch.zeros(1, device=dev).expand(kB, self.size_h, 16, 16)

        hi_dlos = torch.zeros(1, device=dev).expand(kB, self.size_h, 16, 16)
        ci_dlos = torch.zeros(1, device=dev).expand(kB, self.size_h, 16, 16)
        oi_dlos = torch.zeros(1, device=dev).expand(kB, self.size_h, 16, 16)
        hg_dlos = torch.zeros(1, device=dev).expand(kB, self.size_h, 16, 16)
        cg_dlos = torch.zeros(1, device=dev).expand(kB, self.size_h, 16, 16)
        og_dlos = torch.zeros(1, device=dev).expand(kB, self.size_h, 16, 16)

        ug_jlos = torch.zeros(1, device=dev).expand(kB, self.size_h, 16, 16)
        ug_dlos = torch.zeros(1, device=dev).expand(kB, self.size_h, 16, 16)

        # --- Inference/generation
        (ug_jlos, ug_dlos,
         pr_means_jlos, pr_logvs_jlos, pr_means_dlos, pr_logvs_dlos,
         po_means_jlos, po_logvs_jlos, po_means_dlos, po_logvs_dlos) = self._forward_mc_unroll7(
            ndraw,
            cnd_repr, qry_jrep, qry_drep, qry_v,
            hi_jlos, ci_jlos, oi_jlos,
            hi_dlos, ci_dlos, oi_dlos,
            hg_jlos, cg_jlos, og_jlos,
            hg_dlos, cg_dlos, og_dlos,
            ug_jlos, ug_dlos,
            pr_means_jlos, pr_logvs_jlos, pr_means_dlos, pr_logvs_dlos,
            po_means_jlos, po_logvs_jlos, po_means_dlos, po_logvs_dlos)

        # --- Decoding
        dec_jlos = self.dop_jdecoder(ug_jlos)
        dec_dlos = self.dop_ddecoder(ug_dlos)

        return (dec_jlos, dec_dlos,
                pr_means_jlos, pr_logvs_jlos, pr_means_dlos, pr_logvs_dlos,
                po_means_jlos, po_logvs_jlos, po_means_dlos, po_logvs_dlos)

    def predict(self, cnd_x, cnd_v, qry_v, ndraw=7):
        pass


if __name__ == '__main__':
    net = GeRN()
    x = torch.randn(1, 3, 1, 240, 320)
    v = torch.randn(1, 3, 4, 1, 1)
    j = torch.randn(1, 1, 256, 256)
    d = torch.randn(1, 1, 256, 256)
    q = torch.randn(1, 4, 1, 1)

    dj, dd, _, _, _, _, _, _, _, _ = net(x, v, j, d, q)
    print(dj.size())
    print(dd.size())
