import torch
import torch.nn as nn
from .utils import count_parameters, init_parameters
from .model import *


class GeRN(torch.jit.ScriptModule):
    def __init__(self, nr=256, nq=128, nh=128, nv=4):
        super(GeRN, self).__init__()
        gain = nn.init.calculate_gain('leaky_relu')

        # --- default sizes
        self.size_r = nr
        self.size_q = nq
        self.size_h = nh
        self.size_v = nv

        # --- representaiton operators
        self.rop_deprepr = DepRepresentationEncoder()
        self.rop_losrepr = LosRepresentationEncoder()

        # --- inference operators
        self.iop_posterior = GaussianFactor()
        self.iop_state = RecurrentCell(nr + nq + nh, nh)

        # --- generation operators
        self.gop_prior = GaussianFactor()
        self.gop_state = RecurrentCell(nr + nh, nh)
        self.gop_delta = GeneratorDelta()
        self.conv2d = nn.Sequential(
            nn.Conv2d(nr + nh + nv, nr + nh, (3, 3), padding=1),
            nn.LeakyReLU(.01),
            GroupNorm2d(nr + nh, 8))
        self.conv2d.apply(partial(init_parameters, gain=gain))

        # --- decoding operators
        self.dop_base = Decoder()

        print('{}: {:,} trainable parameters.'.format(
            self.__class__.__name__, count_parameters(self)))

    @torch.jit.script_method
    def _forward_mc_loop(self, ndraw, cnd_repr, qry_jrep, qry_drep, qry_v, hi_jlos, ci_jlos, oi_jlos, hi_dlos, ci_dlos, oi_dlos, hg_jlos, cg_jlos, og_jlos, hg_dlos, cg_dlos, og_dlos, ug_jlos, ug_dlos, pr_means_jlos, pr_logvs_jlos, pr_means_dlos, pr_logvs_dlos, po_means_jlos, po_logvs_jlos, po_means_dlos, po_logvs_dlos):
        # type: (int, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, List[Tensor], List[Tensor], List[Tensor], List[Tensor], List[Tensor], List[Tensor], List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor, List[Tensor], List[Tensor], List[Tensor], List[Tensor], List[Tensor], List[Tensor], List[Tensor], List[Tensor],]

        for ast in range(ndraw):
            _, pr_mean_jlos, pr_logv_jlos = self.gop_prior(hg_jlos)
            _, pr_mean_dlos, pr_logv_dlos = self.gop_prior(hg_dlos)

            inp_ij = torch.cat([cnd_repr, qry_jrep, hg_jlos], dim=1)
            inp_id = torch.cat([cnd_repr, qry_drep, hg_dlos], dim=1)

            hi_jlos, ci_jlos, oi_jlos = self.iop_state(
                inp_ij, hi_jlos, ci_jlos, oi_jlos)
            hi_dlos, ci_dlos, oi_dlos = self.iop_state(
                inp_id, hi_dlos, ci_dlos, oi_dlos)

            po_z_jlos, po_mean_jlos, po_logv_jlos = self.iop_posterior(hi_jlos)
            po_z_dlos, po_mean_dlos, po_logv_dlos = self.iop_posterior(hi_dlos)

            inp_gj = torch.cat([cnd_repr, po_z_jlos], dim=1)
            inp_gd = torch.cat([cnd_repr, po_z_dlos, qry_v], dim=1)
            inp_gd = self.conv2d(inp_gd)

            hg_jlos, cg_jlos, og_jlos = self.gop_state(
                inp_gj, hg_jlos, cg_jlos, og_jlos)
            hg_dlos, cg_dlos, og_dlos = self.gop_state(
                inp_gd, hg_dlos, cg_dlos, og_dlos)

            ug_jlos = ug_jlos + self.gop_delta(ug_jlos, hg_jlos)
            ug_dlos = ug_dlos + self.gop_delta(ug_dlos, hg_dlos)

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

    # @torch.jit.script_method
    # def _predict_mc_loop(self, asteps, rwn_aggr, qry_v, h_gop, c_gop, o_gop, u_gop, prior_means=[], prior_logvs=[]):
    #     # type: (int, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, List[Tensor], List[Tensor]) -> Tuple[Tensor, List[Tensor], List[Tensor]]
    #     for ast in range(asteps):
    #         prior_z, prior_mean, prior_logv = self.gop_prior(h_gop)

    #         input_gop = torch.cat([rwn_aggr, prior_z, qry_v], dim=1)
    #         h_gop, c_gop, o_gop = self.gop_state(
    #             input_gop, h_gop, c_gop, o_gop)
    #         u_gop = u_gop + self.gop_delta(u_gop, h_gop)

    #         # collect means and log variances
    #         prior_means.append(prior_mean), prior_logvs.append(prior_logv)

    #     return u_gop, prior_means, prior_logvs

    def forward(self, cnd_x, cnd_v, qry_jlos, qry_dlos, qry_v, ndraw=7):
        """Forward method
        Args:
            cnd_x (torch.tensor): A 5-way tensor representing contextual sensors (pixels)
            cnd_v (torch.tenosr): A 5-way tensor representing contextual sensors (viewpoint)
            qry_x (torch.tenosr): Description
            qry_v (torch.tenosr): Description
            ndraw (int, optional): Description
        Returns:
            torch.tenosr: Description
        """
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

        # --- Conditional filtered and aggregated representations
        cnd_x = cnd_x.view(kB * kN, kC, kH, kW)
        cnd_v = cnd_v.view(kB * kN, -1, 1, 1)
        cnd_repr = self.rop_deprepr(cnd_x, cnd_v)
        cnd_repr = cnd_repr.view(kB, kN, -1, 1, 1).sum(dim=1)
        cnd_repr = cnd_repr.expand(-1, -1, 16, 16)

        # --- Query representation
        qry_jrep = self.rop_losrepr(qry_jlos, index=2)
        qry_drep = self.rop_losrepr(qry_dlos, qry_v, index=1)
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
         po_means_jlos, po_logvs_jlos, po_means_dlos, po_logvs_dlos) = self._forward_mc_loop(
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
        dec_jlos = self.dop_base(ug_jlos)
        dec_dlos = self.dop_base(ug_dlos)

        return (dec_jlos, dec_dlos,
                pr_means_jlos, pr_logvs_jlos, pr_means_dlos, pr_logvs_dlos,
                po_means_jlos, po_logvs_jlos, po_means_dlos, po_logvs_dlos)

    def predict(self, cnd_x, cnd_v, qry_v, ndraw=7):
        pass


if __name__ == '__main__':
    pass
