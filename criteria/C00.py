import torch.nn as nn
from torch.nn.functional import binary_cross_entropy
from torch.distributions import Normal, kl_divergence


def kl_div(po_means, po_logvs, pr_means, pr_logvs, batch_size):
    kldiv = 0.
    ndraw = len(po_means)

    for n in range(ndraw):
        q = Normal(po_means[n], (0.5 * po_logvs[n]).exp())
        p = Normal(pr_means[n], (0.5 * pr_logvs[n]).exp())
        kldiv += kl_divergence(q, p).mean() * po_means[n].size(0) / batch_size

    kldiv = kldiv / ndraw

    return kldiv


def Criteria(xprm):

    class _Criteria(object):

        @xprm.capture
        def __init__(self, model, wL1, wL2, wKL):
            self.wL1 = wL1
            self.wL2 = wL2
            self.wKL = wKL
            self.params = []
            self._bce_jlos = 0
            self._bce_dlos = 0
            self._kld_jlos = 0
            self._kld_dlos = 0

            for module in model.modules():

                if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
                    self.params.append(module.weight)

        @property
        def l1regl(self):
            if self.wL1 > 0:
                return self.wL1 * sum(map(lambda p: p.abs().sum(), self.params))

            else:
                return 0.

        @property
        def l2regl(self):
            if self.wL2 > 0.:
                return self.wL2 * sum(map(lambda p: p.pow(2).sum(), self.params))

            else:
                return 0.

        @property
        def bce_jlos(self):
            return self._bce_jlos

        @property
        def bce_dlos(self):
            return self._bce_dlos

        @property
        def kld_jlos(self):
            return self._kld_jlos

        @property
        def kld_dlos(self):
            return self._kld_dlos

        def balance_loss(self, x, y):
            d = (x + y).item()
            r1 = x.item() / d
            r2 = y.item() / d
            return r1 * x + r2 * y

        def loss(self, trg_jlos, trg_dlos, dec_jlos, dec_dlos, wgt_jlos, wgt_dlos,
                 pr_means_jlos, pr_logvs_jlos, pr_means_dlos, pr_logvs_dlos,
                 po_means_jlos, po_logvs_jlos, po_means_dlos, po_logvs_dlos,
                 batch_size, isd):

            bce_jlos = binary_cross_entropy(dec_jlos, trg_jlos, wgt_jlos, reduction='none')
            bce_jlos = bce_jlos.mean() * trg_jlos.size(0) / batch_size
            kld_jlos = kl_div(po_means_jlos, po_logvs_jlos, pr_means_jlos, pr_logvs_jlos, batch_size)

            bce_dlos = binary_cross_entropy(dec_dlos, trg_dlos, wgt_dlos, reduction='none')
            bce_dlos = bce_dlos.mean() * trg_dlos.size(0) / batch_size
            kld_dlos = kl_div(po_means_dlos, po_logvs_dlos, pr_means_dlos, pr_logvs_dlos, batch_size)

            self._bce_jlos = bce_jlos.item()
            self._bce_dlos = bce_dlos.item()
            self._kld_jlos = kld_jlos.item()
            self._kld_dlos = kld_dlos.item()

            ret = isd * self.balance_loss(bce_jlos, bce_dlos)
            ret += self.wKL * (kld_jlos + kld_dlos)

            return ret

    return _Criteria
