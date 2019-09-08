import torch.nn as nn
from torch.nn import functional as F


def kl_divergence(po_means, po_logvs, pr_means, pr_logvs):
    kldiv = 0.
    num_batch = po_means[0].size(0)
    ndraw = len(po_means)

    for qp in zip(po_means, po_logvs, pr_means, pr_logvs):

        qm, qv, pm, pv = map(lambda v: v.view(num_batch, -1), qp)

        kldiv += (-pv + qv).exp().sum(1) + \
            (qm - pm).pow(2).div(pv.exp()).sum(1) + pv.sum(1) - qv.sum(1)

    kldiv = kldiv.mean() / ndraw
    return kldiv


# dec_jlos, dec_dlos,
# pr_means_jlos, pr_logvs_jlos, pr_means_dlos, pr_logvs_dlos,
# po_means_jlos, po_logvs_jlos, po_means_dlos, po_logvs_dlos

class GernCriterion(object):

    def __call__(self, trg_jlos, trg_dlos, dec_jlos, dec_dlos,
                 pr_means_jlos, pr_logvs_jlos, pr_means_dlos, pr_logvs_dlos,
                 po_means_jlos, po_logvs_jlos, po_means_dlos, po_logvs_dlos):

        bce_jlos = F.binary_cross_entroy(dec_jlos, trg_jlos, reduction='mean')
        bce_dlos = F.binary_cross_entroy(dec_dlos, trg_dlos, reduction='mean')
        kld_jlos = kl_divergence(
            po_means_jlos, po_logvs_jlos, pr_means_jlos, pr_logvs_jlos)
        kld_dlos = kl_divergence(
            po_means_dlos, po_logvs_dlos, pr_means_dlos, pr_logvs_dlos)

        return bce_jlos, bce_dlos, kld_jlos, kld_dlos
