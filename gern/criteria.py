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

    def call_for_dlos(self, trg_dlos, dec_dlos, wgt_dlos, pr_means_dlos, pr_logvs_dlos, po_means_dlos, po_logvs_dlos):
        bce_dlos = F.binary_cross_entropy(
            dec_dlos, trg_dlos, wgt_dlos, reduction='none')
        bce_dlos = bce_dlos.mean() * trg_dlos.size(0) / self.batch_size
        kld_dlos = kl_divergence(
            po_means_dlos, po_logvs_dlos, pr_means_dlos, pr_logvs_dlos)

        return bce_dlos, kld_dlos

    def call_for_jlos(self, trg_jlos, dec_jlos, wgt_jlos, pr_means_jlos, pr_logvs_jlos, po_means_jlos, po_logvs_jlos):
        bce_jlos = F.binary_cross_entropy(
            dec_jlos, trg_jlos, wgt_jlos, reduction='none')
        bce_jlos = bce_jlos.mean() * trg_jlos.size(0) / self.batch_size
        kld_jlos = kl_divergence(
            po_means_jlos, po_logvs_jlos, pr_means_jlos, pr_logvs_jlos)

        return bce_jlos, kld_jlos

    def __call__(self, trg_jlos, trg_dlos, dec_jlos, dec_dlos, wgt_jlos, wgt_dlos,
                 pr_means_jlos, pr_logvs_jlos, pr_means_dlos, pr_logvs_dlos,
                 po_means_jlos, po_logvs_jlos, po_means_dlos, po_logvs_dlos,
                 batch_size):
        self.batch_size = batch_size
        bce_jlos, kld_jlos = self.call_for_jlos(
            trg_jlos, dec_jlos, wgt_jlos, pr_means_jlos, pr_logvs_jlos, po_means_jlos, po_logvs_jlos)
        bce_dlos, kld_dlos = self.call_for_dlos(
            trg_dlos, dec_dlos, wgt_dlos, pr_means_dlos, pr_logvs_dlos, po_means_dlos, po_logvs_dlos)
        return bce_jlos, bce_dlos, kld_jlos, kld_dlos
