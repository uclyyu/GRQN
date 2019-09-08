from .. import gern
import unittest
import torch


class TestGern(unittest.TestCase):
    def setUp(self):
        self.nr = 256
        self.nq = 128
        self.nh = 128
        self.nv = 4
        self.net = gern.GeRN(self.nr, self.nq, self.nh, self.nv)

    def test_forward(self):
        cnd_x = torch.randn(1, 3, 1, 320, 240)
        cnd_v = torch.randn(1, 3, 4, 1, 1)
        qry_x = torch.randn(1, 1, 256, 256)
        qry_v = torch.randn(1, 4, 1, 1)

        (dec_jlos, dec_dlos,
         pr_means_jlos, pr_logvs_jlos, pr_means_dlos, pr_logvs_dlos,
         po_means_jlos, po_logvs_jlos, po_means_dlos, po_logvs_dlos) = self.net(
            cnd_x, cnd_v,
            qry_x, qry_v, ndraw=1)

        self.assertEqual(dec_jlos.size(), torch.Size([1, 1, 320, 240]))
        self.assertEqual(dec_dlos.size(), torch.Size([1, 1, 320, 240]))
        self.assertEqual(pr_means_jlos[-1].size(),
                         torch.Size([1, self.nh, 16, 16]))
        self.assertEqual(pr_logvs_jlos[-1].size(),
                         torch.Size([1, self.nh, 16, 16]))
        self.assertEqual(pr_means_dlos[-1].size(),
                         torch.Size([1, self.nh, 16, 16]))
        self.assertEqual(pr_logvs_dlos[-1].size(),
                         torch.Size([1, self.nh, 16, 16]))
        self.assertEqual(po_means_jlos[-1].size(),
                         torch.Size([1, self.nh, 16, 16]))
        self.assertEqual(po_logvs_jlos[-1].size(),
                         torch.Size([1, self.nh, 16, 16]))
        self.assertEqual(po_means_dlos[-1].size(),
                         torch.Size([1, self.nh, 16, 16]))
        self.assertEqual(po_logvs_dlos[-1].size(),
                         torch.Size([1, self.nh, 16, 16]))


if __name__ == '__main__':
    unittest.main()
