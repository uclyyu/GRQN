from gern.data import GernDataLoader, GernDataset
from torch.utils.data import DataLoader


if __name__ == '__main__':
    dataset = GernDataset('/home/yen/data/gern/phase/train')
    loader = DataLoader(dataset, shuffle=True, drop_last=True,
                        batch_size=120, num_workers=0)
    k = 0
    while True:
        k += 1
        for n, (ctx_x, ctx_v, qry_jlos, qry_dlos, qry_v, wgt_jlos, wgt_dlos) in enumerate(loader):
            ctx_x = ctx_x.cuda()
            ctx_v = ctx_v.cuda()
            qry_jlos = qry_jlos.cuda()
            qry_dlos = qry_dlos.cuda()
            qry_v = qry_v.cuda()
            wgt_jlos = wgt_jlos.cuda()
            wgt_dlos = wgt_dlos.cuda()
            print(k, n)
            pass
