import os
import time
import torch
import pymongo
from gridfs import GridFS
from datetime import datetime
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter


def Worker(xprm):

    class _Worker(object):

        @xprm.capture
        def __init__(self, DefaultModel, DefaultData):
            self.model = DefaultModel()
            self.data = DefaultData()

        @xprm.capture
        def restore(self, restore, db_addr, device, _run):
            rid = restore['run']
            rep = restore['epoch']
            dbn = restore['db']

            if rid is None:
                rid = _run._id

            client = pymongo.MongoClient(db_addr)
            fs = GridFS(client[dbn])
            runs = client[dbn].runs
            run_entry = runs.find_one({'_id': rid})
            ckpt_interval = run_entry['config']['ckpt_interval']

            if rep < 0:
                index = len(run_entry['artifacts']) - 1

            else:
                index = rep + 1
                index = int(index / ckpt_interval - 1)

            file_id = run_entry['artifacts'][index]['file_id']
            file_name = run_entry['artifacts'][index]['name']

            self.model.load_state_dict(
                torch.load(fs.get(file_id),
                    map_location=device))

            print(f"Restored RUN {restore['run']} checkpoint: {file_name}")

        @xprm.capture
        def train(self, _run, DefaultScheduler, DefaultCriteria,
                  paths, num_epoch, lr, batch_size, batch_split, ndraw, device, ckpt_interval, seed):
            model = self.model = self.model.to(device)
            optimiser = AdamW(model.parameters(), lr['start'])
            scheduler = DefaultScheduler(optimiser)
            criteria = DefaultCriteria(model)
            writer = SummaryWriter(os.path.join(paths['tensorboard'], f'{_run._id:03d}'))
            since = time.time()

            for epoch in range(num_epoch):
                bce_jlos_total = {'train': 0, 'test': 0}
                bce_dlos_total = {'train': 0, 'test': 0}
                kld_jlos_total = {'train': 0, 'test': 0}
                kld_dlos_total = {'train': 0, 'test': 0}

                elapsed = int(time.time() - since)
                since = time.time()
                print(f'\n{datetime.today()} *** Epoch {epoch:05d} (+{elapsed:,}s): ', end='')

                for phase in ['train', 'test']:

                    print(f'* {phase} ', end='', flush=True)

                    for n, pkg in enumerate(self.data.get_loader(phase)):

                        for (ctx_x, ctx_v,
                             qry_jlos, qry_dlos, qry_v,
                             wgt_jlos, wgt_dlos) in self.data.unpack(pkg):
                            ctx_x = ctx_x.to(device)
                            ctx_v = ctx_v.to(device)
                            qry_v = qry_v.to(device)
                            qry_jlos = qry_jlos.to(device)
                            qry_dlos = qry_dlos.to(device)
                            wgt_jlos = wgt_jlos.to(device)
                            wgt_dlos = wgt_dlos.to(device)

                            with torch.set_grad_enabled(phase == 'train'):

                                (dec_jlos, dec_dlos,
                                 pr_means_jlos, pr_logvs_jlos, pr_means_dlos, pr_logvs_dlos,
                                 po_means_jlos, po_logvs_jlos, po_means_dlos, po_logvs_dlos) = model(
                                    ctx_x, ctx_v, qry_jlos, qry_dlos, qry_v, ndraw)
                                loss = criteria.loss(
                                    qry_jlos, qry_dlos, dec_jlos, dec_dlos, wgt_jlos, wgt_dlos,
                                    pr_means_jlos, pr_logvs_jlos, pr_means_dlos, pr_logvs_dlos,
                                    po_means_jlos, po_logvs_jlos, po_means_dlos, po_logvs_dlos,
                                    batch_size, scheduler.isd)

                                if phase == 'train':
                                    loss.backward()

                            bce_jlos_total[phase] += criteria.bce_jlos
                            bce_dlos_total[phase] += criteria.bce_dlos
                            kld_jlos_total[phase] += criteria.kld_jlos
                            kld_dlos_total[phase] += criteria.kld_dlos

                        if phase == 'train':
                            regl = criteria.l1regl + criteria.l2regl

                            if isinstance(regl, torch.Tensor):
                                regl.backward()

                            optimiser.step()
                            optimiser.zero_grad()
                            scheduler.step(epoch)

                    bce_jlos_total[phase] /= (n + 1)
                    bce_dlos_total[phase] /= (n + 1)
                    kld_jlos_total[phase] /= (n + 1)
                    kld_dlos_total[phase] /= (n + 1)
                    writer.add_image(f'jlos/decoder/{phase}', dec_jlos[0], epoch)
                    writer.add_image(f'dlos/decoder/{phase}', dec_dlos[0], epoch)
                    writer.add_image(f'jlos/true/{phase}', qry_jlos[0], epoch)
                    writer.add_image(f'dlos/true/{phase}', qry_dlos[0], epoch)

                    if phase == 'train' and (epoch + 1) % ckpt_interval == 0:
                        file = f'/tmp/_checkpoint_{seed}.pth'
                        name = f'ckpt-epoch-{epoch:06d}.pth'
                        mime = 'application/octet-stream'
                        torch.save(model.state_dict(), file)
                        _run.add_artifact(file, name, content_type=mime)

                writer.add_scalars('bce/jlos', bce_jlos_total, epoch)
                writer.add_scalars('bce/dlos', bce_dlos_total, epoch)
                writer.add_scalars('kld/jlos', kld_jlos_total, epoch)
                writer.add_scalars('kld/dlos', kld_dlos_total, epoch)
                _run.log_scalar('bce-jlos-train', bce_jlos_total['train'], epoch)
                _run.log_scalar('bce-dlos-train', bce_dlos_total['train'], epoch)
                _run.log_scalar('kld-jlos-train', kld_jlos_total['train'], epoch)
                _run.log_scalar('kld_dlos-train', kld_dlos_total['train'], epoch)
                _run.log_scalar('bce-jlos-test', bce_jlos_total['test'], epoch)
                _run.log_scalar('bce-dlos-test', bce_dlos_total['test'], epoch)
                _run.log_scalar('kld-jlos-test', kld_jlos_total['test'], epoch)
                _run.log_scalar('kld_dlos-test', kld_dlos_total['test'], epoch)

    return _Worker
