from gern.gern import GeRN
from gern.data import GernDataLoader, GernDataset
from gern.criteria import GernCriterion
from gern.scheduler import LearningRateScheduler, PixelStdDevScheduler
from gern.utils import get_params_l2, get_params_l1
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from datetime import datetime
import torch
import argparse
import os
import json
import sys
import time
from numpy import random


def wtag(*strings):
    return '/'.join(strings)


def balance_loss(x, y):
    d = (x + y).item()
    r1 = x.item() / d
    r2 = y.item() / d
    return r1 * x + r2 * y


def trainer(args, model, criterion, optimiser, lr_scheduler, sd_scheduler, writer):
    # --- Load checkpoint if from_epoch > 0
    if args.from_epoch > 0:
        load_name = os.path.join(
            args.chkptdir, 'chkpt_{:08d}.pth').format(args.from_epoch - 1)
        model.load_state_dict(torch.load(load_name))
        print('Loaded checkpoint: {}'.format(load_name))

    # --- Build data loader
    dataloaders = {'train': DataLoader(GernDataset(args.rootdir_train), shuffle=True, drop_last=True,
                                       batch_size=args.train_batch_size, num_workers=args.data_worker),
                   'test': GernDataLoader(args.rootdir_test, subset_size=args.test_subset_size,
                                          batch_size=args.test_batch_size, num_workers=args.data_worker)}
    batch_sizes = {'train': args.train_batch_size,
                   'test': args.test_batch_size}

    since = time.time()
    today = datetime.today()
    epoch = args.from_epoch
    for epoch in range(args.total_epochs):
        # --- Time information
        elapsed = time.time() - since
        since = time.time()
        epoch_string = '\n--- Epoch {:5d} (+{:.0f}s) {} {} {}'.format(
            epoch, elapsed, '-' * 25, today, datetime.today())
        print(epoch_string, flush=True)

        bce_jlos_total = {'train': 0, 'test': 0}
        bce_dlos_total = {'train': 0, 'test': 0}
        kld_jlos_total = {'train': 0, 'test': 0}
        kld_dlos_total = {'train': 0, 'test': 0}
        regulariser_total = 0.

        for phase in ['train', 'test']:

            if phase == 'train':
                print('**train**', end="", flush=True)
                model.train()
            else:
                print('**test**', end="", flush=True)
                model.eval()

            for nbat, (ctx_x, ctx_v, qry_jlos, qry_dlos, qry_v, wgt_jlos, wgt_dlos) in enumerate(dataloaders[phase]):
                print('.', end="", flush=True)
                # --- Context size
                k = random.randint(1, 6)

                with torch.set_grad_enabled(phase == 'train'):
                    # --- Model inputs
                    ctx_x = ctx_x[:, :k].to(args.target_device)
                    ctx_v = ctx_v[:, :k].to(args.target_device)
                    qry_jlos = qry_jlos[:, k - 1].to(args.target_device)
                    qry_dlos = qry_dlos[:, k - 1].to(args.target_device)
                    qry_v = qry_v[:, k - 1].to(args.target_device)
                    wgt_jlos = None  # wgt_jlos[:, slice(
                        # k - 1, k)].to(args.target_device)
                    wgt_dlos = None  # wgt_dlos[:, slice(
                        # k - 1, k)].to(args.target_device)

                    (dec_jlos, dec_dlos,
                     pr_means_jlos, pr_logvs_jlos, pr_means_dlos, pr_logvs_dlos,
                     po_means_jlos, po_logvs_jlos, po_means_dlos, po_logvs_dlos) = model(
                        ctx_x, ctx_v, qry_jlos, qry_dlos, qry_v)

                    bce_jlos, bce_dlos, kld_jlos, kld_dlos = criterion(
                        qry_jlos, qry_dlos, dec_jlos, dec_dlos, wgt_jlos, wgt_dlos,
                        pr_means_jlos, pr_logvs_jlos, pr_means_dlos, pr_logvs_dlos,
                        po_means_jlos, po_logvs_jlos, po_means_dlos, po_logvs_dlos,
                        batch_sizes[phase])

                split_loss = sd_scheduler.weight * \
                    balance_loss(bce_jlos, bce_dlos) + \
                    args.kl_weight * (kld_jlos + kld_dlos)

                # --- Gradient accumulation
                bce_jlos_total[phase] += bce_jlos.item()
                bce_dlos_total[phase] += bce_dlos.item()
                kld_jlos_total[phase] += kld_jlos.item()
                kld_dlos_total[phase] += kld_dlos.item()

                if phase == 'train':
                    l1 = get_params_l1(model)
                    (split_loss + args.l2_weight * l1).backward()
                    regulariser_total += l1.item()

                # --- Gradient step & reset
                if phase == 'train':
                    optimiser.step()
                    optimiser.zero_grad()

                    lr = lr_scheduler.step(epoch)
                    sd_scheduler.step(epoch)
                    writer.add_scalar(wtag('epoch', 'lr'), lr, epoch)

            # --- Log progress (total)
            bce_jlos_total[phase] /= (nbat + 1)
            bce_dlos_total[phase] /= (nbat + 1)
            kld_jlos_total[phase] /= (nbat + 1)
            kld_dlos_total[phase] /= (nbat + 1)
            regulariser_total /= (nbat + 1)

            if phase == 'train':
                writer.add_scalar(
                    wtag('total', 'weight_l2', phase), regulariser_total, epoch)

            # --- Make training checkpoint
            if phase == 'train' and (epoch + 1) % args.checkpoint_interval == 0:
                save_name = os.path.join(
                    args.chkptdir, 'chkpt_{:08d}.pth'.format(epoch))
                torch.save(model.state_dict(), save_name)

            # --- Save origin and decoded images epoch
            writer.add_image(wtag('total', 'dec_jlos', phase),
                             dec_jlos[0], epoch)
            writer.add_image(wtag('total', 'dec_dlos', phase),
                             dec_dlos[0], epoch)
            writer.add_image(wtag('total', 'qry_jlos', phase),
                             qry_jlos[0], epoch)
            writer.add_image(wtag('total', 'qry_dlos', phase),
                             qry_dlos[0], epoch)
            writer.flush()

        # --- add training loss
        writer.add_scalars(
            wtag('total', 'bce_jlos'), bce_jlos_total, epoch)
        writer.add_scalars(
            wtag('total', 'bce_dlos'), bce_dlos_total, epoch)
        writer.add_scalars(
            wtag('total', 'kld_jlos'), kld_jlos_total, epoch)
        writer.add_scalars(
            wtag('total', 'kld_dlos'), kld_dlos_total, epoch)
        writer.flush()


def main(args):
    model = GeRN().to(args.target_device)
    criterion = GernCriterion()
    optimiser = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        args.lr_min, amsgrad=args.enable_amsgrad)
    lr_scheduler = LearningRateScheduler(
        optimiser, args.lr_min, args.lr_max, args.lr_saturate_epoch)
    sd_scheduler = PixelStdDevScheduler(
        args.sd_min, args.sd_max, args.sd_saturate_epoch)

    # if args.target_device == 'cuda':
    #     model = torch.nn.DataParallel(model)

    # --- Tensorboard writer
    writer = SummaryWriter(
        log_dir=args.writerdir, filename_suffix='.r{:03d}'.format(args.run))

    trainer(args, model, criterion, optimiser,
            lr_scheduler, sd_scheduler, writer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Command line argument management
    parser.add_argument('--use-json', type=str, default='')
    parser.add_argument('--generate-default-json', type=str, default='')
    # Main training arguments
    parser.add_argument('--run', type=int, default=0)
    parser.add_argument('--target-device', type=str,
                        default='cuda')
    parser.add_argument('--ndraw', type=int, default=7)
    parser.add_argument('--from-epoch', type=int, default=0)
    parser.add_argument('--total-epochs', type=int, default=2000000)
    parser.add_argument('--lr-min', type=float, default=5e-5)
    parser.add_argument('--lr-max', type=float, default=5e-4)
    parser.add_argument('--lr-saturate-epoch', type=int, default=1600000)
    parser.add_argument('--sd-min', type=float, default=0.7)
    parser.add_argument('--sd-max', type=float, default=2.0)
    parser.add_argument('--sd-saturate-epoch', type=int, default=1600000)
    parser.add_argument('--enable-amsgrad', type=bool, default=False)
    parser.add_argument('--kl-weight', type=float, default=0.001)
    parser.add_argument('--l2-weight', type=float, default=0.001)
    # Dataset settings
    parser.add_argument('--rootdir-train', type=str)
    parser.add_argument('--rootdir-test', type=str)
    parser.add_argument('--train-subset-size', type=int, default=64)
    parser.add_argument('--test-subset-size', type=int, default=64)
    parser.add_argument('--train-batch-size', type=int, default=8)
    parser.add_argument('--test-batch-size', type=int, default=8)
    parser.add_argument('--batch-split', type=int, default=1)
    parser.add_argument('--data-worker', type=int, default=4)
    # Trained model and checkpoint output
    parser.add_argument('--savedir', type=str, default='')
    parser.add_argument('--checkpoint-interval', type=int, default=2000)

    args = parser.parse_args()

    # If generate-json is set, write arguments to file.
    if os.path.isdir(args.generate_default_json):
        json_name = os.path.join(
            args.generate_default_json, 'trainer_arguments.json')
        delattr(args, 'generate_default_json')
        delattr(args, 'use_json')
        with open(json_name, 'w') as jw:
            json.dump(vars(args), jw)
        sys.exit(0)

    # If an argument .json file is assigned, update args.
    if os.path.isfile(args.use_json):
        with open(args.use_json, 'r') as jr:
            json_data = json.load(jr)
            json_data['use_json'] = args.use_json
            args.__dict__.update(json_data)
            print('Loaded trainer argument from file.')

    # Assert savedir
    assert os.path.isdir(args.savedir), 'Not a valid pathname.'

    chkptdir = os.path.join(args.savedir, 'checkpoints')
    if not os.path.isdir(chkptdir):
        os.makedirs(chkptdir)
    setattr(args, 'chkptdir', chkptdir)

    writerdir = os.path.join(args.savedir, 'board')
    if not os.path.isdir(writerdir):
        os.makedirs(writerdir)
    setattr(args, 'writerdir', writerdir)

    # --- --- ---
    main(args)
