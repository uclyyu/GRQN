from gern.gern import GeRN
from gern.data import GernDataLoader
from gern.criteria import GernCriterion
from gern.scheduler import LearningRateScheduler
from torch.utils.tensorboard import SummaryWriter
import torch
import argparse
import os
import json
import sys
import time
from numpy import random


def wtag(*strings):
    return '/'.join(strings)


def trainer(args, model, criterion, optimiser, lr_scheduler, writer):
    # --- Load checkpoint if from_epoch > 0
    if args.from_epoch > 0:
        load_name = os.path.join(
            args.chkptdir, 'chkpt_{:08d}.pth').format(args.from_epoch - 1)
        model.load_state_dict(torch.load(load_name))
        print('Loaded checkpoint: {}'.format(load_name))

    # --- Build data loader
    dataloaders = {'train': GernDataLoader(args.rootdir_train, subset_size=args.train_subset_size,
                                           batch_size=args.batch_size, num_workers=args.data_worker),
                   'test': GernDataLoader(args.rootdir_test, subset_size=args.test_subset_size,
                                          batch_size=args.batch_size, num_workers=args.data_worker)}

    since = time.time()
    for epoch in range(args.from_epoch, args.total_epochs):

        # --- Information
        elapsed = time.time() - since
        since = time.time()
        epoch_string = '\n\n--- Epoch {:5d} (+{:.0f}s) {}'.format(
            epoch, elapsed, '-' * 50)
        print(epoch_string)

        # --- Alternating between training and testing phases
        for phase in ['train', 'test']:
            if phase == 'train':
                lr = lr_scheduler.step(epoch)
                writer.add_scalar(wtag('epoch', 'lr'), lr, epoch)
                model.train()

            else:
                model.eval()

            bce_jlos_epoch = 0.
            bce_dlos_epoch = 0.
            kld_jlos_epoch = 0.
            kld_dlos_epoch = 0.

            optimiser.zero_grad()
            # --- Iterate over the current subset
            with torch.set_grad_enabled(phase == 'train'):
                for i, (ctx_x, ctx_v, qry_jlos, qry_dlos, qry_v) in enumerate(dataloaders[phase]):
                    k = random.randint(1, 6)

                    # --- Model inputs
                    ctx_x = ctx_x[:, :k].to(args.target_device)
                    ctx_v = ctx_v[:, :k].to(args.target_device)
                    qry_jlos = qry_jlos[:, k - 1].to(args.target_device)
                    qry_dlos = qry_dlos[:, k - 1].to(args.target_device)
                    qry_v = qry_v[:, k - 1].to(args.target_device)

                    # --- Forward pass
                    (dec_jlos, dec_dlos,
                     pr_means_jlos, pr_logvs_jlos, pr_means_dlos, pr_logvs_dlos,
                     po_means_jlos, po_logvs_jlos, po_means_dlos, po_logvs_dlos) = model(
                        ctx_x, ctx_v, qry_jlos, qry_dlos, qry_v)

                    bce_jlos, bce_dlos, kld_jlos, kld_dlos = criterion(
                        qry_jlos, qry_dlos, dec_jlos, dec_dlos,
                        pr_means_jlos, pr_logvs_jlos, pr_means_dlos, pr_logvs_dlos,
                        po_means_jlos, po_logvs_jlos, po_means_dlos, po_logvs_dlos)

                    total_loss = bce_jlos + bce_dlos + kld_jlos + kld_dlos
                    bce_jlos_epoch += bce_jlos.item()
                    bce_dlos_epoch += bce_dlos.item()
                    kld_jlos_epoch += kld_jlos.item()
                    kld_dlos_epoch += kld_dlos.item()

                    # --- Gradient step
                    if phase == 'train':
                        total_loss.backward()
                        optimiser.step()

                    # --- Reset
                    optimiser.zero_grad()

                bce_jlos_epoch /= (i + 1)
                bce_dlos_epoch /= (i + 1)
                kld_jlos_epoch /= (i + 1)
                kld_dlos_epoch /= (i + 1)

                # --- Make training checkpoint
                if phase == 'train':
                    if epoch == 0 or ((epoch + 1) % args.checkpoint_interval) == 0:
                        save_name = os.path.join(
                            args.chkptdir, 'chkpt_{:08d}.pth'.format(epoch))
                        torch.save(model.state_dict(), save_name)

                # --- Log epoch progress
                writer.add_scalar(
                    wtag('epoch', 'bce_jlos', phase), bce_jlos_epoch, epoch)
                writer.add_scalar(
                    wtag('epoch', 'bce_dlos', phase), bce_dlos_epoch, epoch)
                writer.add_scalar(
                    wtag('epoch', 'kld_jlos', phase), kld_jlos_epoch, epoch)
                writer.add_scalar(
                    wtag('epoch', 'kld_dlos', phase), kld_dlos_epoch, epoch)

                # --- Save origin and decoded images every test epoch
                if phase == 'test':
                    writer.add_image(wtag('epoch', 'dec_jlos'),
                                     dec_jlos[0], epoch)
                    writer.add_image(wtag('epoch', 'dec_dlos'),
                                     dec_dlos[0], epoch)
                    writer.add_image(wtag('epoch', 'qry_jlos'),
                                     qry_jlos[0], epoch)
                    writer.add_image(wtag('epoch', 'qry_dlos'),
                                     qry_dlos[0], epoch)


def main(args):
    model = GeRN().to(args.target_device)
    criterion = GernCriterion()
    optimiser = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        args.lr_min, amsgrad=args.enable_amsgrad)
    lr_scheduler = LearningRateScheduler(
        optimiser, args.lr_min, args.lr_max, args.lr_saturate_epoch)

    # if args.target_device == 'cuda':
    #     model = torch.nn.DataParallel(model)

    # --- Tensorboard writer
    writer = SummaryWriter(
        log_dir=args.writerdir, filename_suffix='.r{:03d}'.format(args.run))

    trainer(args, model, criterion, optimiser,
            lr_scheduler, writer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Command line argument management
    parser.add_argument('--use-json', type=str, default='')
    parser.add_argument('--generate-default-json', type=str, default='')
    # Main training arguments
    parser.add_argument('--run', type=int, default=0)
    parser.add_argument('--target-device', type=str,
                        default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--ndraw', type=int, default=7)
    parser.add_argument('--from-epoch', type=int, default=0)
    parser.add_argument('--total-epochs', type=int, default=2000000)
    parser.add_argument('--lr-min', type=float, default=5e-5)
    parser.add_argument('--lr-max', type=float, default=5e-4)
    parser.add_argument('--lr-saturate-epoch', type=int, default=1600000)
    parser.add_argument('--enable-amsgrad', type=bool, default=False)
    # Dataset settings
    parser.add_argument('--rootdir-train', type=str)
    parser.add_argument('--rootdir-test', type=str)
    parser.add_argument('--train-subset-size', type=int, default=64)
    parser.add_argument('--test-subset-size', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=8)
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
