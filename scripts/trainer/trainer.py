import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from gern.gern import GeRN
from gern.data import GernDataLoader
from gern.criteria import GernCriterion
from gern.scheduler import LearningRateScheduler, PixelStdDevScheduler
from torch import distributed
from torch.utils import checkpoint as ptcp
from glob import glob
from tensorboardX import SummaryWriter
import torch, argparse, os, json, sys, time
import matplotlib.pyplot as plt
import numpy as np


def initialise_process(args):
	distributed.init_process_group(
		backend=args.backend,
		init_method=args.init_method,
		rank=args.local_rank,
		world_size=args.world_size)
	group = distributed.new_group(list(range(args.world_size)))
	return group

def average_gradients(args, model, group):
	world_size = float(args.world_size)
	for m in model.modules():
			if hasattr(m, 'grad'):
				if m.requires_grad:
					if m.grad is None:
						print(m)
	for param in filter(lambda p: p.requires_grad, model.parameters()):
		distributed.all_reduce(param.grad.data, op=distributed.ReduceOp.SUM, group=group)
		param.grad.data /= (world_size * args.batch_size)

def log_best_json(args, epoch, accuracy, l_percept, l_heatmap, l_classifier, l_aggregate, l_kldiv):
	save_name = os.path.join(args.bestdir, 'stats.json')
	with open(save_name, 'w') as jw:
		data = {'epoch': epoch, 'accuracy': accuracy, 
				'l_percept': l_percept, 'l_heatmap': l_heatmap, 
				'l_classifier': l_classifier, 'l_aggregate': l_aggregate, 'l_kldiv': l_kldiv }
		json.dump(data, jw)

def split_batch(C, Q, L, size=1, dim=0):
	cnd_x, cnd_m, cnd_k, cnd_v = C
	qry_x, qry_m, qry_k, qry_v = Q
	for cx, cm, ck, cv, qx, qm, qk, qv, l in zip(
		cnd_x.split(size, dim=dim), cnd_m.split(size, dim=dim), cnd_k.split(size, dim=dim), cnd_v.split(size, dim=dim), 
		qry_x.split(size, dim=dim), qry_m.split(size, dim=dim), qry_k.split(size, dim=dim), qry_v.split(size, dim=dim),
		L.split(size, dim=dim)):

		yield cx, cm, ck, cv, qx, qm, qk, qv, l

def plot_rgbv(dec, orig):
	d = dec[-1].permute(1, 2, 0).contiguous().detach().cpu().numpy()
	o = orig[-1].permute(1, 2, 0).contiguous().detach().cpu().numpy()
	figure = plt.figure(figsize=(10, 5))
	ax = plt.subplot(121)
	ax.set_title('Decoded')
	ax.set_axis_off()
	plt.imshow(d, vmin=d.min(), vmax=d.min())
	ax = plt.subplot(122)
	ax.set_title('Original')
	ax.set_axis_off()
	plt.imshow(o, vmin=o.min(), vmax=o.min())
	return figure

def plot_heat(dec, orig):
	d = dec[-1, 0].detach().cpu().numpy()
	o = orig[-1, 0].detach().cpu().numpy()
	figure = plt.figure(figsize=(10, 5))
	ax = plt.subplot(121)
	ax.set_title('Decoded')
	ax.set_axis_off()
	plt.imshow(d, vmin=d.min(), vmax=d.min())
	ax = plt.subplot(122)
	ax.set_title('Original')
	ax.set_axis_off()
	plt.imshow(o, vmin=o.min(), vmax=o.min())
	return figure

def train_distributed(args, model, criterion, optimiser, lr_scheduler, sd_scheduler, writer, group):

	best_correct = 0.	
	# --- Load checkpoint if from_epoch > 0
	if args.from_epoch > 0:
		assert args.from_epoch % args.checkpoint_interval == 0
		load_name = os.path.join(args.chkptdir, 'chkpt_{:08d}.pth').format(args.from_epoch - 1)
		model.load_state_dict(torch.load(load_name))
		writer.add_text(
			args.tag_general,
			'Loaded checkpoint: {}'.format(load_name))

		# Load best model statisitcs
		json_name = os.path.join(args.bestdir, 'stats.json')
		with open(json_name, 'r') as jr:
			json_data = json.load(jr)
			best_correct = json_data['accuracy']

	# --- Build data loader
	subset_sizes = {'train': args.train_subset_size, 'test': args.test_subset_size}
	dataloaders = {'train': GernDataLoader(args.rootdir_train, subset_size=subset_sizes['train'], batch_size=args.batch_size, num_workers=args.data_worker),
				  'test':  GernDataLoader(args.rootdir_test,  subset_size=subset_sizes['test'], batch_size=args.batch_size, num_workers=args.data_worker)}
	
	since = time.time()
	for epoch in range(args.from_epoch, args.total_epochs):

		print('[{}]Epoch {} (+{:.0f}s)'.format(args.local_rank, epoch, time.time() - since))
		since = time.time()

		# alternating between training and testing phases
		for phase in ['train', 'test']:
			if phase == 'train':
				lr_scheduler.step(epoch)
				sd_scheduler.step(epoch)
				model.train()
			else:
				model.eval()

			epoch_loss = 0.
			epoch_correct = 0.

			# zero parameter gradients
			optimiser.zero_grad()

			# Iterate over batches in the current subset
			for C, Q, L in dataloaders[phase]:
				# copy to device
				C = [c.to(args.target_device) for c in C]
				Q = [q.to(args.target_device) for q in Q]
				L = L.to(args.target_device)

				# --- Forward pass
				batch_loss = 0.
				batch_correct = 0.
				with torch.set_grad_enabled(phase == 'train'):
					# Split batch (if necessary)
					for cx, cm, ck, cv, qx, qm, qk, qv, lab in split_batch(C, Q, L, size=1, dim=0):
						gern_output = model(cx, cm, ck, cv, qx, qm, qk, qv, asteps=args.asteps, rsteps=args.max_rsteps)
						gern_target = model.make_target(qx, qm, qk, qv, lab, rsteps=args.max_rsteps)

						weighted_loss, correct = criterion(gern_output, gern_target, args.criterion_weights)

						batch_loss += np.array(criterion.item())
						epoch_loss += np.array(criterion.item())
						batch_correct += correct
						epoch_correct += correct

						# accumulate gradients (training phase only)
						if phase == 'train':
							weighted_loss.backward()

				# --- Step optimiser (training phase only)
				if phase == 'train':
					average_gradients(args, model, group)
					optimiser.step()

				# --- Log batch progress
				batch_loss = batch_loss / args.batch_size
				batch_correct = batch_correct / args.batch_size

				writer.add_scalar(args.tags_loss['percept']   ('batch', phase), batch_loss[0], epoch)
				writer.add_scalar(args.tags_loss['heatmap']   ('batch', phase), batch_loss[1], epoch)
				writer.add_scalar(args.tags_loss['classifier']('batch', phase), batch_loss[2], epoch)
				writer.add_scalar(args.tags_loss['aggregator']('batch', phase), batch_loss[3], epoch)
				writer.add_scalar(args.tags_loss['divergence']('batch', phase), batch_loss[4], epoch)
				writer.add_scalar(args.tags_loss['accuracy']  ('batch', phase), batch_correct, epoch)
			
			epoch_loss = epoch_loss / subset_sizes[phase]
			epoch_correct = epoch_correct / subset_sizes[phase]

			# --- Save best model
			is_best = False
			if phase == 'test' and epoch_correct > best_correct:
				is_best = True
				best_correct = epoch_correct
				save_name = os.path.join(args.bestdir, 'model.pth')
				torch.save(model.state_dict(), save_name)
				
				writer.add_scalar(args.tags_loss['percept']   ('best', phase), epoch_loss[0], epoch)
				writer.add_scalar(args.tags_loss['heatmap']   ('best', phase), epoch_loss[1], epoch)
				writer.add_scalar(args.tags_loss['classifier']('best', phase), epoch_loss[2], epoch)
				writer.add_scalar(args.tags_loss['aggregator']('best', phase), epoch_loss[3], epoch)
				writer.add_scalar(args.tags_loss['divergence']('best', phase), epoch_loss[4], epoch)
				writer.add_scalar(args.tags_loss['accuracy']  ('best', phase), epoch_correct, epoch)
				log_best_json(args, epoch, best_correct, *epoch_loss)

			# --- Make training checkpoint
			if phase == 'train':
				if epoch == 0 or ((epoch + 1) % args.checkpoint_interval) == 0:
					save_name = os.path.join(args.chkptdir, 'chkpt_{:08d}.pth'.format(epoch))
					torch.save(model.state_dict(), save_name)

			# --- Log epoch progress
			writer.add_scalar(args.tags_loss['percept']   ('epoch', phase), epoch_loss[0], epoch)
			writer.add_scalar(args.tags_loss['heatmap']   ('epoch', phase), epoch_loss[1], epoch)
			writer.add_scalar(args.tags_loss['classifier']('epoch', phase), epoch_loss[2], epoch)
			writer.add_scalar(args.tags_loss['aggregator']('epoch', phase), epoch_loss[3], epoch)
			writer.add_scalar(args.tags_loss['divergence']('epoch', phase), epoch_loss[4], epoch)
			writer.add_scalar(args.tags_loss['accuracy']  ('epoch', phase), epoch_correct, epoch)

			# --- Save origin and decoded images every epoch
			writer.add_figure(args.tags_figure['rgbv'](phase), plot_rgbv(gern_output.rgbv, gern_target.rgbv), epoch)
			writer.add_figure(args.tags_figure['heat'](phase), plot_rgbv(gern_output.heat, gern_target.heat), epoch)


def main(args):
	# --- Setting up trainer
	group = initialise_process(args)
	model = GeRN().to(args.target_device)
	criterion = GernCriterion().to(args.target_device)
	optimiser = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), args.lr_min)
	lr_scheduler = LearningRateScheduler(optimiser, args.lr_min, args.lr_max, args.lr_saturate_epoch)
	sd_scheduler = PixelStdDevScheduler(args.criterion_weights, 0, args.sd_min, args.sd_max, args.sd_saturate_epoch)

	# --- Tensorboard writer
	writerdir = os.path.join(args.savedir, 'board', 'rank', '{:02d}'.format(args.local_rank))
	if args.coward:
		assert not os.path.isdir(writerdir), 'Cowardly avoid overwriting run folder.'
	try:
		os.makedirs(writerdir)
	except FileExistsError:
		pass

	writer = SummaryWriter(log_dir=writerdir, filename_suffix='.r{:03d}'.format(args.run))

	# --- Add tensorboard tags
	setattr(args, 'tags_loss',{
		'percept': 	  lambda stage, phase: 'run/{:03d}/loss/{}/percept/{}'.format(args.run, stage, phase),
		'heatmap': 	  lambda stage, phase: 'run/{:03d}/loss/{}/heatmap/{}'.format(args.run, stage, phase),
		'classifier': lambda stage, phase: 'run/{:03d}/loss/{}/classifier/{}'.format(args.run, stage, phase),
		'aggregator': lambda stage, phase: 'run/{:03d}/loss/{}/aggregator/{}'.format(args.run, stage, phase),
		'divergence': lambda stage, phase: 'run/{:03d}/loss/{}/divergence/{}'.format(args.run, stage, phase),
		'accuracy':	  lambda stage, phase: 'run/{:03d}/loss/{}/accuracy/{}'.format(args.run, stage, phase)
		})
	setattr(args, 'tags_figure', {
		'rgbv': lambda phase: 'run/{:03d}/figure/rgbv/{}'.format(args.run, phase),
		'heat': lambda phase: 'run/{:03d}/figure/heat/{}'.format(args.run, phase)
		})
	setattr(args, 'tag_general', 'run/{:03d}/general'.format(args.run))


	# --- Sanity-checks: checkpoint
	chkptdir = os.path.join(args.savedir, 'checkpoints', 'rank', '{:02d}', 'run', '{:03d}').format(args.local_rank, args.run)
	if args.coward:
		assert not os.path.isdir(chkptdir), 'Cowardly avoid overwriting checkpoints: {}'.format(chkptdir)
	else:
		# create checkpoint folder
		try:
			os.makedirs(chkptdir)
		except FileExistsError:
			writer.add_text(
				args.tag_general, 
				'Re-using checkpoint folder.')
	setattr(args, 'chkptdir', chkptdir)

	# --- Sanity-checks: best model
	bestdir = os.path.join(args.savedir, 'best', 'rank', '{:02d}', 'run', '{:03d}').format(args.local_rank, args.run)
	if args.coward:
		assert not os.path.isdir(bestdir), 'Cowardly avoid overwriting best models.'
	else:
		try:
			os.makedirs(bestdir)
		except FileExistsError:
			writer.add_text(
				args.tag_general, 
				'Reusing best model folder.')
	setattr(args, 'bestdir', bestdir)

	# --- Get the latest epoch if forcing checkpoint.
	if args.force_checkpoint:
		latest_epoch = 0
		for checkpoint in glob(os.path.join(args.chkptdir, '*.pth')):
			epoch = int(checkpoint.split('_')[-1].split('.')[0])
			latest_epoch = max(epoch, latest_epoch)
			args.from_epoch = latest_epoch + 1  # 
		writer.add_text(
			args.tag_general, 
			'Resuming epoch {}'.format(args.from_epoch))

	# --- Training procedure
	if args.local_rank is None:
		raise NotImplementedError
	else:	
		train_distributed(args, model, criterion, optimiser, lr_scheduler, sd_scheduler, writer, group)


if __name__ == '__main__':
	UNDEFINED = '<UNDEFINED>'
	parser = argparse.ArgumentParser()
	# Command line argument management
	parser.add_argument('--use-json', type=str, default=UNDEFINED)
	parser.add_argument('--generate-default-json', type=str, default=UNDEFINED)
	# Settings for PyTorch distributed module
	parser.add_argument('--local_rank', type=int, default=0)
	parser.add_argument('--world-size', type=int, default=1)
	parser.add_argument('--backend', type=str, default='nccl')
	parser.add_argument('--init-method', type=str, default='tcp://192.168.1.116:23456')
	# Main training arguments
	parser.add_argument('--run', type=int, default=0)
	parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
	parser.add_argument('--asteps', type=int, default=7)
	parser.add_argument('--max-rsteps', type=int, default=100)
	parser.add_argument('--criterion-weights', nargs=5, type=float, default=[2.0, 1.0, 1.0, 1.0, 1.0])
	parser.add_argument('--from-epoch', type=int, default=0)
	parser.add_argument('--total-epochs', type=int, default=2000000)
	parser.add_argument('--lr-min', type=float, default=5e-5)
	parser.add_argument('--lr-max', type=float, default=5e-4)
	parser.add_argument('--lr-saturate-epoch', type=int, default=1600000)
	parser.add_argument('--sd-min', type=float, default=0.7)
	parser.add_argument('--sd-max', type=float, default=2.0)
	parser.add_argument('--sd-saturate-epoch', type=int, default=200000)
	# Dataset settings
	parser.add_argument('--rootdir-train', type=str, default=UNDEFINED)
	parser.add_argument('--rootdir-test', type=str, default=UNDEFINED)
	parser.add_argument('--train-subset-size', type=int, default=64)
	parser.add_argument('--test-subset-size', type=int, default=64)
	parser.add_argument('--batch-size', type=int, default=8)
	parser.add_argument('--data-worker', type=int, default=4)
	# Trained model and checkpoint output
	parser.add_argument('--coward', action='store_true')
	parser.add_argument('--savedir', type=str, default=UNDEFINED)
	parser.add_argument('--force-checkpoint', action='store_true')
	parser.add_argument('--checkpoint-interval', type=int, default=2000)

	args = parser.parse_args()

	# If generate-json is set, write arguments to file.
	if os.path.isdir(args.generate_default_json):
		json_name = os.path.join(args.generate_default_json, 'trainer_arguments.json')
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

	# Assert rank and world size
	assert args.local_rank <= args.world_size, 'Rank exceeds world size.'

	# Set target device
	if args.device == 'cpu':
		target_device = 'cpu'
	elif args.device == 'cuda':
		target_device = 'cuda:{}'.format(args.local_rank)
	setattr(args, 'target_device', target_device)

	# --- --- ---
	main(args)
