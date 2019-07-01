# Trainer script for the BallTube example
import torch
from gern.gern import GeRN
from gern.data import BallTubeDataLoader
from gern.criteria import GernCriterion
from gern.scheduler import LearningRateScheduler, PixelStdDevScheduler
from torch import distributed
from torch.utils import checkpoint as ptcp
from glob import glob
from tensorboardX import SummaryWriter
from tqdm import tqdm
import torch, argparse, os, json, sys, time
import matplotlib.pyplot as plt
import numpy as np


def apply_regularizer(r=[]):
	def _apply_regularizer(module):
		if hasattr(module, 'regularizer'):
			r.append(module.regularizer())
	return _apply_regularizer


def gather_regularizer(model):
	r = []
	model.apply(apply_regularizer(r))
	reg, num = zip(*r)
	return sum(reg) / sum(num)


def log_best_json(args, epoch, accuracy, l_rgbv, l_classifier, l_kldiv):
	save_name = os.path.join(args.bestdir, 'stats.json')
	with open(save_name, 'w') as jw:
		data = {'epoch': epoch, 'accuracy': accuracy, 
				'l_rgbv': l_rgbv, 'l_classifier': l_classifier, 'l_kldiv': l_kldiv }
		json.dump(data, jw)


def plot_rgbv(pred, targ, tindex):
	d = pred[0, tindex[0]].permute(1, 2, 0).contiguous().detach().cpu().numpy()
	o = targ[0, tindex[0]].permute(1, 2, 0).contiguous().detach().cpu().numpy()
	d = (d - d.min()) / (d.max() - d.min())
	o = (o - o.min()) / (o.max() - o.min())
	figure = plt.figure(figsize=(10, 5))
	ax = plt.subplot(121)
	ax.set_title('Decoded')
	ax.set_axis_off()
	plt.imshow(d)
	ax = plt.subplot(122)
	ax.set_title('Original')
	ax.set_axis_off()
	plt.imshow(o)
	return figure


def train(args, model, criterion, optimiser, lr_scheduler, sd_scheduler, writer):

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
	dataloaders = {'train': BallTubeDataLoader(args.rootdir_train, subset_size=subset_sizes['train'], batch_size=1, num_workers=args.data_worker),
				   'test':  BallTubeDataLoader(args.rootdir_test,  subset_size=subset_sizes['test'],  batch_size=1, num_workers=args.data_worker)}
	
	since = time.time()
	for epoch in range(args.from_epoch, args.total_epochs):

		# --- Information
		elapsed = time.time() - since
		since = time.time()
		epoch_string = '\n\n--- Epoch {:5d} (+{:.0f}s) {}'.format(epoch, elapsed, '-' * 50)
		print(epoch_string)

		# --- Alternating between training and testing phases
		for phase in ['train', 'test']:
			if phase == 'train':
				lr_scheduler.step(epoch)
				sd_scheduler.step(epoch)
				model.train()
			else:
				model.eval()

			epoch_loss = 0.
			epoch_correct = 0.
			epoch_regularizer = 0.
			batch_loss = 0.
			batch_correct = 0.
			batch_regularizer = 0.


			optimiser.zero_grad()
			# --- Iterate over the current subset
			with torch.set_grad_enabled(phase == 'train'):
				for i, (cx, cv, qx, qv, label) in enumerate(dataloaders[phase]):
					
					# --- Model inputs
					cx, cv = [c.to(args.target_device) for c in [cx, cv]]
					qx, qv = [q.to(args.target_device) for q in [qx, qv]]
					label = label.to(args.target_device)

					# --- Forward pass
					model_output = model(cx, cv, qx, qv)
					dec_rgbv, cat_logits, prior_means, prior_logvs, posterior_means, posterior_logvs = model_output

					weighted_loss = criterion(label, qx, dec_rgbv, cat_logits, 
								  		   	  prior_means, prior_logvs, posterior_means, posterior_logvs, 
											  sd_scheduler.weights)
					regularizer = gather_regularizer(model)

					# --- Accumulate gradients
					if phase == 'train':
						((weighted_loss + 0.01 * regularizer) / args.batch_size).backward()

					batch_loss += np.array(criterion.item())
					epoch_loss += np.array(criterion.item())
					batch_correct += criterion.accuracy
					epoch_correct += criterion.accuracy
					batch_regularizer += regularizer
					epoch_regularizer += regularizer

					if (i + 1) % args.batch_size == 0:
						# --- Step optimiser
						if phase == 'train':
							optimiser.step()

						# --- Log batch progress
						batch_loss = batch_loss / args.batch_size
						batch_correct = batch_correct / args.batch_size
						batch_regularizer = batch_regularizer / args.batch_size
						writer.add_scalar(args.tags_loss['rgbv']       ('batch', phase), batch_loss[0], epoch)
						writer.add_scalar(args.tags_loss['classifier'] ('batch', phase), batch_loss[1], epoch)
						writer.add_scalar(args.tags_loss['divergence'] ('batch', phase), batch_loss[2], epoch)
						writer.add_scalar(args.tags_loss['accuracy']   ('batch', phase), batch_correct, epoch)
						writer.add_scalar(args.tags_loss['regularizer']('batch', phase), batch_regularizer, epoch)

						# --- Reset 
						optimiser.zero_grad()
						batch_loss = 0.
						batch_correct = 0.
				
				epoch_loss = epoch_loss / subset_sizes[phase]
				epoch_correct = epoch_correct / subset_sizes[phase]
				epoch_regularizer = epoch_regularizer / subset_sizes[phase]

				# --- Save best model
				is_best = False
				if phase == 'test' and epoch_correct > best_correct:
					is_best = True
					best_correct = epoch_correct
					save_name = os.path.join(args.bestdir, 'model.pth')
					torch.save(model.state_dict(), save_name)
					
					writer.add_scalar(args.tags_loss['rgbv']      ('best', phase), epoch_loss[0], epoch)
					writer.add_scalar(args.tags_loss['classifier']('best', phase), epoch_loss[1], epoch)
					writer.add_scalar(args.tags_loss['divergence']('best', phase), epoch_loss[2], epoch)
					writer.add_scalar(args.tags_loss['accuracy']  ('best', phase), epoch_correct, epoch)
					log_best_json(args, epoch, best_correct, *epoch_loss)

				# --- Make training checkpoint
				if phase == 'train':
					if epoch == 0 or ((epoch + 1) % args.checkpoint_interval) == 0:
						save_name = os.path.join(args.chkptdir, 'chkpt_{:08d}.pth'.format(epoch))
						torch.save(model.state_dict(), save_name)

				# --- Log epoch progress
				writer.add_scalar(args.tags_loss['rgbv']       ('epoch', phase), epoch_loss[0], epoch)
				writer.add_scalar(args.tags_loss['classifier'] ('epoch', phase), epoch_loss[1], epoch)
				writer.add_scalar(args.tags_loss['divergence'] ('epoch', phase), epoch_loss[2], epoch)
				writer.add_scalar(args.tags_loss['accuracy']   ('epoch', phase), epoch_correct, epoch)
				writer.add_scalar(args.tags_loss['regularizer']('epoch', phase), epoch_regularizer, epoch)

				# --- Save origin and decoded images every epoch
				writer.add_figure(args.tags_figure['rgbv'](phase), plot_rgbv(dec_rgbv, qx, criterion.min_index), epoch)


def main(args):
	# --- Setting up trainer
	model = GeRN(asteps=args.asteps).to(args.target_device)
	criterion = GernCriterion().to(args.target_device)
	optimiser = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), args.lr_min, amsgrad=True)
	lr_scheduler = LearningRateScheduler(optimiser, args.lr_min, args.lr_max, args.lr_saturate_epoch)
	sd_scheduler = PixelStdDevScheduler(args.criterion_weights, 0, args.sd_min, args.sd_max, args.sd_saturate_epoch)

	# --- Tensorboard writer
	writerdir = os.path.join(args.savedir, 'board')
	if args.coward:
		assert not os.path.isdir(writerdir), 'Cowardly avoid overwriting run folder.'
	try:
		os.makedirs(writerdir)
	except FileExistsError:
		pass

	writer = SummaryWriter(writerdir, filename_suffix='.r{:03d}'.format(args.run))

	# --- Add tensorboard tags
	setattr(args, 'tags_loss',{
		'rgbv': 	  lambda stage, phase: 'run/{:03d}/loss/{}/rgbv/{}'.format(args.run, stage, phase),
		'classifier': lambda stage, phase: 'run/{:03d}/loss/{}/classifier/{}'.format(args.run, stage, phase),
		'divergence': lambda stage, phase: 'run/{:03d}/loss/{}/divergence/{}'.format(args.run, stage, phase),
		'accuracy':	  lambda stage, phase: 'run/{:03d}/loss/{}/accuracy/{}'.format(args.run, stage, phase),
		'regularizer': lambda stage, phase: 'run/{:03d}/loss/{}/regularizer/{}'.format(args.run, stage, phase)
		})
	setattr(args, 'tags_figure', {
		'rgbv': lambda phase: 'run/{:03d}/figure/rgbv/{}'.format(args.run, phase)
		})
	setattr(args, 'tag_general', 'run/{:03d}/general'.format(args.run))


	# --- Sanity-checks: checkpoint
	chkptdir = os.path.join(args.savedir, 'checkpoints', 'run', '{:03d}').format(args.run)
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
	bestdir = os.path.join(args.savedir, 'best', 'run', '{:03d}').format(args.run)
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
	train(args, model, criterion, optimiser, lr_scheduler, sd_scheduler, writer)


if __name__ == '__main__':
	UNDEFINED = '<UNDEFINED>'
	parser = argparse.ArgumentParser()
	# Command line argument management
	parser.add_argument('--use-json', type=str, default=UNDEFINED)
	parser.add_argument('--generate-default-json', type=str, default=UNDEFINED)
	# Main training arguments
	parser.add_argument('--run', type=int, default=0)
	parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
	parser.add_argument('--asteps', type=int, default=16)
	parser.add_argument('--criterion-weights', nargs=5, type=float, default=[2.0, 1.0, 0.01])
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
			json.dump(vars(args), jw, sort_keys=True)
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


	# Set target device
	if args.device == 'cpu':
		target_device = 'cpu'
	elif args.device == 'cuda':
		target_device = 'cuda'
	setattr(args, 'target_device', target_device)

	# --- --- ---
	main(args)
