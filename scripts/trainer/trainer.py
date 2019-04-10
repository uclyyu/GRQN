from gern.gern import GeRN
from gern.data import GernDataLoader
from gern.criteria import GernCriterion
from gern.scheduler import LearningRateScheduler, PixelStdDevScheduler
from torch import distributed
from torch.utils import checkpoint as ptcp
from glob import glob
from tensorboardX import SummaryWriter
import torch, argparse, os, json, sys, time


def initialise_process(args):
	distributed.init_process_group(
		backend=args.backend,
		init_method=args.init_method,
		rank=args.rank,
		world_size=args.world_size)

def average_gradients(args, model):
	group = distributed.new_group(range(args.world_size))
	world_size = float(args.world_size)
	for param in filter(lambda p: p.requires_grad, model.parameters()):
		distributed.all_reduce(param.grad.data, op=distributed.ReduceOp.SUM, group=group)
		param.grad.data /= world_size

def log_best_json(args, epoch, accuracy, l_percept, l_heatmap, l_classifier, l_aggregate, l_kldiv):
	save_name = os.path.join(args.bestdir, 'stats.json')
	with open(save_name, 'w') as jw:
		data = {'epoch': epoch, 'accuracy': accuracy, 
				'l_percept': l_percept, 'l_heatmap': l_heatmap, 
				'l_classifier': l_classifier, 'l_aggregate': l_aggregate, 'l_kldiv': l_kldiv }
		json.dump(data, jw)

def train_distributed(args, model, criterion, optimiser, lr_scheduler, sd_scheduler, writer):

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
	dataloaders = {'train': GernDataLoader(args.rootdir_train, subset_size=args.sample_size, batch_size=args.batch_size, num_workers=args.data_worker),
				  'test':  GernDataLoader(args.rootdir_test,  subset_size=args.sample_size, batch_size=args.batch_size, num_workers=args.data_worker)}
	
	since = time.time()
	for epoch in range(args.from_epoch, args.total_epochs):

		writer.add_text(
			args.tag_epoch, 
			'Epoch {} (+{:.0f}s)'.format(epoch, time.time() - since))
		since = time.time()

		# alternating between training and testing phases
		for phase in ['train', 'test']:
			if phase == 'train':
				lr_scheduler.step(epoch)
				sd_scheduler.step(epoch)
				model.train()
			else:
				model.eval()

			running_loss = 0.
			running_correct = 0.

			# zero parameter gradients
			optimiser.zero_grad()

			for C, Q, L in dataloaders[phase]:
				# copy to device
				C = [c.to(args.target_device) for c in C]
				Q = [q.to(args.target_device) for q in Q]
				L = L.to(args.target_device)

				# --- Forward pass
				with torch.set_grad_enabled(phase == 'train'):
					gern_output = model(*C, *Q, asteps=args.asteps, rsteps=args.max_rsteps)
					gern_target = model.make_target(*Q, L, rsteps=args.max_rsteps)

					weighted_loss, correct = criterion(gern_output, gern_target, args.criterion_weights)

					running_loss += sum(criterion.item())
					running_correct += correct

					# accumulate gradients (training phase only)
					if phase == 'train':
						weighted_loss.backward()

			# --- Step optimiser (training phase only)
			if phase == 'train':
				average_gradients(args, model)
				optimiser.step()
			
			epoch_loss = running_loss / args.total_epochs
			epoch_correct = running_correct / args.total_epochs

			# --- Save best model
			is_best = False
			if phase == 'test' and epoch_correct > best_correct:
				is_best = True
				best_correct = epoch_correct
				save_name = os.path.join(args.bestdir, 'model.pth')
				torch.save(model.state_dict(), save_name)
				
				writer.add_text(
					args.tag_best,
					'Best model @{}, correct rate={}'.format(epoch, best_correct))
				writer.add_scalars(
					args.tag_best,
					{'lpercept': criterion.l_percept.item(),
					 'lheatmap': criterion.l_heatmap.item(),
					 'lclassifier': criterion.l_classifier.item(),
					 'laggreate': criterion.l_aggregate.item(),
					 'lkldiv': criterion.l_kldiv.item(),
					 'accuracy': criterion.accuracy},
					 epoch)
				log_best_json(args, epoch, best_correct, *criterion.item())

			# --- Make training checkpoint
			if phase == 'train':
				if epoch == 0 or ((epoch + 1) % args.checkpoint_interval) == 0:
					save_name = os.path.join(args.chkptdir, 'chkpt_{:08d}.pth'.format(epoch))
					torch.save(model.state_dict(), save_name)

			# --- Log training / testing progress
			if phase == 'train':
				tag = args.tag_train
			else:
				tag = args.tag_test
			writer.add_scalars(
					tag,
					{'lpercept': criterion.l_percept.item(),
					 'lheatmap': criterion.l_heatmap.item(),
					 'lclassifier': criterion.l_classifier.item(),
					 'laggreate': criterion.l_aggregate.item(),
					 'lkldiv': criterion.l_kldiv.item(),
					 'accuracy': criterion.accuracy},
					 epoch)


def main(args):
	# --- Setting up trainer
	initialise_process(args)
	model = GeRN().to(args.target_device)
	criterion = GernCriterion().to(args.target_device)
	optimiser = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), args.lr_min, amsgrad=True)
	lr_scheduler = LearningRateScheduler(optimiser, args.lr_min, args.lr_max, args.lr_saturate_epoch)
	sd_scheduler = PixelStdDevScheduler(args.criterion_weights, 0, args.sd_min, args.sd_max, args.sd_saturate_epoch)

	# --- Tensorboard writer
	writerdir = os.path.join(args.savedir, 'board', 'rank', '{:02d}'.format(args.rank))
	if args.coward:
		assert not os.path.isdir(writerdir), 'Cowardly avoid overwriting run folder.'
	try:
		os.makedirs(writerdir)
	except FileExistsError:
		pass

	writer = SummaryWriter(log_dir=writerdir, filename_suffix='.r{:03d}'.format(args.run))

	# --- Add tensorboard tags
	setattr(args, 'tag_general', 'run/{:03d}/general'.format(args.run))
	setattr(args, 'tag_best', 'run/{:03d}/best'.format(args.run))
	setattr(args, 'tag_epoch', 'run/{:03d}/epoch'.format(args.run))
	setattr(args, 'tag_train', 'run/{:03d}/train'.format(args.run))
	setattr(args, 'tag_test', 'run/{:03d}/test'.format(args.run))

	# --- Sanity-checks: checkpoint
	chkptdir = os.path.join(args.savedir, 'checkpoints', 'rank', '{:02d}', 'run', '{:03d}').format(args.rank, args.run)
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
	bestdir = os.path.join(args.savedir, 'best', 'rank', '{:02d}', 'run', '{:03d}').format(args.rank, args.run)
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
	if args.rank is None:
		raise NotImplementedError
	else:	
		train_distributed(args, model, criterion, optimiser, lr_scheduler, sd_scheduler, writer)


if __name__ == '__main__':
	UNDEFINED = '<UNDEFINED>'
	parser = argparse.ArgumentParser()
	# Command line argument management
	parser.add_argument('--use-json', type=str, default=UNDEFINED)
	parser.add_argument('--generate-default-json', type=str, default=UNDEFINED)
	# Settings for PyTorch distributed module
	parser.add_argument('--rank', type=int, default=0)
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
	parser.add_argument('--sample-size', type=int, default=64)
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
	assert args.rank <= args.world_size, 'Rank exceeds world size.'

	# Set target device
	if args.device == 'cpu':
		target_device = 'cpu'
	elif args.device == 'cuda':
		target_device = 'cuda:{}'.format(args.rank)
	setattr(args, 'target_device', target_device)

	# --- --- ---
	main(args)
