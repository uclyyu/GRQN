from gern.gern import GeRN
from gern.data import GernDataLoader
from gern.criteria import GernCriterion
from gern.scheduler import LearningRateScheduler, PixelStdDevScheduler
from loguru import logger
from torch import distributed
from glob import glob
import torch, argparse, os, json, sys, csv, time

_CHECKPOINT_DIR_NAME_ = 'checkpoints'

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
	save_name = os.path.join(args.savedir, 'best_model.json')
	with open(save_name, 'w') as jw:
		data = {'epoch': epoch, 'accuracy': accuracy, 
				'l_percept': l_percept, 'l_heatmap': l_heatmap, 
				'l_classifier': l_classifier, 'l_aggregate': l_aggregate, 'l_kldiv': l_kldiv }
		json.dump(data, jw)

def train_distributed(args, model, criterion, optimiser, lr_scheduler, sd_scheduler, csvwriter):

	best_correct = 0.	
	# --- Load checkpoint if from_epoch > 0
	if args.from_epoch > 0:
		assert args.from_epoch % args.checkpoint_interval == 0
		load_name = os.path.join(args.savedir, _CHECKPOINT_DIR_NAME_, 'model_{:08d}.pth'.format(args.from_epoch - 1))
		model.load_state_dict(torch.load(load_name))
		logger.info('Recovered model from checkpoint {chk}.', chk=args.from_epoch - 1)

		# Load best model statisitcs
		json_name = os.path.join(args.savedir, 'best_model.json')
		with open(json_name, 'r') as jr:
			json_data = json.load(jr)
			best_correct = json_data['accuracy']

	# --- Build data loader
	dataloaders = {'train': GernDataLoader(args.rootdir_train, subset_size=args.sample_size, batch_size=args.batch_size, num_workers=args.data_worker),
				  'test':  GernDataLoader(args.rootdir_test,  subset_size=args.sample_size, batch_size=args.batch_size, num_workers=args.data_worker)}
	
	since = time.time()
	for epoch in range(args.from_epoch, args.total_epochs):

		logger.opt(ansi=True).info('<cyan> ** Epoch {epoch}</cyan> (+ {sec:.0f}s) **', epoch=epoch, sec=time.time() - since)
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
					gern_target = model.make_target(*Q, L)

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
				save_name = os.path.join(args.savedir, 'best_model_rank_{:02d}.pth'.format(args.rank))
				torch.save(model.state_dict(), save_name)
				
				logger.info('Reached a best model at epoch {epoch}.', epoch=epoch)
				log_best_json(args, epoch, best_correct, *criterion.item())

			# --- Make training checkpoint
			if phase == 'train':
				if epoch == 0 or ((epoch + 1) % args.checkpoint_interval) == 0:
					save_name = os.path.join(args.savedir, _CHECKPOINT_DIR_NAME_, 'model_{:08d}.pth'.format(epoch))
					torch.save(model.state_dict(), save_name)
					logger.info('Created training checkpoint at epoch {epoch}.', epoch=epoch)

			# --- Log training / testing progress
			progress = {'epoch': epoch,
						'phase': phase, 
						'l_percept': criterion.l_percept.item(),
						'l_heatmap': criterion.l_heatmap.item(),
						'l_classifier': criterion.l_classifier.item(),
						'l_aggregate': criterion.l_aggregate.item(),
						'l_kldiv': criterion.l_kldiv.item(),
						'accuracy': criterion.accuracy,
						'is_best': is_best}
			csvwriter.writerow(progress)


def main(args):
	# --- Setting up trainer
	initialise_process(args)
	model = GeRN().to(args.target_device)
	criterion = GernCriterion().to(args.target_device)
	optimiser = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), args.lr_min, amsgrad=True)
	lr_scheduler = LearningRateScheduler(optimiser, args.lr_min, args.lr_max, args.lr_saturate_epoch)
	sd_scheduler = PixelStdDevScheduler(args.criterion_weights, 0, args.sd_min, args.sd_max, args.sd_saturate_epoch)

	# --- Logger
	logger_file = os.path.join(args.savedir, 'logger.txt')
	logger.add(logger_file)

	# --- Sanity checks.
	chkpdir = os.path.join(args.savedir, _CHECKPOINT_DIR_NAME_)
	if not os.path.exists(chkpdir):
		logger.info('Adding new directory at {d}.', d=chkpdir)
		os.makedirs(chkpdir)

	# --- Get the latest epoch if forcing checkpoint.
	if args.force_checkpoint:
		latest_epoch = 0
		for checkpoint in glob(os.path.join(args.savedir, _CHECKPOINT_DIR_NAME_, '*.pth')):
			epoch = int(checkpoint.split('_')[-1].split('.')[0])
			latest_epoch = max(epoch, latest_epoch)
		args.from_epoch = latest_epoch + 1
		logger.opt(ansi=True).info('Forcing start from <cyan>epoch {chk}</cyan>.', chk=args.from_epoch)

	# --- Training procedure
	if args.rank is None:
		raise NotImplementedError
	else:
		csv_columns = ['epoch', 'phase', 'l_percept', 'l_heatmap', 'l_classifier', 'l_aggregate', 'l_kldiv', 'accuracy', 'is_best']
		csv_name = os.path.join(args.savedir, 'log_training.csv')
		with open(csv_name, 'w') as csvfile:
			csvwriter = csv.DictWriter(csvfile, fieldnames=csv_columns)
			csvwriter.writeheader()
			train_distributed(args, model, criterion, optimiser, lr_scheduler, sd_scheduler, csvwriter)


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
	# Main training hyperparameters
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
	parser.add_argument('--savedir', type=str, default=UNDEFINED)
	parser.add_argument('--force-checkpoint', type=bool, default=False)
	parser.add_argument('--checkpoint-interval', type=int, default=2000)

	args = parser.parse_args()

	# If generate-json is set, write arguments to file.
	if os.path.exists(args.generate_default_json):
		json_name = os.path.join(args.generate_default_json, 'trainer_arguments.json')
		logger.opt(ansi=True).info('Generate default .json at <cyan>{name}</cyan>.', name=json_name)
		args.generate_default_json = None
		with open(json_name, 'w') as jw:
			json.dump(vars(args), jw)
		sys.exit(0)

	# If an argument .json file is assigned, update args.
	if os.path.exists(args.use_json):
		with open(args.use_json, 'r') as jr:
			json_data = json.load(jr)
			json_data['use_json'] = args.use_json
			args.__dict__.update(json_data)
			logger.opt(ansi=True).info('Loaded trainer arguments from <cyan>{file}</cyan>.', file=args.use_json)

	# Assert savedir
	if not os.path.exists(args.savedir):
		logger.exception('--savedir "{path}" is not a valid path name.', path=args.savedir)

	# Assert rank and world size
	if args.rank >= args.world_size:
		logger.exception('Rank exceeds world size limit: {rank} >= {ws}.', rank=args.rank, ws=args.world_size)

	# Set target device
	if args.device == 'cpu':
		target_device = 'cpu'
	elif args.device == 'cuda':
		target_device = 'cuda:{}'.format(args.rank)
	setattr(args, 'target_device', target_device)

	# --- --- ---
	main(args)
