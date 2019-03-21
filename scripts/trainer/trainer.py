from gern.gern import GeRN
from gern.data import GernDataLoader
from gern.criteria import GernCriterion
from gern.scheduler import LearningRateScheduler, PixelStdDevScheduler
from loguru import logger
from torch import distributed
import torch, argparse, os, json, sys


def initialise_process(args):
	distributed.init_process_group(
		backend=args.backend,
		init_method=args.init_method,
		rank=args.rank,
		world_size=args.world_size)

def average_gradients(model):
	world_size = float(distributed.get_world_size())
	for param in filter(lambda p: p.requires_grad, model.parameters()):
		distributed.all_reduce(param.grad.data, op=distributed.reduce_op.SUM)
		param.grad.data /= world_size

def log_best_json(args, epoch, accuracy, l_percept, l_heatmap, l_classifier, l_aggregate, l_kldiv):
	save_name = os.path.join(args.savedir, 'best_model.json')
	with open(save_name, 'w') as jw:
		data = {'epoch': epoch, 'accuracy': accuracy, 
				'l_percept': l_percept, 'l_heatmap': l_heatmap, 
				'l_classifier': l_classifier, 'l_aggregate': l_aggregate, 'l_kldiv': l_kldiv }
		json.dump(data, jw)

def train_distributed(args, model, criterion, optimiser, lr_scheduler, sd_scheduler, csvwriter):
	# --- Set target cuda device 
	device = 'cuda:{}'.format(args.rank)
	
	best_correct = 0.	
	# --- Load checkpoint if from_epoch > 0
	if args.from_epoch > 0:
		assert (args.from_epoch + 1) % args.checkpoint_interval == 0
		load_name = os.path.join(args.savedir, 'checkpoint', 'model_{:08d}.pth'.format(args.from_epoch))
		model.load_state_dict(torch.load(load_name))
		logger.info('Resumed training from checkpoint {chk}.', chk=args.from_epoch)

		# Load best model statisitcs
		json_name = os.path.join(args.savedir, 'best_model.json')
		with open(json_name, 'r') as jr:
			json_data = json.load(jr)
			best_correct = json_data['accuracy']

	# --- Build data loader
	dataloader = {'train': GernDataLoader(args.rootdir_train, subset_size=args.sample_size, batch_size=args.batch_size, num_workers=args.data_worker),
				  'test':  GernDataLoader(args.rootdir_test,  subset_size=args.sample_size, batch_size=args.batch_size, num_workers=args.data_worker)}
	
	for epoch in range(args.from_epoch, args.total_epochs):

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
				C = [c.to(device) for c in C]
				Q = [q.to(device) for q in Q]
				L = L.to(device)

				# --- Forward pass
				with torch.set_grad_enabled(phase == 'train'):
					gern_output = model(*C, *Q, asteps=args.asteps, rsteps=args.rsteps)
					gern_target = model.make_target(*Q, L)

					weighted_loss, correct = criterion(outputs, target, args.criterion_weights)

					running_loss += sum(criterion.item())
					running_correct += correct

					# accumulate gradients (training phase only)
					if phase == 'train':
						weighted_loss.backward()

			# --- Step optimiser (training phase only)
			if phase == 'train':
				average_gradients(model)
				optimiser.step()
			
			epoch_loss = running_loss / args.total_epochs
			epoch_correct = running_correct / args.total_epochs

			# --- Save best model
			is_best = False
			if phase == 'test' and epoch_correct > best_correct:
				is_best = True
				best_correct = epoch_correct
				save_name = os.path.join(args.savedir, 'best_model_{:02d}.pth'.format(args.rank))
				torch.save(model.state_dict(), save_name)
				
				logger.info('Reached a best model at epoch {epoch}.', epoch=epoch)
				log_best_json(args, epoch, best_correct, *criterion.item())

			# --- Make training checkpoint
			if phase == 'train':
				if epoch == 0 or ((epoch + 1) % args.checkpoint_interval) == 0:
					save_name = os.path.join(args.savedir, 'checkpoints', 'model_{:08d}.pth'.format(epoch))
					torch.save(model.state_dict(), save_name)
					logger.info('Created training checkpoint at epoch {epoch}.', epoch=epoch)

			# --- Log training / testing progress
			progress = {'phase': phase, 
						'l_percept': criterion.l_percept,
						'l_heatmap': criterion.l_heatmap,
						'l_classifier': criterion.l_classifier,
						'l_aggregate': criterion.l_aggregate,
						'l_kldiv': criterion.l_kldiv,
						'accuracy': criterion.accuracy,
						'is_best': is_best}
			csvwriter.writerow(progress)


def main(args):
	# --- Setting up trainer
	initialise_process(args)
	model = GeRN().cuda(args.rank)
	criterion = GernCriterion().cuda(args.rank)
	optimiser = torch.optim.Adam(model.parameters(), args.lr_min, amsgrad=True)
	lr_scheduler = LearningRateScheduler(optimiser, args.lr_min, args.lr_max, args.lr_saturate_epoch)
	sd_scheduler = PixelStdDevScheduler(args.criterion_weights, args.sd_min, args.sd_max, args.sd_saturate_epoch)

	# --- Sanity checks.
	if not os.path.exists(args.savedir):
		os.makedirs(os.path.join(args.savedir, 'checkpoint'))

	# --- Get the latest epoch if forcing checkpoint.
	if args.force_checkpoint:
		latest_epoch = 0
		for checkpoint in glob(os.path.join(args.savedir, 'checkpoint', '*.pth')):
			epoch = int(checkpoint.split('_')[-1].split('.')[0])
			latest_epoch = max(epoch, latest_epoch)
		args.from_epoch = latest_epoch

	# --- Training procedure
	if args.rank is None:
		raise NotImplementedError
	else:
		csv_columns = ['phase', 'l_percept', 'l_heatmap', 'l_classifier', 'l_aggregate', 'l_kldiv', 'accuracy', 'is_best']
		csv_name = os.path.join(args.savedir, 'log_training.csv')
		with open(csv_name, 'w') as csvfile:
			csvwriter = csv.DictWriter(csvfile, fieldnames=csv_columns)
			csvwriter.writeheader()
			train_distributed(args, model, criterion, optimiser, lr_scheduler, sd_scheduler, csvwriter)


if __name__ == '__main__':
	UNDEFINED = '<UNDEFINED>'
	parser = argparse.ArgumentParser()
	# offer to generate default configuration file in json format. 
	parser.add_argument('--use-json', type=str, default=UNDEFINED)
	parser.add_argument('--generate-json', type=str, default=UNDEFINED)
	# on distributed
	parser.add_argument('--rank', type=int, default=0)
	parser.add_argument('--world-size', type=int, default=1)
	parser.add_argument('--backend', type=str, default='nccl')
	parser.add_argument('--init-method', type=str, default='tcp://192.168.1.116:23456')
	# on model
	parser.add_argument('--asteps', type=int, default=7)
	parser.add_argument('--rsteps', default=None)
	parser.add_argument('--criterion-weights', nargs=5, type=float, default=[2.0, 1.0, 1.0, 1.0, 1.0])
	# lr
	parser.add_argument('--from-epoch', type=int, default=0)
	parser.add_argument('--total-epochs', type=int, default=2e6)
	parser.add_argument('--lr-min', type=float, default=5e-5)
	parser.add_argument('--lr-max', type=float, default=5e-4)
	parser.add_argument('--lr-saturate-epoch', type=int, default=1.6e6)
	parser.add_argument('--sd-min', type=float, default=0.7)
	parser.add_argument('--sd-max', type=float, default=2.0)
	parser.add_argument('--sd-saturate-epoch', type=int, default=2e5)
	# load/save
	parser.add_argument('--rootdir-train', type=str, default=UNDEFINED)
	parser.add_argument('--rootdir-test', type=str, default=UNDEFINED)
	parser.add_argument('--sample-size', type=int, default=64)
	parser.add_argument('--batch-size', type=int, default=8)
	parser.add_argument('--data-worker', type=int, default=4)
	parser.add_argument('--savedir', type=str, default=UNDEFINED)
	parser.add_argument('--force-checkpoint', type=bool, default=False)
	parser.add_argument('--checkpoint-interval', type=int, default=2000)

	args = parser.parse_args()

	# If generate-json is set, write arguments to file.
	if os.path.exists(args.generate_json):
		json_name = os.path.join(args.generate_json, 'trainer_arguments.json')
		args.generate_json = None
		with open(json_name, 'w') as jw:
			json.dump(vars(args), jw)
		sys.exit(0)

	# If an argument .json file is assigned, update args.
	if os.path.exists(args.use_json):
		with open(args.use_json, 'r') as jr:
			logger.info('Loading trainer arguments from {file}.', file=args.use_json)
			json_data = json.load(jr)
			args.__dict__.update(json_data)

	if args.savedir == UNDEFINED:
		logger.exception('--savedir must be a valid path.')

	main(args)




