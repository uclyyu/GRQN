from gern.gern import GeRN
from gern.data import GernDataLoader
from gern.criteria import GernCriterion
from gern.scheduler import LearningRateScheduler, PixelStdDevScheduler
from loguru import logger
from torch import distributed
import torch, argparse


# TODO:
# 2. Adding saving outlet and 

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


def train_distributed(args, model, criterion, optimiser, lr_scheduler, sd_scheduler):
	if args.rank is None:
		device = 'cpu'
	elif isinstance(args.rank, int):
		device = 'cuda:{}'.format(args.rank)
	else:
		raise
	# Load model if from_epoch > 0

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
					gern_output = model(*C, *Q, asteps=args.ar_steps)
					gern_target = model.make_target(*Q, L)

					weighted_loss, correct = criterion(outputs, target, args.criterion_weights)

					running_loss += sum(criterion.item()[:-1])
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

			# --- Copy best model

			# --- Save trained model
			if epoch == 0 or (epoch % save_model_every) == 0:
				pass


def main(args):
	initialise_process(args)
	model = GeRN().cuda(args.rank)
	criterion = GernCriterion().cuda(args.rank)
	optimiser = torch.optim.Adam(model.parameters(), args.lr_min, amsgrad=True)
	lr_scheduler = LearningRateScheduler(optimiser, args.lr_min, args.lr_max, args.lr_saturate_epoch)
	sd_scheduler = PixelStdDevScheduler(args.criterion_weights, args.sd_min, args.sd_max, args.sd_saturate_epoch)

	train_distributed(args, model, criterion, optimiser, lr_scheduler, sd_scheduler)



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	# offer to generate default configuration file in json format. 
	parser.add_argument('--use-json', type=str, default='<UNDEFINED>')
	parser.add_argument('--generate-json', metavar='PATH', type=str, default='<UNDEFINED>')
	# on distributed
	parser.add_argument('--rank')
	parser.add_argument('--world-size')
	parser.add_argument('--backend')
	parser.add_argument('--init-method')
	# - 
	# on model
	parser.add_argument('--ar-steps', type=int, default=7)
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
	parser.add_argument('--rootdir-train', type=str, default='<UNDEFINED>')
	parser.add_argument('--rootdir-test', type=str, default='<UNDEFINED>')
	parser.add_argument('--sample-size', type=int, default=64)
	parser.add_argument('--batch-size', type=int, default=8)
	parser.add_argument('--data-worker', type=int, default=4)
	parser.add_argument('--savedir', type=str, default='<UNDEFINED>')
