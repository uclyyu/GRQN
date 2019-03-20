from gern.gern import GeRN
from gern.data import GernDataLoader
from gern.criteria import GernCriterion
from gern.scheduler import LearningRateScheduler, InversePrecisionScheduler
from loguru import logger
from torch import distributed
import torch, argparse


# TODO:
# 1. Scheduling methods for learning rate and perceptual loss inverse precision
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


def train_distributed(args, model, criterion, optimiser, lr_scheduler, ip_scheduler):
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
	
	for epoch in range(args.from_epoch, args.total_epoch):

		# alternating between training and testing phases
		for phase in ['train', 'test']:
			if phase == 'train':
				lr_scheduler.step(epoch)
				ip_scheduler.step(epoch)
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
			# . track gradients in training phase only
			with torch.set_grad_enabled(phase == 'train'):
				gern_output = model(*C, *Q, asteps=args.ar_steps)
				gern_target = model.make_target(*Q, L)

				weighted_loss, correct = criterion(outputs, target, weights)

				running_loss += sum(criterion.item()[:-1])
				running_correct += correct


				# . A
				if phase == 'train':
					weight_loss.backward()



		# --- Optimise only in 
		if phase == 'train':
			pass
		# --- Average gradients
		average_gradients(model)


		epoch_loss
		epoch_accuracy

		# --- Save trained model
		if epoch == 0 or (epoch % save_model_every) == 0:
			pass




def main(args):
	initialise_process(args)




def anneal_sigma(epoch, epoch_max, sigma_min=0.7, sigma_max=2.0):
	return max(sigma_min + (sigma_max - sigma_min) * (1 - epoch / epoch_max), sigma_min)



def anneal_lr(optimiser, epoch, n=1.6e6, lr_min=5e-5, lr_max=5e-4):
	lr = max(lr_min + (lr_max - lr_min) * (1 - epoch / n), lr_min)
	for param_group in optimiser.param_groups:
		param_group['lr'] = lr


class OrderNamespaceAction(argparse.Action):
	def __call__(self, parser, namespace, values, option_string=None):
		if not 'ordered_args' in namespace:
			setattr(namespace, 'ordered_args', [])
		previous = namespace.ordered_args
		previous.append((self.dest, values))
		setattr(namespace, 'ordered_args', previous)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	# offer to generate default configuration file in json format. 
	parser.add_argument('--use-json')
	parser.add_argument('--generate-json', metavar='PATH')
	# on distributed
	parser.add_argument('--rank')
	parser.add_argument('--world-size')
	parser.add_argument('--backend')
	parser.add_argument('--init-method')
	# - 
	# on model
	parser.add_argument('--ar-steps')
	parser.add_argument('--criterion-weights', nargs=5, type=float)
	# lr
	parser.add_argument('--from-epoch')
	parser.add_argument('--total-epochs')
	parser.add_argument('--lr-min')
	parser.add_argument('--lr-max')
	parser.add_argument('--sigma-min')
	parser.add_argument('--sigma-max')
	# load/save
	parser.add_argument('--rootdir-train')
	parser.add_argument('--rootdir-test')
	parser.add_argument('--sample-size')
	parser.add_argument('--batch-size')
	parser.add_argument('--data-worker')
	parser.add_argument('--savedir')

	# parser.add_argument('--backend', type=str, dest='backend', default='nccl')
	# parser.add_argument('--init-method', type=str, dest='init', default='tcp://192.168.1.101:23456')
	# parser.add_argument('--rank', dest='rank', type=int)
	# parser.add_argument('--world-size', dest='size', type=int)
