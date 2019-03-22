from loguru import logger


class LearningRateScheduler(object):
	def __init__(self, optimiser, lrmin, lrmax, end_epoch):
		assert lrmin < lrmax
		self.optimiser = optimiser
		self.lrmin = lrmin
		self.lrmax = lrmax
		self.end_epoch = end_epoch

	def step(self, epoch):
		lr = self.lrmin + (self.lrmax - self.lrmin) * (1 - epoch / self.end_epoch)
		lr = max(lr, self.lrmin)
		for param_group in self.optimiser.param_groups:
			param_group['lr'] = lr
		logger.opt(ansi=True).info('Stepping learning rate = {new:.8f}.', new=lr)


class PixelStdDevScheduler(object):
	def __init__(self, weights, index, sdmin, sdmax, end_epoch):
		assert sdmin < sdmax
		assert index < len(weights)
		self.weights = weights
		self.index = index
		self.sdmin = sdmin
		self.sdmax = sdmax
		self.end_epoch = end_epoch

	def step(self, epoch):
		sd = self.sdmin + (self.sdmax - self.sdmin) * (1 - epoch / self.end_epoch)
		sd = max(sd, self.sdmin)
		self.weights[self.index] = 1. / sd
		logger.opt(ansi=True).info('Stepping pixel-sd = {new:.8f}.', new=sd)

