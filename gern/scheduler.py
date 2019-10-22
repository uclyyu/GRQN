class LearningRateScheduler(object):
    def __init__(self, optimiser, lrmin, lrmax, end_epoch):
        assert lrmin < lrmax
        self.optimiser = optimiser
        self.lrmin = lrmin
        self.lrmax = lrmax
        self.end_epoch = end_epoch

    def step(self, epoch):
        lr = self.lrmin + (self.lrmax - self.lrmin) * \
            (1 - epoch / self.end_epoch)
        lr = max(lr, self.lrmin)
        for param_group in self.optimiser.param_groups:
            param_group['lr'] = lr

        return lr


class PixelStdDevScheduler(object):
    def __init__(self, sdmin, sdmax, end_epoch):
        assert sdmin < sdmax
        self.sdmin = sdmin
        self.sdmax = sdmax
        self.end_epoch = end_epoch
        self._weight = 1. / self.sdmax

    @property
    def weight(self):
        return self._weight

    def step(self, epoch):
        sd = self.sdmin + (self.sdmax - self.sdmin) * \
            (1 - epoch / self.end_epoch)
        sd = max(sd, self.sdmin)

        self._weight = 1. / sd
