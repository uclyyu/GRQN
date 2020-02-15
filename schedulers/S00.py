def Scheduler(xprm):

    class _Scheduler(object):

        @xprm.capture
        def __init__(self, optimiser, lr, sd):
            self.optimiser = optimiser
            self._sd = sd['start']
            self._lr = lr['start']

        @xprm.capture
        def _step_lr(self, epoch, lr):
            _lr = lr['end'] + (lr['start'] - lr['end']) * \
                (1 - epoch / lr['saturate_epoch'])
            _lr = max(_lr, lr['end'])
            self._lr = _lr

            for param_group in self.optimiser.param_groups:
                param_group['lr'] = _lr

        @xprm.capture
        def _step_sd(self, epoch, sd):
            _sd = sd['end'] + (sd['start'] - sd['end']) * \
                (1 - epoch / sd['saturate_epoch'])
            _sd = max(_sd, sd['end'])

            self._sd = _sd

        def step(self, epoch):
            self._step_lr(epoch)
            self._step_sd(epoch)

        @property
        def isd(self):
            return 1. / self._sd

        @property
        def lr(self):
            return self._lr

    return _Scheduler
