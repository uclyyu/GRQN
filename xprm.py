#!/usr/bin/env python
import os
import sys
from sacred import Experiment
from sacred.observers import MongoObserver

try:
    GQN_EXPNAME = os.environ['GQN_EXPNAME']
    GQN_EXPADDR = os.environ['GQN_EXPADDR']

except KeyError:
    print("""

    Expect the following environment variables:
    * GQN_EXPNAME (e.g., "GREST-HAS-GQN")
    * GQN_EXPADDR (e.g., "localhost:27001" or leave empty)

    """)

else:
    xprm = Experiment(GQN_EXPNAME)

    if GQN_EXPADDR:
        xprm.observers.append(MongoObserver(url=GQN_EXPADDR, db_name=GQN_EXPNAME))

    print('*** Temporaily adding path at {}'.format(os.getcwd()))
    sys.path.insert(1, os.getcwd())

    @xprm.config
    def cfg_default():
        # task trigger
        tasks = {
            'train': False,
            'restore': False,
            'sandbox': False}
        argv = ' '.join(sys.argv)  # command issued
        db_name = GQN_EXPNAME  # experiment name
        db_addr = GQN_EXPADDR  # MongoDB address

        # path information
        paths = {
            'data': {'train': '../../../../data/gern/phase/train',
                     'test': '../../../../data/gern/phase/test'},
            'tensorboard': f'../../../../data/gern/boards/{GQN_EXPNAME}'}

        # dependency inject
        dinj = {
            'models': '__UNDEFINED__',
            'data': '__UNDEFINED__',
            'criteria': '__UNDEFINED__',
            'schedulers': '__UNDEFINED__',
            'workers': '__UNDEFINED__'}

        num_epoch = None  # total training epoch
        ckpt_interval = None  # saving checkpoint 
        batch_size = None  # batch size
        batch_split = None  # batch split
        subset_size = None  # test set subset size
        shuffle = True  # Pytorch DataLoader argument
        drop_last = True  # Pytorch DataLoader argument
        num_workers = None  # Pytorch DataLoader argument

        wL1 = 0.  # L1 regularisation weight
        wL2 = 0.  # L2 regularisation weight
        wKL = 1.  # ELBO Kl weight
        ndraw = None  # number of prior/posterior factors

        # size information
        sizes = {
            'model': {
                'repr': None,
                'latent': None,
                'hidden': None,
                'query': None},
            'other': {}
        }

        # learning rate
        lr = {
            'start': None,
            'end': None,
            'saturate_epoch': None}

        # fixed pixel standard deviation
        sd = {
            'start': None,
            'end': None,
            'saturate_epoch': None}

        # dropout probability
        drop_pr = {
            'iop_j': {
                'encoder': 0., 'posterior': 0.},
            'iop_d': {
                'encoder': 0., 'posterior': 0.},
            'gop_j': {
                'prior': 0., 'delta': 0.},
            'gop_d': {
                'prior': 0., 'delta': 0.}}

        restore = {
            'db': db_name,
            'run': None, 
            'epoch': -1}

        device = 'cuda:1'  # {cpu | cuda:#}

        try:
            exec(f"from models.{dinj['models']} import Model")
            DefaultModel = Model(xprm)

        except ImportError:
            DefaultModel = Model = None

        try:
            exec(f"from data.{dinj['data']} import Data")
            DefaultData = Data(xprm)

        except ImportError:
            DefaultData = DeafultData = None

        try:
            exec(f"from criteria.{dinj['criteria']} import Criteria")
            DefaultCriteria = Criteria(xprm)

        except ImportError:
            DefaultCriteria = Criteria = None

        try:
            exec(f"from schedulers.{dinj['schedulers']} import Scheduler")
            DefaultScheduler = Scheduler(xprm)

        except ImportError:
            DefaultScheduler = Scheduler = None

        try:
            exec(f"from workers.{dinj['workers']} import Worker")
            DefaultWorker = Worker(xprm)

        except ImportError:
            DefaultWorker = Worker = None

    @xprm.named_config
    def _model_iteration_1_():
        dinj = {
            'models': 'M01',
            'data': 'D00',
            'criteria': 'C00',
            'schedulers': 'S00',
            'workers': 'W00'}

        wL1 = 0.
        wL2 = 0.
        wKL = 1e-5
        ndraw = 7
        num_epoch = 2000
        ckpt_interval = 5
        batch_size = 256
        batch_split = 16
        subset_size = 2560
        shuffle = True
        drop_last = True
        num_workers = 4

        lr = {
            'start': 1e-3,
            'end': 1e-4,
            'saturate_epoch': 2000}

        sd = {
            'start': 2.0,
            'end': 0.7,
            'saturate_epoch': 1500}

        drop_pr = {
            'iop_j': {
                'encoder': 0., 'posterior': 0.},
            'iop_d': {
                'encoder': 0., 'posterior': 0.},
            'gop_j': {
                'prior': 0., 'delta': 0.},
            'gop_d': {
                'prior': 0., 'delta': 0.}}

        sizes = {
            'model': {
                'repr': 128,
                'latent': 32,
                'hidden': 32,
                'query': 4},
            'other': {}}

    @xprm.command
    def sandbox(_run, DefaultModel, DefaultData, DefaultCriteria, DefaultScheduler, DefaultWorker):
        # sandbox task
        worker = DefaultWorker()
        worker.restore()
        pass

    @xprm.main
    def main(tasks, DefaultWorker):

        if tasks['sandbox']:
            sandbox()

        else:
            worker = DefaultWorker()

            if tasks['restore']:
                worker.restore()

            if tasks['train']:
                worker.train()

if __name__ == '__main__':
    xprm.run_commandline()
