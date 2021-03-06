from torch.optim.lr_scheduler import *
import numpy as np

class LR(object):
    def __init__(self, optimizer, params):
        super(LR, self).__init__()
        self.params = params
        self.optimizer = optimizer
        self.max_clr_epochs = 0
        self.set_scheduler(self.params.lr_type)

    def set_scheduler(self, lr_type):
        if lr_type==None: self.scheduler = None
        elif lr_type=='decay': self.scheduler = ExponentialLR(self.optimizer, self.params.lr_exp_decay_rate)
        elif lr_type=='reduce': self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=self.params.lr_mul_decay_rate,
                                                                               patience=self.params.reduce_patiene_epoch, verbose=True,
                                               threshold=0.001, threshold_mode='abs', cooldown=1, min_lr=1e-10, eps=1e-08)
        elif self.params.lr_type=='step': self.scheduler = MultiStepLR(self.optimizer, self.params.steps, self.params.lr_mul_decay_rate)
        elif self.params.lr_type=='clr':
            self.scheduler = CyclicLR(self.optimizer, self.params)
            self.decay_phase_start = False
            self.max_clr_epochs = self.params.clr_max_cycle_epochs * self.params.clr_step_epochs * 2


    def step(self, epoch=None, val_mrr = 0.0):
        if self.params.lr_type == None or self.params.lr_type == 'standard': return
        elif self.params.lr_type == 'decay': self.scheduler.step(epoch - self.max_clr_epochs)
        elif self.params.lr_type == 'reduce': self.scheduler.step(val_mrr, epoch - self.max_clr_epochs)
        elif self.params.lr_type == 'step': self.scheduler.step(epoch - self.max_clr_epochs)
        elif self.params.lr_type == 'clr':
            if epoch < self.max_clr_epochs: self.scheduler.step()
            else:
                self.params.lr_type = self.params.lr_type_after_max_clr_epochs
                self.set_scheduler(self.params.lr_type)


class CyclicLR(object):
    """Sets the learning rate of each parameter group according to
    cyclical learning rate policy (CLR). The policy cycles the learning
    rate between two boundaries with a constant frequency, as detailed in
    the paper `Cyclical Learning Rates for Training Neural Networks`_.
    The distance between the two boundaries can be scaled on a per-iteration
    or per-cycle basis.
    Cyclical learning rate policy changes the learning rate after every batch.
    `batch_step` should be called after a batch has been used for training.
    To resume training, save `last_batch_iteration` and use it to instantiate `CycleLR`.
    This class has three built-in policies, as put forth in the paper:
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each
        cycle iteration.
    This implementation was adapted from the github repo: `bckenstler/CLR`_
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        base_lr (float or list): Initial learning rate which is the
            lower boundary in the cycle for eachparam groups.
            Default: 0.001
        max_lr (float or list): Upper boundaries in the cycle for
            each parameter group. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore
            max_lr may not actually be reached depending on
            scaling function. Default: 0.006
        step_size (int): Number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch. Default: 2000
        mode (str): One of {triangular, triangular2, exp_range}.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
            Default: 'triangular'
        gamma (float): Constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
            Default: 1.0
        scale_fn (function): Custom scaling policy defined by a single
            argument lambda function, where
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored
            Default: None
        scale_mode (str): {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on
            cycle number or cycle iterations (training
            iterations since start of cycle).
            Default: 'cycle'
        last_batch_iteration (int): The index of the last batch. Default: -1
    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> scheduler = torch.optim.CyclicLR(optimizer)
        >>> data_loader = torch.utils.data.DataLoader(...)
        >>> for epoch in range(10):
        >>>     for batch in data_loader:
        >>>         scheduler.batch_step()
        >>>         train_batch(...)
    .. _Cyclical Learning Rates for Training Neural Networks: https://arxiv.org/abs/1506.01186
    .. _bckenstler/CLR: https://github.com/bckenstler/CLR
    """

    def __init__(self, optimizer, params):
        self.params    = params

        self.optimizer = optimizer
        self.base_lrs  = [self.params.lr] * len(optimizer.param_groups)
        self.max_lrs   = [self.params.clr_max_lr] * len(optimizer.param_groups)

        self.step_size = self.params.clr_step_epochs * self.params.steps_per_epoch

        self.mode      = self.params.clr_mode
        self.gamma     = self.params.clr_gamma

        if self.mode not in ['triangular', 'triangular2', 'exp_range']:
            raise ValueError('clr mode is invalid')

        if self.mode   == 'triangular' :
            self.scale_fn   = self._triangular_scale_fn
            self.scale_mode = 'cycle'
        elif self.mode == 'triangular2':
            self.scale_fn   = self._triangular2_scale_fn
            self.scale_mode = 'cycle'
        elif self.mode == 'exp_range'  :
            self.scale_fn   = self._exp_range_scale_fn
            self.scale_mode = 'iterations'

        self.step(0)
        self.last_batch_iteration = -1

    def _triangular_scale_fn(self, x):
        return 1.

    def _triangular2_scale_fn(self, x):
        return 1 / (2. ** (x - 1))

    def _exp_range_scale_fn(self, x):
        return self.gamma**(x)

    def step(self, batch_iteration=None):
        if batch_iteration is None:
            batch_iteration = self.last_batch_iteration + 1
        self.last_batch_iteration = batch_iteration
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


    def get_lr(self):
        step_size = float(self.step_size)
        cycle = np.floor(1 + self.last_batch_iteration / (2 * step_size))
        x = np.abs(self.last_batch_iteration / step_size - 2 * cycle + 1)

        lrs = []
        param_lrs = zip(self.optimizer.param_groups, self.base_lrs, self.max_lrs)
        for param_group, base_lr, max_lr in param_lrs:
            base_height = (max_lr - base_lr) * np.maximum(0, (1 - x))
            if self.scale_mode == 'cycle':
                lr = base_lr + base_height * self.scale_fn(cycle)
            else:
                lr = base_lr + base_height * self.scale_fn(self.last_batch_iteration)
            lrs.append(lr)
        return lrs
