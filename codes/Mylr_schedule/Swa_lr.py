from torch.optim.lr_scheduler import _LRScheduler

from torch.optim.swa_utils import SWALR
from codes.Mylr_schedule.MultiStepLRer import MultiSteplr


class My_SWALR(_LRScheduler):
    def __init__(self, kwargs, last_epoch=-1):
        self.count = -1
        self.lr_schedule = MultiSteplr(kwargs)
        self.swa_scheduler = SWALR(kwargs["optimizer"], swa_lr=1e-5, anneal_epochs=8)
        super(My_SWALR, self).__init__(kwargs["optimizer"], last_epoch)

    def step(self, epoch=None):
        if 0 <= self.count <= 12:
            self.lr_schedule.step()
        elif self.count > 12:
            self.swa_scheduler.step()
        else:
            pass
        self.count = self.count + 1
