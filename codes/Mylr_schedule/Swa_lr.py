from torch.optim.lr_scheduler import _LRScheduler

from codes.Mylr_schedule.MultiStepLRer import MultiSteplr
from codes.Mylr_schedule.transformer_schedule import get_polynomial_decay_schedule_with_warmup


class My_SWALR:
    def __init__(self, lr_schedule, SWA_epoch, last_epoch, num_training_steps):
        self.lr_schedule = lr_schedule
        self.SWA_epoch = SWA_epoch
        self.last_epoch = last_epoch
        self.max_epoch = num_training_steps

    def step(self, epoch=None):
        self.lr_schedule.step(epoch)
        self.last_epoch = self.last_epoch + 1
        if self.last_epoch >= self.SWA_epoch:
            pass

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return self.lr_schedule.state_dict()

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.lr_schedule.load_state_dict(state_dict)

    def get_last_lr(self):
        """ Return last computed learning rate by current scheduler.
        """
        return self.lr_schedule.get_last_lr()

    def get_lr(self):
        # Compute learning rate using chainable form of the scheduler
        return self.lr_schedule.get_lr()

    def print_lr(self, is_verbose, group, lr, epoch=None):
        """Display the current learning rate.
        """
        self.lr_schedule.print_lr(is_verbose, group, lr, epoch)
