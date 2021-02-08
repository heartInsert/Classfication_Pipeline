from torch.optim.lr_scheduler import MultiStepLR


class MultiSteplr(MultiStepLR):
    def __init__(self, kwargs):
        super(MultiSteplr, self).__init__(**kwargs)
