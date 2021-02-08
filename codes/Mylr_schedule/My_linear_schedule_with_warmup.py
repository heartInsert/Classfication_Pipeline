from .transformer_schedule import get_linear_schedule_with_warmup
from torch.optim.lr_scheduler import LambdaLR

def linear_schedule_with_warmup(kwargs):
    return get_linear_schedule_with_warmup(**kwargs)


