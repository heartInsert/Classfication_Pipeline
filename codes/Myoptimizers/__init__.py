from codes.Myoptimizers.Adam import MyAdam, MyAdamW
from codes.Myoptimizers.Ranger import ranger

from torch.optim import Adam
from transformers import AdamW

optim_dict = {
    'Adam': Adam,
    'AdamW': AdamW,
    'ranger': ranger
}


def optimizer_call(params, kwargs):
    optimizer_name = kwargs['optimizer_name']
    optimizer_args = kwargs['optimizer_args']
    assert optimizer_name in optim_dict.keys()
    optimizer = optim_dict[optimizer_name](**{'params': params, **optimizer_args})
    return optimizer
