from codes.Mylr_schedule.MultiStepLRer import MultiSteplr
from codes.Mylr_schedule.Swa_lr import My_SWALR
from codes.Mylr_schedule.My_linear_schedule_with_warmup import linear_schedule_with_warmup
from .transformer_schedule import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, \
    get_polynomial_decay_schedule_with_warmup

lrschdule_dict = {
    'MultiStepLR': MultiSteplr,
    'My_SWALR': My_SWALR,
    'Linear_schedule_with_warmup': get_linear_schedule_with_warmup,
    'cosine_schedule_with_warmup': get_cosine_schedule_with_warmup,
    'polynomial_decay_schedule_with_warmup': get_polynomial_decay_schedule_with_warmup,
}


def lrschdule_call(optimizer, kwargs):
    lrschdule_name = kwargs['lrschdule_name']
    lrschdule_args = kwargs['lrschdule_args']
    SWA_args = kwargs['SWA']
    assert lrschdule_name in lrschdule_dict.keys()
    lrschduler = lrschdule_dict[lrschdule_name](**{'optimizer': optimizer, **lrschdule_args})
    # lrschduler = My_SWALR(lrschduler)
    return lrschduler
