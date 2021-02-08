from codes.Myloss_function.cross_entropy import My_Crossentropy, FocalLoss
from codes.Myloss_function.SCEloss import SCELoss
from codes.Myloss_function.label_smooth import LabelSmoothCEloss
import torch.nn as nn

loss_dict = {
    'CrossEntropyLoss': nn.CrossEntropyLoss,
    'FocalLoss': FocalLoss,
    "SCEloss": SCELoss,
    "LabelSmoothCEloss": LabelSmoothCEloss,
}


def loss_call(kwargs):
    loss_name = kwargs['loss_name']
    loss_args = kwargs['loss_args']
    assert loss_name in loss_dict.keys()
    loss_fc = loss_dict[loss_name](**loss_args)
    return loss_fc
