from codes.Mymixway.FMix import FMix
from codes.Mymixway.Cutmix import Cutmix


class Normal_training:
    def __init__(self, loss_fc):
        self.loss_fc = loss_fc

    def __call__(self, model, data, label, is_training):
        logits = model(data)
        loss = self.loss_fc(logits, label)
        return logits, loss


way_dict = {
    'Normal_training': Normal_training,
    'FMix': FMix,
    'Cutmix': Cutmix,
}


def training_call(kwargs):
    way_name = kwargs['training_way_name']
    way_args = kwargs['training_way_args']
    way = way_dict[way_name](**way_args)
    return way
