from codes.Mymodels.Resnet import Resnet50, Resnext50_32x4d
from codes.Mymodels.Efficientnet import efficientnet

# from codes.Mymodels.Bert_chinese import Bert_Chinese_Model

model_dict = {
    'Resnet50': Resnet50,
    'efficientnet': efficientnet,
    'Resnext50_32x4d': Resnext50_32x4d,
}


def model_call(kwargs):
    model_name = kwargs['model_name']
    model = model_dict[model_name](kwargs)
    return model
