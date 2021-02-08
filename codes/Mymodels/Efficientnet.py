import torch
from efficientnet_pytorch import EfficientNet


class efficientnet(torch.nn.Module):
    def __init__(self, kwargs):
        super(efficientnet, self).__init__()
        self.num_class = kwargs['num_class']
        if kwargs['pretrained'] is True:
            self.model = EfficientNet.from_pretrained(model_name=kwargs['model_version'],
                                                      num_classes=kwargs['num_class'])
        else:
            self.model = EfficientNet.from_name(model_name=kwargs['model_version'], num_classes=kwargs['num_class'])

    def forward(self, data):
        if len(data) == 1:
            data = data[0]
        out = self.model(data)
        return out

# params_dict = {
#         # Coefficients:   width,depth,res,dropout
#         'efficientnet-b0': (1.0, 1.0, 224, 0.2),
#         'efficientnet-b1': (1.0, 1.1, 240, 0.2),
#         'efficientnet-b2': (1.1, 1.2, 260, 0.3),
#         'efficientnet-b3': (1.2, 1.4, 300, 0.3),
#         'efficientnet-b4': (1.4, 1.8, 380, 0.4),
#         'efficientnet-b5': (1.6, 2.2, 456, 0.4),
#         'efficientnet-b6': (1.8, 2.6, 528, 0.5),
#         'efficientnet-b7': (2.0, 3.1, 600, 0.5),
#         'efficientnet-b8': (2.2, 3.6, 672, 0.5),
#         'efficientnet-l2': (4.3, 5.3, 800, 0.5),
#     }
