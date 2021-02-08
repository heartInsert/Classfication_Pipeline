from torchvision.models import resnet50, resnext50_32x4d
import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F


class Resnet50(torch.nn.Module):
    def __init__(self, kwargs):
        super(Resnet50, self).__init__()
        self.num_class = kwargs['num_class']
        self.layer1 = resnet50(pretrained=kwargs['pretrained'])
        self.layer2 = torch.nn.Linear(self.layer1.fc.in_features, self.num_class)
        self.layer1.fc = self.layer2

    def forward(self, data):
        if len(data) == 1:
            data = data[0]
        out = self.layer1(data)
        return out


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)


class GeM(torch.nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        # self.p = Parameter(torch.ones(1) * p)
        self.p = p
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p) + ', ' + 'eps=' + str(
            self.eps) + ')'

#
# model = se_resnet50(num_classes=1000, pretrained='imagenet')
# model.avg_pool = GeM()


class Resnext50_32x4d(torch.nn.Module):
    def __init__(self, kwargs):
        super(Resnext50_32x4d, self).__init__()
        self.num_class = kwargs['num_class']
        self.layer1 = resnext50_32x4d(pretrained=kwargs['pretrained'])
        self.layer2 = torch.nn.Linear(self.layer1.fc.in_features, self.num_class)
        self.layer1.fc = self.layer2

    def forward(self, data):
        if isinstance(data, list):
            data = data[0]
        out = self.layer1(data)
        return out
