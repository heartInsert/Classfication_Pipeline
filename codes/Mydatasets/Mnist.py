import torch.utils.data as data
from torchvision.datasets import MNIST
from torch.utils.data import Subset
import json


class Mnist_dataset(data.Dataset):
    def __init__(self, flag: str, kwargs):
        assert flag in ['train', 'test']
        self.kwargs = kwargs
        if flag == 'train':
            index = kwargs['train_index']
            transform = self.kwargs['transform_train']
        else:
            index = kwargs['test_index']
            transform = self.kwargs['transform_test']

        self.mnist = MNIST(self.kwargs['root'], True, download=self.kwargs['download'],
                           transform=transform)
        self.dataset = Subset(self.mnist, index)

    def __getitem__(self, item):
        return self.dataset[item]

    def __len__(self):
        return len(self.dataset)


# rewrite the datast when  you  doing  prediction
class Mnist_dataset_Predict(data.Dataset):
    def __init__(self, flag: str, kwargs):
        assert flag in ['val']
        self.kwargs = kwargs
        transform = self.kwargs['transform_test']
        self.mnist = MNIST(self.kwargs['root'], False, download=self.kwargs['download'],
                           transform=transform)

    def __getitem__(self, item):
        return self.mnist[item]

    def __len__(self):
        return len(self.mnist)
