from codes.Mydatasets.Mnist import Mnist_dataset, Mnist_dataset_Predict
# from codes.Mydatasets.House_Bargin import HouseBargin_dataset, HouseBargin_dataset_Predict
import torch.utils.data as data
from codes.Mydatasets.Classification import classification_dataset, classification_predict_dataset

dataset_dict = {
    'Mnist_dataset': Mnist_dataset,
    'Mnist_dataset_Predict': Mnist_dataset_Predict,
    # 'HouseBargin_dataset': HouseBargin_dataset,
    # 'HouseBargin_dataset_Predict': HouseBargin_dataset_Predict,
    'classification_dataset': classification_dataset,
    'classification_predict_dataset': classification_predict_dataset,
}


def dataset_call(flag, kwargs):
    dataset_name = kwargs['dataset_name']
    assert dataset_name in dataset_dict.keys()
    dataset = dataset_dict[dataset_name](flag, kwargs)
    return dataset
