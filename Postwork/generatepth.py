import torch
from mmcv import Config
from codes.test_fun import LitMNIST
from codes.Mymodels import Resnet50

if __name__=="__main__":
    config_path = '/home/xjz/Desktop/Coding/PycharmProjects/competition/kaggle/cassava_leaf_disease_classification/configs/Resnet50.py'
    checkpoint_path = '/model_weights/2021_0107_23_5307_Resnet50/fold_1/checkpoints/best.ckpt'
    cfg = Config.fromfile(config_path)
    # get model class  and  dataset
    # model = LitMNIST(cfg)
    # model.load_state_dict(torch.load(checkpoint_path)['state_dict'], strict=True)
    # torch.save({"state_dict":model.model_layer.state_dict()},
    #            '/home/xjz/Desktop/Coding/PycharmProjects/competition/kaggle/cassava_leaf_disease_classification/test.ckpt')

    model = Resnet50({'num_class': 5, "pretrained": True})
    state = \
        torch.load(
            '/home/xjz/Desktop/Coding/PycharmProjects/competition/kaggle/cassava_leaf_disease_classification/test.ckpt')['state_dict']
    model.load_state_dict(state, strict=True)
    print()
