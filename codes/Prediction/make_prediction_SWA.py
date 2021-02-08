import os, datetime
from mmcv import Config
from codes.test_fun_SWA import LitMNIST
from codes.Mydatasets import dataset_call
from torch.utils.data import DataLoader
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
from torchvision.datasets import MNIST


def Get_val_dataloader(cfg):
    dataset_val = dataset_call(flag='val', kwargs=cfg.dataset_entity)
    return DataLoader(dataset_val, batch_size=cfg.dataloader_entity['batch_size'] * 2,
                      shuffle=False, num_workers=0,
                      drop_last=False
                      )


def prediction(model, dataloader, predict_way) -> np.array:
    model.eval()
    result = []
    for index, (x, y) in enumerate(dataloader):
        pred = model.SWA(x)
        if predict_way == 'cross_entropy':
            pred = torch.softmax(pred, dim=1)
        if predict_way == 'sigmoid':
            pred = pred.sigmoid()
        pred = pred.detach().cpu().numpy()
        result.extend(pred)
    return np.stack(result)


def make_prediction(model_path, model_folder, model, dataloader, predict_way):
    model_predict = []
    folder_path = os.path.join(model_path, model_folder)
    for fold_n in os.listdir(folder_path):
        if fold_n.startswith('fold_'):
            fold_n_path = os.path.join(folder_path, fold_n)
            prediction_path = os.path.join(fold_n_path, 'prediction.csv')
            checkpoint_path = os.path.join(fold_n_path, 'checkpoints', 'best.ckpt')
            if not os.path.exists(prediction_path):
                # make prediction
                model.swa_model.load_state_dict(torch.load(checkpoint_path)['state_dict'], strict=True)
                result = prediction(model, dataloader, predict_way)
                csv = pd.DataFrame(data=result, columns=['key_{}'.format(i) for i in range(result.shape[1])])
                csv.to_csv(prediction_path, index=0)

            # read csv and make final  prediction
            result_csv = pd.read_csv(prediction_path)
            model_predict.append(result_csv.values)
    model_predict = np.stack(model_predict)
    return model_predict


def main():
    config_paths = ['/home/xjz/Desktop/Coding/PycharmProjects/Anything/Pytorch_lighting/configs/BERT-wwm.py']
    model_folders = ['2020_1022_23_0654_resnet50']
    predict_way = 'cross_entropy'
    assert predict_way in ['cross_entropy', 'sigmoid']
    assert len(config_paths) == len(model_folders)
    final_predicts = []
    for config_path, model_folder in tqdm(zip(config_paths, model_folders), total=len(model_folders)):
        cfg = Config.fromfile(config_path)
        # get model class  and  dataset
        model = LitMNIST(cfg)
        dataloader = Get_val_dataloader(cfg)
        #
        model_path = cfg.logger_entity['weight_savepath']
        model_predict = make_prediction(model_path, model_folder, model, dataloader, predict_way)
        final_predicts.append(model_predict)
    final_predicts = np.concatenate(final_predicts).mean(axis=0)
    predictions = np.argmax(final_predicts, axis=1)
    csv = pd.DataFrame(data=predictions, columns=['prediction'])
    datetime_prefix = datetime.datetime.now().strftime("%Y_%m%d_%H_%M%S")
    csv.to_csv('/home/xjz/Desktop/Coding/PycharmProjects/Anything/Pytorch_lighting/Results/{}_result.csv'.format(
        datetime_prefix), index=0)
    dataset = MNIST('/home/xjz/Desktop/Coding/PycharmProjects/Anything/Pytorch_lighting', False,
                    download=False, transform=None)
    correct = (dataset.targets.numpy() == predictions).sum()
    accuracy = float(correct) / len(dataset)
    print('accuracy is {}'.format(accuracy))


if __name__ == "__main__":
    main()
