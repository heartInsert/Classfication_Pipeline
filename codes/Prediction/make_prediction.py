import os, datetime
from mmcv import Config
from codes.Mydatasets import dataset_call
from codes.Mymodels import model_call
from torch.utils.data import DataLoader
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd


def Get_predict_dataloader(cfg):
    dataset_val = dataset_call(flag='predict', kwargs=cfg.dataset_entity_predict)
    return DataLoader(dataset_val, batch_size=50, shuffle=False, num_workers=6, drop_last=False)


def prediction(model, dataloader, predict_way) -> np.array:
    with torch.no_grad():
        model.cuda()
        model.eval()
        result = []
        for index, img in enumerate(dataloader):
            img = img.cuda()
            pred = model(img)
            if predict_way == 'cross_entropy':
                pred = torch.softmax(pred, dim=1)
            if predict_way == 'sigmoid':
                pred = pred.sigmoid()
            pred = pred.detach().cpu().numpy()
            result.extend(pred)
            del pred
            pass
    return np.stack(result)


def make_prediction(model_folder_path, model, dataloader, predict_way):
    model_predict = []
    for fold_n in os.listdir(model_folder_path):
        if fold_n.startswith('fold_'):
            fold_n_path = os.path.join(model_folder_path, fold_n)
            prediction_path = os.path.join(fold_n_path, 'prediction.csv')
            checkpoint_path = os.path.join(fold_n_path, 'checkpoints', 'best.ckpt')
            if not os.path.exists(prediction_path):
                model.load_state_dict(torch.load(checkpoint_path)['state_dict'], strict=True)
                result = prediction(model, dataloader, predict_way)
                csv = pd.DataFrame(data=result, columns=['key_{}'.format(i) for i in range(result.shape[1])])
                csv.to_csv(prediction_path, index=0)
                del result
                gc.collect()
            # read csv and make final  prediction
            result_csv = pd.read_csv(prediction_path)
            model_predict.append(result_csv.values)
    model_predict = np.stack(model_predict)
    return model_predict


import gc


def main():
    project_path = '/home/xjz/Desktop/Coding/PycharmProjects/competition/kaggle/cassava_leaf_disease_classification'
    model_folders = ['2021_0210_10_0306_efficientnet']
    data_csv = pd.read_csv('/home/xjz/Desktop/Coding/DL_Data/cassava_leaf_disease_classification/sample_submission.csv')

    model_folder_paths = [os.path.join(project_path, 'model_weights', folder) for folder in model_folders]
    config_paths = []
    for model_folder_path in model_folder_paths:
        files = os.listdir(model_folder_path)
        for file in files:
            if file.endswith('.py'):
                config_paths.append(os.path.join(model_folder_path, file))
    predict_way = 'cross_entropy'
    assert predict_way in ['cross_entropy', 'sigmoid']
    assert len(config_paths) == len(model_folders)
    final_predicts = []
    for config_path, model_folder_path in tqdm(zip(config_paths, model_folder_paths), total=len(model_folders)):
        cfg = Config.fromfile(config_path)
        model = model_call(cfg['model_entity'])
        cfg.dataset_entity_predict['predict_csv'] = data_csv
        dataloader = Get_predict_dataloader(cfg)

        num_TTA = cfg.dataset_entity_predict.get('num_TTA', 1)
        for TTA in range(num_TTA):
            model_predict = make_prediction(model_folder_path, model, dataloader, predict_way)

            final_predicts.append(model_predict)

        del model, dataloader, model_predict
        gc.collect()
    final_predicts = np.concatenate(final_predicts).mean(axis=0)
    predictions = np.argmax(final_predicts, axis=1)
    data_csv['label'] = predictions
    result_csv = data_csv
    datetime_prefix = datetime.datetime.now().strftime("%Y_%m%d_%H_%M%S")
    result_csv.to_csv(os.path.join(project_path, 'Results/{}_result.tsv'.format(datetime_prefix)), index=None)

    print('Done')


if __name__ == "__main__":
    main()
