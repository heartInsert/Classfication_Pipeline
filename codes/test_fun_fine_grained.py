import torch
from torch.nn import functional as F
from torch.optim.swa_utils import AveragedModel, SWALR
import argparse
import copy
from mmcv import Config
from pytorch_lightning import loggers
import datetime, shutil, os
from pytorch_lightning.core.lightning import LightningModule
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, seed_everything
import pytorch_lightning as pl
from codes.Mymodels import model_call
from codes.Mydatasets import dataset_call
from codes.Myoptimizers import optimizer_call
from codes.Mylr_schedule import lrschdule_call
from codes.Mytokenzier import tokenizer_call
from pytorch_lightning.metrics.metric import Metric
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import CosineAnnealingLR


class Loss_Metric(Metric):
    def __init__(self, compute_on_step: bool = True, dist_sync_on_step=False, process_group=None):
        super().__init__(compute_on_step=compute_on_step, dist_sync_on_step=dist_sync_on_step,
                         process_group=process_group)

        self.add_state("loss", default=torch.tensor(0).float(), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, loss: torch.Tensor, num: int):
        self.loss += loss * num
        self.total += num

    def compute(self):
        return self.loss.float() / self.total


class Accuracy_Metric(Metric):
    def __init__(self, compute_on_step: bool = True, dist_sync_on_step=False, process_group=None):
        super().__init__(compute_on_step=compute_on_step, dist_sync_on_step=dist_sync_on_step,
                         process_group=process_group)

        self.add_state("num_acc", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(1e-4), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds = preds.argmax(dim=1)
        self.num_acc += (preds == target).sum()
        self.total += len(target)

    def compute(self):
        accuracy = self.num_acc / self.total
        return accuracy * 100


from codes.Myloss_function import loss_call


class LitMNIST(LightningModule):
    def __init__(self, kwargs):
        super().__init__()
        self.dataset_entity = kwargs['dataset_entity']
        self.dataloader_entity = kwargs['dataloader_entity']
        self.optimizer_entity = kwargs['optimzier_entity']
        self.lrschdule_entity = kwargs['lrschdule_entity']
        self.loss_fc = loss_call(kwargs['loss_fc_entity'])
        self.model_layer = model_call(kwargs['model_entity'])
        self.train_loss = Loss_Metric(compute_on_step=False)
        self.val_loss = Loss_Metric(compute_on_step=False)
        self.train_Accuracy = Accuracy_Metric(compute_on_step=False)
        self.val_Accuracy = Accuracy_Metric(compute_on_step=False)

    def train_dataloader(self):
        dataset_train = dataset_call(flag='train', kwargs=self.dataset_entity)
        self.train_loder = DataLoader(dataset_train, batch_size=self.dataloader_entity['batch_size'],
                                      shuffle=self.dataloader_entity['shuffle'],
                                      num_workers=self.dataloader_entity['num_wokers'],
                                      drop_last=self.dataloader_entity['drop_last']
                                      )
        return self.train_loder

    def val_dataloader(self):
        dataset_val = dataset_call(flag='test', kwargs=self.dataset_entity)
        return DataLoader(dataset_val, batch_size=self.dataloader_entity['batch_size'],
                          shuffle=False, num_workers=self.dataloader_entity['num_wokers'],
                          drop_last=False
                          )

    def configure_optimizers(self):
        optimizer = optimizer_call(params=self.parameters(), kwargs=self.optimizer_entity)

        lr_schedule = lrschdule_call(optimizer, self.lrschdule_entity)

        return [optimizer], [lr_schedule]

    def forward(self, data):
        x = self.model_layer(data)
        return x

    def step(self, batch):
        data, label = batch[:-1], batch[-1]
        logits = self(data)
        # compute loss
        loss = self.loss_fc(logits, label)
        return loss, logits, label

    def training_step(self, batch, batch_idx):
        # input_ids, token_type_ids, attention_mask, y = batch
        # data_dict = {'input_ids': input_ids, 'token_type_ids': token_type_ids, 'attention_mask': attention_mask}
        # logits = self(data_dict)
        # # compute loss
        # loss = self.loss_fc(logits, y)
        loss, logits, y = self.step(batch)
        # add log
        self.log('train_loss_step', loss, on_step=True, on_epoch=False, prog_bar=False, logger=True)
        self.train_loss(loss, len(y))
        self.train_Accuracy(logits, y)
        return loss

    def training_epoch_end(self, outputs):
        self.log('train_loss_epoch', self.train_loss.compute())
        self.log('train_Accuracy_epoch', self.train_Accuracy.compute())
        self.log('lr', self.optimizers().param_groups[0]['lr'])
        self.train_loss.reset()
        self.train_Accuracy.reset()

    def validation_step(self, batch, batch_idx):
        # input_ids, token_type_ids, attention_mask, y = batch
        # data_dict = {'input_ids': input_ids, 'token_type_ids': token_type_ids, 'attention_mask': attention_mask}
        # logits = self(data_dict)
        # # compute loss
        # loss = self.loss_fc(logits, y)
        loss, logits, y = self.step(batch)
        # add log
        self.val_loss(loss, len(y))
        self.val_Accuracy(logits, y)
        return loss

    def validation_epoch_end(self, outputs):
        self.log('val_loss_epoch', self.val_loss.compute())
        self.log('val_Accuracy_epoch', self.val_Accuracy.compute())
        self.val_loss.reset()
        self.val_Accuracy.reset()


from torchvision.datasets import MNIST
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
import scikitplot as skplt
import matplotlib.pyplot as plt
import copy


def Get_fold_splits(fold, kwargs):
    csv = pd.read_csv(kwargs['data_csv_path'])
    labels = csv['label'].values
    X = csv.values
    y = labels
    skl = StratifiedKFold(n_splits=fold, shuffle=True, random_state=2020)
    # for train, test in skl.split(X,y):
    #     print('Train: %s | test: %s' % (train, test),'\n')

    index_generator = skl.split(X, y)
    return csv, index_generator


def inferrence(model, dataloader):
    pred = []
    target = []
    model.cuda()
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            data, label = batch[:-1], batch[-1]
            data = [d.cuda() for d in data]
            logits = model(data)
            pred.extend(logits.argmax(1).detach().cpu().numpy().tolist())
            target.extend(label.numpy().tolist())
    del model, dataloader
    return pred, target


def plot_confusion_matrix(pred, target, normalize, save_path):
    if plt.isinteractive():
        plt.ioff()
    fig, axes = plt.subplots(1, 1, figsize=None)
    plot = skplt.metrics.plot_confusion_matrix(target, pred, normalize=normalize, ax=axes)
    fig.savefig(save_path, dpi=300)


if __name__ == "__main__":

    config_path = r'/home/xjz/Desktop/Coding/PycharmProjects/competition/kaggle/cassava_leaf_disease_classification/configs/Resnet50Ranger_0_1.py'
    seed_everything(2020)
    # get config
    cfg = Config.fromfile(config_path)
    # get  stuff
    model_name = cfg.model_entity['model_name']
    datetime_prefix = datetime.datetime.now().strftime("%Y_%m%d_%H_%M%S")
    experiment_name = datetime_prefix + '_' + model_name
    weight_folder = os.path.join(cfg.logger_entity['weight_savepath'], experiment_name)
    #  copy  config
    os.makedirs(weight_folder)
    shutil.copy(config_path, os.path.join(weight_folder, os.path.basename(config_path)))
    # get dataset length
    csv, index_generator = Get_fold_splits(cfg.kfold, cfg.dataset_entity)
    pred_list, target_list = [], []
    # for  loop  code
    for n_fold, (train_index, test_index) in enumerate(index_generator, start=1):
        #  testing
        if n_fold <= 2:
            if cfg.Train_mode is not True:
                train_index = train_index[:200]
                test_index = test_index[:200]
            train_csv = csv.iloc[train_index]
            test_csv = csv.iloc[test_index]
            # binary classifier
            train_csv = train_csv[train_csv['label'].isin([0, 4])]
            test_csv = test_csv[test_csv['label'].isin([0, 4])]
            train_csv.loc[train_csv['label'] == 4, 'label'] = 1
            test_csv.loc[test_csv['label'] == 4, 'label'] = 1
            cfg['dataset_entity']['train_csv'] = train_csv
            cfg['dataset_entity']['test_csv'] = test_csv

            model = LitMNIST(cfg)
            checkpoint_callback = pl.callbacks.ModelCheckpoint(**cfg.logger_entity.ModelCheckpoint)
            tb_logger = loggers.TensorBoardLogger(save_dir=cfg.logger_entity['weight_savepath'], name=experiment_name,
                                                  version='fold_{}'.format(n_fold))
            cfg.trainer_entity['checkpoint_callback'] = checkpoint_callback
            cfg.trainer_entity['logger'] = tb_logger
            trainer = Trainer(**cfg.trainer_entity)
            trainer.fit(model)
            if cfg.Train_mode is not None:
                model_pth_name = os.path.basename(checkpoint_callback.best_model_path)
                best_model_path = checkpoint_callback.best_model_path.replace(model_pth_name, 'best.ckpt')
                model.load_state_dict(torch.load(checkpoint_callback.best_model_path)['state_dict'], strict=True)
                torch.save({"state_dict": model.model_layer.state_dict()}, best_model_path)
                # delete original
                os.remove(checkpoint_callback.best_model_path)
                # confusion matrix
                pred, target = inferrence(model.model_layer, copy.deepcopy(trainer.val_dataloaders[0]))

                plot_confusion_matrix(pred, target, normalize=False,
                                      save_path=os.path.join(os.path.dirname(checkpoint_callback.dirpath),
                                                             'confusion_matrix.jpg'))
                plot_confusion_matrix(pred, target, normalize=True,
                                      save_path=os.path.join(os.path.dirname(checkpoint_callback.dirpath),
                                                             'confusion_matrix_normalize.jpg'))
                # add pred and target to global
                pred_list.extend(pred)
                target_list.extend(target)

            del model, trainer, checkpoint_callback
    plot_confusion_matrix(pred_list, target_list, normalize=False,
                          save_path=os.path.join(weight_folder, 'global_confusion_matrix.jpg'))
    plot_confusion_matrix(pred_list, target_list, normalize=True,
                          save_path=os.path.join(weight_folder, 'global_confusion_matrix_normalize.jpg'))
