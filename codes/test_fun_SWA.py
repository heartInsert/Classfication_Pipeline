import torch
from torch.nn import functional as F
from torch.optim.swa_utils import AveragedModel, SWALR
import argparse
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
        self.train_acc = pl.metrics.Accuracy(compute_on_step=False)
        self.val_acc = pl.metrics.Accuracy(compute_on_step=False)

        self.swa_model = AveragedModel(self.model_layer)

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
                          drop_last=self.dataloader_entity['drop_last']
                          )

    def on_save_checkpoint(self, checkpoint) -> None:
        checkpoint['state_dict'] = self.swa_model.state_dict()
        pass

    def configure_optimizers(self):
        optimizer = optimizer_call(params=self.parameters(), kwargs=self.optimizer_entity)

        lr_schedule = lrschdule_call(optimizer, self.lrschdule_entity)

        return [optimizer], [lr_schedule]

    def forward(self, x):
        x = self.model_layer(x)
        return x

    def SWA(self, x):
        x = self.swa_model(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        # compute loss
        loss = self.loss_fc(logits, y)
        # add log
        self.log('train_loss_step', loss, on_step=True, on_epoch=False, prog_bar=False, logger=True)
        self.train_loss(loss, len(y))
        self.train_acc(logits, y)
        return loss

    def training_epoch_end(self, outputs):
        self.log('train_loss_epoch', self.train_loss.compute())
        self.log('train_acc_epoch', self.train_acc.compute())
        self.log('lr', self.optimizers().param_groups[0]['lr'])
        self.train_loss.reset()
        self.train_acc.reset()

        # swa
        self.swa_model.update_parameters(self.model_layer)

        #

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        # compute loss
        loss = self.loss_fc(logits, y)
        # add log
        self.val_loss(loss, len(y))
        self.val_acc(logits, y)
        return loss

    def validation_epoch_end(self, outputs):
        self.log('val_loss_epoch', self.val_loss.compute())
        self.log('valid_acc_epoch', self.val_acc.compute())
        self.val_loss.reset()
        self.val_acc.reset()


from torchvision.datasets import MNIST
from sklearn.model_selection import StratifiedKFold, KFold
import numpy as np


def Get_fold_splits(kwargs) -> list:
    dataset = MNIST(root=kwargs['root'], train=True, download=False)
    skl = StratifiedKFold(n_splits=5, shuffle=True, random_state=2020)
    labels = dataset.train_labels.cpu().numpy()
    X = np.ones((len(labels)), )
    y = labels
    index_generator = skl.split(X, y)
    return index_generator


if __name__ == "__main__":

    # path = '/home/xjz/Desktop/Coding/PycharmProjects/Anything/Pytorch_lighting/model_weights/2020_1021_22_4233_resnet50/fold_1/checkpoints/epoch=3.ckpt'

    config_path = 'configs/BERT-wwm.py'
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
    index_generator = Get_fold_splits(cfg.dataset_entity)
    # for  loop  code
    for n_fold, (train_index, test_index) in enumerate(index_generator, start=1):
        cfg['dataset_entity']['train_index'] = train_index.tolist()
        cfg['dataset_entity']['test_index'] = test_index.tolist()
        model = LitMNIST(cfg)

        checkpoint_callback = pl.callbacks.ModelCheckpoint(**cfg.logger_entity.ModelCheckpoint)

        tb_logger = loggers.TensorBoardLogger(save_dir=cfg.logger_entity['weight_savepath'], name=experiment_name,
                                              version='fold_{}'.format(n_fold))

        cfg.trainer_entity['checkpoint_callback'] = checkpoint_callback
        cfg.trainer_entity['logger'] = tb_logger
        trainer = Trainer(**cfg.trainer_entity)
        trainer.fit(model)
        # swa
        torch.optim.swa_utils.update_bn(model.train_loder, model.swa_model)
        trainer.save_checkpoint(checkpoint_callback.best_model_path)
        if cfg.Train_mode is True:
            best_modelpth = checkpoint_callback.best_model_path.replace(
                os.path.basename(checkpoint_callback.best_model_path), 'best.ckpt')
            os.rename(checkpoint_callback.best_model_path, best_modelpth)
        del model, trainer, checkpoint_callback
        pass
        # temp = torch.load(r'lightning_logs\version_10\checkpoints\best.ckpt')
        # model.load_state_dict(temp['state_dict'])
