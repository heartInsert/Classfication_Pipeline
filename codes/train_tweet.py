import pytorch_lightning as pl
from models import my_Roberta
import torch
from transformers import AdamW
from torch.utils.data import DataLoader
from dataset import TweetDataset
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
from transformers import RobertaTokenizerFast, RobertaConfig
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import os
from transformers import CONFIG_NAME


from torch.utils.data import Dataset
import numpy as np
import torch


class TweetDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, max_len=120):
        self.df = df
        self.max_len = max_len
        self.labeled = 'selected_text' in df  # 判断是train 还是test的df
        # self.tokenizer = tokenizers.ByteLevelBPETokenizer(
        #     vocab_file='../input/roberta-base/vocab.json',
        #     merges_file='../input/roberta-base/merges.txt',
        #     lowercase=True,
        #     add_prefix_space=True)
        self.tokenizer = tokenizer
        self.ids_all, self.masks_all, self.tweets_all, self.offsets_all = [], [], [], []
        self.start_idx_all, self.end_idx_all = [], []
        self.tokenizing()

    def tokenizing(self):
        for index, row in self.df.iterrows():
            ids, masks, tweet, offsets = self.get_input_data(row)
            self.ids_all.append(ids)
            self.masks_all.append(masks)
            self.tweets_all.append(tweet)
            self.offsets_all.append(offsets)
            if self.labeled:
                start_idx, end_idx = self.get_target_idx(row, tweet, offsets)
                self.start_idx_all.append(start_idx)
                self.end_idx_all.append(end_idx)
        pass

    def __getitem__(self, index):
        data = {}
        data['ids'] = self.ids_all[index]
        data['masks'] = self.masks_all[index]
        data['tweet'] = self.tweets_all[index]
        data['offsets'] = self.offsets_all[index]
        if self.labeled:
            data['start_idx'] = self.start_idx_all[index]
            data['end_idx'] = self.end_idx_all[index]
        return data

    def __len__(self):
        return len(self.df)

    def get_input_data(self, row):
        tweet = " " + " ".join(row.text.lower().split())
        encoding = self.tokenizer.encode_plus(tweet)[0]
        sentiment_id = self.tokenizer.encode_plus(row.sentiment)[0].ids
        ids = [0] + sentiment_id + [2, 2] + encoding.ids + [2]
        offsets = [(0, 0)] * 4 + encoding.offsets + [(0, 0)]
        pad_len = self.max_len - len(ids)
        if pad_len > 0:
            ids += [1] * pad_len
            offsets += [(0, 0)] * pad_len
        ids = torch.tensor(ids)
        masks = torch.where(ids != 1, torch.tensor(1), torch.tensor(0))
        offsets = torch.tensor(offsets)
        return ids, masks, tweet, offsets

    def get_target_idx(self, row, tweet, offsets):
        selected_text = " " + " ".join(row.selected_text.lower().split())
        len_st = len(selected_text) - 1
        idx0 = None
        idx1 = None
        for ind in (i for i, e in enumerate(tweet) if e == selected_text[1]):
            if " " + tweet[ind: ind + len_st] == selected_text:
                idx0 = ind
                idx1 = ind + len_st - 1
                break
        char_targets = [0] * len(tweet)
        if idx0 != None and idx1 != None:
            for ct in range(idx0, idx1 + 1):
                char_targets[ct] = 1
        target_idx = []
        for j, (offset1, offset2) in enumerate(offsets):
            if sum(char_targets[offset1: offset2]) > 0:
                target_idx.append(j)
        start_idx = target_idx[0]
        end_idx = target_idx[-1]
        return start_idx, end_idx



# 经常改的参数需要外置，不需要经常改的参数可以放在系统内

class lightModel(pl.LightningModule):
    def __init__(self, model, tokenizer, df_dict: dict, max_length: int):
        # 一个封闭的控制流程
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.train_df = df_dict['train']
        self.test_df = df_dict['test']
        self.max_length = max_length
        self.loss_fun = nn.CrossEntropyLoss()

    def configure_optimizers(self):  # lr scheduler ?
        optimizer = self.get_optimizer(self.model, lr=1e-3, weight_decay=1e-3)
        lr_schedu = MultiStepLR(optimizer, milestones=[2, 4], gamma=0.1)
        return [optimizer], [lr_schedu]

    def train_dataloader(self):
        train_dataset = TweetDataset(self.train_df, self.tokenizer, self.max_length)
        return DataLoader(train_dataset, shuffle=True, batch_size=30, num_workers=6)

    def val_dataloader(self):
        test_dataset = TweetDataset(self.test_df, self.tokenizer, self.max_length)
        return DataLoader(test_dataset, shuffle=False, batch_size=30, num_workers=6)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.compute_loss(logits, y)
        # add logging
        tensorboard_logs = {'loss': loss, 'xjz': loss * 2}
        return {'loss': loss, 'log': tensorboard_logs}

    def forward(self, x):
        y = self.model(x)
        return y

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.compute_loss(logits, y)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def compute_loss(self, x, y):
        loss = self.loss_fun(x, y)
        return loss

    def get_optimizer(self, model, lr, weight_decay):
        background_param_id = list(map(id, model.transformer.parameters()))
        optimizer_grouped_parameters = [
            # backgroud transformer use  0.1 * weight_decay   0.1* lr
            {'params': [p for p in model.parameters() if id(p) in background_param_id],
             'weight_decay': 0.1 * weight_decay, 'lr': 0.1 * lr},
            {'params': [p for p in model.parameters() if id(p) not in background_param_id],
             'weight_decay': weight_decay, 'lr': lr}
        ]
        return AdamW(optimizer_grouped_parameters)


from pytorch_lightning.callbacks import Callback


class entity():
    def __init__(self, model_class, config_class, tokenizer_class, model_name):
        self.model_class, self.config_class, self.tokenizer_class, self.model_name = model_class, config_class, tokenizer_class, model_name


def get_model(model_class, path_or_modelname):
    model = model_class(path_or_modelname)
    return model


def get_tokenizer(tokenizer_class, dir_or_modelname):
    tokenizer = tokenizer_class.from_pretrained(dir_or_modelname)
    return tokenizer


def model_tokenizer(model_entities, model_name):
    model_entity = model_entities[model_name]
    model = get_model(model_entity.model_class, model_entity.model_name)
    tokenzier = get_tokenizer(model_entity.tokenizer_class, model_entity.model_name)
    return model, tokenzier


def save_model_config(model_to_save, checkpoint_fold):
    output_config_file = os.path.join(checkpoint_fold, CONFIG_NAME)
    model_to_save.transformer.config.to_json_file(output_config_file)


def save_tokenizer(tokenizer, checkpoint_fold):
    tokenizer.save_pretrained(checkpoint_fold)


def main():
    model_entities = {
        "myRoberta": entity(model_class=my_Roberta, config_class=RobertaConfig,
                            tokenizer_class=RobertaTokenizerFast, model_name="roberta-base")
    }
    train_csv = pd.read_csv(f'{data_dir}\\train.csv')
    train_csv = train_csv.dropna()
    test_csv = pd.read_csv(f'{data_dir}\\test.csv')
    pl.seed_everything(seed)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_csv, train_csv.sentiment), start=1):
        checkpoint_fold = checkpoint_dir + '/fold{}'.format(fold)
        if not os.path.exists(checkpoint_fold):
            os.makedirs(checkpoint_fold)
        print(f'Fold: {fold}')
        #
        model_nlp, tokenizer = model_tokenizer(model_entities, model_name)
        save_model_config(model_nlp, checkpoint_fold)
        save_tokenizer(tokenizer, checkpoint_fold)
        #
        df_dict = {'train': train_csv.iloc[train_idx], "test": train_csv.iloc[val_idx]}
        model_lighting = lightModel(model_nlp, tokenizer, df_dict, max_length=max_length)
        checkpoint_callback = pl.callbacks.ModelCheckpoint(filepath=checkpoint_fold, verbose=True,
                                                           save_top_k=2, monitor='val_loss', mode='min')
        trainer = pl.Trainer(fast_dev_run=True, gpus=1, max_epochs=epochs, checkpoint_callback=checkpoint_callback)
        trainer.fit(model_lighting)
    pass





if __name__ == "__main__":
    model_name = 'myRoberta'
    data_dir = r'C:\Users\Administrator\Desktop\DL_Data\Tweet_sentiment_extract'
    checkpoint_dir = 'checkpoints' + '/' + model_name
    max_length = 120
    seed = 2009
    n_folds = 10
    epochs = 5
    main()
    print()



