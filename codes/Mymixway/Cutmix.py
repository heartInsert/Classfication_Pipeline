import torch.nn as nn
import numpy as np
import torch


class Cutmix:
    def __init__(self, beta=1, prob=1, loss_fc=nn.CrossEntropyLoss()):
        self.beta = beta
        self.prob = prob
        self.loss_fc = loss_fc
        self.cutmix = False
        self.target_a = None
        self.target_b = None
        self.lam = None

    def rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def __call__(self, model, data, label, is_training):
        r = np.random.rand(1)
        if is_training and self.beta > 0 and r < self.prob:
            self.cutmix = True
            # generate mixed sample
            lam = np.random.beta(self.beta, self.beta)
            rand_index = torch.randperm(data.size()[0]).cuda()
            self.target_a = label
            self.target_b = label[rand_index]
            bbx1, bby1, bbx2, bby2 = self.rand_bbox(data.size(), lam)
            data[:, :, bbx1:bbx2, bby1:bby2] = data[rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            self.lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))
        else:
            self.cutmix = False
            self.target_a = None
            self.target_b = None
            self.lam = None
        logits = model(data)
        loss = self.loss(logits, label)
        return logits, loss

    def loss(self, logits, label):
        if self.cutmix:
            loss = self.loss_fc(logits, self.target_a) * self.lam + self.loss_fc(logits, self.target_b) * (
                    1. - self.lam)
        else:
            loss = self.loss_fc(logits, label)
        return loss
