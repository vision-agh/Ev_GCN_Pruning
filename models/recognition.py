import torch
import lightning as L

from torchmetrics import Accuracy
from torchmetrics.classification import ConfusionMatrix

from typing import Dict, Tuple
from torch.nn.functional import softmax

from models.model import MyModel

import wandb
import numpy as np
import matplotlib.pyplot as plt


class LNRecognition(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.lr = cfg.lr
        self.weight_decay = cfg.weight_decay

        self.batch_size = cfg.batch_size
        self.num_classes = cfg.num_classes

        self.model = MyModel(cfg)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task="multiclass", num_classes=cfg.num_classes)

        if cfg.num_classes > 3:
            self.accuracy_top_3 = Accuracy(task="multiclass", num_classes=cfg.num_classes, top_k=3).to(self.device)

        self.save_hyperparameters()

        self.val_pred = None
        self.train_pred = None
        self.pred = []
        self.target = []

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

    def forward(self, data):
        x = self.model(data['x'], data['pos'], data['edge_index'], data['batch'])
        return x

    def training_step(self, batch, batch_idx):
        outputs = self.forward(data=batch)
        loss = self.criterion(outputs, batch['label'].cuda())

        y_prediction = torch.argmax(outputs, dim=-1)
        accuracy = self.accuracy(preds=y_prediction.cpu(), target=batch['label'].cpu())

        self.log('train_loss', loss, on_epoch=True, logger=True, batch_size=self.batch_size)
        self.log('train_acc', accuracy, on_epoch=True, logger=True, batch_size=self.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.forward(data=batch)
        loss = self.criterion(outputs, batch['label'].cuda())

        y_prediction = torch.argmax(outputs, dim=-1)
        accuracy = self.accuracy(preds=y_prediction.cpu(), target=batch['label'].cpu())
        
        self.log('val_loss', loss, on_epoch=True, logger=True, batch_size=self.batch_size)
        self.log('val_acc', accuracy, on_epoch=True, logger=True, batch_size=self.batch_size)

        if self.num_classes > 3:
            pred = softmax(outputs, dim=-1)
            top_3 = self.accuracy_top_3(preds=pred.unsqueeze(0).to(self.device), target=torch.tensor([batch['y']]).to(self.device))
            self.log('val_acc_top_3', top_3, on_epoch=True, logger=True, batch_size=self.batch_size)

    def test_step(self, batch, batch_idx):
        outputs = self.forward(data=batch)
        loss = self.criterion(outputs, target=torch.tensor(batch['label']).long().to('cuda'))

        y_prediction = torch.argmax(outputs, dim=-1)
        accuracy = self.accuracy(preds=y_prediction.cpu().unsqueeze(0), target=torch.tensor([batch['label']]))

        self.log('test_loss', loss, on_epoch=True, logger=True, batch_size=self.batch_size)
        self.log('test_acc', accuracy, on_epoch=True, logger=True, batch_size=self.batch_size)