import torch
import lightning as L

from torchmetrics import Accuracy
from torchmetrics.classification import ConfusionMatrix

from typing import Dict, Tuple
from torch.nn.functional import softmax

from models.model_tiny import MyModelTiny
from utils.structured_pruning import structured_pruning

import wandb
import numpy as np
import matplotlib.pyplot as plt


class LNRecognitionTiny(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.lr = cfg.lr
        self.weight_decay = cfg.weight_decay

        self.batch_size = cfg.batch_size
        self.num_classes = cfg.num_classes

        self.model = MyModelTiny(cfg)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task="multiclass", num_classes=cfg.num_classes)

        if cfg.num_classes > 3:
            self.accuracy_top_3 = Accuracy(task="multiclass", num_classes=cfg.num_classes, top_k=3)

        self.save_hyperparameters()

        self.val_pred = None
        self.train_pred = None
        self.pred = []
        self.target = []

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
        # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1 if epoch < 20 else 0.1)
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}

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
            top_3 = self.accuracy_top_3(preds=pred, target=batch['label'])
            self.log('val_acc_top_3', top_3, on_epoch=True, logger=True, batch_size=self.batch_size)

    def test_step(self, batch, batch_idx):
        outputs = self.forward(data=batch)
        loss = self.criterion(outputs, target=torch.tensor(batch['label']).long().to('cuda'))

        y_prediction = torch.argmax(outputs, dim=-1)
        accuracy = self.accuracy(preds=y_prediction.cpu().unsqueeze(0), target=torch.tensor([batch['label']]))

        self.log('test_loss', loss, on_epoch=True, logger=True, batch_size=self.batch_size)
        self.log('test_acc', accuracy, on_epoch=True, logger=True, batch_size=self.batch_size)

    # def on_validation_epoch_end(self):
    #     if self.trainer.current_epoch == 50:
    #         self.model.calibrate()
    #         print('Calibrating the model')
    # def on_validation_epoch_end(self):
    #     # Pruning
    #     from models.layers.my_pointnet import MyPointNetConv

    #     epochs = {50: 0.1, 60: 0.2, 70: 0.3, 80: 0.4, 90: 0.5}

    #     if self.trainer.current_epoch in epochs.keys():
    #         prune_amount = epochs[self.trainer.current_epoch]
    #         print(f'Pruning {prune_amount} of the model')
    #         for name, module in self.model.named_modules():
    #             if isinstance(module, MyPointNetConv):
    #                 structured_pruning(module, amount=prune_amount)
    #                 print(f'Pruned {name} with amount {prune_amount}')