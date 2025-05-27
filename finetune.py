import lightning as L
import argparse
import multiprocessing as mp

from omegaconf import OmegaConf
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from data.ncars import NCars
from data.mnist import MNIST
from data.cifar import CIFAR
from data.ncaltech import NCaltech
from models.recognition import LNRecognition

from utils.structured_pruning import structured_pruning    

import torch
import os

def main():
    cfg = OmegaConf.load('configs/mnist_tiny.yaml')

    print(cfg)

    cfg.lr = cfg.lr * 0.01

    dm = MNIST(cfg)
    dm.setup()

    cfg.conv1.num_bits = 8
    cfg.conv2.num_bits = 8
    cfg.conv3.num_bits = 8
    cfg.conv4.num_bits = 8
    cfg.conv5.num_bits = 8

    model = LNRecognition.load_from_checkpoint('checkpoints/mnist-dvs_3_tiny-v1.ckpt', cfg=cfg).cuda()
    
    model.model.calibrate()
    print(model)

    # structured_pruning(model.model.conv1, (cfg.conv1.out_channels - 18)/cfg.conv1.out_channels)
    # structured_pruning(model.model.conv2, (cfg.conv2.out_channels - 33)/cfg.conv2.out_channels)
    # structured_pruning(model.model.conv3, (cfg.conv3.out_channels - 72)/cfg.conv3.out_channels)
    # structured_pruning(model.model.conv4, (cfg.conv4.out_channels - 57)/cfg.conv4.out_channels)
    # structured_pruning(model.model.conv5, (cfg.conv5.out_channels - 120)/cfg.conv5.out_channels)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        dirpath='checkpoints/',
        filename=f'{cfg.data_name}_{cfg.radius}_finetune',
        save_top_k=1,
        mode='max',
    )
    
    wandb_logger = WandbLogger(project='event_recognition_pruning', 
                               name=f'{cfg.data_name}_{cfg.radius}_finetune', 
                               log_model='all')

    lr_monitor = LearningRateMonitor(logging_interval='step') 
    trainer = L.Trainer(max_epochs=10, 
                        log_every_n_steps=cfg.log_every_n_steps, 
                        gradient_clip_val=cfg.gradient_clip_val, 
                        accumulate_grad_batches=cfg.accumulate_grad_batches, 
                        logger=wandb_logger,
                        callbacks=[lr_monitor, checkpoint_callback],
                        )
    
    trainer.fit(model, dm)

    model.model.eval().cuda()
    acc = 0
    itere = 0

    for batch_data in dm.test_dataloader():
        for k, v in batch_data.items():
            if isinstance(v, torch.Tensor):
                batch_data[k] = v.cuda()

        out = model(batch_data)
        label = batch_data['label']

        out = torch.argmax(out, dim=-1)

        accuracy = (out == label).sum().item()
        acc += accuracy
        itere += label.size(0)

    print(f'Average Accuracy: {acc / itere}')

    model.model.quantize()

    acc = 0
    itere = 0

    for batch_data in dm.test_dataloader():
        for k, v in batch_data.items():
            if isinstance(v, torch.Tensor):
                batch_data[k] = v.cuda()

        out = model(batch_data)
        label = batch_data['label']

        out = torch.argmax(out, dim=-1)

        accuracy = (out == label).sum().item()
        acc += accuracy
        itere += label.size(0)

    print(f'Average Accuracy: {acc / itere}')
    
    # Save the model
    os.makedirs(f'weights_{cfg.data_name}_tiny', exist_ok=True)
    torch.save(model.model.state_dict(), f'weights_{cfg.data_name}_tiny/model.pth')

if __name__ == '__main__':
    L.seed_everything(42, workers=True)
    mp.set_start_method('fork', force=True)
    main()