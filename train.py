import lightning as L
import argparse
import multiprocessing as mp

from omegaconf import OmegaConf
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from data.ncars import NCars
from data.mnist import MNIST
from data.cifar import CIFAR
from models.recognition import LNRecognition


def main():
    cfg = OmegaConf.load('configs/cifar.yaml')

    print(cfg)

    if cfg.data_name == 'ncars':
        dm = NCars(cfg)
    elif cfg.data_name == 'mnist-dvs':
        dm = MNIST(cfg)
    elif cfg.data_name == 'cifar10-dvs':
        dm = CIFAR(cfg)
    dm.setup()

    model = LNRecognition(cfg)
    
    print(model)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints/',
        filename=f'{cfg.data_name}_{cfg.radius}',
        save_top_k=1,
        mode='min',
    )
    
    wandb_logger = WandbLogger(project='event_recognition_pruning', 
                               name=f'{cfg.data_name}_{cfg.radius}', 
                               log_model='all')

    lr_monitor = LearningRateMonitor(logging_interval='step') 
    trainer = L.Trainer(max_epochs=cfg.max_epochs, 
                        log_every_n_steps=cfg.log_every_n_steps, 
                        gradient_clip_val=cfg.gradient_clip_val, 
                        accumulate_grad_batches=cfg.accumulate_grad_batches, 
                        logger=wandb_logger,
                        callbacks=[lr_monitor, checkpoint_callback],
                        )
    
    trainer.fit(model, dm)

if __name__ == '__main__':
    L.seed_everything(42, workers=True)
    mp.set_start_method('fork', force=True)
    main()