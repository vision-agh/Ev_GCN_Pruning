import lightning as L
import argparse
import multiprocessing as mp

from omegaconf import OmegaConf
from lightning.pytorch.loggers.wandb import WandbLogger

from data.ncars import NCars
from models.recognition import LNRecognition


def main(args):
    cfg = OmegaConf.load('configs/ncars.yaml')

    dm = NCars(cfg)
    dm.setup()

    model = LNRecognition(cfg)

    wandb_logger = WandbLogger(project='event_recognition_pruning', 
                               name=f'{cfg.data_name}_{cfg.radius}', 
                               log_model='all')
    wandb_logger.watch(model)

    trainer = L.Trainer(max_epochs=cfg.max_epochs, 
                        log_every_n_steps=cfg.log_every_n_steps, 
                        gradient_clip_val=cfg.gradient_clip_val, 
                        accumulate_grad_batches=cfg.accumulate_grad_batches, 
                        logger=wandb_logger)
    
    trainer.fit(model, dm)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ncars')

    mp.set_start_method('spawn', force=True)
    args = parser.parse_args()
    main(args)