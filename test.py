from models.recognition import LNRecognition
from data.mnist import MNIST

from omegaconf import OmegaConf
from utils.structured_pruning import structured_pruning
from utils.precompute_space import precompute_space, generate_configs

import lightning as L
import torch
import time
import numpy as np
import pandas as pd

def main():
    L.seed_everything(42, workers=True)
    cfg = OmegaConf.load('configs/mnist.yaml')
    dm = MNIST(cfg)
    dm.setup()

    model = LNRecognition.load_from_checkpoint('checkpoints/mnist-dvs_3.ckpt', cfg=cfg).cuda()
    model.model.eval()


    structured_pruning(model.model.conv1, (cfg.conv1.out_channels - 15)/cfg.conv1.out_channels)
    structured_pruning(model.model.conv2, (cfg.conv2.out_channels - 15)/cfg.conv2.out_channels)
    structured_pruning(model.model.conv3, (cfg.conv3.out_channels - 15)/cfg.conv3.out_channels)
    structured_pruning(model.model.conv4, (cfg.conv4.out_channels - 15)/cfg.conv4.out_channels)
    structured_pruning(model.model.conv5, (cfg.conv5.out_channels - 15)/cfg.conv5.out_channels)

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
    print(config)
    print(f'Average Accuracy: {acc / itere} with {brams} BRAMs')
    print(f'Time taken: {time.time() - t_start} seconds')


if __name__ == '__main__':
    main()