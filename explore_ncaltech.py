from models.recognition import LNRecognition
from data.mnist import MNIST
from data.ncars import NCars
from data.cifar import CIFAR
from data.ncaltech import NCaltech

from omegaconf import OmegaConf
from utils.structured_pruning import structured_pruning
from utils.precompute_space_ncaltech import precompute_space_ncaltech, generate_configs_ncaltech

import lightning as L
import torch
import time
import numpy as np
import pandas as pd

def main():
    L.seed_everything(42, workers=True)
    cfg = OmegaConf.load('configs/ncaltech.yaml')

    if cfg.data_name == 'ncars':
        dm = NCars(cfg)
    elif cfg.data_name == 'mnist-dvs':
        dm = MNIST(cfg)
    elif cfg.data_name == 'cifar10-dvs':
        dm = CIFAR(cfg)
    elif cfg.data_name == 'ncaltech101':
        dm = NCaltech(cfg)
    dm.setup()

    model = LNRecognition.load_from_checkpoint(f'checkpoints/{cfg.data_name}_3.ckpt', cfg=cfg).cuda()
    model.model.eval()

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
    print(f'Average Accuracy: {acc / itere} for floating point')

    precomputed_space = precompute_space_ncaltech(cfg, depth_start=0, depth_end=2)
    configs = generate_configs_ncaltech(precomputed_space)

    print(f'Number of configurations: {len(configs)}')

    results = []

    for config in configs:
        t_start = time.time()

        cfg.conv1.num_bits = config['conv1_bits']
        cfg.conv2.num_bits = config['conv2_bits']
        cfg.conv3.num_bits = config['conv3_bits']
        cfg.conv4.num_bits = config['conv4_bits']
        cfg.conv5.num_bits = config['conv5_bits']

        model = LNRecognition.load_from_checkpoint(f'checkpoints/{cfg.data_name}_3.ckpt', cfg=cfg).cuda()
        model.model.eval()
        model.model.calibrate()

        structured_pruning(model.model.conv1, (cfg.conv1.out_channels - config['conv1_pruning'])/cfg.conv1.out_channels)
        structured_pruning(model.model.conv2, (cfg.conv2.out_channels - config['conv2_pruning'])/cfg.conv2.out_channels)
        structured_pruning(model.model.conv3, (cfg.conv3.out_channels - config['conv3_pruning'])/cfg.conv3.out_channels)
        structured_pruning(model.model.conv4, (cfg.conv4.out_channels - config['conv4_pruning'])/cfg.conv4.out_channels)
        structured_pruning(model.model.conv5, (cfg.conv5.out_channels - config['conv5_pruning'])/cfg.conv5.out_channels)

        brams = (config['conv1_bram'] + config['conv2_bram'] + config['conv3_bram'] + config['conv4_bram'] + config['conv5_bram'])
        m = (config['conv1_m'] + config['conv2_m'] + config['conv3_m'] + config['conv4_m'] + config['conv5_m'])

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
        print(f'Average Accuracy: {acc / itere} with {brams} BRAMs and {m} multipliers')
        print(f'Time taken: {time.time() - t_start} seconds')

        results.append({
            'conv1_pruning': config['conv1_pruning'],
            'conv2_pruning': config['conv2_pruning'],
            'conv3_pruning': config['conv3_pruning'],
            'conv4_pruning': config['conv4_pruning'],
            'conv5_pruning': config['conv5_pruning'],
            'conv1_bits': config['conv1_bits'],
            'conv2_bits': config['conv2_bits'],
            'conv3_bits': config['conv3_bits'],
            'conv4_bits': config['conv4_bits'],
            'conv5_bits': config['conv5_bits'],
            'conv1_bram': config['conv1_bram'],
            'conv2_bram': config['conv2_bram'],
            'conv3_bram': config['conv3_bram'],
            'conv4_bram': config['conv4_bram'],
            'conv5_bram': config['conv5_bram'],
            'm': m,
            'brams': brams,
            'accuracy': acc / itere,
        })
        print('----------------------------------')

    # Save the results to a file
    df = pd.DataFrame(results)
    df.to_csv(f'results_{cfg.data_name}.csv', index=False)


if __name__ == '__main__':
    main()