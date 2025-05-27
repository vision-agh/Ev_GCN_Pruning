from models.recognition_tiny import LNRecognitionTiny
from data.mnist import MNIST

from omegaconf import OmegaConf
from utils.structured_pruning import structured_pruning
from utils.precompute_space import precompute_space, generate_configs
from utils.generate_outputs import events_out
import lightning as L
import torch
import time
import numpy as np
import pandas as pd
import os

def main():
    L.seed_everything(42, workers=True)
    cfg = OmegaConf.load('configs/mnist_tiny.yaml')
    cfg.batch_size = 1
    cfg.num_workers = 0
    cfg.debug = True
    dm = MNIST(cfg)
    dm.setup()

    model = LNRecognitionTiny(cfg=cfg).cuda()
    model.model.eval()
    model.model.calibrate()
    model.model.quantize()

    print(model)
    
    model.model.load_state_dict(torch.load('weights_mnist-dvs_tiny/model.pth'))

    model.model.conv1.get_parameters('weights_mnist-dvs_tiny/conv1.txt')
    model.model.conv2.get_parameters('weights_mnist-dvs_tiny/conv2.txt')
    model.model.conv3.get_parameters('weights_mnist-dvs_tiny/conv3.txt')
    model.model.conv4.get_parameters('weights_mnist-dvs_tiny/conv4.txt')
    model.model.conv5.get_parameters('weights_mnist-dvs_tiny/conv5.txt')

    # save the float weights and biases of linear layers
    model.model.linear1.linear.weight.detach().cpu().numpy().tofile('weights_mnist-dvs_tiny/linear1_w.bin')
    model.model.linear1.linear.bias.detach().cpu().numpy().tofile('weights_mnist-dvs_tiny/linear1_b.bin')
    model.model.linear2.linear.weight.detach().cpu().numpy().tofile('weights_mnist-dvs_tiny/linear2_w.bin')
    model.model.linear2.linear.bias.detach().cpu().numpy().tofile('weights_mnist-dvs_tiny/linear2_b.bin')
    
    acc = 0
    itere = 0

    os.makedirs('outputs', exist_ok=True)

    for batch_data in dm.test_dataloader():
        for k, v in batch_data.items():
            if isinstance(v, torch.Tensor):
                batch_data[k] = v.cuda()

        events_out(batch_data['real_events'][0],
                   cfg,
                   'outputs/events.txt')

        out = model(batch_data)
        label = batch_data['label']

        out = torch.argmax(out, dim=-1)

        print(label)

        break

        accuracy = (out == label).sum().item()
        acc += accuracy
        itere += label.size(0)

    # print(f'Average Accuracy: {acc / itere}')

if __name__ == '__main__':
    main()