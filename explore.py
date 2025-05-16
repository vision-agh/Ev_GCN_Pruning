from models.recognition import LNRecognition
from data.mnist import MNIST

from omegaconf import OmegaConf
from utils.structured_pruning import structured_pruning

import lightning as L
import torch

def main():
    L.seed_everything(42, workers=True)
    cfg = OmegaConf.load('configs/mnist.yaml')
    dm = MNIST(cfg)
    dm.setup()

    cfg.num_bits = 6

    model = LNRecognition.load_from_checkpoint('checkpoints/mnist-dvs_3-v4.ckpt', cfg=cfg).cuda()
    model.model.eval()
    # model.model.calibrate()

    # for idx, batch_data in enumerate(dm.train_dataloader()):
    #     for k, v in batch_data.items():
    #         if isinstance(v, torch.Tensor):
    #             batch_data[k] = v.cuda()

    #     out = model(batch_data)

    #     if idx == 100:
    #         break
    
        
    # model.model.eval()
    

    # structured_pruning(model.model.conv5, 1/128)


    acc = 0
    itere = 0
    for batch_data in dm.val_dataloader():
        for k, v in batch_data.items():
            if isinstance(v, torch.Tensor):
                batch_data[k] = v.cuda()

        out = model(batch_data)
        label = batch_data['label']

        out = torch.argmax(out, dim=-1)

        accuracy = (out == label).sum().item()
        acc += accuracy
        itere += label.size(0)

        # print(f'Accuracy: {accuracy.item()}')
    print(f'Average Accuracy: {acc / itere}')


if __name__ == '__main__':
    main()