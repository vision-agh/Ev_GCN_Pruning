import multiprocessing as mp

from omegaconf import OmegaConf

from data.ncars import NCars

from models.layers.my_pointnet import MyPointNetConv
from models.layers.my_max_pool import MyGraphPooling

import torch

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

    cfg = OmegaConf.load('configs/ncars.yaml')

    lm = NCars(cfg)
    lm.setup()

    layer1 = MyPointNetConv(4, 16, False, 8, True).cuda()
    layer2 = MyPointNetConv(19, 32, False, 8, False).cuda()
    pool = MyGraphPooling(16, max_dimension=128)
    layer3 = MyPointNetConv(35, 32, False, 8, False).cuda()
    layer4 = MyPointNetConv(35, 64, False, 8, False).cuda()

    for i, batch in enumerate(lm.train_dataloader()):
        x = batch['x'].cuda()
        pos = batch['pos'].cuda()
        edge_index = batch['edge_index'].cuda()

        batch = batch['batch'].cuda()

        x = layer1(x, pos, edge_index)
        x = layer2(x, pos, edge_index)

        x, pos, edge_index, batch = pool(x, pos, edge_index, batch)

        x = layer3(x, pos, edge_index)
        x = layer4(x, pos, edge_index)



    print('done')

# import torch

# x = torch.randn(1024, 1).cuda()
# pos = torch.rand(1024, 3).cuda()

# edge_index = torch.randint(0, 1024, (1024, 2)).cuda()
# mask = edge_index[:, 0] != edge_index[:, 1]
# edge_index = edge_index[mask, :]    
# edge_index = torch.unique(edge_index, dim=0)
# edge_index = torch.cat((edge_index, torch.arange(1024, device=edge_index.device).unsqueeze(1).expand(-1, 2)), dim=0)

# batch = torch.randint(0, 3, (1024,), device=edge_index.device).long()


# layer = layer.cuda()
# out = layer(x, pos, edge_index)
# print(out)


# layer.calibrate()

# out = layer(x, pos, edge_index)
# print(out)

# print(layer.observer_output.scale)


# layer.quantize()

# out = layer(x, pos, edge_index)

# print(layer.observer_output.dequantize_tensor(out))



# pool = MyGraphPooling(2, max_dimension=16)


# pos *= 16 


# print(x, pos, edge_index, batch)
# pos, x, edge_index, batch = pool(x, pos, edge_index, batch)
# print(x, pos, edge_index, batch)


