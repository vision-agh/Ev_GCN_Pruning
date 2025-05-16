from models.layers.my_pointnet import MyPointNetConv
from models.layers.my_max_pool import MyGraphPooling
from data.ncars import NCars

from omegaconf import OmegaConf
import torch

import matplotlib.pyplot as plt
import numpy as np

def main():
    cfg = OmegaConf.load('configs/ncars.yaml')
    cfg.batch_size = 2
    dm = NCars(cfg)
    dm.setup()
    
    model = MyPointNetConv(3, 16, False, cfg.num_bits, True)
    pool = MyGraphPooling([6, 5, 5])
    
    for batch in dm.val_dataloader():
        x = batch['x']
        pos = batch['pos']
        edge_index = batch['edge_index']
        batch_idx = batch['batch']
        
        print(x.shape)
        print(pos.shape)
        print(edge_index.shape)
        print(batch_idx.shape)

        x, pos, edge_index, batch_idx = pool(x, pos, edge_index, batch_idx)

        x = x[batch_idx == 0]
        pos = pos[batch_idx == 0]

        valid_indices = torch.where(batch_idx == 0)[0]

        # get edges where both nodes are in the same batch
        mask = (torch.any(edge_index[:,0].unsqueeze(1) == valid_indices, dim=1) & 
                torch.any(edge_index[:,1].unsqueeze(1) == valid_indices, dim=1))
        edge_index = edge_index[mask]

        print(x.shape)
        print(pos.shape)
        print(edge_index.shape)

        break

        # for i in range(max(batch_idx) + 1):
        #     mask = batch_idx == i
        #     x_i = x[mask]
        #     pos_i = pos[mask]
        #     edge_index_i = edge_index[:, mask]
        #     batch_idx_i = batch_idx[mask]

        #     print(x_i.shape)
        #     print(pos_i.shape)
        #     print(edge_index_i.shape)
        #     print(batch_idx_i.shape)

        #     fig = plt.figure(figsize=(10, 10))
        #     ax = fig.add_subplot(111, projection='3d')
        #     ax.scatter(pos_i[:, 1], pos_i[:, 2], pos_i[:, 0], c=x[:, 0], cmap='bwr', marker='o', s=2)

        #     # plot edges
        #     for i in range(edge_index.shape[0]):
        #         x1, y1, z1 = pos_i[edge_index[i, 0], 1], pos_i[edge_index[i, 0], 2], pos_i[edge_index[i, 0], 0]
        #         x2, y2, z2 = pos_i[edge_index[i, 1], 1], pos_i[edge_index[i, 1], 2], pos_i[edge_index[i, 1], 0]
        #         ax.plot([x1, x2], [y1, y2], [z1, z2], color='gray', alpha=0.5)

        #     ax.set_xlabel('X Label')
        #     ax.set_ylabel('Y Label')
        #     ax.set_zlabel('Z Label')

        #     plt.title('3D Scatter Plot of Point Cloud')
        #     plt.show(block=False)
        #     plt.pause(0.1)

        # x, pos, edge_index, batch_idx = pool(x, pos, edge_index, batch_idx)

        # print(x.shape)
        # print(pos.shape)
        # print(edge_index.shape)
        # print(batch_idx.shape)

        # fig = plt.figure(figsize=(10, 10))
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(pos[:, 1], pos[:, 2], pos[:, 0], c=x[:, 0], cmap='bwr', marker='o', s=2)
        # # plot edges
        # for i in range(edge_index.shape[0]):
        #     x1, y1, z1 = pos[edge_index[i, 0], 1], pos[edge_index[i, 0], 2], pos[edge_index[i, 0], 0]
        #     x2, y2, z2 = pos[edge_index[i, 1], 1], pos[edge_index[i, 1], 2], pos[edge_index[i, 1], 0]
        #     ax.plot([x1, x2], [y1, y2], [z1, z2], color='gray', alpha=0.5)

        # ax.set_xlabel('X Label')
        # ax.set_ylabel('Y Label')
        # ax.set_zlabel('Z Label')

        # plt.title('3D Scatter Plot of Point Cloud')
        # plt.show()


if __name__ == '__main__':
    main()