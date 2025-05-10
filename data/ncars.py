import os
import glob
import numpy as np
import torch
import lightning as L

from torch.utils.data import DataLoader
from data.base.event_ds import EventDS

class NCars(L.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
            
    def setup(self, stage=None):
        self.train_data = self.generate_ds('train')
        self.test_data = self.generate_ds('test')

    def generate_ds(self, mode: str):
        files = glob.glob(os.path.join(self.cfg.data_dir, 
                                        self.cfg.data_name, 
                                        mode, 
                                        '*', 
                                        'events.txt'))
        return EventDS(files, 
                       self.cfg, 
                       mode=mode)

    def train_dataloader(self):
        return DataLoader(self.train_data, 
                          batch_size=self.cfg.batch_size, 
                          num_workers=self.cfg.num_workers, 
                          shuffle=True, 
                          collate_fn=self.collate_fn, 
                          persistent_workers=False)
    
    def val_dataloader(self):
        return DataLoader(self.test_data, 
                          batch_size=self.cfg.batch_size, 
                          num_workers=self.cfg.num_workers, 
                          shuffle=False, 
                          collate_fn=self.collate_fn, 
                          persistent_workers=False)
    
    def test_dataloader(self):
        return DataLoader(self.test_data, 
                          batch_size=self.cfg.batch_size, 
                          num_workers=self.cfg.num_workers, 
                          shuffle=False, 
                          collate_fn=self.collate_fn, 
                          persistent_workers=False)
    
    def collate_fn(self, data_list):
        x = torch.cat([data['x'] for data in data_list], dim=0)
        pos = torch.cat([data['pos'] for data in data_list], dim=0)

        edge_index = []
        offset = 0
        for data in data_list:
            edge_index.append(data['edge_index'] + offset)
            offset += data['x'].shape[0]
        edge_index = torch.cat(edge_index, dim=0)

        label = torch.tensor([data['label'] for data in data_list])

        batch = torch.cat([torch.full((data['x'].shape[0],), i) for i, data in enumerate(data_list)], dim=0)
        batch = batch.long()

        return {
            'x': x,
            'pos': pos,
            'edge_index': edge_index,
            'batch': batch,
            'label': label
        }