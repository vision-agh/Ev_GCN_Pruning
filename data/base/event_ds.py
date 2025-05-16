import torch
from torch.utils.data import Dataset
# from data.base.augmentation import RandomZoom, RandomCrop, RandomTranslate, Crop
import matrix_neighbour

import numpy as np

class EventDS(Dataset):
    def __init__(self, 
                 files, 
                 cfg, 
                 reader=None,
                 mode=None):
        
        self.files = files
        self.cfg = cfg
        self.mode = mode
        self.reader = reader

    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, index: int):
        events_file = self.files[index]
        events, label = self.reader(events_file, self.cfg)

        if self.mode == 'train':
            events['x'] = events['x'] + np.random.randint(-5, 5)
            events['y'] = events['y'] + np.random.randint(-5, 5)
            mask = (events['x'] >= 0) & (events['x'] < self.cfg.WIDTH) & \
                    (events['y'] >= 0) & (events['y'] < self.cfg.HEIGHT)
            for key in events.keys():
                events[key] = events[key][mask]


            if np.random.rand() < 0.2:
                events['p'] = events['p'] * (-1)

        # Normalize x y and t to [0, 128]
        events['x'] = np.floor(events['x'] / self.cfg.org_WIDTH * self.cfg.WIDTH)
        events['y'] = np.floor(events['y'] / self.cfg.org_HEIGHT * self.cfg.HEIGHT)
        events['t'] = np.floor((events['t'] / ( self.cfg.time_window )) * self.cfg.T)

        events = np.column_stack((events['x'], events['y'], events['t'], events['p']))

        x, pos, edge_index = matrix_neighbour.generate_edges(events.astype(np.int32), 
                                                             self.cfg.radius, 
                                                             self.cfg.WIDTH, 
                                                             self.cfg.HEIGHT)

        x = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
        pos = torch.tensor(pos, dtype=torch.float32)
        edge_index = torch.tensor(edge_index, dtype=torch.long)

        return {
            'x': x,
            'pos': pos,
            'edge_index': edge_index,
            'label': label,
        }