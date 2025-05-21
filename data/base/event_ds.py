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
            # Randomly rotate the events
            if self.cfg.rotate_angle != 0:
                events = self.RandomRotate(events)
            events = self.RandomHFlip(events)

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
    
    def RandomRotate(self, events):
        angle = np.random.randint(-self.cfg.rotate_angle, self.cfg.rotate_angle)
        angle = np.deg2rad(angle)

        x, y = events['x'], events['y']

        events['x'] = np.floor( x * np.cos(angle) - y * np.sin(angle) )
        events['y'] = np.floor( x * np.sin(angle) + y * np.cos(angle) )

        mask = (events['x'] >= 0) & (events['x'] < self.cfg.org_WIDTH) & \
                (events['y'] >= 0) & (events['y'] < self.cfg.org_HEIGHT)
        for key in events.keys():
            events[key] = events[key][mask]

        return events

    def RandomZoom(self, events):
        zoom = np.random.uniform(1 - self.cfg.zoom_scale, 1 + self.cfg.zoom_scale)
        events['x'] = int(events['x'] * zoom)
        events['y'] = int(events['y'] * zoom)

        mask = (events['x'] >= 0) & (events['x'] < self.cfg.WIDTH) & \
                (events['y'] >= 0) & (events['y'] < self.cfg.HEIGHT)
        for key in events.keys():
            events[key] = events[key][mask]

        return events
    
    def RandomHFlip(self, events):
        if np.random.rand() < self.cfg.hflip:
            events['x'] = np.floor(self.cfg.org_WIDTH - 1 - events['x'])

        mask = (events['x'] >= 0) & (events['x'] < self.cfg.org_WIDTH) & \
                (events['y'] >= 0) & (events['y'] < self.cfg.org_HEIGHT)
        for key in events.keys():
            events[key] = events[key][mask]

        return events