import torch
from torch.utils.data import Dataset
# from data.base.augmentation import RandomZoom, RandomCrop, RandomTranslate, Crop
import matrix_neighbour

import numpy as np

class EventDS(Dataset):
    def __init__(self, 
                 files, 
                 cfg, 
                 mode='train'):
        
        self.files = files
        self.cfg = cfg

        # if mode == 'test' or mode == 'val':
        #     self.random_crop = RandomCrop([0.75, 0.75], p=0.2, width=dim[0], height=dim[1])
        #     self.zoom = RandomZoom([1, 1.5], subsample=True, width=dim[0], height=dim[1])
        #     self.translate = RandomTranslate([0.1, 0.1, 0], width=dim[0], height=dim[1])
        #     self.crop = Crop([0,0], [1, 1], width=dim[0], height=dim[1])

    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, index: int):
        events_file = self.files[index]
        annotation_file = events_file.replace('events.txt', 'is_car.txt')

        events = np.loadtxt(events_file)

        # Extract and process events data
        all_x, all_y, all_ts, all_p = events.T
        all_ts *= 1e+6  # Convert to seconds
        all_p[all_p == 0] = -1

        # Create dictionary for events
        events = {
            'x': all_x,
            'y': all_y,
            't': all_ts,
            'p': all_p
        }
        
        # Filter events by time window
        mask = events['t'] < self.cfg.time_window * 1e+6
        for key in events:
            events[key] = events[key][mask]

        # Normalize x y and t to [0, 128]
        events['x'] = (events['x'] / self.cfg.width) * 128
        events['y'] = (events['y'] / self.cfg.height) * 128
        events['t'] = (events['t'] / ( self.cfg.time_window * 1e+6 )) * 128

        events = np.column_stack((events['x'], events['y'], events['t'], events['p']))

        x, pos, edge_index = matrix_neighbour.generate_edges(events.astype(np.int32), self.cfg.radius, 128, 128)

        x = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
        pos = torch.tensor(pos, dtype=torch.float32)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        label = np.loadtxt(annotation_file).item()

        return {
            'x': x,
            'pos': pos,
            'edge_index': edge_index,
            'label': label,
        }