import os
import glob
import numpy as np
import torch
import lightning as L

from torch.utils.data import DataLoader
from data.base.event_ds import EventDS

class MNIST(L.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        
    def setup(self, stage=None):
        all_files = glob.glob(os.path.join(self.cfg.data_dir,
                                            self.cfg.data_name,
                                            '*',
                                            '*.aedat'))
        

        # split the files into train and test sets 
        num_train = int(len(all_files) * 0.8)
        # shuffle the files
        np.random.shuffle(all_files)
        train_files = all_files[:num_train]
        test_files = all_files[num_train:]
        self.train_data = EventDS(train_files,
                                 self.cfg,
                                 reader=self.load_events,
                                 mode='train')
        
        self.test_data = EventDS(test_files,
                                self.cfg,
                                reader=self.load_events,
                                mode='test')
        

    def train_dataloader(self):
        return DataLoader(self.train_data, 
                          batch_size=self.cfg.batch_size, 
                          num_workers=self.cfg.num_workers, 
                          shuffle=True, 
                          collate_fn=self.collate_fn, 
                          persistent_workers=True)
    
    def val_dataloader(self):
        return DataLoader(self.test_data, 
                          batch_size=self.cfg.batch_size, 
                          num_workers=self.cfg.num_workers, 
                          shuffle=False, 
                          collate_fn=self.collate_fn, 
                          persistent_workers=True)
    
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

        label = torch.tensor([data['label'] for data in data_list], dtype=torch.long)

        batch = torch.cat([torch.full((data['x'].shape[0],), i) for i, data in enumerate(data_list)], dim=0)
        batch = batch.long()

        return {
            'x': x,
            'pos': pos,
            'edge_index': edge_index,
            'batch': batch,
            'label': label
        }
    
    @staticmethod
    def load_events(file: str,
                    cfg):
        label = int(file.split('/')[-2])

        with open(file, 'rb') as fp:
            t, x, y, p = load_events(fp,
                        x_mask=0xfE,
                        x_shift=1,
                        y_mask=0x7f00,
                        y_shift=8,
                        polarity_mask=1,
                        polarity_shift=None)

            events = {'t': t, 'x': 127 - y, 'y': 127 - x, 'p': 1 - 2 * p.astype(int)}
        
        middle_t = len(events['t']) // 2
        middle_t = events['t'][middle_t]
        mask1 = events['t'] < middle_t + cfg.time_window // 2
        mask2 = events['t'] > middle_t - cfg.time_window // 2
        mask = mask1 & mask2
        events = {k: v[mask] for k, v in events.items()}

        events['t'] = events['t'] - events['t'][0]

        return events, label
    

############################################################################################################
# CIFAR10-DVS READER
############################################################################################################
    
EVT_DVS = 0  # DVS event type
EVT_APS = 1  # APS event

def read_bits(arr, mask=None, shift=None):
    if mask is not None:
        arr = arr & mask
    if shift is not None:
        arr = arr >> shift
    return arr


y_mask = 0x7FC00000
y_shift = 22

x_mask = 0x003FF000
x_shift = 12

polarity_mask = 0x800
polarity_shift = 11

valid_mask = 0x80000000
valid_shift = 31


def skip_header(fp):
    p = 0
    lt = fp.readline()
    ltd = lt.decode().strip()
    while ltd and ltd[0] == "#":
        p += len(lt)
        lt = fp.readline()
        try:
            ltd = lt.decode().strip()
        except UnicodeDecodeError:
            break
    return p


def load_raw_events(fp,
                    bytes_skip=0,
                    bytes_trim=0,
                    filter_dvs=False,
                    times_first=False):
    p = skip_header(fp)
    fp.seek(p + bytes_skip)
    data = fp.read()
    if bytes_trim > 0:
        data = data[:-bytes_trim]
    data = np.fromstring(data, dtype='>u4')
    if len(data) % 2 != 0:
        print(data[:20:2])
        print('---')
        print(data[1:21:2])
        raise ValueError('odd number of data elements')
    raw_addr = data[::2]
    timestamp = data[1::2]
    if times_first:
        timestamp, raw_addr = raw_addr, timestamp
    if filter_dvs:
        valid = read_bits(raw_addr, valid_mask, valid_shift) == EVT_DVS
        timestamp = timestamp[valid]
        raw_addr = raw_addr[valid]
    return timestamp, raw_addr


def parse_raw_address(addr,
                      x_mask=x_mask,
                      x_shift=x_shift,
                      y_mask=y_mask,
                      y_shift=y_shift,
                      polarity_mask=polarity_mask,
                      polarity_shift=polarity_shift):
    polarity = read_bits(addr, polarity_mask, polarity_shift).astype(np.bool_)
    x = read_bits(addr, x_mask, x_shift)
    y = read_bits(addr, y_mask, y_shift)
    return x, y, polarity


def load_events(
        fp,
        filter_dvs=False,
        **kwargs):
    timestamp, addr = load_raw_events(
        fp,
        filter_dvs=filter_dvs,
    )
    x, y, polarity = parse_raw_address(addr, **kwargs)
    return timestamp, x, y, polarity