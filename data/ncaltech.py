import os
import glob
import numpy as np
import torch
import lightning as L

from torch.utils.data import DataLoader
from data.base.event_ds import EventDS

class NCaltech(L.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        
    def setup(self, stage=None):
        all_files = glob.glob(os.path.join(self.cfg.data_dir,
                                            self.cfg.data_name,
                                            'events',
                                            '*',
                                            '*.bin'))
        

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
        label = file.split('/')[-2]
        label = ncaltech_dict[label]

        f = open(file, 'rb')
        raw_data = np.fromfile(f, dtype=np.uint8)
        f.close()
        raw_data = np.uint32(raw_data)
        all_y = raw_data[1::5]
        all_x = raw_data[0::5]
        all_p = (raw_data[2::5] & 128) >> 7
        all_ts = ((raw_data[2::5] & 127) << 16) | (raw_data[3::5] << 8) | (raw_data[4::5])
        all_ts = all_ts  
        all_p = all_p.astype(np.float64)
        all_p[all_p == 0] = -1
        
        events = {}
        events['x'] = all_x
        events['y'] = all_y
        events['t'] = all_ts
        events['p'] = all_p
        
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

ncaltech_dict = {
    'ferry': 0,
    'umbrella': 1,
    'cougar_face': 2,
    'starfish': 3,
    'binocular': 4,
    'stegosaurus': 5,
    'trilobite': 6,
    'buddha': 7,
    'snoopy': 8,
    'minaret': 9,
    'stop_sign': 10,
    'mandolin': 11,
    'cellphone': 12,
    'camera': 13,
    'gerenuk': 14,
    'metronome': 15,
    'tick': 16,
    'octopus': 17,
    'pizza': 18,
    'scorpion': 19,
    'dolphin': 20,
    'chair': 21,
    'cougar_body': 22,
    'flamingo_head': 23,
    'brontosaurus': 24,
    'crocodile_head': 25,
    'soccer_ball': 26,
    'headphone': 27,
    'accordion': 28,
    'mayfly': 29,
    'beaver': 30,
    'pagoda': 31, 
    'cannon': 32,
    'euphonium': 33,
    'helicopter': 34,
    'chandelier': 35,
    'stapler': 36,
    'revolver': 37,
    'airplanes': 38,
    'wheelchair': 39,
    'pigeon': 40,
    'crayfish': 41,
    'llama': 42,
    'kangaroo': 43,
    'strawberry': 44,
    'watch': 45,
    'hawksbill': 46,
    'dragonfly': 47,
    'butterfly': 48,
    'dollar_bill': 49,
    'pyramid': 50,
    'inline_skate': 51,
    'nautilus': 52,
    'rhino': 53,
    'yin_yang': 54,
    'crocodile': 55,
    'wrench': 56,
    'crab': 57,
    'lamp': 58,
    'flamingo': 59,
    'schooner': 60,
    'panda': 61,
    'water_lilly': 62,
    'ceiling_fan': 63,
    'car_side': 64,
    'grand_piano': 65,
    'joshua_tree': 66,
    'dalmatian': 67,
    'cup': 68,
    'platypus': 69,
    'menorah': 70,
    'brain': 71,
    'Motorbikes': 72,
    'Faces_easy': 73,
    'saxophone': 74,
    'windsor_chair': 75,
    'sea_horse': 76,
    'sunflower': 77,
    'scissors': 78,
    'Leopards': 79,
    'laptop': 80,
    'ibis': 81,
    'anchor': 82,
    'okapi': 83,
    'ketch': 84,
    'wild_cat': 85,
    'rooster': 86,
    'barrel': 87,
    'elephant': 88,
    'gramophone': 89,
    'emu': 90,
    'garfield': 91,
    'lotus': 92,
    'bonsai': 93,
    'bass': 94,
    'ewer': 95,
    'ant': 96,
    'electric_guitar': 97,
    'hedgehog': 98,
    'lobster': 99
}