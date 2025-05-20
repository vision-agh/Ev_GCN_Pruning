import torch
from torch.nn import Module

from models.quantisation.observer import Observer, FakeQuantize

class MyGraphPoolOut2D(Module):
    def __init__(self, 
                 pool_size: int = 4, 
                 max_dimension: int = 256,
                 num_bits:int = 8):
        
        super(MyGraphPoolOut2D, self).__init__()

        self.pool_size = pool_size
        self.max_dimension = max_dimension
        self.grid_size = max_dimension // pool_size

        '''Modes for calibration and quantization'''

        self.register_buffer('calib_mode', torch.tensor(False, requires_grad=False))
        self.register_buffer('quantize_mode', torch.tensor(False, requires_grad=False))

        '''Initialize quantization observers for input, weight and output tensors.'''
        self.observer_input = Observer(num_bits=num_bits)
        self.num_bits = num_bits

    def forward(self,
                x: torch.Tensor,            # [N, F]
                pos: torch.Tensor,          # [N, 3]
                batch: torch.Tensor         # [N]
                ):
        
        if self.calib_mode.item() and not self.quantize_mode.item():
            return self.forward_calib(x, pos, batch)
        elif self.quantize_mode.item():
            return self.forward_quant(x, pos, batch)
        elif not self.calib_mode.item() and not self.quantize_mode.item():
            return self.forward_float(x, pos, batch)
        else:
            raise ValueError('Invalid mode')
        
    def forward_float(self, 
                x: torch.Tensor, 
                pos: torch.Tensor,
                batch: torch.Tensor):
        
        max_batch = batch.max() + 1
        qpos = torch.div(pos[:,:2], self.pool_size, rounding_mode='floor').long()
        key = torch.cat([batch.unsqueeze(1), qpos], dim=1)

        unique_keys, inv = torch.unique(key, dim=0, return_inverse=True)
        new_batch = unique_keys[:, 0]
        uniq_qpos = unique_keys[:, 1:]

        pooled_x = torch.zeros((uniq_qpos.size(0), x.size(1)), dtype=x.dtype, device=x.device)
        output_x = torch.zeros((max_batch, self.grid_size ** 2, x.size(1)), dtype=x.dtype, device=x.device)

        pooled_x = pooled_x.scatter_reduce(0, inv.unsqueeze(1).expand(-1, x.size(1)), x, reduce="amax", include_self=False)
        indices_1d = uniq_qpos[:, 0] * self.grid_size + uniq_qpos[:, 1]
        
        output_x[new_batch, indices_1d] = pooled_x
        output_x = output_x.flatten(start_dim=1)
        return output_x
    
    def forward_calib(self, 
                x: torch.Tensor, 
                pos: torch.Tensor,
                batch: torch.Tensor):
        
        max_batch = batch.max() + 1
        qpos = torch.div(pos[:,:2], self.pool_size, rounding_mode='floor').long()
        key = torch.cat([batch.unsqueeze(1), qpos], dim=1)

        unique_keys, inv = torch.unique(key, dim=0, return_inverse=True)
        new_batch = unique_keys[:, 0]
        uniq_qpos = unique_keys[:, 1:]

        pooled_x = torch.zeros((uniq_qpos.size(0), x.size(1)), dtype=x.dtype, device=x.device)
        output_x = torch.zeros((max_batch, self.grid_size ** 2, x.size(1)), dtype=x.dtype, device=x.device)

        pooled_x = pooled_x.scatter_reduce(0, inv.unsqueeze(1).expand(-1, x.size(1)), x, reduce="amax", include_self=False)
        indices_1d = uniq_qpos[:, 0] * self.grid_size + uniq_qpos[:, 1]
        
        output_x[new_batch, indices_1d] = pooled_x
        output_x = output_x.flatten(start_dim=1)
        return output_x
    
    def calibrate(self):
        self.calib_mode.fill_(True)

    def quantize(self,
               observer_input: Observer = None,
               observer_output: Observer = None):
        
        self.quantize_mode.fill_(True)
        '''Freeze model - quantize weights/bias and calculate scales'''
        if observer_input is not None:
            self.observer_input = observer_input

    def forward_quant(self, 
                x: torch.Tensor, 
                pos: torch.Tensor,
                batch: torch.Tensor):
        
        max_batch = batch.max() + 1
        qpos = torch.div(pos[:,:2], self.pool_size, rounding_mode='floor').long()
        key = torch.cat([batch.unsqueeze(1), qpos], dim=1)

        unique_keys, inv = torch.unique(key, dim=0, return_inverse=True)
        new_batch = unique_keys[:, 0]
        uniq_qpos = unique_keys[:, 1:]

        pooled_x = torch.zeros((uniq_qpos.size(0), x.size(1)), dtype=x.dtype, device=x.device)
        output_x = torch.zeros((max_batch, self.grid_size ** 2, x.size(1)), dtype=x.dtype, device=x.device) + self.observer_input.zero_point

        pooled_x = pooled_x.scatter_reduce(0, inv.unsqueeze(1).expand(-1, x.size(1)), x, reduce="amax", include_self=False)
        indices_1d = uniq_qpos[:, 0] * self.grid_size + uniq_qpos[:, 1]
        
        output_x[new_batch, indices_1d] = pooled_x
        output_x = output_x.flatten(start_dim=1)
        return output_x
    
    def __repr__(self):
        return f"{self.__class__.__name__}(pool_size={self.pool_size}, max_dimension={self.max_dimension})"