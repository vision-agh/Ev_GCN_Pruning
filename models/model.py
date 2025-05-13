import torch
import torch.nn as nn

from models.layers.my_max_pool import MyGraphPooling
from models.layers.my_pool_out import MyGraphPoolOut
from models.layers.my_pool_out_2d import MyGraphPoolOut2D
from models.layers.my_pointnet import MyPointNetConv
from models.layers.my_linear import MyLinear


class MyModel(nn.Module):
    def __init__(self, cfg):
        super(MyModel, self).__init__()
        self.cfg = cfg

        self.conv1 = MyPointNetConv(3, 16, False, cfg.num_bits, True)

        self.pool1 = MyGraphPooling([6, 5, 5])

        self.conv2 = MyPointNetConv(18, 32, False, cfg.num_bits, False)
        self.conv3 = MyPointNetConv(34, 32, False, cfg.num_bits, False)

        self.pool2 = MyGraphPooling([2, 2, 2])

        self.conv4 = MyPointNetConv(34, 32, False, cfg.num_bits, False)
        self.conv5 = MyPointNetConv(34, 32, False, cfg.num_bits, False)

        self.pool_out = MyGraphPoolOut2D(2, max_dimension=10)

        # Linear layers
        self.linear1 = MyLinear(input_dim=32 * 5 ** 2,
                                output_dim=2,
                                bias=True,
                                num_bits=cfg.num_bits)
        

        # Modes for calibration and quantization
        self.register_buffer('calib_mode', torch.tensor(False, requires_grad=False))
        self.register_buffer('quantize_mode', torch.tensor(False, requires_grad=False))
    
    def forward(self, x: torch.Tensor,
                pos: torch.Tensor,
                edge_index: torch.Tensor,
                batch: torch.Tensor):
        
        '''Forward pass of the model.'''
        x = self.conv1(x, pos[:,:2], edge_index)
        x, pos, edge_index, batch = self.pool1(x, pos, edge_index, batch)

        x = self.conv2(x, pos[:,:2], edge_index)
        x = self.conv3(x, pos[:,:2], edge_index)
        x, pos, edge_index, batch = self.pool2(x, pos, edge_index, batch)

        x = self.conv4(x, pos[:,:2], edge_index)
        x = self.conv5(x, pos[:,:2], edge_index)

        x = self.pool_out(x, pos, batch)

        x = self.linear1(x)

        if self.quantize_mode.item():
            x = self.linear1.observer_output.dequantize_tensor(x)
        return x
    
    def calibrate(self):
        '''Calibrate the model.'''
        self.calib_mode.fill_(True)

        self.conv1.calibrate()
        self.conv2.calibrate()
        self.conv3.calibrate()
        self.conv4.calibrate()
        self.conv5.calibrate()
        self.pool_out.calibrate()
        self.linear1.calibrate()

    def quantize(self):
        '''Quantize the model.'''
        self.quantize_mode.fill_(True)

        self.conv1.quantize()
        self.conv2.quantize(observer_input=self.conv1.observer_output)
        self.conv3.quantize(observer_input=self.conv2.observer_output)
        self.conv4.quantize(observer_input=self.conv3.observer_output)
        self.conv5.quantize(observer_input=self.conv4.observer_output)
        self.pool_out.quantize(observer_input=self.conv5.observer_output)
        self.linear1.quantize(observer_input=self.conv5.observer_output)