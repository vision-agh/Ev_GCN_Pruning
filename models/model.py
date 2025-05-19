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

        self.conv1 = MyPointNetConv(cfg.conv1.in_channels,
                                    cfg.conv1.out_channels, 
                                    cfg.conv1.bias, 
                                    cfg.conv1.num_bits,
                                    first_layer=True)

        self.pool1 = MyGraphPooling(pool_size=cfg.pool1.grid)

        self.conv2 = MyPointNetConv(cfg.conv2.in_channels,
                                    cfg.conv2.out_channels, 
                                    cfg.conv2.bias, 
                                    cfg.conv2.num_bits,
                                    first_layer=False)
        
        self.conv3 = MyPointNetConv(cfg.conv3.in_channels,
                                    cfg.conv3.out_channels, 
                                    cfg.conv3.bias, 
                                    cfg.conv3.num_bits,
                                    first_layer=False)

        self.pool2 = MyGraphPooling(pool_size=cfg.pool2.grid)

        self.conv4 = MyPointNetConv(cfg.conv4.in_channels,
                                    cfg.conv4.out_channels, 
                                    cfg.conv4.bias, 
                                    cfg.conv4.num_bits,
                                    first_layer=False)
        
        self.conv5 = MyPointNetConv(cfg.conv5.in_channels,
                                    cfg.conv5.out_channels, 
                                    cfg.conv5.bias, 
                                    cfg.conv5.num_bits,
                                    first_layer=False)

        self.pool_out = MyGraphPoolOut2D(pool_size=cfg.pool_out.pool_size, 
                                         max_dimension=cfg.pool_out.max_dim)

        # Linear layers
        self.linear1 = MyLinear(input_dim=cfg.linear1.in_features,
                                output_dim=cfg.linear1.out_features,
                                bias=cfg.linear1.bias,
                                num_bits=cfg.linear1.num_bits)
        

        self.linear2 = MyLinear(input_dim=cfg.linear2.in_features,
                                output_dim=cfg.linear2.out_features,
                                bias=cfg.linear2.bias,
                                num_bits=cfg.linear2.num_bits)

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
        x = torch.relu(x)
        # x = torch.dropout(x, p=0.2, train=self.training)
        x = self.linear2(x)

        if self.quantize_mode.item():
            x = self.linear1.observer_output.dequantize_tensor(x)
        return torch.log_softmax(x, dim=-1)
    
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