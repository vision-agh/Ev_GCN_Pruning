import torch
import torch.nn.utils.prune as prune

def structured_pruning(layer, amount=0.5):
    if hasattr(layer.linear, 'weight_mask'):
        prune.remove(layer.linear, 'weight')

    if hasattr(layer.norm, 'bias_mask'):
        prune.remove(layer.norm, 'bias')

    prune.ln_structured(layer.linear, name='weight', amount=amount, n=2, dim=0)
    
    mask = layer.linear.weight_mask.detach().clone()
    non_active_channels = mask.sum(dim=1) != 0

    prune.custom_from_mask(layer.norm, 'bias', mask=non_active_channels)
    prune.custom_from_mask(layer.norm, 'weight', mask=non_active_channels)