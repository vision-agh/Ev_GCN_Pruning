import torch

def graph_gen_out(
    x: torch.Tensor,
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    cfg,
    name: str,
) -> dict:
    """
    Generate graph output for the model.
    
    Args:
        x (torch.Tensor): Input tensor.
        pos (torch.Tensor): Position tensor.
        edge_index (torch.Tensor): Edge index tensor.
        batch (torch.Tensor): Batch tensor.
        cfg (dict): Configuration dictionary.
    
    Returns:
        dict: Dictionary containing the graph output.
    """
    
    if cfg.debug:
        with open(name, 'w') as f:
                for idx, (features, position) in enumerate(zip(torch.flip(x, [1]), pos)):
                    f.write(str(features.to(torch.int32).tolist()) + " " + str(position.to(torch.int32).tolist()) + "\n")
                    neighbours = edge_index[:, 1][edge_index[:, 0] == idx]
                    for neighbour in neighbours:
                        f.write("     " + str(pos[neighbour.item()].to(torch.int32).tolist()) + "\n")


def conv_first_gen_out(
    x: torch.Tensor,
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    cfg,
    name: str,
) -> dict:
    """
    Generate convolution output for the model.
    
    Args:
        x (torch.Tensor): Input tensor.
        pos (torch.Tensor): Position tensor.
        edge_index (torch.Tensor): Edge index tensor.
        batch (torch.Tensor): Batch tensor.
        cfg (dict): Configuration dictionary.
    
    Returns:
        dict: Dictionary containing the convolution output.
    """
    if cfg.debug:
        with open(name, 'w') as f:
            for features, position in zip(torch.flip(x, [1]), pos):
                f.write(str(position.to(torch.int32).tolist()) + " " + str(features.to(torch.int32).tolist()) + "\n")

def conv_gen_out(
    x: torch.Tensor,
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    cfg,
    name: str,
) -> dict:
    """
    Generate convolution output for the model.
    
    Args:
        x (torch.Tensor): Input tensor.
        pos (torch.Tensor): Position tensor.
        edge_index (torch.Tensor): Edge index tensor.
        batch (torch.Tensor): Batch tensor.
        cfg (dict): Configuration dictionary.
    
    Returns:
        dict: Dictionary containing the convolution output.
    """
    if cfg.debug:
        with open(name, 'w') as f:
            indices = pos[:, 2] * 64 * 64 + pos[:, 1] * 64 + pos[:, 0]  # Make sure these multipliers scale correctly to preserve order
            sorted_indices = torch.argsort(indices)
            fliped_features = torch.flip(x, [1])
            for i, j in zip(pos[sorted_indices], fliped_features[sorted_indices]):
                f.write(str(i.to(torch.int32).tolist())+str(j.to(torch.int32).tolist())+"\n")


def events_out(events,
                cfg,
                name: str) -> dict:
    """
    Generate events output for the model.
    
    Args:
        events (torch.Tensor): Input tensor.
        cfg (dict): Configuration dictionary.
    
    Returns:
        dict: Dictionary containing the events output.
    """
    if cfg.debug:
        with open(name, 'w') as f:
            for x, y, t, p in zip(events['x'], events['y'], events['t'], events['p']):
                f.write(f"{x} {y} {t} {p}\n")


        