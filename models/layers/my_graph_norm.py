from typing import Optional

import torch
from torch import Tensor


class GraphNorm(torch.nn.Module):
    r"""GraphNorm implemented *purely* with PyTorch,
    using :pymeth:`torch.Tensor.scatter_reduce_` (no PyG `scatter` helper).

    Reference:
        *GraphNorm: A Principled Approach to Accelerating Graph Neural
        Network Training* (Zhao **et al.**, NeurIPS 2020)
        <https://arxiv.org/abs/2009.03294>

    Args:
        in_channels (int): Size of each input feature vector.
        eps (float, optional): Small constant for numerical stability.
            (default: :obj:`1e-5`)
    """

    def __init__(self, in_channels: int, eps: float = 1e-5):
        super().__init__()
        self.in_channels = in_channels
        self.eps = eps

        self.weight = torch.nn.Parameter(torch.ones(in_channels))
        self.bias = torch.nn.Parameter(torch.zeros(in_channels))
        self.mean_scale = torch.nn.Parameter(torch.ones(in_channels))

    # --------------------------------------------------------------------- #
    # Main logic – **only** torch / scatter_reduce below
    # --------------------------------------------------------------------- #

    def reset_parameters(self):
        r"""Reinitialize learnable parameters."""
        self.weight.data.fill_(1)
        self.bias.data.fill_(0)
        self.mean_scale.data.fill_(1)
        self.eps = 1e-5

    def forward(
        self,
        x: Tensor,
        batch: Tensor = None,
        batch_size: Optional[int] = None,
    ) -> Tensor:
        r"""Normalize batched node features.

        Args:
            x (Tensor): Node features :math:`\mathbf{X} \in \mathbb{R}^{N\times C}`.
            batch (LongTensor, optional): Batch vector
                :math:`\mathbf{b}\in \{0,\dots,B-1\}^N`. If :obj:`None`,
                all nodes belong to a single graph. (default: :obj:`None`)
            batch_size (int, optional): Number of graphs :math:`B`.
                Computed automatically if omitted. (default: :obj:`None`)
        """
        if batch is None:                                   # single-graph case
            batch = x.new_zeros(x.size(0), dtype=torch.long)
            batch_size = 1

        if batch_size is None:                              # infer B
            batch_size = int(batch.max()) + 1

        C = x.size(1)                                       # feature dim

        # ---- per-graph mean ------------------------------------------------
        index = batch.unsqueeze(-1).expand(-1, C)           # (N,C) graph ids
        mean = torch.zeros(batch_size, C,
                           dtype=x.dtype, device=x.device)
        mean.scatter_reduce_(0, index, x,
                             reduce='mean', include_self=False)

        # ---- center & rescale ---------------------------------------------
        centered = x - mean.index_select(0, batch) * self.mean_scale

        # ---- per-graph variance (of centered features) --------------------
        var = torch.zeros(batch_size, C,
                          dtype=x.dtype, device=x.device)
        var.scatter_reduce_(0, index, centered.pow(2),
                            reduce='mean', include_self=False)

        std = (var + self.eps).sqrt().index_select(0, batch)

        # ---- affine transform ---------------------------------------------
        return self.weight * centered / std + self.bias