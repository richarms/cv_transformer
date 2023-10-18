import torch
import torch.nn as nn
from torch.nn import functional as F

from typing import Optional

class ComplexLayerNorm(nn.Module):
    """ Complex LayerNorm with an optional bias."""

    def __init__(self, ndim: int, bias: Optional):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
