import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import to_2tuple

class Downsample(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

class LanguageGate(nn.Module):
    def __init__(self, dim):
        """
        Initializes the LanguageGate module.

        Args:
            dim (int): The dimensionality of the input and output features.
        """
        super(LanguageGate, self).__init__()

        self.lang_gate = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.ReLU(),
            nn.Linear(dim, dim, bias=False),
            nn.Tanh())

    def forward(self, x, x_residual):
        """
        Defines the forward pass of the LanguageGate module.

        Args:
            x (batch_size, dim): The input tensor.
            x_residual (batch_size, dim): The residual tensor.

        Returns:
            torch.Tensor (batch_size, dim): The sum of the input tensor and the gated residual tensor.
        """
        return x + (self.lang_gate(x_residual) * x_residual)
