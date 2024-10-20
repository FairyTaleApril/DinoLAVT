import torch
from torch import nn
from torch.nn import functional as F


class MaskPredictor(nn.Module):
    def __init__(self, config, layer_id, factor=2):
        super(MaskPredictor, self).__init__()
        num_layers = len(config.depths)
        in_feat_dim = int(config.token_size * 2 ** (num_layers - 1 - layer_id))
        hidden_size = in_feat_dim // factor

        self.seq = nn.Sequential(
            nn.Conv2d(hidden_size + in_feat_dim, hidden_size, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(),
            nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU()
        )
        self.conv = nn.Conv2d(hidden_size, 2, 1) if layer_id < num_layers - 2 else nn.Identity()

    def forward(self, x, f):
        """
        Forward pass for the MaskPredictor module.

        Args:
            x (B, C, H, W): Input tensor.
            f (B, C', H', W'): Feature tensor to concatenate with x.

        Returns:
            torch.Tensor: Output tensor after processing, shape depends on is_last:
                          - If is_last is True, shape will be (B, 2, H'', W'').
                          - If is_last is False, shape will be (B, hidden_size, H'', W'').
        """
        if x.size(-2) < f.size(-2) or x.size(-1) < f.size(-1):
            x = F.interpolate(input=x, size=(f.size(-2), f.size(-1)), mode='bilinear', align_corners=True)
        x = torch.cat([x, f], dim=1)
        x = self.seq(x)
        x = self.conv(x)
        return x
