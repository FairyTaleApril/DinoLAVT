import torch.nn as nn
from timm.models.vision_transformer import Block

from models.language_gate import LanguageGate
from models.down_sample import DownSample
from models.pwam import PWAM


class LAVTEncoder(nn.Module):
    def __init__(self, args, config, layer_id, drop_path, norm_layer):
        super().__init__()
        self.dim = int(args.img_token_size * 2 ** layer_id)
        self.vit_blocks = nn.ModuleList([
            Block(self.dim, config.num_heads[layer_id], config.mlp_ratio, qkv_bias=True,
                  proj_drop=config.proj_drop, attn_drop=config.attn_drop,
                  drop_path=drop_path[i], norm_layer=norm_layer) for i in range(config.depths[layer_id])])
        self.fusion = PWAM(self.dim, self.dim, 768, self.dim, self.dim,
                           config.num_heads_fusion[layer_id], config.fusion_drop)
        self.lang_gate = LanguageGate(self.dim)
        self.down_sample = DownSample(self.dim, norm_layer) if layer_id < len(config.depths) - 1 else None

    def forward(self, x, H, W, l, l_mask):
        for blk in self.vit_blocks:
            x = blk(x)

        # PWAM fusion
        x_residual = self.fusion(x, l, l_mask)
        # apply a gate on the residual
        x = self.lang_gate(x, x_residual)

        if self.down_sample is not None:
            x_down = self.down_sample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x_residual, H, W, x_down, Wh, Ww
        else:
            return x_residual, H, W, x, H, W
