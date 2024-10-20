import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from timm.models.vision_transformer import Block
import omegaconf

from models.lavt_encoder import LAVTEncoder
from models.mask_predictor import MaskPredictor
from models.language_gate import LanguageGate
from models.pwam import PWAM


# class LavtLayer(nn.Module):
#     def __init__(self, c, args, dim, depth, num_heads, drop_path,
#                  downsample, num_heads_fusion, norm_layer):
#         super().__init__()
#         self.dim = dim
#         self.grad_ckpt = args.grad_checkpointing
#         self.vit_blocks = nn.ModuleList([
#             Block(dim, num_heads, c.mlp_ratio, qkv_bias=True,
#                   proj_drop=c.proj_drop, attn_drop=c.attn_drop,
#                   drop_path=drop_path[i], norm_layer=norm_layer) for i in range(depth)])
#         self.fusion = PWAM(dim, dim, 768, dim, dim, num_heads=num_heads_fusion, dropout=c.fusion_drop)
#         self.lang_gate = LanguageGate(dim)
#         if downsample is not None:
#             self.downsample = downsample(dim, norm_layer)
#         else:
#             self.downsample = None
#
#     def forward(self, x, H, W, l, l_mask):
#         """ Forward function.
#
#         Args:
#             x: Input feature, tensor size (B, H*W, C).
#             H, W: Spatial resolution of the input feature.
#         """
#         for blk in self.attn_blocks:
#             if self.grad_ckpt:
#                 x = checkpoint(blk, x)
#             else:
#                 x = blk(x)
#
#         # PWAM fusion
#         x_residual = self.fusion(x, l, l_mask)
#         # apply a gate on the residual
#         x = x + (self.res_gate(x_residual) * x_residual)
#
#         if self.downsample is not None:
#             x_down = self.downsample(x, H, W)
#             Wh, Ww = (H + 1) // 2, (W + 1) // 2
#             return x_residual, H, W, x_down, Wh, Ww
#         else:
#             return x_residual, H, W, x, H, W


class Lavt(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.norm_layer = nn.LayerNorm
        config = omegaconf.OmegaConf.load(f'configs/{args.model}.yaml')
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, sum(config.depths))]
        self.num_layers = len(config.depths)
        self.lavt_encoders = nn.ModuleList([
            LAVTEncoder(config, i, dpr[i], self.norm_layer) for i in range(self.num_layers)])
        self.lavt_decoders = nn.ModuleList([
            MaskPredictor(config, i) for i in range(self.num_layers - 1)])

        # self.layers = nn.ModuleList([
        #     LavtLayer(c.layer_configs, args, int(c.token_size * 2 ** i), depths[i], c.num_heads[i],
        #               dpr[sum(depths[:i]):sum(depths[:i + 1])], Downsample if (i < self.num_layers - 1) else None,
        #               c.num_heads_fusion[i], self.norm_layer) for i in range(self.num_layers)])


    def forward(self, x, l, l_mask):
        Wh, Ww = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)

        residuals = []
        for i in range(self.num_layers):
            x_residual, H, W, x, Wh, Ww = self.lavt_encoders[i](x, Wh, Ww, l, l_mask)
            x_residual = self.norm_layer(x_residual)  # output of a Block has shape (B, H*W, dim)

            x_residual = x_residual.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
            residuals.append(x_residual)

        x = residuals[-1]
        for i in range(self.num_layers - 1):
            x = self.lavt_decoders[i](x, residuals[i])

        return x
