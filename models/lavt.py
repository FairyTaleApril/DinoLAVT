import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from timm.models.vision_transformer import Block
import omegaconf

from models.networks import Downsample, LanguageGate
from models.pwam import PWAM


class LavtLayer(nn.Module):
    def __init__(self, c, args, dim, depth, num_heads, drop_path,
                 downsample, num_heads_fusion, norm_layer):
        super().__init__()
        self.dim = dim
        self.grad_ckpt = args.grad_checkpointing
        self.vit_blocks = nn.ModuleList([
            Block(dim, num_heads, c.mlp_ratio, qkv_bias=True,
                  proj_drop=c.proj_drop, attn_drop=c.attn_drop,
                  drop_path=drop_path[i], norm_layer=norm_layer
            )
            for i in range(depth)])

        self.fusion = PWAM(dim,  # both the visual input and for combining, num of channels
                           dim,  # v_in
                           768,  # l_in
                           dim,  # key
                           dim,  # value
                           num_heads=num_heads_fusion,
                           dropout=c.fusion_drop)

        self.lang_gate = LanguageGate(dim)

        if downsample is not None:
            self.downsample = downsample(dim, norm_layer)
        else:
            self.downsample = None

    def forward(self, x, H, W, l, l_mask):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        for blk in self.attn_blocks:
            if self.grad_ckpt:
                x = checkpoint(blk, x)
            else:
                x = blk(x)

        # PWAM fusion
        x_residual = self.fusion(x, l, l_mask)
        # apply a gate on the residual
        x = x + (self.res_gate(x_residual) * x_residual)

        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x_residual, H, W, x_down, Wh, Ww
        else:
            return x_residual, H, W, x, H, W

class Lavt(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.norm_layer = nn.LayerNorm
        model_cfg = f'configs/{args.model}.yaml'
        c = omegaconf.OmegaConf.load(model_cfg)
        depths = c.depths
        self.num_layers = len(depths)
        dpr = [x.item() for x in torch.linspace(0, c.drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList([
            LavtLayer(
                c=c.layer_configs,
                args=args,
                dim=int(c.token_size * 2 ** i),
                depth=depths[i],
                num_heads=c.num_heads[i],
                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                downsample=Downsample if (i < self.num_layers - 1) else None,
                num_heads_fusion=c.num_heads_fusion[i],
                norm_layer=self.norm_layer
            )
            for i in range(self.num_layers)])

    def forward(self, x, l, l_mask):
        Wh, Ww = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)

        outs = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww, l, l_mask)
            x_out = self.norm_layer(x_out)  # output of a Block has shape (B, H*W, dim)

            out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
            outs.append(out)

        return tuple(outs)