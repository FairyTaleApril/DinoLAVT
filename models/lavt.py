import omegaconf
import torch
import torch.nn as nn
from torch.nn import functional as F

from models.lavt_encoder import LAVTEncoder
from models.mask_predictor import MaskPredictor
from models.patch_embed import PatchEmbed


class Lavt(nn.Module):
    def __init__(self, args):
        super().__init__()
        norm_layer = nn.LayerNorm
        config = omegaconf.OmegaConf.load(f'configs/{args.model}.yaml')
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, sum(config.depths))]
        self.num_layers = len(config.depths)
        self.lavt_encoders = nn.ModuleList([
            LAVTEncoder(config, i, dpr, norm_layer) for i in range(self.num_layers)])
        self.lavt_decoders = nn.ModuleList([
            MaskPredictor(self.lavt_encoders[-(i + 1)].dim, i == (self.num_layers - 2)) for i in range(self.num_layers - 1)])

        self.pos_drop = nn.Dropout(config.drop_rate)
        self.num_features = [int(config.token_size * 2 ** i) for i in range(self.num_layers)]
        self.norm_layers = nn.ModuleList([
            norm_layer(num) for num in self.num_features
        ])

        self.patch_embed = PatchEmbed()

        # self.layers = nn.ModuleList([
        #     LavtLayer(c.layer_configs, args, int(c.token_size * 2 ** i), depths[i], c.num_heads[i],
        #               dpr[sum(depths[:i]):sum(depths[:i + 1])], Downsample if (i < self.num_layers - 1) else None,
        #               c.num_heads_fusion[i], self.norm_layer) for i in range(self.num_layers)])

    def forward(self, x, l, l_mask, img):
        x = self.patch_embed(img)
        Wh, Ww = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)

        residuals = []
        for i in range(self.num_layers):
            x_residual, H, W, x, Wh, Ww = self.lavt_encoders[i](x, Wh, Ww, l, l_mask)
            x_residual = self.norm_layers[i](x_residual)  # output of a Block has shape (B, H*W, dim)

            x_residual = x_residual.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
            residuals.append(x_residual)

        x = residuals[-1]
        for i in range(self.num_layers - 1):
            x = self.lavt_decoders[i](x, residuals[-(i + 2)])

        input_shape = img.shape[-2:]
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=True)

        return x
