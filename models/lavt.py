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
        self.img_embed_model = args.img_embed_model

        config = omegaconf.OmegaConf.load(f'configs/{args.model}.yaml')
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, sum(config.depths))]
        self.num_layers = len(config.depths)

        self.lavt_encoders = nn.ModuleList([
            LAVTEncoder(args, config, i, dpr, nn.LayerNorm) for i in range(self.num_layers)])
        self.lavt_decoders = nn.ModuleList([
            MaskPredictor(self.lavt_encoders[-(i + 1)].dim, i == (self.num_layers - 2)) for i in range(self.num_layers - 1)])

        self.pos_drop = nn.Dropout(config.drop_rate)
        self.num_features = [int(args.img_token_size * 2 ** i) for i in range(self.num_layers)]
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(num) for num in self.num_features
        ])

        self.patch_embed = PatchEmbed() if self.img_embed_model == 'patch_embed' else None

    def forward(self, x, l, l_mask, img):
        if self.img_embed_model == 'patch_embed':
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
