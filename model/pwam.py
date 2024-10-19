import torch
import torch.nn as nn
import torch.nn.functional as F


class VisionLanguageAttention(nn.Module):
    def __init__(self, v_in_channels, l_in_channels, key_channels, value_channels, num_heads=1):
        super(VisionLanguageAttention, self).__init__()

        # x shape: (B, H*W, v_in_channels)
        # l shape: (B, l_in_channels, N_l)
        # l_mask shape: (B, N_l, 1)
        self.v_in_channels = v_in_channels
        self.l_in_channels = l_in_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        self.num_heads = num_heads

        # Queries: visual features: (B, H*W, v_in_channels)
        self.compute_query = nn.Sequential(
            nn.Conv1d(self.v_in_channels, self.key_channels, kernel_size=1, stride=1),
            nn.InstanceNorm1d(self.key_channels))

        # Keys: language features: (B, l_in_channels, #words)
        # Avoid any form of spatial normalization because a sentence contains many padding 0s
        self.compute_key = nn.Conv1d(self.l_in_channels, self.key_channels, kernel_size=1, stride=1)

        # Values: language features: (B, l_in_channels, #words)
        self.compute_value = nn.Conv1d(self.l_in_channels, self.value_channels, kernel_size=1, stride=1)

        # Out projection
        self.out_project = nn.Sequential(
            nn.Conv1d(self.value_channels, self.out_channels, kernel_size=1, stride=1),
            nn.InstanceNorm1d(self.out_channels))

    def forward(self, x, l, l_mask):
        # x shape: (B, H*W, v_in_channels)
        # l shape: (B, l_in_channels, N_l)
        # l_mask shape: (B, N_l, 1)
        B, HW, n_l = x.size(0), x.size(1), l.size(-1)
        l_mask = l_mask.permute(0, 2, 1)  # (B, N_l, 1) -> (B, 1, N_l)

        query = self.compute_query(x.permute(0, 2, 1)).permute(0, 2, 1)  # (B, H*W, key_channels)
        key = self.compute_key(l) * l_mask  # (B, key_channels, N_l)
        value = self.compute_value(l) * l_mask  # (B, self.value_channels, N_l)

        # (b, num_heads, H*W, self.key_channels//self.num_heads)
        query = query.reshape(B, HW, self.num_heads, self.key_channels // self.num_heads).permute(0, 2, 1, 3)
        # (b, num_heads, self.key_channels//self.num_heads, N_l)
        key = key.reshape(B, self.num_heads, self.key_channels // self.num_heads, n_l)
        # (b, num_heads, self.value_channels//self.num_heads, N_l)
        value = value.reshape(B, self.num_heads, self.value_channels // self.num_heads, n_l)
        l_mask = l_mask.unsqueeze(1)  # (b, 1, 1, N_l)

        att_map = torch.matmul(query, key)  # (B, self.num_heads, H*W, N_l)
        att_map = (self.key_channels ** -0.5) * att_map  # Scaled dot product
        att_map = att_map + (1e4 * l_mask - 1e4)  # Assign a very small number to padding positions
        att_map = F.softmax(att_map, dim=-1)  # (B, num_heads, H*W, N_l)

        out = torch.matmul(att_map, value.permute(0, 1, 3, 2))  # (B, num_heads, H*W, self.value_channels//num_heads)
        out = out.permute(0, 2, 1, 3).contiguous().reshape(B, HW, self.value_channels)  # (B, H*W, value_channels)
        out = self.out_project(out.permute(0, 2, 1)).permute(0, 2, 1)  # (B, value_channels, H*W)

        return out


class PWAM(nn.Module):
    def __init__(self, dim, v_in_channels, l_in_channels, key_channels, value_channels, num_heads=0, dropout=0.0):
        super(PWAM, self).__init__()

        # x shape: (B, H*W, dim)
        self.vis_project = nn.Sequential(
            nn.Conv1d(dim, dim, 1, 1),
            nn.GELU(),
            nn.Dropout(dropout))

        self.vis_lang_att = VisionLanguageAttention(v_in_channels, l_in_channels, key_channels, value_channels, num_heads)

        self.out_project = nn.Sequential(
            nn.Conv1d(value_channels, value_channels, 1, 1),
            nn.GELU(),
            nn.Dropout(dropout))

    def forward(self, x, l, l_mask):
        # x shape: (B, H*W, dim)
        # l shape: (B, dim, N_l)
        vis = self.vis_project(x.permute(0, 2, 1))  # (B, dim, H*W)
        att = self.vis_lang_att(x, l, l_mask).permute(0, 2, 1)  # (B, dim, H*W)

        out = torch.mul(vis, att)
        out = self.out_project(out).permute(0, 2, 1)  # (B, H*W, dim)

        return out
