import torch.nn as nn


class LanguageGate(nn.Module):
    def __init__(self, dim):
        super(LanguageGate, self).__init__()

        self.res_gate = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.ReLU(),
            nn.Linear(dim, dim, bias=False),
            nn.Tanh())

    def forward(self, x, x_residual):
        return x + (self.res_gate(x_residual) * x_residual)
