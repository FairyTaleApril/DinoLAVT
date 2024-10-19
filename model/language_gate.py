import torch.nn as nn


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
