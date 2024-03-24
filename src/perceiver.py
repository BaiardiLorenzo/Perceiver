import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np


class PerceiverBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        """
        Perceiver block

        :param dim:
        :param num_heads:
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        # Normalization before the cross attention
        self.latent_norm = nn.LayerNorm(dim)
        self.input_norm = nn.LayerNorm(dim)

        # Cross attention
        self.attention = nn.MultiheadAttention(
            dim,
            num_heads,
            dropout=0.0,
            bias=False
        )

        # Project the output of the cross attention
        self.proj = nn.Linear(dim, dim)

        # Dense layer
        self.dense = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim, dim)
        )

    def forward(self, input: Tensor, latent: Tensor) -> Tensor:
        # Normalize the input
        latent_norm = self.latent_norm(latent)
        input_norm = self.input_norm(input)

        # Compute the cross attention
        attention = self.attention(latent_norm, input_norm, input_norm)

        # Project the output of the cross attention
        proj = self.proj(attention)

        # Compute dense layer
        dense = self.dense(proj)

        return dense + latent


class Perceiver(nn.Module):

    def __init__(
            self,
            dim: int,
            depth: int,
            latent_dim: int,
            heads: int,
            latent_blocks: int,
    ):
        """
        Perceiver model

        :param depth:
        :param latent_dim:
        :param dim:
        """
        super().__init__()
        self.depth = depth
        self.dim = dim
        self.latent_dim = latent_dim
        self.heads = heads

        # The latent array is randomly initialized using a truncated normal distribution with
        # mean 0, standard deviation 0.02, and truncation bounds [-2, 2].
        self.latent = nn.Parameter(torch.nn.init.trunc_normal_(
            torch.zeros(self.latent_dim, self.dim),
            mean=0,
            std=0.02,
            a=-2, b=2)
        )

        self.cross_attentions = PerceiverBlock(dim, num_heads=1)
        self.latent_transform = nn.ModuleList([
           PerceiverBlock(dim, num_heads=self.heads) for _ in range(latent_blocks)
        ])

    def forward(self, x: Tensor):
        # Positional encoding
        x = fourier_encode(x, max_freq=10, num_bands=4)

        for _ in range(self.depth):
            self.latent = self.cross_attentions(x, self.latent)
            for block in self.latent_transform:
                self.latent = block(self.latent, self.latent)
        return

