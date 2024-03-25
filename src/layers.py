import torch
from torch import nn, Tensor


class LatentArray(nn.Module):
    def __init__(self, dim: int, latent_dim: int):
        """
        Latent array

        :param dim:
        :param latent_dim:
        """
        super().__init__()
        self.dim = dim
        self.latent_dim = latent_dim

        # The latent array is randomly initialized using a truncated normal distribution with
        # mean 0, standard deviation 0.02, and truncation bounds [-2, 2].
        self.latent = nn.Parameter(torch.nn.init.trunc_normal_(
            torch.zeros(self.latent_dim, self.dim),
            mean=0,
            std=0.02,
            a=-2, b=2)
        )

    def forward(self) -> Tensor:
        return self.latent


class DenseBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0):
        """
        Dense block

        :param dim:
        :param dropout:
        """
        super().__init__()
        self.dim = dim
        self.dropout = dropout
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(dim, dim)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class AttentionBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float = 0.0):
        """
        Attention block

        :param dim:
        :param heads:
        :param dropout:
        """
        super().__init__()
        self.dim = dim
        self.num_heads = heads
        self.dropout = dropout

        # Normalization before the cross attention
        self.latent_norm = nn.LayerNorm(dim)
        self.input_norm = nn.LayerNorm(dim)

        # Cross attention
        self.attention = nn.MultiheadAttention(
            dim,
            heads,
            dropout=0.0,
            bias=False
        )

        # Project the output of the cross attention
        self.proj = nn.Linear(dim, dim)

        # Dense layer
        self.dense = DenseBlock(dim, dropout=dropout)

    def forward(self, x: Tensor, z: Tensor) -> Tensor:
        """
        Forward pass

        :param x: input tensor
        :param z: input/latent tensor
        :return:
        """
        # Normalize the input
        x_norm = self.input_norm(x)
        z_norm = self.latent_norm(z)

        # Compute the cross attention
        a, _ = self.attention(z_norm, x_norm, x_norm)

        # Project the output of the cross attention
        a = self.proj(a)

        # Compute dense layer
        a = self.dense(a)

        return a + z


class PerceiverBlock(nn.Module):
    def __init__(self, dim: int, heads: int, latent_blocks: int = 6, dropout: float = 0.0):
        """
        Perceiver block
        """
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.latent_blocks = latent_blocks
        self.dropout = dropout

        # Cross attention
        self.cross_attentions = AttentionBlock(self.dim, 1, dropout=self.dropout)

        # Latent transformer
        self.latent_transform = nn.ModuleList([
            AttentionBlock(self.dim, self.heads, dropout=self.dropout)
            for _ in range(self.latent_blocks)
        ])

    def forward(self, x: Tensor, z: Tensor) -> Tensor:
        """
        Forward pass

        :param x: input tensor
        :param z: latent tensor
        :return:
        """
        z = self.cross_attention(x, z)
        for latent_transform in self.latent_transform:
            z = latent_transform(z, z)
        return z


class Classifier(nn.Module):
    def __init__(self, dim: int, num_classes: int):
        """
        Classifier

        :param dim:
        :param num_classes:
        """
        super().__init__()
        self.dim = dim
        self.num_classes = num_classes

        # Classifier
        self.classifier = nn.Linear(dim, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass

        :param x:
        :return:
        """
        return self.classifier(x)