import torch

from torch import nn
from torch import Tensor

from src.embedding import create_latent_array
from src.fourier_encode import encoded_position
from src.layer import PerceiverBlock, Classifier


class Perceiver(nn.Module):

    def __init__(
            self,
            dim: int,
            depth: int,
            latent_blocks: int,
            latent_dim: int,
            heads: int,
            num_classes: int,
            embed_dim: int,
            fourier_encode: bool,
            max_freq: int,
            num_bands: int,
            batch_size: int
    ):
        """
        Perceiver model

        :param dim:
        :param depth:
        :param latent_blocks:
        :param latent_dim:
        :param heads:
        :param num_classes:
        :param embed_dim:
        :param fourier_encode:
        :param max_freq:
        :param num_bands:
        :param batch_size:
        """
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.latent_blocks = latent_blocks
        self.heads = heads
        self.latent_dim = latent_dim
        self.embed_dim = embed_dim
        self.fourier_encode = fourier_encode
        self.max_freq = max_freq
        self.num_bands = num_bands
        self.num_classes = num_classes
        self.batch_size = batch_size

        # The latent array
        self.latent = create_latent_array(self.batch_size, self.latent_dim, self.dim)

        # Perceiver block -> @TODO Share the weights for every block
        self.layers = nn.ModuleList([
            PerceiverBlock(dim, heads, latent_blocks)
            for _ in range(depth)
        ])

        # Classifier
        self.classifier = Classifier(dim, num_classes)

    def forward(self, x: Tensor):
        batch, *dims, channels = x.shape

        # Positional encoding
        if self.fourier_encode:
            x = encoded_position(x, dims, self.batch_size, self.max_freq, self.num_bands, x.device, x.dtype)

        x = x.view(batch, torch.prod(torch.tensor(dims)).item(), channels)

        # Repeat the same latent array for all batch_size
        self.latent = self.latents.unsqueeze(0).expand(batch, -1, -1)

        # Compute the perceiver block
        # @TODO Share the weights for every block
        for layer in self.layers:
            x = layer(x, self.latent)

        # Classifier
        x = self.classifier(x)
        return x
