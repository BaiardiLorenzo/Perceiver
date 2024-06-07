import torch

from torch import nn
from torch import Tensor

from src.embedding import create_latent_array
from src.fourier_encode import positional_encoding
from src.layer import PerceiverBlock, Classifier


class Perceiver(nn.Module):

    def __init__(
            self,
            input_dim: int,
            len_shape: int,
            emb_dim: int,
            latent_dim: int,
            batch_size: int,
            num_classes: int,
            depth: int,
            latent_blocks: int,
            heads: int,
            fourier_encode: bool,
            max_freq: int,
            num_bands: int
    ):
        """
        Perceiver model

        :param input_dim: The channel dimension of the input tensor
        :param len_shape: The length of the shape of the input tensor
        :param emb_dim: The channel dimension of the latent tensor
        :param latent_dim: The latent dimension
        :param batch_size: The batch size
        :param num_classes: The number of classes for the classification task
        :param depth: The number of perceiver blocks
        :param latent_blocks: The number of latent blocks in the perceiver block
        :param heads: The number of heads in the multi-head attention in the perceiver block
        :param fourier_encode: Whether to use Fourier encoding
        :param max_freq: The maximum frequency for Fourier encoding
        :param num_bands: The number of bands for Fourier encoding
        """
        super().__init__()
        self.input_dim = input_dim
        self.len_shape = len_shape
        self.emb_dim = emb_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.depth = depth
        self.latent_blocks = latent_blocks
        self.heads = heads
        self.fourier_encode = fourier_encode
        self.max_freq = max_freq
        self.num_bands = num_bands

        # The latent array
        self.latent = create_latent_array(self.latent_dim, self.batch_size, self.emb_dim)

        if fourier_encode:
            self.input_dim = (self.len_shape * (num_bands * 2 + 1)) + self.input_dim

        # Perceiver block -> @TODO Share the weights for every block
        self.layers = nn.ModuleList([
            PerceiverBlock(self.emb_dim, self.input_dim, heads, latent_blocks)
            for _ in range(self.depth)
        ])

        # Classifier
        self.classifier = Classifier(self.emb_dim, self.num_classes)

    def forward(self, x: Tensor):
        _, _, *dims = x.shape

        # Positional encoding
        if self.fourier_encode:
            x = positional_encoding(x, dims, self.batch_size, self.max_freq, self.num_bands)

        # Flatten the input tensor
        x = x.view(x.shape[0], x.shape[1], -1)

        # Permute the input tensor to the shape (M, B, INPUT_DIM)
        xx = x.permute(2, 0, 1)

        # Compute the perceiver block
        # TODO Share the weights for every block
        for layer in self.layers:
            x = layer(xx, self.latent)
            

        # Classifier
        x = self.classifier(x)
        return x
