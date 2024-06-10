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
            emb_dim: int = 512,
            latent_dim: int = 512,
            batch_size: int = 64,
            num_classes: int = 40,
            depth: int = 2,
            latent_blocks: int = 6,
            heads: int = 8,
            fourier_encode: bool = True,
            max_freq: int = 1120,
            num_bands: int = 64
    ):
        """
        Perceiver model

        TODO Write the documentation for the base model of Perceiver

        :param input_dim: The channel dimension of the input tensor
        :param len_shape: The length of the shape of the input tensor
        :param emb_dim: The channel dimension of the latent tensor
        :param latent_dim: The latent dimension
        :param batch_size: The batch size
        :param num_classes: The number of classes for the classification task
        :param depth: The number of perceiver blocks
        :param latent_blocks: The number of latent blocks for every perceiver block
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

        self.layers = PerceiverBlock(self.emb_dim, self.input_dim, self.heads, self.latent_blocks)

        # Classifier
        self.classifier = Classifier(self.emb_dim, self.num_classes)

    def forward(self, x: Tensor, key_mask: Tensor = None) -> Tensor:
        _, *dims, _ = x.shape

        # Positional encoding
        if self.fourier_encode:
            x = positional_encoding(x, dims, self.batch_size, self.max_freq, self.num_bands)

        # Flatten the input tensor
        x = x.view(x.shape[0], x.shape[1], -1)

        # Change the shape of the input tensor to (M, Batch, Input_dim)
        x = x.permute(1, 0, 2)

        # Compute the perceiver block sharing the weights of the model
        for _ in range(self.depth):
            xx = self.layers(x, self.latent, key_mask=key_mask)    

        # Classifier
        xx = self.classifier(xx)
        return xx
