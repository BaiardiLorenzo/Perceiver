from torch import nn
from torch import Tensor

from src.embedding import PositionalFourierEmbedding, create_latent_array
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
        self.num_bands = num_bands
        self.num_classes = num_classes
        self.batch_size = batch_size

        self.positional_embedding = PositionalFourierEmbedding(dim, embed_dim=self.embed_dim, num_bands=self.num_bands)

        # The latent array
        self.latent = create_latent_array(self.batch_size, self.latent_dim, self.dim)

        # Perceiver block -> @TODO Share the weights for every block
        self.layers = nn.ModuleList([
            PerceiverBlock(dim, heads, latent_blocks)
            for _ in range(depth)
        ])

        # self.block = PerceiverBlock(dim, heads, latent_blocks)

        # Classifier
        self.classifier = Classifier(dim, num_classes)

    def forward(self, x: Tensor):
        # Positional encoding
        x = self.positional_embedding(x)

        # Compute the perceiver block
        # @TODO Share the weights for every block
        for layer in self.layers:
            x = layer(x, self.latent)

        # Compute the perceiver block share the weights
        # for _ in range(self.depth):
        #     x = self.block(x, self.latent)

        # Classifier
        x = self.classifier(x)
        return x
