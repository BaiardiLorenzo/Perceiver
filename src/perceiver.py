import torch

from torch import nn
from torch import Tensor

from src.embedding import create_latent_array
from src.positional_embedding import positional_embedding
from src.layer import PerceiverBlock, Classifier


class Perceiver(nn.Module):

    def __init__(
            self,
            input_dim: int,
            len_shape: int = 1,
            emb_dim: int = 512,
            latent_dim: int = 512,
            num_classes: int = 40,
            depth: int = 2,
            latent_blocks: int = 6,
            heads: int = 8,
            fourier_encode: bool = True,
            max_freq: int = 1120,
            num_bands: int = 64
    ):
        """
        Perceiver model:
        - Latent Array
        - Positional Encoding
        - Perceiver Block: Cross-Attention and Transformer
        - Classifier

        We build our architecture from two components:
        (i) a cross-attention module that maps a byte array (e.g. an
        pixel array) and a latent array to a latent array, and (ii) a
        Transformer tower that maps a latent array to a latent array.
        The size of the byte array is determined by the input data
        and is generally large (e.g. ImageNet images at resolution
        224 have 50,176 pixels), while the size of the latent array
        is a hyperparameter which is typically much smaller (e.g.
        we use 512 latents on ImageNet). Our model applies the
        cross-attention module and the Transformer in alternation.
        This corresponds to projecting the higher-dimensional byte
        array through a lower-dimension attention bottleneck before
        processing it with a deep Transformer, and then using the
        resulting representation to query the input again. The model
        can also be seen as performing a fully end-to-end clustering
        of the inputs with latent positions as cluster centres, lever
        aging a highly asymmetric cross-attention layer. Because
        we optionally share weights between each instance of the
        Transformer tower (and between all instances of the cross
        attention module but the first), our model can be interpreted
        as a recurrent neural network (RNN), but unrolled in depth
        using the same input, rather than in time. 
        All attention modules in the Perceiver are non-causal: we use no masks.

        :param input_dim: The channel dimension of the input tensor
        :param len_shape: The length of the shape of the input tensor (default: 1)
        :param emb_dim: The channel dimension of the latent tensor
        :param latent_dim: The latent dimension
        :param num_classes: The number of classes for the classification task
        :param depth: The number of perceiver blocks
        :param latent_blocks: The number of latent blocks for every perceiver block
        :param heads: The number of heads in the multi-head attention in the perceiver block
        :param fourier_encode: Whether to use Fourier encoding (default: True)
        :param max_freq: The maximum frequency for Fourier encoding
        :param num_bands: The number of bands for Fourier encoding
        """
        super().__init__()
        self.input_dim = input_dim
        self.len_shape = len_shape
        self.emb_dim = emb_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.depth = depth
        self.latent_blocks = latent_blocks
        self.heads = heads
        self.fourier_encode = fourier_encode
        self.max_freq = max_freq
        self.num_bands = num_bands

        # The latent array
        self.latent = create_latent_array(self.latent_dim, self.emb_dim)

        # Add the Fourier encoding to the input tensor
        if fourier_encode:
            self.input_dim = (self.len_shape * (num_bands * 2 + 1)) + self.input_dim

        # Perceiver block
        self.layers = PerceiverBlock(self.emb_dim, self.input_dim, self.heads, self.latent_blocks)

        # Classifier
        self.classifier = Classifier(self.emb_dim, self.num_classes)

    def forward(self, x: Tensor, key_mask: Tensor=None) -> Tensor:
        """
        Forward pass:
        - Positional Encoding
        - Flatten the input tensor 
        - Repeat the latent tensor to match the batch size
        - Compute the perceiver block sharing the weights of the model
        - Classifier

        :param x: input tensor [Batch, [Dims], Channels]
        :param key_mask: key padding mask for the unbatched input tensor
        :return: output tensor [Batch, Num_classes]
        """

        # Positional encoding
        if self.fourier_encode:
            x = positional_embedding(x, self.max_freq, self.num_bands)

        # Flatten the input tensor
        x = x.view(x.shape[0], x.shape[1], -1)

        # Repeat the latent tensor to match the batch size
        latent = self.latent.expand(-1, x.shape[0], -1)

        # Change the shape of the input tensor to [Length, Batch, Input_dim]
        x = x.permute(1, 0, 2)

        # Compute the perceiver block sharing the weights of the model
        for _ in range(self.depth):
            xx = self.layers(x, latent, key_mask=key_mask)    

        # Classifier
        xx = self.classifier(xx)
        return xx
