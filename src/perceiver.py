from torch import nn
from torch import Tensor

from typing import Optional

from src.latent_array import get_latents_array
from src.positional_encoding import ff_positional_encoding
from src.layer import PerceiverBlock, Decoder


class Perceiver(nn.Module):
    """
    From the paper:
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
    """
    def __init__(
        self,
        input_dim: int,
        latent_length: int = 512,
        latent_dim: int = 1024,
        num_classes: int = 40,
        latent_blocks: int = 6,
        heads: int = 8,
        perceiver_block: int = 2,
        share_weights: bool = False,
        ff_pos_encoding: bool = True,
        input_shapes: int = 1,
        max_freq: int = 1120,
        num_bands: int = 64,
    ):
        """
        Initialize the Perceiver model:
            - Latent array
            - n Perceiver blocks
            - Decoder

        Args:
            input_dim (int): The input dimension
            latent_length (int): The latent length. Defaults to 512.
            latent_dim (int): The latent dimension. Defaults to 1024.
            num_classes (int): The number of classes. Defaults to 40.
            latent_blocks (int): The number of latent blocks. Defaults to 6.
            heads (int): The number of heads. Defaults to 8.
            perceiver_block (int): The number of perceiver blocks. Defaults to 2.
            share_weights (bool): Share the weights of the model. Defaults to False.
            ff_pos_encoding (bool): Use Fourier encoding. Defaults to True.
            input_shapes (int): The input shapes. Defaults to 1.
            max_freq (int): The maximum frequency. Defaults to 1120.
            num_bands (int): The number of bands. Defaults to 64.
        """
        super().__init__()
        self.ff_pos_encoding = ff_pos_encoding
        self.max_freq = max_freq
        self.num_bands = num_bands

        # The latent array
        self.latent_array = get_latents_array(latent_length, latent_dim)

        # Change the input dimension if Fourier encoding is used
        # Input_dim = [Input_dim + len(Dims)*(num_bands * 2 + 1)] if Fourier encoding is used
        if self.ff_pos_encoding:
            input_dim = (input_shapes * (num_bands * 2 + 1)) + input_dim
        
        # Perceiver blocks
        # If share_weights is True, we share the weights of the model
        if share_weights:
            pb = PerceiverBlock(latent_dim, input_dim, latent_blocks, heads)
            self.layers = nn.ModuleList([pb for _ in range(perceiver_block)])
        else:
            self.layers = nn.ModuleList([PerceiverBlock(latent_dim, input_dim, latent_blocks, heads) for _ in range(perceiver_block)])

        # Decoder
        self.decoder = Decoder(latent_dim, num_classes)


    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass:
            - Fourier Positional Encoding to the input tensor (if ff_pos_encoding is True)
            - Compute the perceiver blocks
            - Decoder

        Args:
            x (Tensor): input tensor [Batch, [Dims], Channels]
            mask (Tensor, optional): mask tensor. Defaults to None.
        """
        # Fourier Positional Encoding
        if self.ff_pos_encoding:
            x = ff_positional_encoding(x, self.max_freq, self.num_bands)

        # Flatten the input tensor
        # [Batch, [Dims], Channels] -> [Batch, Input_Length, Input_dim]
        x = x.view(x.shape[0], -1, x.shape[-1])

        # Repeat the latent tensor to match the batch size
        # [Emb_length, Emb_dim] -> [Batch, Emb_length, Emb_dim]
        latent = self.latent_array.repeat(x.shape[0], 1, 1)
             
        # Compute the perceiver block 
        for layer in self.layers:
            latent = layer(x, latent, mask)

        # Classifier
        return self.decoder(latent)