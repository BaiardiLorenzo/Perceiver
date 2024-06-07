import torch 
import torch.nn as nn
import numpy as np
import math

from math import pi
from torch import Tensor


def fourier_encode(x: Tensor, max_freq: int, num_bands: int) -> Tensor:
    """
    Encodes the input tensor using Foruier features and concatenates to the original tensor
    
    :param x: tensor to encode
    :param max_freq: maximum frequency
    :param num_bands: number of bands
    :return: tensor with fourier features concatenated
    """

    x = x.unsqueeze(-1) # Add a dimension at the end
    device, dtype, xx = x.device, x.dtype, x

    # Create the sequenze of frequencies equally spaced from 1 to max_freq / 2
    freqs = torch.linspace(1, max_freq / 2, num_bands, device = device, dtype = dtype)

    # Expand the freqs tensor to match the shape of the input tensor
    freqs = freqs.view((1,) * (len(x.shape) - 1) + (-1,))

    # Compute the fourier features
    x = x * freqs * pi

    # Concatenate the fourier features
    x = torch.cat([x.sin(), x.cos()], dim=-1)
    x = torch.cat((x, xx), dim=-1)
    return x


def positional_encoding(x: Tensor, dims: list, batch_size: int, max_freq: int, num_bands: int) -> Tensor:
    # Create a list of positions for each axis
    """
    xd is the value of the input position along the dth dimension (e.g. for images d = 2 and for video d = 3). 
    xd takes values in [−1, 1] for each dimension
    """
    # Create a list of positions for each axis
    pos = torch.stack(list(torch.meshgrid(
        *(torch.linspace(-1, 1, steps=size, dtype=x.dtype) for size in dims),
        indexing="ij",
        )), dim=-1)
    
    # Encode the positions with a fourier feature encoding
    enc_pos = fourier_encode(pos, max_freq, num_bands)
                        
    # Expand the encoded positions to match the shape of the input tensor
    # Tensor of shape (B, Channels, (Dimensions))
    enc_pos = enc_pos.view(*enc_pos.shape[:(len(dims))], -1)
    # Change the position of the final dimension to the first dimension of the tensor
    enc_pos = enc_pos.permute(-1, *range(len(enc_pos.shape) - 1))
    # Expand the tensor to match the shape of the input tensor
    enc_pos = enc_pos.unsqueeze(0).expand(batch_size, *enc_pos.shape)

    # Concatenate the encoded positions to the input tensor
    x = torch.cat((x, enc_pos), dim=1)
    return x
