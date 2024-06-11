import torch 

from math import pi
from torch import Tensor


def fourier_encode(x: Tensor, max_freq: int, num_bands: int) -> Tensor:
    """
    Encodes the input tensor using Fourier features and concatenates to the original tensor

    We parametrize the frequency encoding to take the values
    [sin(f_k*pi*x_d), cos(f_k*pi*x_d)], where the frequency f_k is the k-th
    band of a bank of frequencies spaced equally between 1 and u/2. 
    u/2 can be naturally interpreted as the Nyquist frequency corresponding 
    to a target sampling rate of u.
    
    :param x: tensor to compute the fourier features
    :param max_freq: maximum frequency
    :param num_bands: number of bands
    :return: tensor with fourier features concatenated
    """

    # Add the dimension for the fourier channel
    x = x.unsqueeze(-1) 
    xx = x

    # Create the sequenze of frequencies equally spaced from 1 to max_freq / 2
    freqs = torch.linspace(1, max_freq / 2, num_bands, dtype=x.dtype, device=x.device)

    # Expand the freqs tensor to match the shape of the input tensor
    freqs = freqs.view((1,) * (len(x.shape) - 1) + (-1,))

    # Compute the fourier features
    x = x * freqs * pi

    # Concatenate the fourier features
    x = torch.cat([x.sin(), x.cos()], dim=-1)
    x = torch.cat((x, xx), dim=-1)
    return x


def positional_embedding(x: Tensor, max_freq: int, num_bands: int) -> Tensor:
    """
    Add positional embedding to the input tensor using Fourier encoding

    xd is the value of the input position along the dth dimension (e.g. for images d = 2 and for video d = 3). 
    xd takes values in [−1, 1] for each dimension.
    For signals with irregular or very fine sampling, such as ModelNet40 point clouds, the maximum band can also be 
    treated as a hyperparameter.

    :param x: input tensor of the shape [Batch, [Dims], Channels]
    :param max_freq: maximum frequency
    :param num_bands: number of bands
    :return: tensor with positional embedding [Batch, [Dims], Channels + len(Dims)*(num_bands * 2 + 1)]
    """
    batch_size, *dims, _, dtype, device = *x.shape, x.dtype, x.device

    # Create a list of positions for each axis
    xd = torch.stack(list(torch.meshgrid(
        *(torch.linspace(-1, 1, steps=dim, dtype=dtype, device=device) for dim in dims),
        indexing="ij",
        )), dim=-1)
    
    # Encode the positions with a fourier feature encoding
    enc_pos = fourier_encode(xd, max_freq, num_bands)

    # Expand the encoded positions to match the shape of the input tensor: 
    # [[Dims], len(Dims)*(num_bands * 2 + 1)]
    enc_pos = enc_pos.view(*enc_pos.shape[:(len(dims))], -1)

    # [Batch, [Dims], len(Dims)*(num_bands * 2 + 1)]
    enc_pos = enc_pos.unsqueeze(0).expand(batch_size, *enc_pos.shape)

    # Concatenate the encoded positions to the input tensor
    x = torch.cat((x, enc_pos), dim=-1)
    return x
