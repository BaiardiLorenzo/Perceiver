import torch
import numpy as np
from torch import Tensor


def fourier_encode(x: Tensor, max_freq: int, num_bands: int = 4) -> Tensor:
    """
    Encodes the input tensor using fourier features

    :param x:
    :param max_freq:
    :param num_bands:
    :return:
    """
    x = x.unsqueeze(-1)  # Add a dimension at the end

    freq_scale = torch.linespace(1., max_freq / 2, num_bands)

    # Compute the fourier features
    x_f = x * freq_scale * np.pi

    x = torch.cat([torch.sin(x_f), torch.cos(x_f)], dim=-1)
    x = torch.cat((x_f, x), dim=-1)
    return x


def fourier_encode(x: Tensor, num_bands: int = 8) -> Tensor:
    """
    Encodes the input tensor using fourier features

    :param x:
    :param num_bands:
    :return:
    """

    res = input * f * np.pi

    return torch.cat([torch.sin(res), torch.cos(res)], dim=-1)