import math

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


# Example parameters: shape=(28, 28), bands=8
def fourier_features(shape, bands):
    # This first "shape" refers to the shape of the input data, not the output of this function
    dims = len(shape)  # Number of dimensions in the input data: if shape(28,28), dims=2

    # Every tensor we make has shape: (bands, dimension, x, y, etc...)

    # Pos is computed for the second tensor dimension
    # (aptly named "dimension"), with respect to all
    # following tensor-dimensions ("x", "y", "z", etc.)
    linspaces = [torch.linspace(-1.0, 1.0, steps=n) for n in list(shape)]
    meshgrid = torch.meshgrid(*linspaces)
    list_meshgrid = list(meshgrid)
    pos = torch.stack(list_meshgrid)
    pos = pos.unsqueeze(0).expand((bands,) + pos.shape)

    # Band frequencies are computed for the first
    # tensor-dimension (aptly named "bands") with
    # respect to the index in that dimension
    band_frequencies = (torch.logspace(
        math.log(1.0),
        math.log(shape[0]/2),
        steps=bands,
        base=math.e
    )).view((bands,) + tuple(1 for _ in pos.shape[1:])).expand(pos.shape)

    # For every single value in the tensor, let's compute:
    #             freq[band] * pi * pos[d]

    # We can easily do that because every tensor is the
    # same shape, and repeated in the dimensions where
    # it's not relevant (e.g. "bands" dimension for the "pos" tensor)
    result = (band_frequencies * math.pi * pos).view((dims * bands,) + shape)

    # Use both sin & cos for each band, and then add raw position as well
    # TODO: raw position
    result = torch.cat([
        torch.sin(result),
        torch.cos(result),
    ], dim=0)

    return result


def main():
    # Example usage
    fourier_features((28, 28), 8)


if __name__ == "__main__":
    main()
