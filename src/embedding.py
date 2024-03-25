import math
import torch
from torch import nn
import numpy as np
from torch import Tensor


class PositionalFourierEmbedding(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int, num_bands: int = 4):
        """
        Positional Fourier Embedding

        :param input_dim:
        :param embed_dim:
        :param num_bands:
        """
        super().__init__()
        self.shape = input_dim
        self.embed_dim = embed_dim
        self.num_bands = num_bands

        # Compute the fourier features
        self.fourier_features = self.fourier_features(self.input_dim, self.num_bands)
        self.linear = nn.Linear(self.input_dim + self.fourier_features.shape[0], self.num_bands)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass

        :param x:
        :return:
        """
        # Flatten the input tensor
        x = x.view(x.shape[0], -1)

        # Add the fourier features
        x = torch.cat([x, self.fourier_features], dim=0)

        # Apply the linear layer
        x = self.linear(x)

        return x

    def fourier_features(self, shape, bands):
        # This first "shape" refers to the shape of the input data, not the output of this function
        dims = len(shape)

        # Every tensor we make has shape: (bands, dimension, x, y, etc...)

        # Pos is computed for the second tensor dimension
        # (aptly named "dimension"), with respect to all
        # following tensor-dimensions ("x", "y", "z", etc.)
        pos = torch.stack(
            list(
                torch.meshgrid(
                    *(torch.linspace(-1.0, 1.0, steps=n) for n in list(shape))
                )
            )
        )
        pos = pos.unsqueeze(0).expand((bands,) + pos.shape)

        # Band frequencies are computed for the first
        # tensor-dimension (aptly named "bands") with
        # respect to the index in that dimension
        band_frequencies = (
            (
                torch.logspace(
                    math.log(1.0), math.log(shape[0] / 2), steps=bands, base=math.e
                )
            )
            .view((bands,) + tuple(1 for _ in pos.shape[1:]))
            .expand(pos.shape)
        )

        # For every single value in the tensor, let's compute:
        #             freq[band] * pi * pos[d]

        # We can easily do that because every tensor is the
        # same shape, and repeated in the dimensions where
        # it's not relevant (e.g. "bands" dimension for the "pos" tensor)
        result = (band_frequencies * math.pi * pos).view((dims * bands,) + shape)

        # Use both sin & cos for each band, and then add raw position as well
        # TODO: raw position
        result = torch.cat(
            [
                torch.sin(result),
                torch.cos(result),
            ],
            dim=0,
        )

        return result


def test_fourier_encode(x: Tensor, max_freq: int, num_bands: int = 4) -> Tensor:
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
    return x
