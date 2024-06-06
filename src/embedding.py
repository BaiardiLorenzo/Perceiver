import torch
from torch import nn


def create_latent_array(
        batch_size: int,
        channel_dim: int,
        latent_dim: int,
        mean: float = 0,
        std: float = 0.02,
        a: float = -2,
        b: float = 2) -> nn.Parameter:
    """
    Create a latent array

    The latent array is randomly initialized using a truncated normal distribution with
    mean 0, standard deviation 0.02, and truncation bounds [-2, 2].

    :param batch_size: B
    :param channel_dim: D
    :param latent_dim: N
    :param mean: default value is 0
    :param std: default value is 0.02
    :param a: lower bound, default value is -2
    :param b: upper bound, default value is 2
    :return:
    """
    return nn.Parameter(
        torch.nn.init.trunc_normal_(
            torch.zeros(batch_size, channel_dim, latent_dim),
            mean=mean,
            std=std,
            a=a, b=b
        )
    )


