import torch
from torch import nn


def get_latents_array(
        latent_length: int,
        latent_dim: int,
        mean: float = 0,
        std: float = 0.02,
        a: float = -2,
        b: float = 2) -> nn.Parameter:
    """
    Create a latent array of shape [latent_length, latent_dim] for the Perceiver model

    From the paper:
        The latent array is randomly initialized using a truncated normal distribution with
        mean 0, standard deviation 0.02, and truncation bounds [-2, 2].

    Args:
        latent_length (int): The length of the latent array
        latent_dim (int): The dimension of the latent array
        mean (float, optional): The mean of the truncated normal distribution. Defaults to 0.
        std (float, optional): The standard deviation of the truncated normal distribution. Defaults to 0.02.
        a (float, optional): The lower bound of the truncated normal distribution. Defaults to -2.
        b (float, optional): The upper bound of the truncated normal distribution. Defaults to 2.
    
    Returns:
        nn.Parameter: The latent array
    """
    return nn.Parameter(
        torch.nn.init.trunc_normal_(
            torch.zeros(latent_length, latent_dim),
            mean=mean,
            std=std,
            a=a,
            b=b
        )
    )