import unittest

import torch
from torch import nn

from src.perceiver import Perceiver

class TestPerceiver(unittest.TestCase):

    def test_perceiver_forward(self):
        """
        Test the perceiver forward pass
        :return:
        """

        # [batch_size, channels, height, width]
        B, C, H, W = 32, 3, 4, 5

        # [batch_size, emb_dim, latent_dim]
        D, N = 64, 128

        # Create a tensor of shape (B, C, H, W)
        x = torch.zeros((B, C, H, W))

        # Create a perceiver model
        perceiver = Perceiver(
            input_dim=C,
            len_shape=len([H, W]),
            emb_dim=D,
            latent_dim=N,
            batch_size=B,
            num_classes=10,
            depth=2,
            latent_blocks=2,
            heads=4,
            fourier_encode=True,
            max_freq=10,
            num_bands=2
        )
        y = perceiver(x)


if __name__ == '__main__':
    unittest.main()