import unittest

import torch
from torch import nn

from src.perceiver import Perceiver
from src.embedding import create_latent_array


class TestPerceiver(unittest.TestCase):

    def test_perceiver_forward(self):
        """
        Test the perceiver forward pass
        :return:
        """
        B, C, H, W = 32, 3, 4, 5
        M = H * W

        D = 2  # Channel dimension for the latent tensor
        N = 8  # Latent dimension

        # Create a tensor of shape (B, C, M)
        x = torch.zeros((B, C, M))

        # Create a latent tensor of shape (B, D, N)
        z = create_latent_array(B, D, N)

        # Create a perceiver model
        perceiver = Perceiver(
            dim=20,
            depth=2,
            latent_blocks=2,
            latent_dim=5,
            heads=4,
            num_classes=10,
            embed_dim=20,
            fourier_encode=False,
            max_freq=1120,
            num_bands=64,
            batch_size=512
        )
        y = perceiver(x, z)

        self.assertEqual(y.shape, (5, 20))

    
    def test_perceiver_fourier_encode_forward(self):
        """
        Test the perceiver forward pass with fourier encoding
        :return:
        """
        B, C, H, W = 32, 3, 4, 5
        M = H * W

        D = 2  # Channel dimension for the latent tensor
        N = 8  # Latent dimension

        # Create a tensor of shape (B, C, M)
        x = torch.zeros((B, C, M))

        # Create a perceiver model
        perceiver = Perceiver(
            dim=20,
            depth=2,
            latent_blocks=2,
            latent_dim=5,
            heads=4,
            num_classes=10,
            embed_dim=20,
            fourier_encode=True,
            max_freq=1120,
            num_bands=64,
            batch_size=512
        )
        y = perceiver(x)

        self.assertEqual(y.shape, (5, 20))


if __name__ == '__main__':
    unittest.main()