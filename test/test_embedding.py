import unittest

import torch
from torch import nn

from src.embedding import create_latent_array


class TestPositionalEmbedding(unittest.TestCase):

    def test_byte_array_shape(self):
        """
        Test that the byte array dimension is correct BxCxHxW -> BxCx(H*W)
        M = Height * Width
        BxCxHxW -> BxCxM

        :return:
        """
        batch_size = 2
        channels = 3
        height = 4
        width = 5
        m = height * width

        # Create a tensor of shape (B, C, H, W)
        x = torch.zeros((batch_size, channels, height, width))

        # Reshape the tensor to (B, C, H*W)
        x = x.view(batch_size, channels, m)

        self.assertEqual(x.shape, (batch_size, channels, m))

    def test_latent_array_shape(self):
        """
        Test that the latent array shape is correct
        BxCxM -> BxCxM

        :return:
        """
        batch_size = 32
        d_channels = 3
        latent_dim = 5

        # Create a latent tensor of shape (B, D, N)
        z = create_latent_array(batch_size, d_channels, latent_dim)

        self.assertEqual(z.shape, (batch_size, d_channels, latent_dim))


if __name__ == '__main__':
    unittest.main()
