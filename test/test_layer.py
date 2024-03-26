import unittest
import torch
from torch import nn, Tensor

from src.embedding import create_latent_array
from src.layer import DenseBlock, AttentionBlock


class TestFeedForwardBlock(unittest.TestCase):

    def test_output_shape(self):
        """
        Test the dense block
        :return:
        """
        batch_size = 32
        channels = 3
        height = 4
        width = 5
        m = height * width

        # Create a tensor of shape (B, C, M)
        x = torch.rand((batch_size, channels, m))

        # Create a dense block
        dense_block = DenseBlock(m)

        # Return a tensor of shape (B, C, M)
        y = dense_block(x)

        self.assertEqual(y.shape, (batch_size, channels, m))


class TestAttentionBlock(unittest.TestCase):

    def test_layer_norm_output_shape(self):
        """
        Test the layer norm
        :return:
        """
        batch_size = 32
        channels = 3
        n = 20

        # Create a latent tensor of shape (B, D, N)
        z = create_latent_array(batch_size, channels, n)

        # Create a layer norm
        layer_norm = nn.LayerNorm(n)

        # Return a tensor of shape (B, D, N)
        z = layer_norm(z)

        self.assertEqual(z.shape, (batch_size, channels, n))

    def test_latent_output_shape(self):
        """
        Test the attention block
        :return:
        """
        batch_size = 32
        channels = 16
        latent_dim = 8
        heads = 4

        # Create a tensor of shape (B, D, N)
        x = torch.rand((batch_size, channels, latent_dim))

        # Change the shape of tensors (N, B, D)
        x = x.permute(2, 0, 1)

        # Create an attention block
        attention_block = AttentionBlock(channels, channels, heads)

        # Return a tensor of shape (B, D, N)
        y = attention_block(x, x)

        # Change the shape of tensors (B, D, N)
        y = y.permute(1, 2, 0)

        self.assertEqual(y.shape, (batch_size, channels, latent_dim))

    def test_cross_attention_output_shape(self):
        """
        Test the attention block
        :return:
        """
        batch_size = 32
        channels = 3
        height = 4
        width = 5
        m = height * width
        heads = 1

        d_channel = 2
        latent_dim = 8

        # Create a tensor of shape (B, C, M)
        x = torch.zeros((batch_size, channels, m))

        # Create a latent tensor of shape (B, D, N)
        z = create_latent_array(batch_size, d_channel, latent_dim)

        # Change the shape of tensors (N, B, D)
        x = x.permute(2, 0, 1)
        z = z.permute(2, 0, 1)

        # Create an attention block
        attention_block = AttentionBlock(d_channel, channels, heads)
        z = attention_block(x, z)

        # Change the shape of tensors (B, D, N)
        z = z.permute(1, 2, 0)

        self.assertEqual(z.shape, (batch_size, d_channel, latent_dim))


if __name__ == '__main__':
    unittest.main()
