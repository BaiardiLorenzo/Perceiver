import unittest
import torch
from torch import nn, Tensor

from src.embedding import create_latent_array
from src.layer import DenseBlock, AttentionBlock, LatentBlock, PerceiverBlock


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

    def test_create_cross_attention(self):
        """
        Test for the creation of the attention block
        :return:
        """
        B, C, H, W = 32, 3, 4, 5
        M = H * W
        heads = 1

        D = 2  # Channel dimension for the latent tensor
        N = 8  # Latent dimension

        # Create a tensor of shape (B, C, M)
        x = torch.zeros((B, C, M))

        # Create a latent tensor of shape (B, D, N)
        z = create_latent_array(B, D, N)

        # Normalization before the cross attention
        latent_norm = nn.LayerNorm(N)
        input_norm = nn.LayerNorm(M)

        # Cross attention
        attention = nn.MultiheadAttention(
            N,
            1,
            kdim=M,
            vdim=M,
            dropout=0.0,
            bias=False,
            batch_first=True
        )
        
        x = input_norm(x)
        z = latent_norm(z)

        # Compute the cross attention
        a, _ = attention(query=z, key=x, value=x)

        # Add residual connection
        res = a + z

        self.assertEqual(res.shape, (B, D, N))

        dense = DenseBlock(N)

        # Compute dense layer
        out = dense(res)

        # Add residual connection
        out = out + res

        self.assertEqual(out.shape, (B, D, N))

    
    def test_attention_block(self):
        """
        Test the attention block
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

        # Create an attention block
        attention_block = AttentionBlock(N, M, heads=1)

        # Return a tensor of shape (B, D, N)
        y = attention_block(x, z)

        self.assertEqual(y.shape, (B, D, N))

    
    def test_latent_block(self):
        """
        Test the latent block
        :return:
        """
        B, D, N = 32, 2, 8

        # Create a latent tensor of shape (B, D, N)
        z = create_latent_array(B, D, N)

        # Normalize the latent tensor
        latent_norm = nn.LayerNorm(N)
        z = latent_norm(z)

        # Create a attention block
        attention_block = LatentBlock(N, heads=1, latent_blocks=6)

        # Return a tensor of shape (B, D, N)
        y = attention_block(z)

        self.assertEqual(y.shape, (B, D, N))

    
    def test_perceiver_block(self):
        """
        Test the Perciever block: Cross-attention and Latent transformer
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

        # Create a perceiver block
        perceiver_block = PerceiverBlock(N, M, heads=8, latent_blocks=6)

        # Return a tensor of shape (B, D, N)
        y = perceiver_block(x, z)

        self.assertEqual(y.shape, (B, D, N))



if __name__ == '__main__':
    unittest.main()
