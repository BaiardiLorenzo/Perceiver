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
        B, M, C = 32, 20, 3

        # Create a tensor of shape (B, M, C)
        x = torch.zeros((B, M, C))

        # Create a dense block
        dense_block = DenseBlock(C)

        # Return a tensor of shape (B, M, C)
        y = dense_block(x)

        self.assertEqual(y.shape, (B, M, C))
        
        # Create a tensor of shape (M, B, C)
        x = torch.zeros((M, B, C))

        # Return a tensor of shape (M, B, C)
        y = dense_block(x)

        self.assertEqual(y.shape, (M, B, C))


class TestAttentionBlock(unittest.TestCase):

    def test_layer_norm_output_shape(self):
        """
        Test the layer norm
        :return:
        """
        # [batch_size, channels, dim]
        B, D, N = 32, 3, 20

        # Create a latent tensor of shape (B, D, N)
        z = create_latent_array(B, D, N)

        # Permute the tensor to shape (N, B, D)
        z = z.permute(2, 0, 1)

        # Create a layer norm
        layer_norm = nn.LayerNorm(D)

        # Return a tensor of shape (N, B, D)
        z = layer_norm(z)

        self.assertEqual(z.shape, (N, B, D))


    def test_latent_output_shape(self):
        """
        Test the attention block
        :return:
        """
        # [batch_size, channels, dim]
        B, D, N = 32, 16, 8
        heads = 8

        # Create a tensor of shape (B, D, N)
        x = torch.randn((B, D, N))

        # Change the shape of tensors (N, B, D)
        x = x.permute(2, 0, 1)

        # Create an attention block
        attention_block = AttentionBlock(D, D, heads=heads)

        # Return a tensor of shape (N, B, D)
        y = attention_block(x, x)

        self.assertEqual(y.shape, (N, B, D))


    def test_create_cross_attention(self):
        """
        Test for the creation of the attention block
        :return:
        """
        # [batch_size, channels, dim]
        B, C, M = 32, 3, 20

        # [batch_size, channels, dim]
        D, N = 2, 8

        # Create a tensor of shape (M, B, C)
        x = torch.zeros((M, B, C))

        # Create a latent tensor of shape (B, D, N)
        z = create_latent_array(B, D, N)

        # Permute the tensor to shape (N, B, D)
        z = z.permute(2, 0, 1)

        # Normalization before the cross attention
        latent_norm = nn.LayerNorm(D)
        input_norm = nn.LayerNorm(C)

        # Cross attention
        attention = nn.MultiheadAttention(
            D,  # Embedding dimension
            1,
            kdim=C,
            vdim=C,
            dropout=0.0,
            bias=False,
        )
        
        x = input_norm(x)
        z = latent_norm(z)

        # Compute the cross attention
        a, _ = attention(query=z, key=x, value=x)

        # Add residual connection
        res = a + z

        self.assertEqual(res.shape, (N, B, D))

        dense = DenseBlock(D)

        # Compute dense layer
        out = dense(res)

        # Add residual connection
        out = out + res

        self.assertEqual(out.shape, (N, B, D))


    def test_attention_block(self):
        """
        Test the attention block
        :return:
        """
        # [batch_size, channels, dim]
        B, C, M = 32, 64, 20

        # [batch_size, channels, dim]
        D, N = 8, 8  

        # Create a tensor of shape (M, B, C)
        x = torch.zeros((M, B, C))

        # Create a latent tensor of shape (B, D, N)
        z = create_latent_array(B, D, N)

        # Permute the tensor to shape (N, B, D)
        z = z.permute(2, 0, 1)

        # Create an attention block
        attention_block = AttentionBlock(D, C, heads=8)

        # Return a tensor of shape (N, B, D)
        y = attention_block(x, z)

        self.assertEqual(y.shape, (N, B, D))

    
    def test_latent_block(self):
        """
        Test the latent block
        :return:
        """
        # [batch_size, channels, dim]
        B, D, N = 32, 8, 64

        # Create a latent tensor of shape (B, D, N)
        z = create_latent_array(B, D, N)

        # Permute the tensor to shape (N, B, D)
        z = z.permute(2, 0, 1)

        # Normalize the latent tensor
        latent_norm = nn.LayerNorm(D)

        z = latent_norm(z)

        # Create a attention block
        attention_block = LatentBlock(D, heads=8, latent_blocks=6)

        # Return a tensor of shape (N, B, D)
        y = attention_block(z)

        self.assertEqual(y.shape, (N, B, D))

    
    def test_perceiver_block(self):
        """
        Test the Perciever block: Cross-attention and Latent transformer
        :return:
        """
        # [batch_size, channels, dim]
        B, C, M = 32, 8, 20

        # [batch_size, channels, dim]
        D, N = 8, 8 

        # Create a tensor of shape (M, B, C)
        x = torch.zeros((M, B, C))

        # Create a latent tensor of shape (B, D, N)
        z = create_latent_array(B, D, N)

        # Permute the tensor to shape (N, B, D)
        z = z.permute(2, 0, 1)

        # Create a perceiver block
        perceiver_block = PerceiverBlock(D, C, heads=8, latent_blocks=6)

        # Return a tensor of shape (N, B, D)
        y = perceiver_block(x, z)

        self.assertEqual(y.shape, (N, B, D))



if __name__ == '__main__':
    unittest.main()
