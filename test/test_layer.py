import unittest
import torch
from torch import nn

from src.latent_array import get_latents_array
from src.layer import DenseBlock, MultiHeadAttention, LatentTransformer, PerceiverBlock


class DenseBlockTest(unittest.TestCase):

    def test_dense_block(self):
        """
        Test the dense block layer

        :return: tensor of correct shape and values
        """
        # [Batch, Length, Channels]
        B, M, C = 32, 20, 3
        
        # ------------------------------------------------
        # Create a dense block
        dense_block = DenseBlock(C)

        # Create a tensor of shape [Batch, Length, Channels]
        x = torch.randn((B, M, C))

        y = dense_block(x)

        # Return a tensor of shape [Batch, Length, Channels]
        self.assertEqual(y.shape, (B, M, C))

        # Assert that the values of the input tensor are not all zeros or NaN
        self.assertTrue(torch.any(y != 0))
        self.assertFalse(torch.any(torch.isnan(y)))


class AttentionBlockTest(unittest.TestCase):

    def test_layer_norm(self):
        """
        Test the layer norm

        :return: tensor of correct shape and values
        """
        # [Batch, Length, Channels]
        B, N, D = 32, 20, 3

        # ------------------------------------------------
        # Create a layer norm
        layer_norm = nn.LayerNorm(D)

        # Create a latent tensor of shape [Batch, Length, Channels]
        z = get_latents_array(N, D)
        z = z.repeat(B, 1, 1)

        # Compute the layer norm
        y = layer_norm(z)

        # Return a tensor of shape [Batch, Length, Channels]
        self.assertEqual(y.shape, (B, N, D))


    def test_create_cross_attention(self):
        """
        Test for the creation of the attention block

        :return: tensor of correct shape and values
        """
        # [Batch, Length, Channels]
        B, M, C = 32, 20, 3
        N, D = 8, 8

        # Create a tensor of shape [Batch, Length, Channels]
        x = torch.randn((B, M, C))

        # Create a latent tensor of shape [Batch, Length, Channels]
        z = get_latents_array(N, D)
        z = z.repeat(B, 1, 1)

        # Normalization before the cross attention
        latent_norm = nn.LayerNorm(D)
        input_norm = nn.LayerNorm(C)

        # Cross attention
        mha = MultiHeadAttention(
            query_dim=D,
            kv_dim=C,
        )

        # Dense block
        dense = DenseBlock(D)
        
        # Normalize the input and latent tensors
        x = input_norm(x)
        z = latent_norm(z)

        # Compute the cross attention
        a = mha(z, x)

        # Return a tensor of shape [Batch, Length, Channels]
        self.assertEqual(a.shape, (B, N, D))

        # Assert that the values of the input tensor are not all zeros or NaN
        self.assertTrue(torch.any(a != 0))
        self.assertFalse(torch.any(torch.isnan(a)))

        # Add residual connection
        res = a + z

        # Compute dense layer
        out = dense(res)

        # Add residual connection
        out = out + res

        # Return a tensor of shape [Batch, Length, Channels]
        self.assertEqual(out.shape, (B, N, D))

        # Assert that the values of the input tensor are not all zeros or NaN
        self.assertTrue(torch.any(out != 0))
        self.assertFalse(torch.any(torch.isnan(out)))


class LatentTransformerBlockTest(unittest.TestCase):

    def test_latent_transformer(self):
        """
        Test the latent transformer

        :return: tensor of correct shape and values
        """
        # [Batch, Length, Channels]
        B, N, D = 32, 64, 16
        
        heads = 8
        latent_blocks = 6

        # Create a attention block
        latent_transformer = LatentTransformer(
            latent_dim=D,
            heads=heads,
            latent_blocks=latent_blocks
        )

        # Create a latent tensor of shape [Batch, Length, Channels]
        z = get_latents_array(N, D)
        z = z.repeat(B, 1, 1)
        
        y = latent_transformer(z)

        # Return a tensor of shape [Batch, Length, Channels]
        self.assertEqual(y.shape, (B, N, D))

        # Assert that the values of the input tensor are not all zeros or NaN
        # print(f"Result: {y}")
        self.assertTrue(torch.any(y != 0))
        self.assertFalse(torch.any(torch.isnan(y)))

    
class PerceiverBlockTest(unittest.TestCase):

    def test_perceiver_block(self):
        """
        Test the Perciever block: Cross-attention and Latent transformer

        :return: tensor of correct shape and values
        """
        # [Batch, Length, Channels]
        B, M, C = 32, 20, 3
        N, D = 8, 8 

        heads = 8
        latent_blocks = 6

        # Create a perceiver block
        perceiver_block = PerceiverBlock(
            latent_dim=D,
            input_dim=C,
            heads=heads,
            latent_blocks=latent_blocks
        )

        # Create a tensor of shape [Batch, Length, Channels]
        x = torch.randn((B, M, C))

        # Create a latent tensor of shape [Batch, Length, Channels]
        z = get_latents_array(N, D)
        z = z.repeat(B, 1, 1)
        
        y = perceiver_block(x, z)

        # Return a tensor of shape [Batch, Length, Channels]
        self.assertEqual(y.shape, (B, N, D))

        # Assert that the values of the input tensor are not all zeros or NaN
        # print(f"Result: {y}")
        self.assertTrue(torch.any(y != 0))
        self.assertFalse(torch.any(torch.isnan(y)))


if __name__ == '__main__':
    unittest.main()
