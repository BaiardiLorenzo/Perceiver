import unittest
import torch
from torch import nn, Tensor

from src.embedding import create_latent_array
from src.layer import DenseBlock, AttentionBlock, LatentTransformerBlock, PerceiverBlock


class DenseBlockTest(unittest.TestCase):

    def test_dense_block(self):
        """
        Test the dense block layer

        :return: tensor of correct shape and values
        """
        # [Length, Batch, Channels]
        M, B, C = 20, 32, 3

        # ------------------------------------------------
        # Create a dense block
        dense_block = DenseBlock(C)

        # Create a tensor of shape [Length, Batch, Channels]
        x = torch.randn((M, B, C))
        
        y = dense_block(x)

        # Return a tensor of shape [Length, Batch, Channels]
        self.assertEqual(y.shape, (M, B, C))

        # Assert that the values of the input tensor are not all zeros or NaN
        self.assertTrue(torch.any(x != 0))
        self.assertFalse(torch.any(torch.isnan(x)))
        
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
        # [Length, Batch, Channels]
        N, B, D = 20, 32, 3

        # ------------------------------------------------
        # Create a layer norm
        layer_norm = nn.LayerNorm(D)

        # Create a latent tensor of shape [Length, Batch, Channels]
        z = create_latent_array(N, D, B)

        y = layer_norm(z)

        # Return a tensor of shape [Length, Batch, Channels]
        self.assertEqual(y.shape, (N, B, D))


    def test_create_cross_attention(self):
        """
        Test for the creation of the attention block

        :return: tensor of correct shape and values
        """
        # [Length, Batch, Channels]
        M, B, C = 20, 32, 3
        D, N = 8, 8

        # ------------------------------------------------
        # Normalization before the cross attention
        latent_norm = nn.LayerNorm(D)
        input_norm = nn.LayerNorm(C)

        # Cross attention
        attention = nn.MultiheadAttention(
            embed_dim=D,  
            num_heads=1,
            kdim=C,
            vdim=C
        )

        # Dense block
        dense = DenseBlock(D)

        # Create a tensor of shape [Length, Batch, Channels]
        x = torch.randn((M, B, C))

        # Create a latent tensor of shape [Length, Batch, Channels]
        z = create_latent_array(N, D, B)
        
        x = input_norm(x)
        z = latent_norm(z)

        # Compute the cross attention
        a, _ = attention(query=z, key=x, value=x)

        # Return a tensor of shape [Length, Batch, Channels]
        self.assertEqual(a.shape, (N, B, D))

        # Assert that the values of the input tensor are not all zeros or NaN
        self.assertTrue(torch.any(a != 0))
        self.assertFalse(torch.any(torch.isnan(a)))

        # Add residual connection
        res = a + z

        # Compute dense layer
        out = dense(res)

        # Add residual connection
        out = out + res

        # Return a tensor of shape [Length, Batch, Channels]
        self.assertEqual(out.shape, (N, B, D))

        # Assert that the values of the input tensor are not all zeros or NaN
        self.assertTrue(torch.any(out != 0))
        self.assertFalse(torch.any(torch.isnan(out)))

        # ------------------------------------------------
        # Normalization before the cross attention
        latent_norm = nn.LayerNorm(D)
        input_norm = nn.LayerNorm(C)

        # Cross attention
        attention = nn.MultiheadAttention(
            embed_dim=D,  
            num_heads=1,
            kdim=C,
            vdim=C,
            batch_first=True
        )

        # Dense block
        dense = DenseBlock(D)

        # Create a tensor of shape [Batch, Length, Channels]
        x = torch.randn((B, M, C))

        # Create a latent tensor of shape [Length, Batch, Channels]
        z = create_latent_array(N, D, B)

        # Permute latent tensor to shape [Batch, Length, Channels]
        z = z.permute(1, 0, 2)

        x = input_norm(x)
        z = latent_norm(z)

        # Compute the cross attention
        a, _ = attention(query=z, key=x, value=x)

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


    def test_attention_block(self):
        """
        Test the attention block

        :return: tensor of correct shape and values
        """
        # [Length, Batch, Channels]
        M, B, C = 20, 32, 3
        N, B, D = 64, 32, 16

        # ------------------------------------------------
        # Create an attention block
        attention_block = AttentionBlock(
            emb_dim=D, 
            input_dim=C
        )

        # Create a tensor of shape [Length, Batch, Channels]
        x = torch.randn((M, B, C))

        # Create a latent tensor of shape [Length, Batch, Channels]
        z = create_latent_array(N, D, B)

        y = attention_block(x, z)

        # Return a tensor of shape [Length, Batch, Channels]
        self.assertEqual(y.shape, (N, B, D))

        # Assert that the values of the input tensor are not all zeros or NaN
        self.assertTrue(torch.any(y != 0))
        self.assertFalse(torch.any(torch.isnan(y)))

        # ------------------------------------------------
        # TODO Add the batch_first parameter to the attention block

        # Create an attention block
        # attention_block = AttentionBlock(D, C)

        # Create a tensor of shape [Batch, Length, Channels]
        # x = torch.randn((B, M, C))

        # Create a latent tensor of shape [Length, Batch, Channels]
        # z = create_latent_array(N, D, B)

        # Permute latent tensor to shape [Batch, Length, Channels]
        # z = z.permute(1, 0, 2)

        # y = attention_block(x, z)

        # Return a tensor of shape [Batch, Length, Channels]
        # self.assertEqual(y.shape, (B, M, C))

        # Assert that the values of the input tensor are not all zeros or NaN
        # self.assertTrue(torch.any(x != 0))
        # self.assertFalse(torch.any(torch.isnan(x)))


    def test_attention_key_mask(self):
        """
        Test the MultiheadAttention with key padding mask

        :return: tensor of correct shape and values
        """
        # [Length, Batch, Channels]
        M, B, C = 20, 32, 3
        D, N = 8, 8

        # ------------------------------------------------
        # Create a tensor of shape [Length, Batch, Channels]
        x = torch.randn((M, B, C))

        # Create a latent tensor of shape [Length, Batch, Channels]
        z = create_latent_array(N, D, B)

        # Create a key padding mask
        key_padding_mask = torch.zeros((B, M), dtype=torch.bool)

        # Multihead attention
        attention = nn.MultiheadAttention(
            embed_dim=D,
            num_heads=1,
            kdim=C,
            vdim=C,
        )

        # Compute the cross attention
        a, _ = attention(query=z, key=x, value=x, key_padding_mask=key_padding_mask)

        # Return a tensor of shape [Length, Batch, Channels]
        self.assertEqual(a.shape, (N, B, D))

        # Assert that the values of the input tensor are not all zeros or NaN
        # print(f"Result: {a}")
        self.assertTrue(torch.any(a != 0))
        self.assertFalse(torch.any(torch.isnan(a)))


class LatentTransformerBlockTest(unittest.TestCase):

    def test_latent_transformer(self):
        """
        Test the latent transformer

        :return: tensor of correct shape and values
        """
        # [Length, Batch, Channels]
        N, B, D = 64, 32, 16
        
        heads = 8
        latent_blocks = 6

        # Create a attention block
        latent_transformer = LatentTransformerBlock(
            emb_dim=D, 
            heads=heads, 
            latent_blocks=latent_blocks
        )

        # Create a latent tensor of shape [Length, Batch, Channels]
        z = create_latent_array(N, D, B)
        
        y = latent_transformer(z)

        # Return a tensor of shape [Length, Batch, Channels]
        self.assertEqual(y.shape, (N, B, D))

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
        # [Length, Batch, Channels]
        M, B, C = 20, 32, 3
        N, D = 8, 8 

        heads = 8
        latent_blocks = 6

        # Create a perceiver block
        perceiver_block = PerceiverBlock(
            emb_dim=D, 
            input_dim=C, 
            heads=heads, 
            latent_blocks=latent_blocks
        )

        # Create a tensor of shape [Length, Batch, Channels]
        x = torch.randn((M, B, C))

        # Create a latent tensor of shape [Length, Batch, Channels]
        z = create_latent_array(N, D, B)
        
        y = perceiver_block(x, z)

        # Return a tensor of shape [Length, Batch, Channels]
        self.assertEqual(y.shape, (N, B, D))

        # Assert that the values of the input tensor are not all zeros or NaN
        # print(f"Result: {y}")
        self.assertTrue(torch.any(y != 0))
        self.assertFalse(torch.any(torch.isnan(y)))


if __name__ == '__main__':
    unittest.main()
