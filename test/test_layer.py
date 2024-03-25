import unittest
import torch
from torch import nn

from src.perceiver import DenseBlock, AttentionBlock


class TestDenseBlock(unittest.TestCase):

    def test_output_shape(self):
        """
        Test the dense block
        :return:
        """
        # Create a tensor of shape (C, H, W)
        x = torch.zeros((3, 4, 5))
        x = x.view(x.shape[0], -1)

        # Create a dense block
        dense_block = DenseBlock(20)
        y = dense_block(x)

        self.assertEqual(y.shape, (3, 20))


class TestAttentionBlock(unittest.TestCase):

    def test_latent_output_shape(self):
        """
        Test the attention block
        :return:
        """
        # Create a tensor of shape (C, H, W)
        x = torch.zeros((3, 4, 5))
        x = x.view(x.shape[0], -1)

        # Create an attention block
        attention_block = AttentionBlock(20, 4)
        y = attention_block(x, x)

        self.assertEqual(y.shape, (3, 20))

    def test_cross_attention_output_shape(self):
        """
        Test the attention block
        :return:
        """
        # Create a tensor of shape (C, H, W) -> (C, M)
        x = torch.zeros((3, 4, 5))
        x = x.view(x.shape[0], -1)

        # Create a latent tensor of shape
        latent_dim = 5
        dim = 20
        z = nn.Parameter(torch.nn.init.trunc_normal_(
            torch.zeros(latent_dim, dim),
            mean=0,
            std=0.02,
            a=-2, b=2)
        )

        # Create an attention block
        attention_block = AttentionBlock(20, 4)
        y = attention_block(x, z)

        self.assertEqual(y.shape, (5, 20))