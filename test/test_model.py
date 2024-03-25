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

        # Create a perceiver model
        perceiver = Perceiver(dim=20, depth=2, latent_blocks=2, latent_dim=5, heads=4, num_classes=10)
        y = perceiver(x, z)

        self.assertEqual(y.shape, (5, 20))

