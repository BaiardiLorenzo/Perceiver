import unittest
import torch
from torch import nn

from src.perceiver import DenseBlock, AttentionBlock, Perceiver


class TestPositionalEmbedding(unittest.TestCase):

    def test_byte_array_shape(self):
        """
        Test that the byte array dimension is correct CxHxW -> CxH*W
        C = channels, M = Height * Width
        :return:
        """
        # Create a tensor of shape (C, H, W)
        x = torch.zeros((3, 4, 5))

        # Flatten the tensor to (C, H*W)
        x = x.view(x.shape[0], -1)

        self.assertEqual(x.shape, (3, 20))


if __name__ == '__main__':
    unittest.main()
