import unittest
import os 
import sys
import torch
import math

from torch import nn
from src.fourier_encode import fourier_encode
from src.fourier_encode import positional_encoding

class TestFourierEncode(unittest.TestCase):

    def test_fourier_encode(self):
        """
        Test that the fourier_encode function returns the correct shape of the positional embedding


        :return:
        """
        max_freq = 1120
        num_bands = 64

        B, C, H, W = 32, 3, 4, 5
        M = H * W

        # Create a tensor of shape (B, C, H, W)
        # x = torch.randn((B, C, H, W))
        x = torch.randn((B, C, M))

        dtype = x.dtype
        b, _, *dims = x.shape # (B, C, (DIMENSIONS))

        # Create a list of positions for each axis
        pos = torch.stack(list(torch.meshgrid(
            *(torch.linspace(-1, 1, steps=size, dtype=dtype) for size in dims),
            indexing="ij",
            )), dim=-1)
        
        # (len(Dims), Dims)
        test_dim = torch.Size(dims + [len(dims)]) 
        print("Position: ", pos.shape, test_dim) 
        self.assertEqual(pos.shape, test_dim)
        
        # Encode the positions with a fourier feature encoding
        enc_pos = fourier_encode(pos, max_freq, num_bands)

        # (len(Dims), Dims, num_bands * 2 + 1)
        test_dim = torch.Size(dims + [len(dims)] + [num_bands * 2 + 1]) 
        print("Encoded Position: ", enc_pos.shape, test_dim)
        self.assertEqual(enc_pos.shape, test_dim)
                         
        # Expand the encoded positions to match the shape of the input tensor
        # Tensor of shape (B, Channels, (Dimensions))
        enc_pos = enc_pos.view(*enc_pos.shape[:(len(dims))], -1)
        print("Expanded Encoded Position: ", enc_pos.shape)
        # Change the position of the final dimension to the first dimension of the tensor
        enc_pos = enc_pos.permute(-1, *range(len(enc_pos.shape) - 1))
        print("Expanded Encoded Position: ", enc_pos.shape)
        # Expand the tensor to match the shape of the input tensor
        enc_pos = enc_pos.unsqueeze(0).expand(b, *enc_pos.shape)

        # (Batch, (num_bands * 2 + 1) * len(Dims), [Dims])
        test_dim = torch.Size([b] + [(num_bands * 2 + 1)*len(dims)] + dims)
        print("Expanded Encoded Position: ", enc_pos.shape, test_dim)
        self.assertEqual(enc_pos.shape, test_dim)

        # Concatenate the encoded positions to the input tensor
        x = torch.cat((x, enc_pos), dim=1)

        # (Batch, (num_bands * 2 + 1) * len(Dims) + Channels, [Dims])
        test_dim = torch.Size([b] + [(num_bands * 2 + 1) * len(dims) + 3] + dims)
        print("Result: ", x.shape, test_dim)
        self.assertEqual(x.shape, test_dim)

    
    def test_positional_embedding(self):
        """
        Test the positional embedding function

        :return:
        """
        B, C, H, W = 32, 3, 4, 5
        M = H * W

        # x = torch.randn((B, C, H, W))
        x = torch.randn((B, C, M))
        batch, channels, *dims = x.shape

        x = positional_encoding(x, dims, batch, 1120, 64)
        print("Positional Encoding: ", x.shape)
        self.assertEqual(x.shape, torch.Size([batch, channels + 64 * 2 + 1, M]))


if __name__ == '__main__':
    unittest.main()